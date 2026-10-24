#!/usr/bin/env python
"""Greek invoice → structured JSON, OCR-first.

The slow version asked a vision model to read the whole page (~19 s). That was
only ever necessary because PaddleOCR was assumed to have no Greek. It does:
`el_PP-OCRv5_mobile_rec` reads Greek text lines in about half a second a page.

So the work splits properly now:

  * el_PP-OCRv5  reads the Greek words and gives every line a box   (~0.5 s)
  * geometry     turns labels + boxes into fields (label → value to its right)
  * PP-OCRv5     the Latin recogniser, a second opinion on every digit
  * arithmetic   the document proving itself (net + ΦΠΑ = σύνολο, …)
  * the VLM      wakes up ONLY for fields the page could not prove

A clean invoice never touches the vision model and lands in about a second.
A damaged one costs the old 19 s — which is the right way round.
"""
import argparse
import difflib
import json
import math
import os
import re
import sys
import time
import unicodedata

from extract import (FIELDS, LOOKALIKE, arithmetic_checks, ask_vlm,
                     confirm_against_ocr, digits, fold, norm_num, paddle_lines,
                     parse_json)

EL_MODEL = os.environ.get("EL_REC_MODEL", "el_PP-OCRv5_mobile_rec")
# below this median line score the page is damaged enough that the words, not
# the numbers, are what need a second opinion (measured: fax/photocopy/bad JPEG
# land under it, phone photo and crooked scan stay above)
TEXT_VERIFY_BELOW = float(os.environ.get("TEXT_VERIFY_BELOW", "0.995"))
NUM_RE = re.compile(r"-?\d[\d.,]*")
# an amount may carry a thousands separator that printed as a space ("1 185,69"),
# and must never be confused with the VAT RATE sitting on the same line ("24%")
MONEY_RE = re.compile(r"-?\d{1,3}(?:[.\s]\d{3})+(?:,\d{1,2})?(?!\s*%)"
                      r"|-?\d+(?:[.,]\d{1,2})?(?!\s*%)")
# the Greek recogniser knows Ο and 0 are different letters and still mixes them
# up inside numbers (ΑΦΜ 3868334Ο7). Only touch runs that are already numeric.
DIGIT_LOOKALIKE = str.maketrans({"Ο": "0", "O": "0", "ο": "0", "о": "0",
                                 "Ι": "1", "I": "1", "l": "1", "|": "1"})
DIGITISH = re.compile(r"[0-9ΟOοоΙIl|][0-9ΟOοоΙIl|.,]{2,}")


def fix_digits(t):
    return DIGITISH.sub(lambda m: m.group(0).translate(DIGIT_LOOKALIKE), str(t or ""))

DATE_RE = re.compile(r"\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b")
VAT9_RE = re.compile(r"\b(\d{9})\b")


# ---------------------------------------------------------------- the reader

def greek_lines(path, cache=True):
    """every text line on the page, with its box; cached per image"""
    cf = path + ".el.json"
    if cache and os.path.exists(cf):
        return json.load(open(cf))
    from paddleocr import PaddleOCR
    global _OCR
    try:
        ocr = _OCR
    except NameError:
        ocr = _OCR = PaddleOCR(
            text_recognition_model_name=EL_MODEL,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=os.environ.get("PADDLE_DEVICE", "gpu:0"),
        )
    r = ocr.predict(path)[0].json["res"]
    out = []
    for poly, txt, sc in zip(r["dt_polys"], r["rec_texts"], r["rec_scores"]):
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        # the top edge tells us how crooked the page is
        (ax, ay), (bx, by) = [tuple(map(float, p)) for p in poly[:2]]
        ang = math.atan2(by - ay, bx - ax) if bx != ax else 0.0
        out.append({"text": txt, "score": float(sc), "ang": ang,
                    "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)})
    out.sort(key=lambda l: (round(l["y0"] / 10), l["x0"]))
    if cache:
        json.dump(out, open(cf, "w"), ensure_ascii=False)
    return out


def pdf_lines(path, dpi=200):
    """A PDF from accounting software already contains its text, with exact
    coordinates. Reading that layer is both faster than OCR and exact — the same
    geometry parser runs on it unchanged. Returns [] for a scanned PDF."""
    import subprocess
    import xml.etree.ElementTree as ET
    try:
        xml = subprocess.run(["pdftotext", "-bbox-layout", path, "-"],
                             capture_output=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    k = dpi / 72.0
    out = []
    for i, page in enumerate(root.iter("{http://www.w3.org/1999/xhtml}page")):
        off = i * 4000          # stack pages so one parse sees them in order
        for ln in page.iter("{http://www.w3.org/1999/xhtml}line"):
            words = [w.text or "" for w in ln.findall("x:word", ns)]
            txt = " ".join(w for w in words if w).strip()
            if not txt:
                continue
            out.append({"text": txt, "score": 1.0, "ang": 0.0,
                        "x0": float(ln.get("xMin")) * k,
                        "y0": float(ln.get("yMin")) * k + off,
                        "x1": float(ln.get("xMax")) * k,
                        "y1": float(ln.get("yMax")) * k + off})
    out.sort(key=lambda l: (round(l["y0"] / 10), l["x0"]))
    return out


def pdf_to_png(path, dpi=200, out_dir=None):
    """a scanned PDF still has to be rendered before anything can read it"""
    import glob as _glob
    import subprocess
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(path)), "_pages")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, os.path.basename(path)[:-4])
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", path, stem], check=True)
    return sorted(_glob.glob(stem + "-*.png"))


def reading_order(lines, min_gutter=28, min_band=10):
    """Put the lines in the order a human reads them.

    UNVALIDATED, and nothing calls it yet: pdftotext cannot referee it, because
    its own output is not in reading order either. Keep it off until there is a
    document set whose true order is known.

    A gazette page, a contract and most invoices are laid out in columns, and
    OCR hands back boxes in raster order — so the right-hand column interleaves
    with the left and every sentence is cut in half. This is the classic XY-cut:
    slice the page at horizontal gaps into bands, slice each band at vertical
    gutters into columns, recurse. A full-width heading simply forms its own
    band and stays where it belongs.
    """
    def cut(block, depth=0):
        if len(block) <= 1 or depth > 6:
            return sorted(block, key=lambda l: (rowy(l), l["x0"]))
        # horizontal gaps first: bands
        bands, cur = [], []
        for l in sorted(block, key=rowy):
            if cur and rowy(l) - max(x["y1"] for x in cur) > min_band:
                bands.append(cur)
                cur = []
            cur.append(l)
        bands.append(cur)
        if len(bands) > 1:
            return [l for b in bands for l in cut(b, depth + 1)]
        # no band split: look for a vertical gutter
        xs = sorted(block, key=lambda l: l["x0"])
        edge, gut = xs[0]["x1"], None
        for l in xs[1:]:
            if l["x0"] - edge > min_gutter:
                gut = (edge + l["x0"]) / 2
                break
            edge = max(edge, l["x1"])
        if gut is None:
            return sorted(block, key=lambda l: (rowy(l), l["x0"]))
        left = [l for l in block if (l["x0"] + l["x1"]) / 2 < gut]
        right = [l for l in block if (l["x0"] + l["x1"]) / 2 >= gut]
        if not left or not right:
            return sorted(block, key=lambda l: (rowy(l), l["x0"]))
        return cut(left, depth + 1) + cut(right, depth + 1)

    for l in lines:
        l.setdefault("yc", (l["y0"] + l["y1"]) / 2)
    return cut(list(lines))


# ------------------------------------------------------------------ geometry

def lab(s):
    """a label, reduced to something matchable.

    Accents and punctuation go, and Greek capitals fold onto their Latin
    lookalikes — a recogniser that prints ΑΦΜ as "A.Φ.M." is not making a
    mistake a human would notice, and must not cost us the field.
    """
    s = unicodedata.normalize("NFD", str(s or "").upper())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.translate(LOOKALIKE)
    return re.sub(r"[^A-ZΑ-Ω]", "", s)


def has_label(line, *wanted, cutoff=0.78):
    """Does this line carry one of these labels, allowing for OCR slips.

    Fuzzy matching only earns its keep on labels long enough to be distinctive:
    sliding a 5-letter window over a long line will hit 0.8 similarity by
    accident, which is how ΤΜΗΜΑ once matched "Αριθμός Ερωτήματος:66/2002".
    Short labels must appear literally.
    """
    t = lab(line["text"])
    for w in wanted:
        w = lab(w)
        if not w:
            continue
        if w in t:
            return True
        if len(w) < 7:
            continue
        for i in range(0, max(1, len(t) - len(w) + 1)):
            if difflib.SequenceMatcher(None, w, t[i:i + len(w)]).ratio() >= cutoff:
                return True
    return False


def lab_keep_digits(s):
    """like lab(), but numbers survive — for "27ης Μαΐου 2020" """
    s = unaccent(s).translate(LOOKALIKE)
    return re.sub(r"[^A-ZΑ-Ω0-9]", " ", s)


def unaccent(s):
    """accents and case gone, everything else — digits included — kept"""
    s = unicodedata.normalize("NFD", str(s or "").upper())
    return "".join(c for c in s if not unicodedata.combining(c))


def find(lines, *wanted, after=None, **kw):
    for l in lines:
        if after is not None and rowy(l) < after:
            continue
        if has_label(l, *wanted, **kw):
            return l
    return None


def deskew(lines):
    """A page fed crooked into a scanner puts the right-hand column half a row
    off from the left-hand one, and the table silently shifts by a row. Rather
    than rotate the image and read it twice, measure the skew from the text
    boxes and compare rows along that slope."""
    angs = sorted(l["ang"] for l in lines if l["x1"] - l["x0"] > 40 and abs(l["ang"]) < 0.2)
    th = angs[len(angs) // 2] if angs else 0.0
    if abs(th) < 0.002:
        th = 0.0
    for l in lines:
        cx, cy = (l["x0"] + l["x1"]) / 2, (l["y0"] + l["y1"]) / 2
        l["yc"] = cy - cx * math.tan(th)
    return th


def rowy(l):
    return l.get("yc", (l["y0"] + l["y1"]) / 2)


def same_row(a, b, tol=0.6):
    ha = a["y1"] - a["y0"]
    return abs(rowy(a) - rowy(b)) < ha * tol


def right_of(lines, anchor):
    """the nearest line to the right on the same row — i.e. the value"""
    cands = [l for l in lines
             if l is not anchor and same_row(anchor, l) and l["x0"] >= anchor["x1"] - 8]
    return min(cands, key=lambda l: l["x0"]) if cands else None


def above(lines, anchor, max_gap=90):
    """the nearest line sitting over this one — old forms print the number
    above its own label (`292` over `Αριθ. Γνωμοδ.`)"""
    cands = [l for l in lines
             if rowy(l) < rowy(anchor) - 2 and rowy(anchor) - rowy(l) < max_gap
             and min(l["x1"], anchor["x1"]) - max(l["x0"], anchor["x0"]) > -40]
    return max(cands, key=rowy) if cands else None


def below(lines, anchor, max_gap=120):
    cands = [l for l in lines if rowy(l) > rowy(anchor) + 2
             and rowy(l) - rowy(anchor) < max_gap]
    return min(cands, key=lambda l: (rowy(l), l["x0"])) if cands else None


def tail_after_label(line, *wanted):
    """'Αριθμός: 8/7464' → '8/7464'"""
    t = line["text"]
    if ":" in t:
        return t.split(":", 1)[1].strip()
    for w in wanted:
        m = re.search(re.escape(w), t, re.I)
        if m:
            return t[m.end():].strip(" .:-")
    return ""


def first_money(*texts):
    """the first thing on this line that is actually an amount"""
    for t in texts:
        if not t:
            continue
        m = MONEY_RE.search(fix_digits(str(t).replace("€", " ")))
        if m:
            v = norm_num(m.group(0).replace(" ", ""))
            if v is not None:
                return v
    return None


def first_num(*texts):
    for t in texts:
        if not t:
            continue
        m = NUM_RE.search(fix_digits(str(t).replace("€", " ")))
        if m:
            v = norm_num(m.group(0))
            if v is not None:
                return v
    return None


# -------------------------------------------------------------- field layout

def read_fields(lines):
    """labels + boxes → the ten fields, plus where each one came from"""
    f, src = {}, {}

    def put(k, value, line):
        f[k] = value
        src[k] = line

    # --- ΑΦΜ: the first one belongs to the issuer, the one under
    # ΣΤΟΙΧΕΙΑ ΠΕΛΑΤΗ to the customer
    client_hdr = find(lines, "ΣΤΟΙΧΕΙΑ ΠΕΛΑΤΗ", "ΠΕΛΑΤΗΣ", "ΣΤΟΙΧΕΙΑΠΕΛΑΤΗ")
    cut = rowy(client_hdr) if client_hdr else None
    afm_lines = [l for l in lines if has_label(l, "ΑΦΜ") and VAT9_RE.search(fix_digits(l["text"]))]
    for l in afm_lines:
        v = VAT9_RE.search(fix_digits(l["text"])).group(1)
        if cut is not None and rowy(l) >= cut:
            if "client_vat" not in f:
                put("client_vat", v, l)
        elif "issuer_vat" not in f:
            put("issuer_vat", v, l)
    if "client_vat" not in f and len(afm_lines) > 1:
        l = afm_lines[1]
        put("client_vat", VAT9_RE.search(fix_digits(l["text"])).group(1), l)

    # --- issuer: the biggest type above its own ΑΦΜ is the company name
    top = rowy(src["issuer_vat"]) if "issuer_vat" in src else 500
    cands = [l for l in lines if rowy(l) <= top + 4
             and not has_label(l, "ΤΙΜΟΛΟΓΙΟ", "ΑΡΙΘΜΟΣ", "ΗΜΕΡΟΜΗΝΙΑ", "ΑΦΜ", "ΔΟΥ")
             and len(lab(l["text"])) >= 3]
    if cands:
        l = max(cands, key=lambda l: (l["y1"] - l["y0"], -l["y0"]))
        put("issuer_name", l["text"].strip(), l)

    # --- client: first real line under the ΣΤΟΙΧΕΙΑ ΠΕΛΑΤΗ header
    if client_hdr:
        for l in lines:
            if rowy(l) > rowy(client_hdr) + 2 and len(lab(l["text"])) >= 3 \
                    and not has_label(l, "ΑΦΜ", "ΔΟΥ"):
                put("client_name", l["text"].strip(), l)
                break

    # --- number and date
    l = find(lines, "ΑΡΙΘΜΟΣ", "ΑΡΙΘ")
    if l:
        v = tail_after_label(l, "Αριθμός", "Αριθ")
        if not v:
            r = right_of(lines, l)
            v = r["text"].strip() if r else ""
        v = re.sub(r"[.,;:·\-\s]+$", "", v)   # photocopy speckle lands as a trailing dot
        if v:
            put("invoice_number", v, l)
    l = find(lines, "ΗΜΕΡΟΜΗΝΙΑ")
    m = DATE_RE.search(fix_digits(l["text"])) if l else None
    if not m:
        for c in lines:
            m = DATE_RE.search(fix_digits(c["text"]))
            if m:
                l = c
                break
    if m:
        put("invoice_date", m.group(1).replace(".", "/").replace("-", "/"), l)

    # --- the money block: label on the left, value to its right
    for key, words in (("net_amount", ("ΚΑΘΑΡΗ ΑΞΙΑ", "ΚΑΘΑΡΗΑΞΙΑ", "ΣΥΝΟΛΟ ΚΑΘΑΡΗΣ ΑΞΙΑΣ")),
                       ("vat_amount", ("ΦΠΑ",)),
                       ("gross_amount", ("ΣΥΝΟΛΟ ΠΛΗΡΩΜΗΣ", "ΓΕΝΙΚΟ ΣΥΝΟΛΟ", "ΠΛΗΡΩΤΕΟ"))):
        l = find(lines, *words)
        if not l:
            continue
        if key == "vat_amount":
            m = re.search(r"(\d{1,2}(?:[.,]\d+)?)\s*%", l["text"])
            if m:
                put("vat_rate", norm_num(m.group(1)), l)
        r = right_of(lines, l)
        v = first_money(r["text"] if r else None)
        if v is None and ":" in l["text"]:
            v = first_money(l["text"].split(":", 1)[1])
        if v is not None:
            put(key, v, r if (r and v == first_money(r["text"])) else l)

    if f.get("gross_amount") is None:
        l = find(lines, "ΣΥΝΟΛΟ")
        r = right_of(lines, l) if l else None
        if r:
            put("gross_amount", first_money(r["text"]), r)

    return f, src


def read_line_items(lines):
    """the table, by column position rather than by guessing"""
    hdr = {}
    for key, words in (("desc", ("ΠΕΡΙΓΡΑΦΗ",)), ("qty", ("ΠΟΣΟΤΗΤΑ", "ΠΟΣΟΤ")),
                       ("price", ("ΤΙΜΗ", "ΤΙΜΗ ΜΟΝΑΔΑΣ")), ("total", ("ΑΞΙΑ", "ΣΥΝΟΛΟ"))):
        l = find(lines, *words)
        if l:
            hdr[key] = l
    if len(hdr) < 3 or "desc" not in hdr:
        return []
    head_y = max(rowy(l) for l in hdr.values()) + 4
    stop = find(lines, "ΚΑΘΑΡΗ ΑΞΙΑ", "ΚΑΘΑΡΗΑΞΙΑ", after=head_y)
    stop_y = rowy(stop) if stop else 10 ** 9

    body = [l for l in lines if head_y + 2 < rowy(l) < stop_y - 2]
    rows = []
    for l in sorted(body, key=rowy):
        for r in rows:
            if same_row(r[0], l, tol=0.8):
                r.append(l)
                break
        else:
            rows.append([l])

    def column(l):
        return min(hdr, key=lambda k: abs((l["x0"] + l["x1"]) / 2 -
                                          (hdr[k]["x0"] + hdr[k]["x1"]) / 2))

    items = []
    for r in rows:
        cell = {}
        for l in sorted(r, key=lambda l: l["x0"]):
            c = column(l)
            cell[c] = (cell.get(c, "") + " " + l["text"]).strip()
        if "desc" not in cell or not lab(cell["desc"]):
            continue
        it = {"description": cell["desc"],
              "quantity": first_num(cell.get("qty")),
              "unit_price": first_money(cell.get("price")),
              "line_total": first_money(cell.get("total"))}
        if it["line_total"] is not None or it["unit_price"] is not None:
            items.append(it)
    return items


def prove_rows(items, net):
    """Each row is its own little sum: quantity × price = total. That lets the
    table repair itself — a quantity the scanner lost is recoverable exactly —
    and, more importantly, lets a row that does NOT add up be flagged instead
    of booked."""
    for it in items:
        q, p, t = (norm_num(it.get(k)) for k in ("quantity", "unit_price", "line_total"))
        missing = [k for k, v in (("quantity", q), ("unit_price", p), ("line_total", t))
                   if v is None]
        if not missing:
            it["status"] = ("confident" if abs(q * p - t) <= 0.02 + abs(t) * 1e-4
                            else "needs_review")
            it["evidence"] = ("quantity × price = total ✓" if it["status"] == "confident"
                              else f"{q:g} × {p:g} ≠ {t:g}")
        elif len(missing) == 1 and missing[0] != "line_total" and t is not None:
            other = p if missing[0] == "quantity" else q
            if other:
                v = round(t / other, 2)
                it[missing[0]] = v
                it["status"] = "confident"
                it["evidence"] = f"{missing[0]} recovered from the row arithmetic"
            else:
                it["status"] = "needs_review"
                it["evidence"] = "row incomplete"
        elif len(missing) == 1 and missing[0] == "line_total" and None not in (q, p):
            it["line_total"] = round(q * p, 2)
            it["status"] = "confident"
            it["evidence"] = "total recovered from quantity × price"
        else:
            it["status"] = "needs_review"
            it["evidence"] = "row incomplete"

    tot = sum(norm_num(it.get("line_total")) or 0 for it in items)
    balances = net is not None and items and abs(tot - net) <= 0.02
    return items, balances


TABLE_PROMPT = ("This is the line-item table of a Greek invoice. Return ONLY JSON: "
                '{"descriptions": ["...", ...]} — the ΠΕΡΙΓΡΑΦΗ column, one string '
                "per row, in order, exactly as printed. No other keys.")


def page_quality(lines):
    """how well the recogniser thinks it read this page"""
    sc = sorted(l["score"] for l in lines)
    return sc[len(sc) // 2] if sc else 0.0


def verify_descriptions(path, lines, items, scratch=None):
    """Numbers can prove themselves; words cannot.

    On a fax or a third-generation photocopy the OCR gets the amounts right and
    still turns `Ανταλλακτικά αντλίας` into `Αυταλλακτικά αυτλίας`. Cropping the
    table out and asking the vision model about that strip alone costs ~3 s —
    a quarter of a full-page call — and reads the damaged words better than
    either OCR pass. So: numbers from OCR, words from the model, only when the
    page is bad enough to need it.
    """
    from PIL import Image
    hdr = find(lines, "ΠΕΡΙΓΡΑΦΗ")
    stop = find(lines, "ΚΑΘΑΡΗ ΑΞΙΑ", "ΚΑΘΑΡΗΑΞΙΑ")
    if not (hdr and stop and items):
        return items, 0.0
    im = Image.open(path)
    box = (max(0, int(hdr["x0"] - 30)), max(0, int(hdr["y0"] - 20)),
           im.size[0], max(0, int(stop["y0"] - 10)))
    if box[3] - box[1] < 40:
        return items, 0.0
    scratch = scratch or os.path.join(os.path.dirname(os.path.abspath(path)), "_crop")
    os.makedirs(scratch, exist_ok=True)
    crop_path = os.path.join(scratch, os.path.basename(path) + ".table.png")
    im.crop(box).save(crop_path)

    t = time.time()
    try:
        got = parse_json(ask_vlm(crop_path, TABLE_PROMPT, timeout=120))
    except Exception:
        return items, time.time() - t
    dt = time.time() - t
    desc = [str(d).strip() for d in (got.get("descriptions") or []) if str(d).strip()]
    if len(desc) != len(items):
        for it in items:                       # row count disagrees → say so
            it.setdefault("evidence", "")
            it["status"] = "needs_review"
            it["evidence"] = (it.get("evidence", "") +
                              f"; the vision model counted {len(desc)} rows, OCR {len(items)}")
        return items, dt
    for it, d in zip(items, desc):
        if fold(d) == fold(it.get("description")):
            it["text_evidence"] = "both readers read the same words"
        else:
            it["ocr_description"] = it.get("description")
            it["description"] = d
            it["text_evidence"] = "words re-read by the vision model (damaged page)"
    return items, dt


# ----------------------------------------------------------------- the judge

MONEY = ("net_amount", "vat_amount", "gross_amount")


def money_contradiction(fields):
    """the sum that must hold on every invoice. If it doesn't, nobody is confident.

    This is the rule that turns a bad read into a flag instead of a silent error:
    a reader can agree with itself about a wrong number, but it cannot make the
    page add up.
    """
    net, vat, gross = (norm_num(fields.get(k)) for k in MONEY)
    if None in (net, vat, gross):
        return False
    return abs(net + vat - gross) > 0.02


def printed(value, latin):
    """is this amount actually printed on the page, per the Latin reader?

    A bare "1" matches almost any invoice, so short numbers prove nothing.
    """
    n = norm_num(value)
    if n is None:
        return False
    want = digits(f"{n:.2f}")
    return len(want) >= 3 and want in digits(" ".join(l["text"] for l in latin))


def judge(fields, src, latin, ok, source="ocr"):
    """what can this page prove about each field, and what can it not"""
    out, review = {}, []
    contradicted = money_contradiction(fields)
    proven_money = set() if contradicted else (set(MONEY) if len(ok) >= 2 else set())
    for k in FIELDS:
        v = fields.get(k)
        numeric = k.endswith("_amount") or k == "vat_rate"
        if numeric:
            v = norm_num(v)
        line = src.get(k) or {}
        conf = float(line.get("score", 0.0))
        seen = (printed(v, latin) if numeric and k != "vat_rate"
                else confirm_against_ocr(v, latin, numeric)) if latin else False
        proven = k in proven_money
        if v in (None, ""):
            status, why = "unreadable", "not found on the page"
        elif contradicted and k in MONEY:
            status, why = "needs_review", "the page does not add up (net + ΦΠΑ ≠ σύνολο)"
        elif source == "pdf-text":
            status, why = "confident", "read from the PDF's own text layer"
        elif seen and conf >= 0.80:
            status, why = "confident", "both readers agree"
        elif proven:
            status, why = "confident", "arithmetic consistent"
        elif conf >= 0.98 and not numeric:
            status, why = "confident", f"Greek reader certain ({conf:.2f})"
        else:
            status, why = "needs_review", f"only one reader, score {conf:.2f}"
        out[k] = {"value": v, "status": status, "evidence": why}
        if status != "confident":
            review.append(k)
    return out, review


# ------------------------------------------------------------------ pipeline

def extract(path, use_vlm=True, cache=True):
    t0 = time.time()
    source = "ocr"
    if path.lower().endswith(".pdf"):
        el = pdf_lines(path)
        if len(el) >= 8:                       # a real text layer, not a scan
            source = "pdf-text"
        else:
            pages = pdf_to_png(path)
            if not pages:
                raise SystemExit(f"{path}: nothing to read")
            path = pages[0]                    # one invoice per page, for now
            el = greek_lines(path, cache)
    else:
        el = greek_lines(path, cache)
    t_el = time.time() - t0

    t1 = time.time()
    latin = [] if source == "pdf-text" else paddle_lines(path)
    t_latin = time.time() - t1

    skew = deskew(el)
    fields, src = read_fields(el)
    items = read_line_items(el)
    t_text = 0.0
    if use_vlm and source != "pdf-text" and items:
        quality = page_quality(el)
        if quality < TEXT_VERIFY_BELOW:
            items, t_text = verify_descriptions(path, el, items)
    items, table_ok = prove_rows(items, norm_num(fields.get("net_amount")))
    fields["line_items"] = items
    ok, notes = arithmetic_checks(fields)
    if items and "lines" not in ok:
        notes.append("line items balance to the net ✓" if table_ok
                     else "line items do NOT balance to the net")

    result, review = judge(fields, src, latin or el, ok, source)

    t_vlm, escalated = 0.0, []
    if use_vlm and review:
        t2 = time.time()
        try:
            vf = parse_json(ask_vlm(path))
        except Exception as e:                      # server down → say so, don't guess
            vf, notes = {}, notes + [f"vision model unavailable: {e}"]
        t_vlm = time.time() - t2
        for k in review:
            v = vf.get(k)
            if v is None:
                continue
            numeric = k.endswith("_amount") or k == "vat_rate"
            v = norm_num(v) if numeric else str(v).strip()
            cur = result[k]["value"]
            agree = (abs(v - cur) <= 0.02 if numeric and None not in (v, cur)
                     else fold(v) == fold(cur) if cur is not None else False)
            if agree:
                result[k] = {"value": cur, "status": "confident",
                             "evidence": "OCR and the vision model agree"}
            elif cur is None:
                result[k] = {"value": v, "status": "needs_review",
                             "evidence": "only the vision model saw this"}
            else:
                result[k] = {"value": cur, "status": "needs_review",
                             "evidence": f"readers disagree (OCR {cur!r} vs VLM {v!r})"}
            escalated.append(k)
        review = [k for k, v in result.items() if v["status"] != "confident"]

    return {
        "file": os.path.basename(path),
        "fields": result,
        "line_items": items,
        "source": source,
        "skew_deg": round(math.degrees(skew), 2),
        "checks": notes,
        "needs_review": review,
        "escalated": escalated,
        "timing": {"el_s": round(t_el, 2), "latin_s": round(t_latin, 2),
                   "text_s": round(t_text, 1), "vlm_s": round(t_vlm, 1),
                   "total_s": round(t_el + t_latin + t_text + t_vlm, 2)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-vlm", action="store_true", help="never escalate")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    results, t0 = [], time.time()
    for p in args.images:
        r = extract(p, use_vlm=not args.no_vlm, cache=not args.no_cache)
        results.append(r)
        conf = sum(1 for f in r["fields"].values() if f["status"] == "confident")
        t = r["timing"]
        print(f"{r['file']}: {conf}/{len(FIELDS)} confident, {len(r['needs_review'])} flagged, "
              f"{len(r['line_items'])} items, {t['total_s']}s"
              + (f" (vlm {t['vlm_s']}s)" if t["vlm_s"] else ""), flush=True)
    print(f"--- {len(results)} pages in {time.time() - t0:.1f}s")
    if args.out:
        json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
