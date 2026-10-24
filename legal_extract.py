#!/usr/bin/env python
"""Scanned Greek legal opinion (NSK γνωμοδότηση) → structured JSON.

Same shape as the invoice reader, different document: labels find their values
by position, a second reader checks every digit, and anything that cannot be
confirmed comes back as needs_review instead of as a confident guess.

An invoice can prove itself with arithmetic. A legal opinion cannot — so the
proofs here are different:
  * the opinion number is printed twice on most documents (header and body)
  * a citation has a shape (Ν 2218/1994, π.δ. 18/1989) and a plausible year
  * the two recognisers must agree on every digit that matters

    python legal_extract.py <pdf>... --out legal.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_extract as F
from extract import paddle_lines

FIELDS = ["opinion_number", "opinion_year", "question_number", "date",
          "body", "president", "rapporteur"]

# how Greek statutes are cited, in every spelling OCR produces
KIND = r"(?:Ν\.?Δ|Π\.?Δ|Β\.?Δ|Α\.?Ν|Ν|Κ\.?Ν)"
CITE_RE = re.compile(KIND + r"\.?\s*(\d{1,5})\s*/\s*(\d{4})")
NUMYEAR_RE = re.compile(r"(\d{1,5})\s*/\s*(\d{4})")
LOOSE_NUMYEAR_RE = re.compile(r"(\d{1,5})\D{0,2}((?:19|20)\d{2})\b")
DATE_RE = re.compile(r"(\d{1,2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{4})")
GREEK_MONTHS = {"ΙΑΝΟΥΑΡΙ": 1, "ΦΕΒΡΟΥΑΡΙ": 2, "ΜΑΡΤΙ": 3, "ΑΠΡΙΛΙ": 4,
                "ΜΑΙ": 5, "ΙΟΥΝΙ": 6, "ΙΟΥΛΙ": 7, "ΑΥΓΟΥΣΤ": 8,
                "ΣΕΠΤΕΜΒΡΙ": 9, "ΟΚΤΩΒΡΙ": 10, "ΝΟΕΜΒΡΙ": 11, "ΔΕΚΕΜΒΡΙ": 12}


def pages_of(pdf, limit, dpi, work):
    os.makedirs(work, exist_ok=True)
    stem = os.path.join(work, os.path.basename(pdf)[:-4])
    cmd = ["pdftoppm", "-r", str(dpi), "-f", "1", "-png", pdf, stem]
    if limit:
        cmd[4:4] = ["-l", str(limit)]
    subprocess.run(cmd, check=True, timeout=600)
    base = os.path.basename(stem)
    return [os.path.join(work, f) for f in sorted(os.listdir(work))
            if f.startswith(base + "-") and f.endswith(".png")]


def glued_prefix(text, start):
    """Is the number stuck to a glyph that should not be there?

    `ΓΝΩΜΟΔΟΤΗΣΗ υπ' αριθ. $63/2006` — the 5 of 563 is handwritten on the form,
    read as `$`, and the rest parses cleanly as a perfectly plausible number.
    A digit run touching a junk character is a truncated number until proven
    otherwise.
    """
    if start == 0:
        return False
    c = text[start - 1]
    return not (c.isspace() or c in ".:·,-–—()[]/\\")


def digits_of(s):
    return re.sub(r"\D", "", str(s or ""))


def tail(line, *labels):
    t = line["text"]
    if ":" in t:
        return t.split(":", 1)[1].strip(" .:-")
    for w in labels:
        m = re.search(re.escape(w), t, re.I)
        if m:
            return t[m.end():].strip(" .:-")
    return ""


def value_for(lines, *labels):
    """the text after the label, or the line to its right, or the one below"""
    l = F.find(lines, *labels)
    if not l:
        return None, None
    v = tail(l, *labels)
    if not v:
        r = F.right_of(lines, l)
        v = r["text"].strip(" .:-") if r else ""
        if v:
            return v, r
    if not v:
        b = F.below(lines, l)
        v = b["text"].strip(" .:-") if b else ""
        if v:
            return v, b
    return (v or None), l


def read_header(lines):
    """page 1 carries the identity of the opinion, under Greek labels"""
    out, src = {}, {}

    l = F.find(lines, "ΑΡΙΘΜΟΣ ΕΡΩΤΗΜΑΤΟΣ", "ΕΡΩΤΗΜΑ")
    if l:
        m = NUMYEAR_RE.search(F.fix_digits(l["text"]))
        if m:
            out["question_number"] = f"{m.group(1)}/{m.group(2)}"
            src["question_number"] = l

    for l in lines[:14]:                      # "Αθήνα 18-12-2002" sits in the corner
        m = DATE_RE.search(F.fix_digits(l["text"]))
        if m:
            d, mo, y = (int(x) for x in m.groups())
            if 1 <= d <= 31 and 1 <= mo <= 12 and 1940 <= y <= 2030:
                out["date"] = f"{d:02d}/{mo:02d}/{y}"
                src["date"] = l
                break
    if "date" not in out:                     # "Συνεδρίασις τῆς 19 Δεκεμβρίου 1951"
        for l in lines[:24]:
            t = F.lab(l["text"])          # lab() folds Greek capitals onto Latin
            for name, mo in GREEK_MONTHS.items():
                if F.lab(name) in t:
                    u = F.fix_digits(F.unaccent(l["text"]))
                    m = re.search(r"(\d{1,2})\D{0,12}" + F.lab(name)[:4], F.lab_keep_digits(u))
                    y = re.search(r"(19|20)\d{2}", u)
                    if m and y:
                        out["date"] = f"{int(m.group(1)):02d}/{mo:02d}/{y.group(0)}"
                        src["date"] = l
                    break
            if "date" in out:
                break

    # Γνωμοδότηση αριθ. 706/2002  — the number may also be alone on its line
    l = F.find(lines, "ΓΝΩΜΟΔΟΤΗΣΗ ΑΡΙΘ", "ΑΡΙΘΜΟΣ ΓΝΩΜΟΔΟΤΗΣΕΩΣ", "ΑΡΙΘ ΓΝΩΜΟΔ")
    if l:
        out["_number_label_seen"] = True
        m = NUMYEAR_RE.search(F.fix_digits(l["text"]))
        if not m:
            for c in (F.right_of(lines, l), F.below(lines, l), F.above(lines, l)):
                if not c:
                    continue
                m = NUMYEAR_RE.search(F.fix_digits(c["text"]))
                if m:
                    l = c
                    break
        if not m:
            # the separator is the part that breaks: "3932003" (slash lost),
            # "17ι2026" (slash read as iota). Only accept the split when the
            # trailing year matches the session date, which came off another line.
            for c in (l, F.right_of(lines, l), F.above(lines, l), F.below(lines, l)):
                if not c:
                    continue
                lm = LOOSE_NUMYEAR_RE.search(F.fix_digits(c["text"]))
                if lm and out.get("date", "").endswith(lm.group(2)):
                    m, l = lm, c
                    out["_loose_sep"] = True
                    break
        if m:
            out["opinion_number"], out["opinion_year"] = m.group(1), m.group(2)
            src["opinion_number"] = src["opinion_year"] = l
            if glued_prefix(F.fix_digits(l["text"]), m.start()):
                out["_number_suspect"] = "the number touches a character that is not a digit"
        # NO bare-number fallback. Grabbing the nearest number to the label
        # produced six confident wrong answers (21 for 1308, 1 for 144, 2000
        # for 327) — on a legal citation that is worse than no answer at all.

    l = F.find(lines, "ΟΛΟΜΕΛΕΙΑ", "ΤΡΙΜΕΛΟΥΣ ΕΠΙΤΡΟΠΗΣ", "ΤΜΗΜΑ")
    if l:                                     # the bench is named, not labelled
        out["body"], src["body"] = l["text"].strip(" .-"), l
    for key, labels in (("president", ("ΠΡΟΕΔΡΟΣ", "ΠΡΟΕΔΡΕΥΩΝ", "ΑΝΤΙΠΡΟΕΔΡΟΣ")),
                        ("rapporteur", ("ΕΙΣΗΓΗΤΗΣ", "ΕΙΣΗΓΗΤΡΙΑ", "ΕΙΣΗΓΟΥΜΕΝΟΣ"))):
        v, l = value_for(lines, *labels)
        if v:
            out[key], src[key] = v, l
    return out, src


def read_citations(all_lines):
    """every statute the opinion cites, wherever it appears"""
    seen, out = set(), []
    for l in all_lines:
        t = F.fix_digits(re.sub(r"(?<=[ΝΠΒΑΚ])\s*\.\s*(?=[ΔΝ])", "", l["text"].upper()))
        for m in CITE_RE.finditer(t):
            key = f"{m.group(1)}/{m.group(2)}"
            if key in seen:
                continue
            if not (1821 <= int(m.group(2)) <= 2030):
                continue
            seen.add(key)
            out.append({"cite": key, "text": m.group(0).strip(), "score": l["score"]})
    return out


def reread_number(img, line, dpi_scale=2.0):
    """Look at the number again, with different pixels.

    Both recognisers share one text detector, so when the detector clips a box
    the two agree on the same wrong number — that is how 473/1997 was confidently
    read as 73/1997. Cropping the line with a wide margin and enlarging it gives
    the detector a different problem to solve, which is the only way the second
    opinion is actually independent.
    """
    from PIL import Image
    im = Image.open(img)
    pad_x = max(60, int((line["x1"] - line["x0"]) * 0.35))
    pad_y = max(12, int((line["y1"] - line["y0"]) * 0.6))
    box = (max(0, int(line["x0"] - pad_x)), max(0, int(line["y0"] - pad_y)),
           min(im.size[0], int(line["x1"] + pad_x)),
           min(im.size[1], int(line["y1"] + pad_y)))
    crop = im.crop(box)
    crop = crop.resize((int(crop.size[0] * dpi_scale), int(crop.size[1] * dpi_scale)),
                       Image.LANCZOS)
    path = os.path.join(os.path.dirname(img), "_num.png")
    crop.save(path)
    try:
        text = " ".join(F.fix_digits(x["text"]) for x in F.greek_lines(path, cache=False))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    m = NUMYEAR_RE.search(text) or LOOSE_NUMYEAR_RE.search(text)
    if not m or glued_prefix(text, m.start()):
        return None
    return m.group(1), m.group(2)


def extract(pdf, max_pages=0, dpi=200, work="realtest/_legal", keep=False):
    t0 = time.time()
    imgs = pages_of(pdf, max_pages, dpi, work)
    if not imgs:
        raise SystemExit(f"{pdf}: no pages")

    all_lines, first = [], None
    for i, img in enumerate(imgs):
        lines = F.greek_lines(img, cache=False)
        F.deskew(lines)
        if i == 0:
            first = lines
            latin = paddle_lines(img)
            try:
                os.remove(img + ".paddle.json")
            except OSError:
                pass
        all_lines += lines
        if not keep and i > 0:
            os.remove(img)
    t_ocr = time.time() - t0

    fields, src = read_header(first)
    cites = read_citations(all_lines)

    third = None
    if fields.get("opinion_number") and src.get("opinion_number"):
        try:
            third = reread_number(imgs[0], src["opinion_number"])
        except Exception:
            third = None
    if not keep:
        try:
            os.remove(imgs[0])
        except OSError:
            pass

    # the second reader checks the digits that identify the document
    # tokens, not one long digit string: "73" is inside "473", and that is
    # exactly how a dropped leading digit got confirmed as correct
    latin_text = " ".join(F.fix_digits(l["text"]) for l in latin)
    latin_nums = set(re.findall(r"\d+", latin_text))
    latin_pairs = {f"{a}/{b}" for a, b in NUMYEAR_RE.findall(latin_text)}
    result, review = {}, []
    for k in FIELDS:
        v = fields.get(k)
        line = src.get(k) or {}
        conf = float(line.get("score", 0.0))
        if v in (None, ""):
            if k in ("opinion_number", "opinion_year") and fields.get("_number_label_seen"):
                status, why = "needs_review", "the number is printed but could not be read"
            else:
                status, why = "unreadable", "not found on page 1"
        elif k in ("opinion_number", "opinion_year", "question_number"):
            want = F.digits(v)
            pair = (f"{fields.get('opinion_number')}/{fields.get('opinion_year')}"
                    if k.startswith("opinion") else v)
            agreed = want and (want in latin_nums or pair in latin_pairs)
            if k.startswith("opinion") and fields.get("_number_suspect"):
                agreed = False
            if k.startswith("opinion") and third:
                tv = third[0] if k == "opinion_number" else third[1]
                if fields.get("_loose_sep") and k == "opinion_year":
                    pass                      # the date already proved the year
                elif digits_of(tv) != want:
                    agreed = False
                    why_third = f"a closer look reads {third[0]}/{third[1]}"
                else:
                    why_third = ""
            else:
                why_third = "" if third or not k.startswith("opinion") else \
                    "the number could not be re-read"
            if agreed and not why_third:
                status, why = "confident", (
                    "number and year ran together; the year matches the session date"
                    if fields.get("_loose_sep") else "three looks, same number")
            else:
                status, why = "needs_review", (
                    fields.get("_number_suspect") or why_third or
                    "the second reader did not confirm this number")
        elif conf >= 0.95:
            status, why = "confident", f"clearly printed ({conf:.2f})"
        else:
            status, why = "needs_review", f"faint or damaged print ({conf:.2f})"
        result[k] = {"value": v, "status": status, "evidence": why}
        if status != "confident":
            review.append(k)

    return {
        "file": os.path.basename(pdf),
        "pages": len(imgs),
        "fields": result,
        "citations": cites,
        "needs_review": review,
        "timing": {"ocr_s": round(t_ocr, 1), "per_page_s": round(t_ocr / len(imgs), 2)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-pages", type=int, default=0, help="0 = the whole document")
    args = ap.parse_args()
    results = []
    for p in args.pdfs:
        try:
            r = extract(p, args.max_pages)
        except (subprocess.SubprocessError, SystemExit) as e:
            print(f"{os.path.basename(p)}: FAILED {e}", flush=True)
            continue
        results.append(r)
        f = r["fields"]
        print(f"{r['file']}: #{f['opinion_number']['value']}/{f['opinion_year']['value']} "
              f"{f['date']['value']}  {len(r['citations'])} citations, "
              f"{len(r['needs_review'])} flagged, {r['pages']}p {r['timing']['ocr_s']}s",
              flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
