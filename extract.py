#!/usr/bin/env python
"""Greek invoice → structured JSON, with an explicit refusal when unsure.

Two independent readers look at the same page:
  * PaddleOCR  — weak on Greek words, but exact on digits, dates and VAT ids
  * a local VLM — reads Greek properly and understands the layout

A field is only reported as confident when both agree, or when the arithmetic
of the document proves it. Anything else is handed back for a human to check,
which is the whole point: an accountant can work with "I'm not sure about 3
fields", and cannot work with a wrong number that looks right.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

SERVER = os.environ.get("VLM_SERVER", "http://127.0.0.1:8099")

FIELDS = ["issuer_name", "issuer_vat", "client_name", "client_vat",
          "invoice_number", "invoice_date", "net_amount", "vat_rate",
          "vat_amount", "gross_amount"]

PROMPT = """You are reading a Greek invoice (τιμολόγιο). Return ONLY a JSON object, no commentary.

Keys:
 issuer_name      the company ISSUING the invoice (top of the page)
 issuer_vat       its ΑΦΜ (9 digits)
 client_name      the customer (under ΣΤΟΙΧΕΙΑ ΠΕΛΑΤΗ)
 client_vat       the customer's ΑΦΜ
 invoice_number   the Αριθμός, exactly as printed
 invoice_date     the Ημερομηνία as DD/MM/YYYY
 net_amount       Καθαρή αξία as a number, dot decimal separator
 vat_rate         the ΦΠΑ percentage as a number
 vat_amount       the ΦΠΑ amount as a number
 gross_amount     ΣΥΝΟΛΟ ΠΛΗΡΩΜΗΣ as a number
 line_items       list of {description, quantity, unit_price, line_total}

Write null for any value you cannot read with certainty. Never guess a number."""


def paddle_lines(path):
    """digits-and-structure reader; cached per image"""
    cache = path + ".paddle.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paddleocr import PaddleOCR
    global _LATIN                       # loading the model per page cost more than reading it
    try:
        ocr = _LATIN
    except NameError:
        ocr = _LATIN = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                                 use_textline_orientation=False,
                                 device=os.environ.get("PADDLE_DEVICE", "gpu:0"))
    r = ocr.predict(path)[0]
    out = [{"text": t, "score": float(s)} for t, s in zip(r["rec_texts"], r["rec_scores"])]
    json.dump(out, open(cache, "w"), ensure_ascii=False)
    return out


def ask_vlm(path, prompt=PROMPT, timeout=600):
    import base64
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text": prompt}]}],
        "temperature": 0.0,
        "n_predict": 1200,
        "cache_prompt": False,
    }
    req = urllib.request.Request(SERVER + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}


def norm_num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    s = str(v).strip().replace("€", "").replace(" ", "")
    # 1.487,95 (Greek) and 1,487.95 (English) both appear on real paperwork
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


# Paddle has no Greek model, so it reports Greek capitals as the Latin letters
# that look the same (ΜΑΡΙΝΟΣ → MAPINOΣ). Folding both sides onto one alphabet
# lets the two readers actually be compared instead of always disagreeing.
LOOKALIKE = str.maketrans({
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "α": "A", "β": "B", "ε": "E", "ζ": "Z", "η": "H", "ι": "I", "κ": "K",
    "μ": "M", "ν": "N", "ο": "O", "ρ": "P", "τ": "T", "υ": "Y", "χ": "X",
    "Ά": "A", "Έ": "E", "Ή": "H", "Ί": "I", "Ό": "O", "Ύ": "Y", "Ώ": "Ω",
    "ά": "A", "έ": "E", "ή": "H", "ί": "I", "ό": "O", "ύ": "Y", "ώ": "Ω",
})


def fold(s):
    return re.sub(r"[^A-ZΑ-Ω0-9]", "", str(s or "").upper().translate(LOOKALIKE))


def confirm_against_ocr(value, lines, numeric=False):
    """is the VLM's value actually printed on the page, per the other reader?"""
    if value in (None, ""):
        return False
    blob = " ".join(l["text"] for l in lines)
    if numeric:
        n = norm_num(value)
        if n is None:
            return False
        want = {f"{n:.2f}", f"{n:.2f}".replace(".", ","), f"{n:,.2f}",
                f"{n:,.2f}".replace(",", "@").replace(".", ",").replace("@", "."),
                digits(f"{n:.2f}")}
        if n == int(n):            # a VAT rate prints as "24%", not "24.00"
            want |= {str(int(n)), f"{int(n)}%"}
        blob_d = digits(blob)
        return any(w in blob for w in want) or digits(f"{n:.2f}") in blob_d
    v = str(value).strip()
    if digits(v) and len(digits(v)) >= 6:          # VAT ids, phone-like numbers
        return digits(v) in digits(blob)
    letters, blob_l = fold(v), fold(blob)
    if len(letters) < 4:
        return False
    hits = sum(1 for i in range(0, len(letters) - 3) if letters[i:i + 4] in blob_l)
    return hits >= max(1, (len(letters) - 3) * 0.5)


def arithmetic_checks(f):
    """the document proving itself — stronger evidence than either reader"""
    notes, ok = [], {}
    net, vat, gross = norm_num(f.get("net_amount")), norm_num(f.get("vat_amount")), norm_num(f.get("gross_amount"))
    rate = norm_num(f.get("vat_rate"))
    if None not in (net, vat, gross):
        if abs(net + vat - gross) <= 0.02:
            ok["sum"] = True
            notes.append("net + vat = gross ✓")
        else:
            notes.append(f"net + vat ≠ gross ({net} + {vat} ≠ {gross})")
    if None not in (net, vat, rate) and rate:
        if abs(net * rate / 100 - vat) <= 0.02:
            ok["rate"] = True
            notes.append(f"vat = net × {rate:g}% ✓")
        else:
            notes.append(f"vat ≠ net × {rate:g}%")
    items = f.get("line_items") or []
    tot = 0.0
    for it in items:
        lt = norm_num(it.get("line_total"))
        if lt is None:
            tot = None
            break
        tot += lt
    if tot is not None and net is not None and items:
        if abs(tot - net) <= 0.02:
            ok["lines"] = True
            notes.append("line items sum to net ✓")
        else:
            notes.append(f"line items sum to {tot:.2f}, net says {net:.2f}")
    return ok, notes


def extract(path, skip_paddle=False):
    t0 = time.time()
    lines = [] if skip_paddle else paddle_lines(path)
    t_ocr = time.time() - t0

    t1 = time.time()
    raw = ask_vlm(path)
    fields = parse_json(raw)
    t_vlm = time.time() - t1

    ok, notes = arithmetic_checks(fields)
    result, review = {}, []
    for k in FIELDS:
        v = fields.get(k)
        numeric = k.endswith("_amount") or k == "vat_rate"
        if numeric:
            v = norm_num(v)
        seen = confirm_against_ocr(v, lines, numeric) if lines else False
        proven = bool(ok) and k in ("net_amount", "vat_amount", "gross_amount") and len(ok) >= 2
        status = "confident" if (v is not None and (seen or proven)) else (
            "unreadable" if v is None else "needs_review")
        result[k] = {"value": v, "status": status,
                     "evidence": ("both readers agree" if seen else
                                  "arithmetic consistent" if proven else
                                  "only the vision model saw this")}
        if status != "confident":
            review.append(k)

    return {
        "file": os.path.basename(path),
        "fields": result,
        "line_items": fields.get("line_items") or [],
        "checks": notes,
        "needs_review": review,
        "timing": {"ocr_s": round(t_ocr, 1), "vlm_s": round(t_vlm, 1)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-paddle", action="store_true")
    args = ap.parse_args()
    results = []
    for p in args.images:
        r = extract(p, args.no_paddle)
        results.append(r)
        conf = sum(1 for f in r["fields"].values() if f["status"] == "confident")
        print(f"{r['file']}: {conf}/{len(FIELDS)} confident, "
              f"{len(r['needs_review'])} flagged, "
              f"{r['timing']['ocr_s']}s ocr + {r['timing']['vlm_s']}s vlm", flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
