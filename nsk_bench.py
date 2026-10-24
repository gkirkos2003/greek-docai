#!/usr/bin/env python
"""Read REAL scanned Greek legal documents and check the result against labels
nobody made up.

The NSK corpus (Legal Council of State opinions) is 53,928 scanned opinions, 1951–2026, next to
56,334 harvested metadata records. The opinion number and year are known from the
filename and the index; the provisions cited (Διατάξεις) were catalogued by NSK's
own staff. So for each scan we can ask the only question that matters — did the
reader find what is actually on the page — without inventing ground truth.

    python nsk_bench.py 40
"""
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata

ROOT = os.environ.get("NSK_ROOT", "data/nsk")
WORK = os.environ.get("NSK_WORK", "realtest/_nsk")

LOOKALIKE = str.maketrans({"Ο": "0", "O": "0", "ο": "0", "Ι": "1", "I": "1", "l": "1"})
# how a citation is written in the index vs how it is printed on the page:
#   ΒΔ 451/1962Α45  →  β.δ. 451/1962, άρθρο 45
CITE = re.compile(r"(Ν|ΝΔ|ΒΔ|ΠΔ|ΑΝ|Κ)\s*(\d{1,5})\s*/\s*(\d{4})")


def norm(s):
    s = unicodedata.normalize("NFD", str(s or "").upper())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s)


def digits_only(s):
    return re.sub(r"\D", "", str(s or "").translate(LOOKALIKE))


def load_index():
    idx = {}
    path = os.path.join(ROOT, "opinions.jsonl")
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        prov = []
        for p in (r.get("provisions") or []):
            if "$(document" in p or "p_p_id" in p or len(p) > 200:
                p = p.split("Πρόεδρος")[0]
            m = CITE.search(norm(p))
            if m:
                prov.append((m.group(1), m.group(2), m.group(3)))
        idx[r["id"]] = {"subject": r.get("subject") or "", "provisions": prov}
    return idx


def sample(n, seed=5):
    years = sorted(y for y in os.listdir(os.path.join(ROOT, "pdf")) if y.isdigit())
    rng = random.Random(seed)
    out = []
    # walk the whole range, not the first n years — the 1950s scans are the
    # worst in the corpus and would flatter nothing but themselves
    order = [years[round(i * (len(years) - 1) / max(n - 1, 1))] for i in range(n)]
    while len(out) < n and years:
        y = order[len(out)] if len(out) < len(order) else years[len(out) % len(years)]
        files = [f for f in os.listdir(os.path.join(ROOT, "pdf", y)) if f.endswith(".pdf")]
        if not files:
            years.remove(y)
            continue
        out.append(os.path.join(ROOT, "pdf", y, rng.choice(files)))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    pages = int(os.environ.get("NSK_PAGES", "2"))
    os.makedirs(WORK, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fast_extract as F

    idx = load_index()
    rows = []
    for pdf in sample(n):
        if len(rows) >= n:
            break
        base = os.path.basename(pdf)[:-4]
        m = re.match(r"(\d+)_(\d{4})_", base)
        if not m:
            continue
        num, year = m.group(1), m.group(2)
        meta = idx.get(f"{num}/{year}", {})
        stem = os.path.join(WORK, base)
        t = time.time()
        try:
            subprocess.run(["pdftoppm", "-r", "200", "-f", "1", "-l", str(pages),
                            "-png", pdf, stem], check=True, timeout=300)
        except subprocess.SubprocessError:
            continue
        imgs = sorted(p for p in os.listdir(WORK) if p.startswith(base + "-"))
        text = []
        for img in imgs:
            p = os.path.join(WORK, img)
            text += [l["text"] for l in F.greek_lines(p, cache=False)]
            os.remove(p)
        dt = time.time() - t
        blob = norm("\n".join(text))
        blob_d = digits_only(blob)

        found_num = f"{num}/{year}" in blob or (
            digits_only(num) in blob_d and digits_only(year) in blob_d)
        found_year = digits_only(year) in blob_d
        prov = meta.get("provisions") or []
        hits = sum(1 for (_, pn, py) in prov
                   if digits_only(pn) + digits_only(py) in blob_d.replace(" ", "")
                   or (digits_only(pn) in blob_d and digits_only(py) in blob_d))
        rows.append({"file": base, "year": int(year), "chars": len(blob),
                     "pages": len(imgs), "seconds": round(dt, 1),
                     "number_found": bool(found_num), "year_found": bool(found_year),
                     "provisions": len(prov), "provisions_found": hits})
        r = rows[-1]
        print(f"{base}: {r['chars']:6} chars  num {'✓' if found_num else '✗'}  "
              f"year {'✓' if found_year else '✗'}  "
              f"cites {hits}/{len(prov)}  {r['seconds']}s", flush=True)

    if not rows:
        print("nothing read")
        return
    ok_n = sum(r["number_found"] for r in rows)
    ok_y = sum(r["year_found"] for r in rows)
    withp = [r for r in rows if r["provisions"]]
    cited = sum(r["provisions_found"] for r in withp)
    totp = sum(r["provisions"] for r in withp)
    empty = sum(1 for r in rows if r["chars"] < 200)
    print(f"\n{len(rows)} real scanned opinions, {rows[0]['pages']} pages each")
    print(f"  opinion number found   {ok_n}/{len(rows)}")
    print(f"  year found             {ok_y}/{len(rows)}")
    print(f"  cited provisions found {cited}/{totp} across {len(withp)} documents")
    print(f"  pages that read as near-empty: {empty}")
    print(f"  seconds per document   {sum(r['seconds'] for r in rows)/len(rows):.1f}")
    json.dump(rows, open("realtest/nsk_bench.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
