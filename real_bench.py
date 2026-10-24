#!/usr/bin/env python
"""Measure the Greek reader on REAL documents, with real ground truth.

Modern ΦΕΚ issues carry their own text layer. That layer is what the typesetter
put there — so rendering a page to an image, reading it back with OCR and
comparing the two is a free, large, honest accuracy test on real Greek
typography. No synthetic invoices, no guessed truth.

Two scores, because they fail differently:
  CER   character error rate over the whole page, order-sensitive — it punishes
        the two-column reading order as if it were a reading mistake
  words bag-of-words recall, which ignores order and asks the question a client
        actually asks: did it get the words right?

    python real_bench.py path/to/fek-pdfs 60
"""
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata


def norm(s):
    """strip accents and case: a reader that writes ΑΘΗΝΑ for Αθήνα is not wrong
    about the word, and Greek scans lose accents constantly"""
    s = unicodedata.normalize("NFD", s.upper())
    return "".join(c for c in s if not unicodedata.combining(c))


def words(s):
    return re.findall(r"[0-9A-ZΑ-Ω]+", norm(s))


def cer(a, b):
    """Levenshtein over characters, row-by-row to keep memory flat"""
    a, b = re.sub(r"\s+", " ", norm(a)).strip(), re.sub(r"\s+", " ", norm(b)).strip()
    if not a:
        return 1.0 if b else 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), 1)


def page_text(pdf, page):
    out = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-layout", pdf, "-"],
                         capture_output=True, timeout=120)
    return out.stdout.decode("utf-8", "replace")


def render(pdf, page, dpi, out_stem):
    subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page),
                    "-png", "-singlefile", pdf, out_stem], check=True, timeout=300)
    return out_stem + ".png"


def pick(root, n, seed=11, min_year=2005):
    """documents new enough to have a text layer, spread across years"""
    years = sorted(y for y in os.listdir(root)
                   if y.isdigit() and int(y) >= min_year)
    rng = random.Random(seed)
    picked = []
    while len(picked) < n and years:
        y = years[len(picked) % len(years)]
        files = os.listdir(os.path.join(root, y))
        if not files:
            years.remove(y)
            continue
        picked.append(os.path.join(root, y, rng.choice(files)))
    return picked


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FEK_ROOT", "data/fek")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    dpi = int(os.environ.get("BENCH_DPI", "200"))
    work = os.environ.get("BENCH_WORK", "realtest/_pages")
    os.makedirs(work, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fast_extract as F

    rows, skipped = [], 0
    for pdf in pick(root, n * 3):
        if len(rows) >= n:
            break
        truth = page_text(pdf, 1)
        if len(words(truth)) < 80:              # a scan, or a cover page: no truth here
            skipped += 1
            continue
        stem = os.path.join(work, os.path.basename(pdf)[:-4])
        try:
            png = render(pdf, 1, dpi, stem)
        except subprocess.SubprocessError:
            skipped += 1
            continue
        t = time.time()
        lines = F.greek_lines(png, cache=False)
        dt = time.time() - t
        got = "\n".join(l["text"] for l in lines)
        F.deskew(lines)
        ordered = "\n".join(l["text"] for l in F.reading_order(lines))

        tw, gw = words(truth), words(got)
        from collections import Counter
        ct, cg = Counter(tw), Counter(gw)
        hit = sum((ct & cg).values())
        rows.append({
            "file": os.path.basename(pdf),
            "chars": len(truth),
            "cer": round(cer(truth, got), 4),
            "cer_ordered": round(cer(truth, ordered), 4),
            "cer_orderfree": round(cer(" ".join(sorted(words(truth))),
                                      " ".join(sorted(words(got)))), 4),
            "word_recall": round(hit / max(len(tw), 1), 4),
            "word_precision": round(hit / max(len(gw), 1), 4),
            "seconds": round(dt, 2),
            "mean_score": round(sum(l["score"] for l in lines) / max(len(lines), 1), 4),
        })
        r = rows[-1]
        print(f"{r['file']}: CER raw {r['cer']:.3f} → xy-cut {r['cer_ordered']:.3f} "
              f"(order-free {r['cer_orderfree']:.3f})  words {r['word_recall']:.3f}  "
              f"{r['seconds']:.1f}s", flush=True)
        os.remove(png)

    if not rows:
        print("no pages with a text layer found")
        return
    med = lambda k: sorted(x[k] for x in rows)[len(rows) // 2]
    print(f"\n{len(rows)} real pages ({skipped} skipped: no text layer)")
    print(f"  CER raw        median {med('cer'):.3f}")
    # NOTE: pdftotext's own output is not in reading order either (-layout
    # interleaves the gazette's columns, plain uses content-stream order), so
    # cer_ordered is recorded but CANNOT be used to judge reading order.
    print(f"  CER xy-cut     median {med('cer_ordered'):.3f}   (not a valid score — see note)")
    print(f"  CER order-free median {med('cer_orderfree'):.3f}   <- pure character accuracy")
    print(f"  word recall    median {med('word_recall'):.3f}   mean "
          f"{sum(x['word_recall'] for x in rows)/len(rows):.3f}")
    print(f"  word precision median {med('word_precision'):.3f}")
    print(f"  seconds/page   median {med('seconds'):.2f}")
    json.dump(rows, open(os.path.join(os.path.dirname(work), "real_bench.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
