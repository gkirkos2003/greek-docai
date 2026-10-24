#!/usr/bin/env python
"""Score the legal reader against labels NSK's own staff wrote.

Ground truth, none of it invented here:
  * the opinion number and year — from the filename and the harvested index
  * the statutes cited (Διατάξεις) — catalogued per opinion by NSK

Headline, as everywhere in this project: SILENT ERRORS — a field that was
reported confidently and is wrong. A refusal costs a minute of somebody's time;
a confident wrong citation ends up in a legal argument.

    python legal_eval.py 24          # sample, read, score
    python legal_eval.py --score legal.json
"""
import json
import os
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_extract as F
from nsk_bench import ROOT, load_index, sample


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


def score(results, idx):
    stats = {"n": 0, "num_right": 0, "num_silent": 0, "num_flagged": 0,
             "year_right": 0, "year_silent": 0, "date_found": 0,
             "body_found": 0, "cites_found": 0, "cites_total": 0,
             "docs_with_cites": 0}
    for r in results:
        m = re.match(r"(\d+)_(\d{4})_", r["file"])
        if not m:
            continue
        num, year = m.group(1), m.group(2)
        meta = idx.get(f"{num}/{year}", {})
        stats["n"] += 1
        f = r["fields"]

        for key, want, rk, sk in (("opinion_number", num, "num_right", "num_silent"),
                                  ("opinion_year", year, "year_right", "year_silent")):
            got, st = f[key]["value"], f[key]["status"]
            ok = digits(got) == digits(want)
            if ok:
                stats[rk] += 1
            elif st == "confident":
                stats[sk] += 1
                print(f"  SILENT ERROR {r['file']} {key}: read {got!r}, really {want!r}")
            if st != "confident" and key == "opinion_number":
                stats["num_flagged"] += 1
        stats["date_found"] += f["date"]["value"] is not None
        stats["body_found"] += f["body"]["value"] is not None

        prov = meta.get("provisions") or []
        if prov:
            stats["docs_with_cites"] += 1
            got = {c["cite"] for c in r["citations"]}
            for (_, pn, py) in prov:
                stats["cites_total"] += 1
                if f"{int(pn)}/{py}" in {f"{int(a.split('/')[0])}/{a.split('/')[1]}"
                                         for a in got}:
                    stats["cites_found"] += 1
    n = max(stats["n"], 1)
    print(f"\n{stats['n']} real scanned opinions")
    print(f"  opinion number right   {stats['num_right']}/{n}"
          f"   SILENT {stats['num_silent']}   refused {stats['num_flagged']}")
    print(f"  year right             {stats['year_right']}/{n}   SILENT {stats['year_silent']}")
    print(f"  date read              {stats['date_found']}/{n}")
    print(f"  bench (τμήμα) read     {stats['body_found']}/{n}")
    print(f"  cited statutes found   {stats['cites_found']}/{stats['cites_total']}"
          f"   across {stats['docs_with_cites']} documents")
    return stats


def main():
    if sys.argv[1:2] == ["--score"]:
        results = json.load(open(sys.argv[2]))
        score(results, load_index())
        return

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    import legal_extract as L
    results, t0 = [], time.time()
    for pdf in sample(n):
        try:
            r = L.extract(pdf)
        except (subprocess.SubprocessError, SystemExit) as e:
            print(f"{os.path.basename(pdf)}: FAILED {e}", flush=True)
            continue
        results.append(r)
        f = r["fields"]
        print(f"{r['file']}: #{f['opinion_number']['value']}/{f['opinion_year']['value']} "
              f"{f['date']['value']}  {len(r['citations'])} cites  "
              f"{r['pages']}p {r['timing']['ocr_s']}s", flush=True)
    os.makedirs("realtest", exist_ok=True)
    json.dump(results, open("realtest/legal.json", "w"), ensure_ascii=False, indent=1)
    print(f"--- {len(results)} documents in {time.time()-t0:.0f}s")
    score(results, load_index())


if __name__ == "__main__":
    main()
