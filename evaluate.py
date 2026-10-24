#!/usr/bin/env python
"""Score extractions against ground truth.

The headline number is not accuracy. It is how often the system was both
WRONG and CONFIDENT — a silent error is the only kind that costs a client
money, and it is the number to put in front of an accountant.
"""
import json
import sys

import extract as E

NUMERIC = {"net_amount", "vat_amount", "gross_amount", "vat_rate"}


def same(key, got, want):
    if got is None:
        return False
    if key in NUMERIC:
        g, w = E.norm_num(got), E.norm_num(want)
        return g is not None and w is not None and abs(g - w) <= 0.02
    g, w = str(got).strip(), str(want).strip()
    if key.endswith("_vat"):
        return E.digits(g) == E.digits(w)
    if key == "invoice_date":
        return E.digits(g) == E.digits(w)
    return g.upper().replace(" ", "") == w.upper().replace(" ", "")


def cosmetic(key, got, want):
    """wrong only by a full stop or a Greek/Latin lookalike — a human reading the
    two side by side would call them the same string. Reported separately,
    because calling it a silent error hides the ones that actually cost money."""
    if got is None or key in NUMERIC:
        return False
    return E.fold(got) == E.fold(want)


def main(results_path, truth_dir="samples"):
    results = json.load(open(results_path))
    stats = {"fields": 0, "right": 0, "silent_errors": 0, "flagged": 0,
             "flagged_but_right": 0, "unreadable": 0, "cosmetic": 0}
    per_field = {}
    for r in results:
        truth = json.load(open(f"{truth_dir}/{r['file'].rsplit('.', 1)[0]}.truth.json"))
        for k, f in r["fields"].items():
            want, got, st = truth.get(k), f["value"], f["status"]
            correct = same(k, got, want)
            cos = not correct and cosmetic(k, got, want)
            d = per_field.setdefault(k, {"right": 0, "silent": 0, "flagged": 0,
                                         "cosmetic": 0, "n": 0})
            if cos:
                stats["cosmetic"] += 1
                d["cosmetic"] += 1
            d["n"] += 1
            stats["fields"] += 1
            if correct:
                stats["right"] += 1
                d["right"] += 1
            if st == "confident" and not correct and not cos:
                stats["silent_errors"] += 1
                d["silent"] += 1
                print(f"  SILENT ERROR {r['file']} {k}: got {got!r}, truth {want!r} "
                      f"({f['evidence']})")
            if st == "needs_review":
                stats["flagged"] += 1
                d["flagged"] += 1
                if correct:
                    stats["flagged_but_right"] += 1
            if st == "unreadable":
                stats["unreadable"] += 1

    # line items matter as much as the header: they are what gets booked
    cells = right_cells = cos_cells = item_silent = 0
    missing_rows = 0
    for r in results:
        truth = json.load(open(f"{truth_dir}/{r['file'].rsplit('.', 1)[0]}.truth.json"))
        T, G = truth.get("line_items", []), r.get("line_items", [])
        missing_rows += abs(len(T) - len(G))
        for i, ti in enumerate(T):
            gi = G[i] if i < len(G) else {}
            for k in ("description", "quantity", "unit_price", "line_total"):
                cells += 1
                got, want = gi.get(k), ti.get(k)
                if k == "description":
                    if str(got or "").upper().replace(" ", "") == str(want).upper().replace(" ", ""):
                        right_cells += 1
                    elif E.fold(got) == E.fold(want):
                        cos_cells += 1
                else:
                    g, w = E.norm_num(got), E.norm_num(want)
                    if g is not None and w is not None and abs(g - w) <= 0.02:
                        right_cells += 1
                    elif gi.get("status") == "confident":
                        item_silent += 1
                        print(f"  SILENT ERROR (line item) {r['file']} row{i} {k}: "
                              f"got {got!r}, truth {want!r}")

    n = stats["fields"] or 1
    print(f"\ndocuments        {len(results)}")
    print(f"fields           {stats['fields']}")
    print(f"correct          {stats['right']}  ({100*stats['right']/n:.1f}%)")
    print(f"SILENT ERRORS    {stats['silent_errors']}  ({100*stats['silent_errors']/n:.2f}%)  <- confident and wrong")
    print(f"cosmetic only    {stats['cosmetic']}  (a missing full stop / Greek-Latin lookalike)")
    print(f"flagged          {stats['flagged']}  (of which {stats['flagged_but_right']} were actually right)")
    print(f"unreadable       {stats['unreadable']}")
    print(f"\nline-item cells  {right_cells}/{cells} right"
          + (f" (+{cos_cells} lookalike-only)" if cos_cells else "")
          + f",  rows missed/extra: {missing_rows}"
          + f",  SILENT in rows: {item_silent}")
    print("\nper field:")
    for k, d in per_field.items():
        print(f"  {k:16} {d['right']}/{d['n']} right   {d['silent']} silent   "
              f"{d['cosmetic']} cosmetic   {d['flagged']} flagged")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.json",
         sys.argv[2] if len(sys.argv) > 2 else "samples")
