#!/usr/bin/env python
"""Turn extraction results into the two things an accountant actually wants:
a spreadsheet they can import, and a short list of what to look at by hand.

    python report.py results.json out/

writes out/invoices.csv, out/lines.csv, out/review.txt
"""
import csv
import json
import os
import sys

HEAD = ["file", "issuer_name", "issuer_vat", "client_name", "client_vat",
        "invoice_number", "invoice_date", "net_amount", "vat_rate",
        "vat_amount", "gross_amount", "status", "to_check"]


def main(results_path, out_dir="out"):
    results = json.load(open(results_path))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "invoices.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEAD)
        for r in results:
            f = r["fields"]
            flags = r["needs_review"]
            w.writerow([r["file"]] + [f[k]["value"] for k in HEAD[1:-2]]
                       + ["OK" if not flags else "CHECK", " ".join(flags)])

    with open(os.path.join(out_dir, "lines.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "invoice_number", "description", "quantity",
                    "unit_price", "line_total", "status"])
        for r in results:
            num = r["fields"]["invoice_number"]["value"]
            for it in r["line_items"]:
                w.writerow([r["file"], num, it.get("description"), it.get("quantity"),
                            it.get("unit_price"), it.get("line_total"),
                            it.get("status", "")])

    checked = 0
    with open(os.path.join(out_dir, "review.txt"), "w") as fh:
        for r in results:
            rows = [it for it in r["line_items"] if it.get("status") != "confident"]
            if not r["needs_review"] and not rows:
                continue
            fh.write(f"\n{r['file']}\n")
            for k in r["needs_review"]:
                f = r["fields"][k]
                fh.write(f"  {k:15} {str(f['value']):>14}   — {f['evidence']}\n")
                checked += 1
            for it in rows:
                fh.write(f"  line  {str(it.get('description'))[:34]:34} "
                         f"{it.get('quantity')} × {it.get('unit_price')} = "
                         f"{it.get('line_total')}   — {it.get('evidence', '')}\n")
                checked += 1
            for c in r["checks"]:
                if "✓" not in c:
                    fh.write(f"  check: {c}\n")

    conf = sum(1 for r in results for f in r["fields"].values()
               if f["status"] == "confident")
    total = sum(len(r["fields"]) for r in results)
    print(f"{len(results)} documents, {total} header fields")
    print(f"  {conf} booked straight through ({100*conf/max(total,1):.1f}%)")
    print(f"  {checked} things to look at by hand  → {out_dir}/review.txt")
    print(f"  {out_dir}/invoices.csv, {out_dir}/lines.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.json",
         sys.argv[2] if len(sys.argv) > 2 else "out")
