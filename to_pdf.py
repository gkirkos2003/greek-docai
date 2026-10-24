#!/usr/bin/env python
"""Re-draw a sample invoice as a REAL PDF — text, not a picture of text.

Most invoices an accountant receives today arrive like this, straight out of
somebody's ERP. They never need OCR at all: the coordinates and the characters
are already in the file. This exists so that claim can be measured instead of
asserted.

    python to_pdf.py samples/invoice_001.truth.json samples_pdf/
"""
import json
import os
import sys

import cairo

PT = 0.36                      # the PNG layout is 200 dpi; a point is 1/72 inch
W, H = 595.0, 842.0            # A4


def draw(truth, out_path):
    surf = cairo.PDFSurface(out_path, W, H)
    c = cairo.Context(surf)
    c.set_source_rgb(0, 0, 0)

    def text(x, y, s, size=9.0, bold=False, right=False):
        c.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL,
                           cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
        c.set_font_size(size)
        s = str(s)
        if right:
            x -= c.text_extents(s).x_advance
        c.move_to(x, y)
        c.show_text(s)

    def euro(v):
        return f"{v:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")

    text(90 * PT, 110 * PT, truth["issuer_name"], 15, bold=True)
    text(90 * PT, 150 * PT, "ΤΙΜΟΛΟΓΙΟ ΠΩΛΗΣΗΣ", 9)
    text(90 * PT, 190 * PT, f"Α.Φ.Μ.: {truth['issuer_vat']}   Δ.Ο.Υ.: Α' ΠΕΙΡΑΙΑ", 9)

    text(960 * PT, 110 * PT, "ΤΙΜΟΛΟΓΙΟ ΠΩΛΗΣΗΣ", 12, bold=True)
    text(960 * PT, 150 * PT, f"Αριθμός: {truth['invoice_number']}", 10)
    text(960 * PT, 180 * PT, f"Ημερομηνία: {truth['invoice_date']}", 10)

    text(90 * PT, 330 * PT, "ΣΤΟΙΧΕΙΑ ΠΕΛΑΤΗ", 10, bold=True)
    text(90 * PT, 365 * PT, truth["client_name"], 11)
    text(90 * PT, 400 * PT, f"Α.Φ.Μ.: {truth['client_vat']}", 9)

    y = 500 * PT
    for label, x in (("ΠΕΡΙΓΡΑΦΗ", 110), ("ΠΟΣΟΤΗΤΑ", 900),
                     ("ΤΙΜΗ", 1130), ("ΑΞΙΑ", 1380)):
        text(x * PT, y, label, 9, bold=True)
    y += 34 * PT
    for it in truth["line_items"]:
        text(110 * PT, y, it["description"], 9)
        text(955 * PT, y, it["quantity"], 9)
        text(1130 * PT, y, euro(it["unit_price"]), 9)
        text(1380 * PT, y, euro(it["line_total"]), 9)
        y += 34 * PT

    y += 50 * PT
    for label, val, bold in (("Καθαρή αξία", truth["net_amount"], False),
                             (f"Φ.Π.Α. {truth['vat_rate']:g}%", truth["vat_amount"], False),
                             ("ΣΥΝΟΛΟ ΠΛΗΡΩΜΗΣ", truth["gross_amount"], True)):
        text(1340 * PT, y, label, 10, bold=bold, right=True)
        text(1564 * PT, y, euro(val) + " €", 10, bold=bold, right=True)
        y += 34 * PT

    c.show_page()
    surf.finish()


def main():
    truths = sys.argv[1:-1] or ["samples/invoice_001.truth.json"]
    out_dir = sys.argv[-1] if len(sys.argv) > 2 else "samples_pdf"
    os.makedirs(out_dir, exist_ok=True)
    for t in truths:
        truth = json.load(open(t))
        stem = os.path.basename(t).replace(".truth.json", "")
        draw(truth, os.path.join(out_dir, stem + ".pdf"))
        import shutil
        shutil.copy(t, os.path.join(out_dir, stem + ".truth.json"))
        print(os.path.join(out_dir, stem + ".pdf"))


if __name__ == "__main__":
    main()
