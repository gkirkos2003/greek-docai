#!/usr/bin/env python
"""Synthetic Greek invoices with known ground truth.

Real client paperwork is what we will measure on in the end, but we need a
labelled set now, and every field here is generated so the answer is known.
"""
import json
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1654, 2339  # A4 at 200 dpi
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
COMPANIES = [
    ("ΑΦΟΙ ΠΑΠΑΔΟΠΟΥΛΟΥ Ο.Ε.", "ΕΜΠΟΡΙΑ ΤΡΟΦΙΜΩΝ", "ΛΕΩΦ. ΑΘΗΝΩΝ 124", "ΠΕΙΡΑΙΑΣ", "18545"),
    ("ΜΑΡΙΝΟΣ ΑΕΒΕ", "ΝΑΥΤΙΛΙΑΚΟΣ ΕΞΟΠΛΙΣΜΟΣ", "ΑΚΤΗ ΜΙΑΟΥΛΗ 57", "ΠΕΙΡΑΙΑΣ", "18536"),
    ("ΤΕΧΝΟΔΟΜΗ Ι.Κ.Ε.", "ΟΙΚΟΔΟΜΙΚΑ ΥΛΙΚΑ", "ΘΕΡΜΟΠΥΛΩΝ 8", "ΘΕΣΣΑΛΟΝΙΚΗ", "54627"),
    ("ΔΕΛΤΑ ΛΟΓΙΣΤΙΚΗ Ε.Ε.", "ΛΟΓΙΣΤΙΚΕΣ ΥΠΗΡΕΣΙΕΣ", "ΕΡΜΟΥ 31", "ΑΘΗΝΑ", "10563"),
    ("ΚΡΗΤΙΚΗ ΔΙΑΝΟΜΗ Μ.Ε.Π.Ε.", "ΔΙΑΝΟΜΗ ΠΟΤΩΝ", "ΚΝΩΣΟΥ 212", "ΗΡΑΚΛΕΙΟ", "71409"),
]
ITEMS = [
    ("Χαρτί Α4 80gr (κιβώτιο)", 5, 24.50), ("Μελάνι εκτυπωτή HP 305", 3, 31.90),
    ("Υπηρεσίες συντήρησης", 1, 450.00), ("Καφές espresso 1kg", 12, 18.75),
    ("Μεταφορικά έξοδα", 1, 85.00), ("Ανταλλακτικά αντλίας", 2, 164.30),
    ("Ελαιόλαδο 5lt", 20, 42.00), ("Συσκευασία δεμάτων", 150, 0.85),
    ("Τσιμέντο 25kg", 40, 7.20), ("Ώρες τεχνικής υποστήριξης", 8, 55.00),
]


def font(size, bold=False):
    f = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, f), size)


def euro(x):
    return f"{x:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")


def make(seed, out_dir):
    rng = random.Random(seed)
    issuer = rng.choice(COMPANIES)
    client = rng.choice([c for c in COMPANIES if c != issuer])
    afm_i = str(rng.randint(100000000, 999999999))
    afm_c = str(rng.randint(100000000, 999999999))
    num = f"{rng.randint(1, 9)}/{rng.randint(100, 9999)}"
    date = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/2026"
    lines = rng.sample(ITEMS, rng.randint(3, 6))
    rows, net = [], 0.0
    for desc, qty, price in lines:
        q = qty if rng.random() < 0.6 else max(1, int(qty * rng.uniform(0.5, 2)))
        total = round(q * price, 2)
        net += total
        rows.append((desc, q, price, total))
    net = round(net, 2)
    vat_rate = rng.choice([24, 24, 24, 13, 6])
    vat = round(net * vat_rate / 100, 2)
    gross = round(net + vat, 2)

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = 90
    d.text((90, y), issuer[0], font=font(40, True), fill="black")
    d.text((90, y + 55), issuer[1], font=font(26), fill="black")
    d.text((90, y + 95), f"{issuer[2]}, {issuer[4]} {issuer[3]}", font=font(26), fill="black")
    d.text((90, y + 135), f"Α.Φ.Μ.: {afm_i}   Δ.Ο.Υ.: {rng.choice(['Α΄ ΠΕΙΡΑΙΑ','ΦΑΕ ΑΘΗΝΩΝ','Β΄ ΘΕΣΣΑΛΟΝΙΚΗΣ'])}",
           font=font(26), fill="black")

    d.rectangle([(1000, 85), (1564, 250)], outline="black", width=3)
    d.text((1030, 110), "ΤΙΜΟΛΟΓΙΟ ΠΩΛΗΣΗΣ", font=font(32, True), fill="black")
    d.text((1030, 160), f"Αριθμός: {num}", font=font(27), fill="black")
    d.text((1030, 200), f"Ημερομηνία: {date}", font=font(27), fill="black")

    y = 320
    d.line([(90, y), (1564, y)], fill="black", width=2)
    d.text((90, y + 20), "ΣΤΟΙΧΕΙΑ ΠΕΛΑΤΗ", font=font(27, True), fill="black")
    d.text((90, y + 65), client[0], font=font(29), fill="black")
    d.text((90, y + 105), f"{client[2]}, {client[4]} {client[3]}", font=font(25), fill="black")
    d.text((90, y + 145), f"Α.Φ.Μ.: {afm_c}", font=font(25), fill="black")

    y = 560
    d.rectangle([(90, y), (1564, y + 50)], fill=(235, 235, 235))
    for x, t in ((110, "ΠΕΡΙΓΡΑΦΗ"), (900, "ΠΟΣΟΤΗΤΑ"), (1130, "ΤΙΜΗ"), (1380, "ΑΞΙΑ")):
        d.text((x, y + 12), t, font=font(26, True), fill="black")
    y += 60
    for desc, q, price, total in rows:
        d.text((110, y), desc, font=font(26), fill="black")
        d.text((955, y), str(q), font=font(26), fill="black")
        d.text((1130, y), euro(price), font=font(26), fill="black")
        d.text((1380, y), euro(total), font=font(26), fill="black")
        y += 48
    d.line([(90, y + 10), (1564, y + 10)], fill="black", width=2)

    y += 50
    for label, val, bold in (("Καθαρή αξία", net, False),
                             (f"Φ.Π.Α. {vat_rate}%", vat, False),
                             ("ΣΥΝΟΛΟ ΠΛΗΡΩΜΗΣ", gross, True)):
        # right-align both, or a long label (ΣΥΝΟΛΟ ΠΛΗΡΩΜΗΣ) runs into the number
        d.text((1340, y), label, font=font(28, bold), fill="black", anchor="ra")
        d.text((1564, y), euro(val) + " €", font=font(28, bold), fill="black", anchor="ra")
        y += 48

    d.text((90, H - 260), f"Τρόπος πληρωμής: {rng.choice(['Μετρητά','Κατάθεση σε τράπεζα','Επιταγή 60 ημερών'])}",
           font=font(25), fill="black")
    d.text((90, H - 210), f"ΜΑΡΚ: {rng.randint(100000000, 999999999)}   "
                          f"UID: {''.join(rng.choice('0123456789ABCDEF') for _ in range(20))}",
           font=font(22), fill=(70, 70, 70))

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"invoice_{seed:03d}")
    # print-and-scan noise, so the test isn't run on a perfect render
    if seed % 3 == 0:
        img = img.rotate(rng.uniform(-0.7, 0.7), resample=Image.BICUBIC, fillcolor="white")
    img.save(stem + ".png", dpi=(200, 200))

    truth = {
        "issuer_name": issuer[0], "issuer_vat": afm_i,
        "client_name": client[0], "client_vat": afm_c,
        "invoice_number": num, "invoice_date": date,
        "net_amount": net, "vat_rate": vat_rate, "vat_amount": vat, "gross_amount": gross,
        "line_items": [{"description": r[0], "quantity": r[1], "unit_price": r[2], "line_total": r[3]} for r in rows],
    }
    json.dump(truth, open(stem + ".truth.json", "w"), ensure_ascii=False, indent=1)
    return stem


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out = sys.argv[2] if len(sys.argv) > 2 else "samples"
    for i in range(1, n + 1):
        print(make(i, out))
