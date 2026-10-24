#!/usr/bin/env python
"""Turn a clean render into paperwork that has actually been through an office.

A number measured on a perfect PNG is not a number anyone should buy. These are
the six things that happen to a real invoice before it reaches an accountant:
it gets faxed, scanned crooked, photographed on a desk, saved as a bad JPEG,
photocopied until the toner thins, or printed on a near-empty cartridge.
"""
import os
import sys

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def _noise(img, sigma):
    a = np.asarray(img).astype(np.float32)
    a += np.random.normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def scan_skew(img):
    """a sheet fed crooked into a scanner, slightly soft"""
    img = img.rotate(1.8, resample=Image.BICUBIC, fillcolor="white", expand=False)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return _noise(img, 6)


def phone_photo(img):
    """photographed on a desk: perspective, uneven light, shadow down one side"""
    w, h = img.size
    img = img.transform((w, h), Image.PERSPECTIVE,
                        _coeffs([(0, 0), (w, 0), (w, h), (0, h)],
                                [(18, 12), (w - 6, 0), (w - 24, h - 9), (9, h - 16)]),
                        resample=Image.BICUBIC, fillcolor="white")
    grad = np.linspace(1.0, 0.62, w)[None, :, None]
    a = np.asarray(img).astype(np.float32) * grad
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return _noise(img.filter(ImageFilter.GaussianBlur(0.8)), 5)


def fax(img):
    """1-bit, low resolution, the way a fax or a cheap scanner delivers it"""
    w, h = img.size
    img = img.convert("L").resize((w // 2, h // 2), Image.BILINEAR)
    img = img.point(lambda p: 255 if p > 150 else 0).convert("RGB")
    return img.resize((w, h), Image.NEAREST)


def bad_jpeg(img, tmp="/tmp/_deg.jpg"):
    """emailed at quality 25 after being resized twice"""
    w, h = img.size
    img.resize((int(w * 0.6), int(h * 0.6)), Image.BILINEAR).save(tmp, quality=25)
    return Image.open(tmp).convert("RGB").resize((w, h), Image.BICUBIC)


def photocopy(img):
    """third-generation copy: grey background, thin strokes, speckle"""
    img = ImageEnhance.Contrast(img).enhance(0.55)
    img = ImageEnhance.Brightness(img).enhance(1.12)
    a = np.asarray(img).astype(np.float32)
    speck = (np.random.rand(*a.shape[:2]) < 0.004)[:, :, None]
    a = np.where(speck, 40, a)
    return _noise(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), 9)


def low_toner(img):
    """printer running out: strokes break up, horizontal banding"""
    a = np.asarray(img.convert("L")).astype(np.float32)
    bands = (np.sin(np.linspace(0, 60, a.shape[0])) * 18)[:, None]
    a = np.clip(a + bands + 38, 0, 255)
    img = Image.fromarray(a.astype(np.uint8)).convert("RGB")
    return _noise(img.filter(ImageFilter.GaussianBlur(0.4)), 4)


def _coeffs(src, dst):
    import numpy as np
    A = []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    A = np.array(A, dtype=np.float64)
    B = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, B)


MODES = {"scan": scan_skew, "photo": phone_photo, "fax": fax,
         "jpeg": bad_jpeg, "copy": photocopy, "toner": low_toner}


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "samples"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "samples_hard"
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(7)
    import shutil
    names = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
    for name in names:
        img = Image.open(os.path.join(src_dir, name)).convert("RGB")
        stem = name[:-4]
        for mode, fn in MODES.items():
            fn(img.copy()).save(os.path.join(out_dir, f"{stem}_{mode}.png"))
            truth = os.path.join(src_dir, stem + ".truth.json")
            if os.path.exists(truth):
                shutil.copy(truth, os.path.join(out_dir, f"{stem}_{mode}.truth.json"))
        print(f"{stem}: {' '.join(MODES)}")


if __name__ == "__main__":
    main()
