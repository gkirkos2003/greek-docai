import sys, time, json
from paddleocr import PaddleOCR

model = sys.argv[1] if len(sys.argv) > 1 else "el_PP-OCRv5_mobile_rec"
imgs  = sys.argv[2:] or ["samples/invoice_001.png"]

ocr = PaddleOCR(
    text_recognition_model_name=model,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu:0",
)
for img in imgs:
    t = time.time()
    res = ocr.predict(img)
    dt = time.time() - t
    for r in res:
        d = r.json["res"] if isinstance(r.json, dict) and "res" in r.json else r.json
        texts  = d.get("rec_texts", [])
        scores = d.get("rec_scores", [])
        print(f"\n=== {img}  model={model}  {dt:.2f}s  lines={len(texts)}")
        for s, tx in zip(scores, texts):
            print(f"  {s:.3f}  {tx}")
