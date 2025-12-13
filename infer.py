from ultralytics import YOLO

# Load model 1 lần khi start app
model = YOLO("best.pt")

def predict(image_path: str):
    results = model(image_path)

    r = results[0]

    if hasattr(r, "probs") and r.probs is not None:
        # YOLO classification
        return {
            "top1": int(r.probs.top1),
            "confidence": float(r.probs.top1conf),
            "probs": r.probs.data.tolist()
        }

    # YOLO detection
    detections = []
    for box in r.boxes:
        detections.append({
            "cls": int(box.cls),
            "conf": float(box.conf),
            "xyxy": box.xyxy.tolist()
        })

    return detections
