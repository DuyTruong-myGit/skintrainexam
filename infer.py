from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("best.pt")

# warm-up
_dummy = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))
model(_dummy)

def predict(image: Image.Image):
    r = model(image)[0]
    probs = r.probs

    cls_id = int(probs.top1)
    conf = float(probs.top1conf)

    return {
        "class_id": cls_id,
        "label": model.names[cls_id],
        "confidence": conf
    }
