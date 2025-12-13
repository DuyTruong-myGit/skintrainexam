import time
import os
from typing import Dict, Any, List
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =========================
# CONFIG
# =========================
MODEL_PATH = os.getenv("MODEL_PATH", "model/best.pt")

# Ngưỡng confidence
MIN_CONFIDENCE = 0.55        # dưới mức này -> unknown_normal
TOPK = 5

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# CLASS MAP (CỐ ĐỊNH)
# THỨ TỰ PHẢI KHỚP VỚI DATASET TRAIN
# =========================
CLASS_MAP = {
    0: {
        "key": "actinic_keratosis",
        "name": "Actinic Keratosis",
        "risk": "moderate",
        "description": "Pre-cancerous lesion caused by sun damage",
        "recommendation": "Consult dermatologist for evaluation"
    },
    1: {
        "key": "basal_cell_carcinoma",
        "name": "Basal Cell Carcinoma",
        "risk": "high",
        "description": "Most common type of skin cancer",
        "recommendation": "Schedule dermatologist appointment"
    },
    2: {
        "key": "dermatofibroma",
        "name": "Dermatofibroma",
        "risk": "low",
        "description": "Benign fibrous skin nodule",
        "recommendation": "Usually harmless"
    },
    3: {
        "key": "melanoma",
        "name": "Melanoma",
        "risk": "critical",
        "description": "Most dangerous form of skin cancer",
        "recommendation": "URGENT medical attention required"
    },
    4: {
        "key": "nevus",
        "name": "Nevus",
        "risk": "low",
        "description": "Common mole",
        "recommendation": "Monitor for changes"
    },
    5: {
        "key": "normal_skin",
        "name": "Normal Skin",
        "risk": "none",
        "description": "Healthy human skin",
        "recommendation": "No action needed"
    },
    6: {
        "key": "pigmented_benign_keratosis",
        "name": "Pigmented Benign Keratosis",
        "risk": "low",
        "description": "Benign pigmented lesion",
        "recommendation": "Generally harmless"
    },
    7: {
        "key": "ringworm",
        "name": "Ringworm",
        "risk": "moderate",
        "description": "Fungal skin infection",
        "recommendation": "Antifungal treatment recommended"
    },
    8: {
        "key": "seborrheic_keratosis",
        "name": "Seborrheic Keratosis",
        "risk": "low",
        "description": "Benign skin growth",
        "recommendation": "No treatment required"
    },
    9: {
        "key": "squamous_cell_carcinoma",
        "name": "Squamous Cell Carcinoma",
        "risk": "high",
        "description": "Common skin cancer",
        "recommendation": "Consult dermatologist urgently"
    },
    10: {
        "key": "unknown_normal",
        "name": "Unknown / Non-skin",
        "risk": "unknown",
        "description": "Image is not related to human skin or diagnosis is unreliable",
        "recommendation": "Please upload a clear photo of human skin lesion"
    },
    11: {
        "key": "vascular_lesion",
        "name": "Vascular Lesion",
        "risk": "low",
        "description": "Blood vessel-related skin lesion",
        "recommendation": "Consult dermatologist if concerned"
    }
}

# =========================
# HELPER: UNKNOWN RESPONSE
# =========================
def unknown_response(confidence: float, topk: List[Dict[str, Any]], inference_ms: float):
    info = CLASS_MAP[10]
    return {
        "success": True,
        "prediction": info["key"],
        "full_name": info["name"],
        "risk_level": info["risk"],
        "confidence": round(confidence, 4),
        "description": info["description"],
        "recommendation": info["recommendation"],
        "top5": topk,
        "inference_time_ms": inference_ms
    }

# =========================
# MAIN INFERENCE FUNCTION
# =========================
def infer_image(image: Image.Image) -> Dict[str, Any]:
    start = time.time()

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Predict
    results = model.predict(image, verbose=False)
    result = results[0]
    probs = result.probs

    # Top-1
    top1_idx = int(probs.top1)
    top1_conf = float(probs.top1conf)

    # Top-K
    topk = []
    for idx, conf in zip(probs.top5, probs.top5conf):
        idx = int(idx)
        info = CLASS_MAP.get(idx)
        if not info:
            continue
        topk.append({
            "class_name": info["key"],
            "full_name": info["name"],
            "confidence": round(float(conf), 4),
            "risk_level": info["risk"]
        })

    inference_ms = round((time.time() - start) * 1000, 2)

    # =========================
    # UNKNOWN LOGIC
    # =========================

    # 1️⃣ Confidence quá thấp
    if top1_conf < MIN_CONFIDENCE:
        return unknown_response(top1_conf, topk, inference_ms)

    # 2️⃣ Model tự đoán ra unknown_normal
    if CLASS_MAP[top1_idx]["key"] == "unknown_normal":
        return unknown_response(top1_conf, topk, inference_ms)

    # =========================
    # NORMAL RESPONSE
    # =========================
    info = CLASS_MAP[top1_idx]

    return {
        "success": True,
        "prediction": info["key"],
        "full_name": info["name"],
        "risk_level": info["risk"],
        "confidence": round(top1_conf, 4),
        "description": info["description"],
        "recommendation": info["recommendation"],
        "top5": topk,
        "inference_time_ms": inference_ms
    }
