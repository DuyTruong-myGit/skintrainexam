import io
import os
import requests
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from infer import infer_image   # 🔥 IMPORT LOGIC CHUẨN

# =========================
# APP CONFIG
# =========================
app = FastAPI(
    title="Skin Disease AI API",
    description="AI-powered skin disease classification (YOLOv11-cls)",
    version="2.0.0"
)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REQUEST MODEL (URL INPUT)
# =========================
class ImageURLRequest(BaseModel):
    image_url: str

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Skin AI",
        "model": "YOLOv11-cls",
        "endpoints": {
            "predict_upload": "POST /predict",
            "predict_url": "POST /predict-url",
            "health": "GET /health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

# =========================
# IMAGE LOADER
# =========================
def load_image_from_upload(file: UploadFile) -> Image.Image:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = file.file.read()
    return Image.open(io.BytesIO(contents))


def load_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot load image from URL")

# =========================
# PREDICT (UPLOAD FILE)
# =========================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict skin disease from uploaded image
    """
    start_time = datetime.now()

    image = load_image_from_upload(file)
    result = infer_image(image)

    result["timestamp"] = datetime.now().isoformat()
    return result

# =========================
# PREDICT (IMAGE URL)
# =========================
@app.post("/predict-url")
async def predict_image_url(payload: ImageURLRequest):
    """
    Predict skin disease from image URL (backend use-case)
    """
    start_time = datetime.now()

    image = load_image_from_url(payload.image_url)
    result = infer_image(image)

    result["timestamp"] = datetime.now().isoformat()
    return result

# =========================
# LOCAL RUN
# =========================
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
