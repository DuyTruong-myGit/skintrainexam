from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from infer import predict

app = FastAPI()

@app.get("/")
def health():
    return {"status": "model-ready"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return predict(image)
