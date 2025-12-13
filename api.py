from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import os
from infer import predict

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    temp_name = f"/tmp/{uuid.uuid4()}.{ext}"

    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict(temp_name)
    os.remove(temp_name)

    return result
