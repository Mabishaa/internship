from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()
model = load_model("./model.h5")

@app.get("/")
def root():
    return {"message": "Image classification API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "apple" if prediction < 0.5 else "orange"
    confidence = float(prediction) if label == "orange" else float(1 - prediction)

    return JSONResponse(content={"prediction": label, "confidence": confidence})
