from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()

MODEL=tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES=['Early Blight', 'Late Blight', 'Healthy']


@app.get("/ping")
async def ping():
    return "Hello, I am alive!"

def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    # bytes=await file.read()
    image=read_file_as_image(await file.read())
    img_batch=np.expand_dims(image, axis=0)
    prediction=MODEL.predict(img_batch)
    pass

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
