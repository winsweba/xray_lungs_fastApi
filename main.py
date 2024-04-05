from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "localhost:8000"
]
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    # allow_origins=origins,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

myModel="lungs_model.h5"
MODEL = tf.keras.models.load_model(myModel)

CLASS_NAMES = ["Normal", "Heart_failure",]

@app.get("/ping")
async def ping():
    return "Hello i am  alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predictions")
async def predictions(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    print(image.shape)
   
    image_batch = np.expand_dims(image, 0)
    
    reshaping_image = tf.image.resize(image_batch, [256, 256,])
    
    predictions =  MODEL.predict(reshaping_image)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }
   


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)