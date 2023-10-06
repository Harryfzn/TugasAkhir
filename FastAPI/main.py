
from fastapi import FastAPI
import uvicorn
from keras.models import load_model
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
import os

app = FastAPI()
model = load_model("model.h5")

@app.get("/")
async def root():
    return {"message": "Connect"}


def predict(filename):
    BATCH_SIZE = 32
    IMAGE_SIZE = (200,200)
    image_path = filename
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    image = np.vstack([x])
    classes = model.predict(image, batch_size=BATCH_SIZE)
    classes = np.argmax(classes)
    
    if classes==0:
        kelas = 'Hawar Daun Alternaria SP'
    elif classes==1:
        kelas = 'Busuk Daun Bremia lactuae '
    elif classes==2:
        kelas = 'Sehat'

    return kelas

@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "PNG")
    if not extension:
        return "Image must be jpg or png format!"
    
    folder = "image"
    file_location = os.path.join(folder, file.filename)
    image = file.file.read()
    with open(file_location, 'wb') as f:
        f.write(image)

    prediction = predict(file_location)
    result = {
        "prediction": prediction,
        "image_name": file.filename,
        "image_path": file_location,
        "message":"prediksi berhasil"
        }
    return result

@app.get("/get_info/")
async def get_info(info_type: str = None):
    
    informasi = {     
        "model":"CNN Arsitektur DenseNet201",
        "fungsi_model":"Memprediksi Penyakit pada Tanaman Selada",
        "nama_pengguna":"Harry Akbar Fauzan",
        "Email":"harryakbar470@gmail.com"
    }
    return informasi

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


