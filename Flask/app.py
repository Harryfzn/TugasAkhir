from flask import Flask, render_template, request
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import numpy as np
import keras
import tensorflow as tf
import configparser as config
from flask import Flask, send_from_directory
from tkinter import image_names
from keras.preprocessing import image
from keras.models import load_model
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

app = Flask(__name__)

model = load_model('model.h5')

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')
    
@app.route('/templates/card.html', methods=['POST'])
def predict():
    IMAGE_SIZE = (200,200)
    BATCH_SIZE = 32

    uploaded = request.files["file-upload-field"]
    image_path = "./static/" + uploaded.filename
    uploaded.save(image_path)
    image_name =  uploaded.filename
    
 
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict(images, batch_size=BATCH_SIZE)
    classes = np.argmax(classes)
    
    if classes==0:
        kelas = 'Bercak Daun'
    elif classes==1:
        kelas = 'Busuk Daun'
    elif classes==2:
        kelas = "Sehat"
    else:
        kelas = 'error'


    return render_template('card.html', kelas = kelas, image_name = image_name, image_path = image_path)

if __name__ == "__main__":
    app.run(port=5000,debug=True)