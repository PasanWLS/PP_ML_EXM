import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from keras import models
from PIL import Image

#creating the flask app
app = Flask(__name__)

#load the h5 model
#the example model given
model = models.load_model("best.h5")

@app.route("/predict", methods=["POST"])
def predict():
    #reading the request
    imagefile = request.files['files']
    if imagefile:
        #reading the user input
        image = Image.open(imagefile)

        #resizing the image read to the size that was used to train the model
        image = image.resize((64,64))

        #saving the user input image
        image_path = "G:\\ML_Exam\\ml-2023-test\\uploadedImages\\" + imagefile.filename
        image.save(image_path)

        #converting the image to an array
        image = np.array(image)

        #converting array to higher dimension
        expanded_image = np.expand_dims(image, axis=0)

        #prediction
        x0predicted = model.predict(expanded_image)

        #reducing the higher dimension array to 1D
        result = np.array(x0predicted).flatten()

        if result>= 0.5:
            response = "Positive"
        elif result<0.5:
            response = "Negative"
        print(result)
        print(response)

    return response
    #renders the result to POSTMAN, HTML page

if __name__ == "__main__":
    app.run(host='192.168.145.1', port=5000)
    #Changes with the IPv4 Address of the device