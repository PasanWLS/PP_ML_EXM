import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras import models
from PIL import Image

# Creating the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = models.load_model("best.h5")

# Folder to save uploaded images
UPLOAD_FOLDER = "G:/ML_Exam/ml-2023-test/uploadedImages"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if the post request has the 'files' part
        if 'files' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        imagefile = request.files['files']
        
        # If user does not select file, browser also submits an empty part without filename
        if imagefile.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Verify if the uploaded file is an image
        try:
            image = Image.open(imagefile)
        except IOError:
            return jsonify({"error": "Invalid image format"}), 400

        # Resize image to the size that was used to train the model
        image = image.resize((64, 64))

        # Save the user input image
        image_path = os.path.join(UPLOAD_FOLDER, imagefile.filename)
        image.save(image_path)

        # Convert the image to an array
        image = np.array(image)

        # Expand dimensions to match the input shape required by the model
        expanded_image = np.expand_dims(image, axis=0)

        # Make a prediction
        x0predicted = model.predict(expanded_image)

        # Flatten the result and determine the classification
        result = np.array(x0predicted).flatten()

        # Example binary classification: adjust for multi-class models
        if result >= 0.5:
            response = "Positive"
        else:
            response = "Negative"

        # Log the result for debugging purposes
        print(result)
        print(response)

        return jsonify({"prediction": response, "confidence": result.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Running the app with IPv4 Address of the device
if __name__ == "__main__":
    app.run(host='192.168.145.1', port=5000)
