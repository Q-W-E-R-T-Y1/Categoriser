import flask
from flask import Flask, jsonify, request
import tensorflow as tf
from PIL import Image
import numpy as np
CATEGORIES = ["cat","dog","rat","tigers"]

app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model('my_model.keras')
def predict_image_class(model, img_path, img_size):
    # Load and resize the image using PIL
    img = Image.open(img_path)
    img = img.resize((img_size, img_size))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Scale the image pixels by 255 (or use a scaler from sklearn here)
    img_array_rescaled = img_array / 255

    # Reshape the data for the model
    reshaped_img = img_array_rescaled.reshape(1, img_size, img_size, 3)

    # Make prediction
    prediction = model.predict(reshaped_img)
    
    # Get the class with highest probability
    predicted_class = np.argmax(prediction)

    return predicted_class
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    file = request.files['file']
    
    # Use your existing function to predict the image class
    predicted_class = predict_image_class(model, file, 512)

    # Return the prediction
    return jsonify({'predicted_class': CATEGORIES[predicted_class]})

if __name__ == '__main__':
    app.run(debug=True)

