import tensorflow as tf
from tensorflow.keras import datasets, layers, models #type:ignore
import os
import scraper
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.preprocessing import image #type:ignore
# Define your dataset directory and categories
# scraper.save_preprocessed_google_image_srcs("dog")
DATADIR = "grouperdata"
CATEGORIES = ["cat","dog","rat","tigers"]  # replace with your categories
# n=len(CATEGORIES)
# # Define a function to create training data
# def create_training_data():
#     X = []
#     y = []
#     for category in CATEGORIES:
#         path = DATADIR + r"/" + category
        
#         class_num = CATEGORIES.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img))
#                 X.append(img_array)
#                 y.append(class_num)
#             except Exception as e:
#                 pass
#     return np.array(X), np.array(y)

# # Call the function to get training data
# X, y = create_training_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Build the CNN model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# # Add dense layers on top
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(n)) # n is the number of classes

# # Compile and train the model
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# history = model.fit(X_train, y_train, epochs=40)


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

# # Use the function to predict an image class
# img_path = "images.jpeg"  # replace with your image path
# predicted_class = predict_image_class(model, img_path, 512)  # replace 512 with your image size
# print(f"The predicted class is: {CATEGORIES[predicted_class]}")
# model.save('my_model.keras')

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Use the function to predict an image class
img_path = "images.jpeg"  # replace with your image path
predicted_class = predict_image_class(model, img_path, 512)  # replace 512 with your image size
print(f"The predicted class is: {CATEGORIES[predicted_class]}")