# Client: client.py
import requests
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define client-side encoder model
def create_encoder_model():
    input_img = Input(shape=(128, 128, 3))  # Resize images to 128x128
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Flatten()(x)
    model = Model(input_img, encoded)
    return model

encoder_model = create_encoder_model()

# Preprocess an image to be passed to the encoder
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Resize image
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Get features from the image
def get_image_features(image_path):
    img_array = preprocess_image(image_path)
    encoded_features = encoder_model.predict(img_array)
    return encoded_features.tolist()

# Send features to the server for classification
def send_features_to_server(features):
    url = 'http://localhost:8080/classify'
    data = {'features': features}
    print("Sending features to server:", data)  # Debugging line
    response = requests.post(url, json=data)
    print("Server response status code:", response.status_code)  # Debugging line
    print("Server response text:", response.text)  # Debugging line
    if response.status_code != 200:
        raise Exception("Error response from server: " + response.text)
    return response.json()

# Route to upload images and get results
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    file.save('uploaded_image.jpg')  # Save uploaded image
    features = get_image_features('uploaded_image.jpg')
    response = send_features_to_server(features)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)