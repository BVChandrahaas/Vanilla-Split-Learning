# Server: server.py
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

app = Flask(__name__)

# Define server-side classifier model
def create_classifier_model():
    encoded_input = Input(shape=(64 * 32 * 32,))  # Adjust input shape based on encoder output
    x = Dense(128, activation='relu')(encoded_input)
    x = Dense(10, activation='softmax')(x)  # Assuming 10 classes for classification
    model = Model(encoded_input, x)
    return model

classifier_model = create_classifier_model()

# Endpoint to process incoming features from client
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400

    client_features = np.array(data['features'])
    try:
        server_outputs = classifier_model.predict(client_features)
        return jsonify({"output": server_outputs.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return error as JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)