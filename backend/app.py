from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

# === [1] Initialize App === #
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# === [2] Load Model and Label Classes === #
model = joblib.load('rfc_model.pkl')
label_classes = np.load('label_classes.npy', allow_pickle=True)

# === [3] Health Check === #
@app.route('/')
def home():
    return jsonify({'message': 'Sign Language Model API is running'})

# === [4] Prediction Endpoint === #
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'landmarks' not in data:
        return jsonify({'error': 'Missing landmark data'}), 400

    try:
        landmarks = np.array(data['landmarks']).reshape(1, -1)
        prediction = model.predict(landmarks)
        label = label_classes[prediction[0]]
        return jsonify({'prediction': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === [5] Run App (Only for local testing) === #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
