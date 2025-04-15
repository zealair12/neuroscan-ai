from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image
import io
from app.utils.preprocess import preprocess_image
import gdown
import os

app = Flask(__name__)
MODEL_PATH = 'app/model/brain_model.pkl'
DRIVE_FILE_ID = '1GievvwojXM4mNNt3n03qiIrjhpFyg_9s'
DOWNLOAD_URL = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'

# Download model from Drive if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)


# Load model from the model folder
with open('app/model/brain_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess image (resize, flatten, etc.)
    processed = preprocess_image(image)

    # Make prediction
    prediction = model.predict([processed])
    confidence = getattr(model, "predict_proba", lambda x: [[1]])([processed])[0]

    return jsonify({
        'prediction': int(prediction[0]),
        'confidence': float(max(confidence))
    })

if __name__ == '__main__':
    app.run(debug=True)
