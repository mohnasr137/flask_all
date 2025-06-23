import os
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)


dl_model_path = os.path.join('model', 'my_model.h5')
coloring_model_path = os.path.join('model', 'autism_binary_coloring_modelv.h5')
handwriting_model_path = os.path.join('model', 'autism_binary_handwritin_modelv.h5')
ml_model_path = 'model/my_ml_model.h5'
scaler_path = 'model/scaler.pkl'


ml_model = tf.keras.models.load_model(ml_model_path)
scaler = joblib.load(scaler_path)


try:
    dl_model = load_model(dl_model_path)
except Exception as e:
    print(f"Error loading DL model: {e}")
    dl_model = None

try:
    coloring_model = load_model(coloring_model_path)
except Exception as e:
    print(f"Error loading Coloring model: {e}")
    coloring_model = None

try:
    handwriting_model = load_model(handwriting_model_path)
except Exception as e:
    print(f"Error loading Handwriting model: {e}")
    handwriting_model = None


def preprocess_image(img_path, target_size=(128, 128)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def predict_image(model, img_array):
    if model is None:
        return "Model not loaded"
    prediction = model.predict(img_array)
    return int(prediction[0][0] > 0.5)

# Child face route
@app.route('/childFace', methods=['POST'])
def predict_child_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files['image']
    img_path = os.path.join('uploads', img_file.filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    img_file.save(img_path)

    try:
        img_array = preprocess_image(img_path, (224, 224))
        prediction = predict_image(dl_model, img_array)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"prediction": prediction})

# Coloring model route
@app.route('/coloring', methods=['POST'])
def predict_coloring():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files['image']
    img_path = os.path.join('uploads', img_file.filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    img_file.save(img_path)

    try:
        img_array = preprocess_image(img_path)
        prediction = predict_image(coloring_model, img_array)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"prediction": prediction})

# Handwriting model route
@app.route('/handWriting', methods=['POST'])
def predict_handwriting():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files['image']
    img_path = os.path.join('uploads', img_file.filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    img_file.save(img_path)

    try:
        img_array = preprocess_image(img_path)
        prediction = predict_image(handwriting_model, img_array)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"prediction": prediction})

# ML Model route
@app.route('/predict', methods=['POST'])
def predict_ml():
    try:
        input_data = request.json['data']
        input_data = np.array(input_data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        prediction = ml_model.predict(input_data_scaled)
        prediction_binary = int(prediction[0][0] > 0.5)
        result = {
            'prediction': prediction_binary,
            'label': 'autistic' if prediction_binary == 1 else 'non-autistic'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
