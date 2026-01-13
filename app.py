import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertModel
import numpy as np
import re
from flask import Flask, request, jsonify, render_template
import traceback
import os

# --- 1. SETUP: Load Model and Tokenizer ---
print("--- Loading final model and tokenizer. This may take a moment... ---")

# Define the text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Load the tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# Load the trained Keras model
model = None
try:
    # *** UPDATED: Now loading the FINAL and BEST model ***
    model_path = 'fake_review_model_final.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CRITICAL ERROR: The model file '{model_path}' was not found. Please run the '3f-final-training.py' notebook.")
        
    custom_objects = {'TFAlbertModel': TFAlbertModel}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"--- Model '{model_path}' loaded successfully! ---")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 2. PREDICTION FUNCTION ---
def predict_review(review_text):
    """Takes raw text, processes it, and returns a prediction."""
    try:
        cleaned = clean_text(review_text)
        inputs = tokenizer(
            cleaned,
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            max_length=256
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        prediction_prob = model.predict([input_ids, attention_mask])
        confidence = prediction_prob[0][0]

        if confidence > 0.5:
            label = "Likely Fake"
        else:
            label = "Likely Genuine"
        return label, confidence
    except Exception:
        print("\n--- AN ERROR OCCURRED INSIDE THE PREDICTION FUNCTION ---")
        print(traceback.format_exc())
        return "Error", 0.0

# --- WARM UP THE MODEL ---
print("--- Warming up the final model... ---")
predict_review("This is a test to initialize the model.")
print("--- Model is ready and running! ---")


# --- 3. FLASK WEB APP ---
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.json['review']
    label, confidence = predict_review(review)
    if label == "Error":
        return jsonify({'error': 'Prediction failed on the server.'}), 500
    return jsonify({
        'prediction_label': label,
        'confidence_score': f"{confidence:.2f}"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=False)
