from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from detect_fake_news import train_model  # Assuming you already have this for training

MAX_LENGTH = 500
app = Flask(__name__)

# Enable CORS for the entire application
CORS(app)  # This will allow all domains to access the API

def load_model_and_tokenizer():
    """Load the trained model and tokenizer."""
    if not os.path.exists("fake_or_true_news_model.keras") or not os.path.exists("tokenizer.json"):
        return None, None
    model = tf.keras.models.load_model('fake_or_true_news_model.keras')
    with open("tokenizer.json", "r") as f:
        tokenizer = tokenizer_from_json(f.read())
    return model, tokenizer

def predict(text, model, tokenizer):
    """Predict whether the given text is fake or true news."""
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    label = "True News" if prediction > 0.5 else "Fake News"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

@app.route('/predict', methods=['POST'])
def api_predict():
    """API endpoint to predict fake or true news."""
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Text is required"}), 400

    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        return jsonify({"error": "Model or tokenizer not found. Please train first."}), 500

    label, confidence = predict(text, model, tokenizer)
    return jsonify({
        "label": label,
        "confidence": f"{confidence:.4f}"
    })

@app.route('/train', methods=['POST'])
def api_train():
    """API endpoint to train the model with new data."""
    data = request.get_json()
    text = data.get('text', '')
    label = data.get('label')

    if text and label is not None:
        train_model(extra_texts=[text], extra_labels=[label])
        return jsonify({"message": "Training completed with new data."}), 200
    return jsonify({"error": "Text and label are required for training."}), 400

@app.route('/view_user_data', methods=['GET'])
def api_view_user_data():
    """API endpoint to view user-labeled training data."""
    if os.path.exists("user_training_data.csv"):
        user_data = pd.read_csv("user_training_data.csv").to_dict(orient='records')
        return jsonify(user_data)
    else:
        return jsonify({"message": "No user training data found."})

@app.route('/clear_user_data', methods=['POST'])
def api_clear_user_data():
    """API endpoint to clear user-labeled training data."""
    if os.path.exists("user_training_data.csv"):
        os.remove("user_training_data.csv")
        return jsonify({"message": "User training data cleared."})
    else:
        return jsonify({"error": "No user training data to clear."})

if __name__ == '__main__':
    app.run(debug=True)
