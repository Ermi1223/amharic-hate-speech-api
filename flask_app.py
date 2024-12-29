# flask_app.py
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model_name = "ermi8/amharic-hate-speech-detection"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Prediction function
def predict_hate_speech(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "ጥላቻ" if prediction == 1 else "መልካም"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = predict_hate_speech(text)
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
