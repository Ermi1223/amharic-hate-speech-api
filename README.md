# Amharic Hate Speech Detection API

This project is an **Amharic language-based hate speech detection** system, designed to identify and classify offensive or harmful language in Amharic text. Built using the **Flask** framework and leveraging **BERT** models, the API detects hate speech in user-submitted text and categorizes it as either **"ጥላቻ" (Hate Speech)** or **"መልካም" (Non-Hate Speech)**.

The API integrates seamlessly into applications where text moderation or automated content review is necessary, especially for platforms serving Ethiopian and Amharic-speaking communities.

## Key Features:
- **Real-time Hate Speech Detection**: Classifies Amharic text into hate speech and non-hate speech categories.
- **Flask-Based API**: A lightweight, easy-to-use API built with Flask for integration into various applications.
- **BERT Model**: Utilizes pre-trained BERT models fine-tuned on an Amharic hate speech dataset for accurate text classification.
- **Simple Interface**: Easy-to-use endpoints that accept text input and return predictions.

## Installation and Usage:
1. Clone the repository:  
   ```bash
   git clone https://github.com/ermi1223/amharic-hate-speech-api.git
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:  
   ```bash
   python app.py
   ```
4. Send POST requests to the `/predict` endpoint with Amharic text to receive predictions.

## Technologies:
- **Flask**: Lightweight web framework for building the API.
- **Transformers (Hugging Face)**: Used for loading the pre-trained BERT model and tokenizer.
- **PyTorch**: Backend framework for the model inference.
- **Streamlit** (if integrating with a front-end): For building the UI to interact with the API.

## Example Request:
```bash
POST http://localhost:5000/predict
Content-Type: application/json
{
  "text": "ሰላም ሰዎች እንዴት ነህ/ነሽ?"
}
```

## Response:
```json
{
  "prediction": "መልካም"
}
```
