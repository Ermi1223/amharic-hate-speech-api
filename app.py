import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "amengemeda/amharic-hate-speech-detection-mBERT"
    # model_name ="ermi8/amharic-hate-speech-detection-mBERT"
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Prediction function
def predict_hate_speech(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "ጥላቻ" if prediction == 1 else "መልካም"

# Streamlit UI
def main():
    st.title("Amharic Hate Speech Detection")
    st.write("Detect whether a given Amharic text is hate speech or not.")

    # Input text
    text = st.text_area("Enter Amharic text:", height=200)

    if st.button("Analyze"):
        if text.strip():
            with st.spinner("Analyzing..."):
                result = predict_hate_speech(text)
            st.success(f"Prediction: {result}")
        else:
            st.error("Please enter some text to analyze.")

# Run the app
if __name__ == "__main__":
    main()
