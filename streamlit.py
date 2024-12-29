# app.py
import streamlit as st
import requests

# Streamlit UI
def main():
    st.title("Amharic Hate Speech Detection")
    st.write("Detect whether a given Amharic text is hate speech or not.")

    # Input text
    text = st.text_area("Enter Amharic text:", height=200)

    if st.button("Analyze"):
        if text.strip():
            with st.spinner("Analyzing..."):
                response = requests.post(
                    "http://127.0.0.1:5000/predict", json={"text": text}
                )

                if response.status_code == 200:
                    result = response.json().get('prediction')
                    st.success(f"Prediction: {result}")
                else:
                    st.error("Error in API response.")
        else:
            st.error("Please enter some text to analyze.")

# Run the app
if __name__ == "__main__":
    main()
