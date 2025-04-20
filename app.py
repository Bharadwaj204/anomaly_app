# app.py
import streamlit as st
import joblib
import re

# Load the trained model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Text preprocessing function
def clean_text(text):
    """Cleans the text by lowercasing and removing punctuation."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Streamlit UI
st.set_page_config(page_title="App Review Anomaly Detector", page_icon="🕵️‍♂️")  # Page title and icon
st.title("🕵️‍♂️ Anomaly Detector for App Reviews")

# Text input area for the review
review = st.text_area("Paste an app review here 👇", height=200)

# Button to trigger prediction
if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        # Clean the input review text
        clean_review = clean_text(review)

        # Transform the review using the pre-trained vectorizer
        vector = vectorizer.transform([clean_review])

        # Predict if the review is genuine or fake using the model
        prediction = model.predict(vector)

        # Display result
        label = "Fake ❌" if prediction[0] == -1 else "Genuine ✅"
        color = "red" if label == "Fake ❌" else "green"

        # Display prediction with color
        st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
