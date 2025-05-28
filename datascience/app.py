import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Constants
MAX_LEN = 200

# Load model and preprocessing tools
@st.cache_resource
def load_all():
    model = load_model("sentiment_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_all()

# Clean input text
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)  # Remove HTML
    text = re.sub(r"[^a-zA-Z ]", "", text)  # Remove non-letters
    return text.lower()

# Predict sentiment
def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)
    label_index = np.argmax(pred, axis=1)[0]
    sentiment = label_encoder.inverse_transform([label_index])[0]
    return sentiment

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")
st.title("🎬 Movie Review Sentiment Classifier")
st.markdown("Enter a movie review below to predict its sentiment using a deep learning model.")

review_input = st.text_area("📝 Your Review", height=200)

if st.button("Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment = predict_sentiment(review_input)
        emoji = {"Positive": "😄", "Neutral": "😐", "Negative": "😠"}.get(sentiment, "")
        st.success(f"**Sentiment:** {sentiment} {emoji}")
