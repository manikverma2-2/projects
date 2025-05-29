import streamlit as st
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved model and tokenizer
model = load_model("sentiment_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 200  # Should match what was used in training

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Remove non-letters
    return text.lower()

def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)
    class_idx = np.argmax(pred, axis=1)[0]
    sentiment = label_encoder.inverse_transform([class_idx])[0]
    return sentiment

# Streamlit UI
st.title("🎭 Sentiment Analysis for Movie Reviews")
review = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if review.strip():
        sentiment = predict_sentiment(review)
        label_map = {-1: "😡 Negative", 0: "😐 Neutral", 1: "😊 Positive"}
        st.success(f"Predicted Sentiment: {label_map[sentiment]}")
    else:
        st.warning("Please enter a review.")
