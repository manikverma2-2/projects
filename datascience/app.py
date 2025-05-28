import streamlit as st
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("sentiment_model.h5")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 200
sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # remove HTML
    text = re.sub(r'[^a-zA-Z ]', '', text)  # remove non-letters
    return text.lower()

def classify_sentiment(pred_probs):
    pred_class = np.argmax(pred_probs)
    return sentiment_labels[pred_class - 1]  # Because we mapped -1 → 0, 0 → 1, 1 → 2

st.title("🎬 IMDB Sentiment Analysis")
st.write("Enter a movie review and get the predicted sentiment!")

user_input = st.text_area("Your Review", "")

if st.button("Analyze Sentiment"):
    if user_input:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        pred_probs = model.predict(padded)
        result = classify_sentiment(pred_probs)
        st.success(f"Predicted Sentiment: **{result}**")
    else:
        st.warning("Please enter a review first.")
