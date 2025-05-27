import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load fine-tuned model
MODEL_PATH = "model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

label_map = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😊"}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    label = torch.argmax(probs).item()
    return label_map[label], probs.tolist()

# Streamlit App UI
st.set_page_config(page_title="Sentiment Classifier", layout="wide")
st.title("🎬 IMDB Sentiment Classifier (BERT)")
st.markdown("Upload a file or type your movie review to classify it as Positive, Neutral, or Negative.")

option = st.radio("Choose input method:", ["Type review", "Upload file"])

if option == "Type review":
    text_input = st.text_area("✍️ Enter your review:", height=150)
    if st.button("Predict Sentiment"):
        if text_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            label, probs = predict_sentiment(text_input)
            st.success(f"Predicted Sentiment: **{label}**")
            st.bar_chart(probs)

elif option == "Upload file":
    uploaded_file = st.file_uploader("📤 Upload a .txt or .csv file with reviews", type=["txt", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if "review" not in df.columns:
                st.error("CSV must contain a 'review' column.")
            else:
                df["Sentiment"], df["Confidence"] = zip(*df["review"].apply(predict_sentiment))
                st.dataframe(df[["review", "Sentiment"]])
        elif uploaded_file.name.endswith(".txt"):
            lines = uploaded_file.read().decode("utf-8").splitlines()
            results = [{"Review": line, "Sentiment": predict_sentiment(line)[0]} for line in lines if line.strip()]
            result_df = pd.DataFrame(results)
            st.dataframe(result_df)

