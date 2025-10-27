import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------- Page Config ---------------------
st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="centered")

# --------------------- Load Tokenizer ---------------------
with open("model/tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())

# --------------------- Load Label Map ---------------------
with open("model/label_encoder.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

# --------------------- Load Model -------------------------
model = tf.keras.models.load_model("model/sentiment_model.keras")

MAX_LEN = 120

# --------------------- Prediction Function ----------------
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, truncating="post")
    pred = model.predict(pad, verbose=0)[0][0]
    label = "positive" if pred >= 0.5 else "negative"
    return label, float(pred)

# --------------------- Streamlit UI -----------------------
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Analyze whether a movie review is **Positive** or **Negative** using AI.")

user_input = st.text_area("📝 Enter movie review here:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review text first.")
    else:
        label, score = predict_sentiment(user_input)

        # Sentiment Emoji
        emoji = "😃" if label == "positive" else "😞"

        st.markdown("---")
        st.subheader(f"Result: {emoji} **{label.upper()}**")

        # Probability Bar
        st.write("### Confidence:")
        st.progress(score if label == "positive" else 1 - score)
        st.write(f"**Confidence Score:** {score:.4f}")

        st.markdown("---")
