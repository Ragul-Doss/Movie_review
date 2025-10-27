import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# --------------------- Load Tokenizer ---------------------
with open("model/tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# --------------------- Load Label Map ---------------------
with open("model/label_encoder.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

# --------------------- Load Model -------------------------
model = tf.keras.models.load_model("model/sentiment_model.keras")

# --------------------- Prediction Function ----------------
MAX_LEN = 120  # must match training

def predict_sentiment(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, truncating="post")
    prediction = model.predict(padded, verbose=0)[0][0]
    label = "positive" if prediction >= 0.5 else "negative"
    return label, float(prediction)

# --------------------- Streamlit UI -----------------------
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and the model will predict whether it's **positive** or **negative**.")

user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if len(user_input.strip()) == 0:
        st.warning("Please enter some text before predicting.")
    else:
        label, score = predict_sentiment(user_input)
        st.success(f"Prediction: **{label.upper()}** ({score:.4f})")
