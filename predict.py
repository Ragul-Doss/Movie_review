import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

# ---- Load tokenizer ----
with open("model/tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

from tensorflow.keras.preprocessing.text import tokenizer_from_json
tokenizer = tokenizer_from_json(tokenizer_data)

# ---- (optional) load label map ----
with open("model/label_encoder.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

# Model constants (must match training)
MAX_LEN = 120
TRUNC_TYPE = "post"

st.title("🎬 Movie Review Sentiment Analysis")
st.write("This app predicts if a movie review is **Positive or Negative** using a trained LSTM model.")

# ---- Load model (.keras format) ----
@st.cache_resource
def load_sentiment_model():
    return tf.keras.models.load_model("model/sentiment_model.keras")

model = load_sentiment_model()

# ---- UI Input ----
user_input = st.text_area("✍️ Enter your movie review here:", "")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # preprocess input exactly like training
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, truncating=TRUNC_TYPE)

        # model prediction
        pred = model.predict(padded)[0][0]  # float probability
        
        if pred > 0.5:
            st.success(f"✅ Positive review ({pred:.2f} confidence)")
        else:
            st.error(f"❌ Negative review ({1 - pred:.2f} confidence)")
