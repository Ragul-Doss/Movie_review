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
st.markdown(
    """
    <h1 style='text-align:center;'>🎬 Movie Review Sentiment</h1>
    <p style='text-align:center; color:gray;'>
        AI-powered sentiment classifier for movie reviews
    </p><br>
    """,
    unsafe_allow_html=True
)

# ---- Card Container ----
with st.container():
    st.markdown(
        """
        <div style="
            background-color: #f9f9f9;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        ">
        """,
        unsafe_allow_html=True
    )

    user_input = st.text_area("📝 Enter your movie review:", height=140)

    predict_btn = st.button("🔍 Predict Sentiment", use_container_width=True)

    if predict_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review text first.")
        else:
            label, score = predict_sentiment(user_input)
            emoji = "😃" if label == "positive" else "😞"

            st.markdown("---")
            st.markdown(f"<h3 style='text-align:center;'>Result: {emoji} <b>{label.upper()}</b></h3>", unsafe_allow_html=True)

            # Show Confidence
            st.write("### Confidence:")
            conf = score if label == "positive" else (1 - score)
            st.progress(conf)
            st.write(f"**Confidence Score:** {score:.4f}")

            st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)
