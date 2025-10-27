#creating the UI of model using streamlit

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizer import tokenizer , max_lenght , trunc_type
import streamlit as st
import numpy as np

st.title("Movie Review Sentiment Analysis")
st.write("This ML model will guess if a given review is positive or negative by using NLP. "
         "The model was trained using Tensorflow on an IMDB dataset of real movie reviews.")

with st.spinner("Loading Model....."):
    new_model = tf.keras.models.load_model('model/sentiment-analysis-model.h5')

pred_review_text = st.text_input("Enter your review")

if pred_review_text.strip() != '':
    with st.spinner("Processing Review..."):
        pred_seq = tokenizer.texts_to_sequences([pred_review_text])
        pred_padded = pad_sequences(pred_seq, maxlen=max_lenght, truncating=trunc_type)

        val = new_model.predict(pred_padded)[0][0]   # single float value

    st.subheader("Prediction")
    if val > 0.5:
        st.success(f"✅ Positive Review ({val:.2f})")
    else:
        st.error(f"❌ Negative Review ({val:.2f})")
