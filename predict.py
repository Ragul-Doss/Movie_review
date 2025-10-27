#creating the UI of model using streamlit

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizer import tokenizer, max_lenght, trunc_type
import streamlit as st
import numpy as np

# ✅ Fix legacy loading issue
tf.keras.config.disable_legacy_model_loading()

st.title("Movie Review Sentiment Analysis")
st.write("This ML model will guess if a given review is positive or negative by using NLP. "
         "This model was trained using Tensorflow and was trained on the imdb-dataset of movie reviews.")

with st.spinner("Loading Model....."):
    new_model = tf.keras.models.load_model('model/sentiment-analysis-model.h5')

pred_review_text = st.text_input("Enter your review")

if pred_review_text.strip() != '':
    with st.spinner("Tokenizing Text....."):
        # Tokenize and pad text
        pred_seq = tokenizer.texts_to_sequences([pred_review_text])
        pred_padded = pad_sequences(pred_seq, maxlen=max_lenght, truncating=trunc_type)

        # ❌ REMOVE expand_dims (this was giving wrong ndim)
        # pred_padded = np.expand_dims(pred_padded, axis=1)

        # ✅ Directly predict (shape will be (1, max_len))
        val = new_model.predict(pred_padded)[0][0]

    st.subheader("The given review was : ")
    if val > 0.5:
        st.success(f"✅ Positive Review ({val:.2f})")
    else:
        st.error(f"❌ Negative Review ({val:.2f})")
