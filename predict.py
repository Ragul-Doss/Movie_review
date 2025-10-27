#creating the UI of model using streamlit

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.models import Model

from tokenizer import tokenizer , max_lenght , trunc_type, vocab_size, embedding_dimension

import streamlit as st
import numpy as np

st.title("Movie Review Sentiment Analysis")
st.write("This ML model will guess if a given review is positive or negative by using NLP. "
         "The model was trained using Tensorflow on an IMDB dataset of real movie reviews.")

with st.spinner("Loading Model....."):
    # Load your base model (missing embedding)
    base_model = tf.keras.models.load_model('model/sentiment-analysis-model.h5')

    # Recreate embedding wrapper
    input_ids = Input(shape=(max_lenght,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dimension, input_length=max_lenght)(input_ids)
    output = base_model(x)

    # Final model that accepts token ids
    new_model = Model(input_ids, output)

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
