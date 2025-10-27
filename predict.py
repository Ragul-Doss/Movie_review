import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.models import Model
from tokenizer import tokenizer, max_lenght, trunc_type, vocab_size, embedding_dimension
import streamlit as st
import numpy as np

st.title("Movie Review Sentiment Analysis")
st.write("This ML model will guess if a given review is positive or negative by using NLP. "
         "This model was trained using Tensorflow and was trained on the imdb-dataset of movie reviews.")

with st.spinner("Loading model..."):
    # load the sub-model (without embedding)
    base_model = tf.keras.models.load_model(
        "model/sentiment-analysis-model.h5",
        safe_mode=False
    )

    # rebuild correct embedding + base model
    inp = Input(shape=(max_lenght,))
    x = Embedding(vocab_size, embedding_dimension, input_length=max_lenght)(inp)
    out = base_model(x)

    new_model = Model(inp, out)

pred_review_text = st.text_input("Enter your review")

if pred_review_text.strip():
    pred_seq = tokenizer.texts_to_sequences([pred_review_text])
    pred_padded = pad_sequences(pred_seq, maxlen=max_lenght, truncating=trunc_type)

    val = new_model.predict(pred_padded)[0][0]

    st.subheader("The given review was : ")
    if val > 0.5:
        st.success(f"✅ Positive Review ({val:.2f})")
    else:
        st.error(f"❌ Negative Review ({val:.2f})")
