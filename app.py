import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


# ---------------------------
# Load Model & Tokenizer
# ---------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("next_word_lstm.h5")


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as handle:
        return pickle.load(handle)


model = load_model()
tokenizer = load_tokenizer()

max_sequence_len = model.input_shape[1] + 1


# ---------------------------
# Prediction Function
# ---------------------------

def predict_next_word(model, tokenizer, text, max_sequence_len):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    prediction = model.predict(token_list, verbose=0)

    predicted_word_index = np.argmax(prediction, axis=1)[0]

    return tokenizer.index_word.get(
        predicted_word_index,
        "Word not found"
    )


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Next Word Prediction using LSTM")

st.markdown(
    """
    Predict the next word using an LSTM model trained on Shakespeare's Hamlet.
    """
)

input_text = st.text_input(
    "Enter a phrase",
    placeholder="To be or not to be"
)

if st.button("Predict Next Word"):

    if input_text.strip():

        next_word = predict_next_word(
            model,
            tokenizer,
            input_text.lower(),
            max_sequence_len
        )

        st.success(f"Predicted Next Word: **{next_word}**")

    else:
        st.warning("Please enter some text.")


st.markdown("---")
st.caption("Built with TensorFlow, LSTM and Streamlit")
