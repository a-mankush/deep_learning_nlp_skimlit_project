import tensorflow as tf
from spacy.lang.en import English
from pprint import pprint
import streamlit as st
import time

from utils import preprocess_text, get_predictions_labels, abstract_to_sentence

# loaded_model = tf.keras.models.load_model("skimLit_8b")

nlp = English()
sentencizer = nlp.add_pipe("sentencizer")


@st.cache_resource()
def get_predictions(inputs, _loaded_model):
    a, b, c, d = preprocess_text(inputs)
    return _loaded_model.predict((a, b, c, d))


@st.cache_resource()
def load_skimlit_model():
    return tf.keras.models.load_model("C:/Users/aman kushwaha/Downloads/aab/modelb8/skimLit_8b")


def main():
    st.header("Welcome to the SkimLit Project")
    abstract = st.text_area("Enter your abstract here")

    if st.button("Predict"):
        if abstract:
            with st.expander("Prediction Process", expanded=True):
                bar = st.progress(0)
                model = load_skimlit_model()
                bar.progress(20)
                predictions = get_predictions(abstract, model)
                bar.progress(80)
                pred_classes = get_predictions_labels(predictions)
                bar.progress(100)
            st.success("Prediction completed!")
            for i, sent in enumerate(abstract_to_sentence(abstract)):
                st.markdown(f"""
                - <i style="color:orange;">&#9872;</i> **`{pred_classes[i]}`**: {sent}
                """, unsafe_allow_html=True)

        else:
            st.error("Please enter an abstract first.")


if __name__ == "__main__":
    main()
