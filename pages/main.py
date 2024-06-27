import streamlit as st
import tensorflow as tf
from spacy.lang.en import English
from utils import abstract_to_sentence, get_predictions_labels, preprocess_text

# loaded_model = tf.keras.models.load_model("MediScan_8b")

nlp = English()
sentencizer = nlp.add_pipe("sentencizer")


@st.cache_data()
def get_predictions(inputs, _loaded_model):
    a, b, c, d = preprocess_text(inputs)
    return _loaded_model.predict((a, b, c, d))


@st.cache_resource()
def load_MediScan_model():
    return tf.keras.models.load_model("skimLit_8b")


def prediction_and_display(abstract):
    bar = st.progress(0)
    model = load_MediScan_model()
    bar.progress(20)
    predictions = get_predictions(abstract, model)
    bar.progress(80)
    pred_classes = get_predictions_labels(predictions)
    bar.progress(100)
    st.success("Prediction completed!")
    for i, sent in enumerate(abstract_to_sentence(abstract)):
        st.markdown(
            f"""
                - <i style="color:orange;">&#9872;</i> **`{pred_classes[i]}`**: {sent}
                """,
            unsafe_allow_html=True,
        )


def main():
    st.header("Welcome to the MediScan")
    abstract = st.text_area("Enter your abstract here")
    with st.expander("Some example text to test the model"):
        """
        ### Example 1:
        This study aims to evaluate the long-term outcome of Children's Friendship Training,
        a parent-assisted social skills intervention for children. Prior research has shown Children's
        Friendship Training to be superior to wait-list control with maintenance of gains at 3-month follow-up.
        Participants were families of children diagnosed with autism spectrum disorder who completed Children's
        Friendship Training 1-5 years earlier. They were recruited through mail, phone, and email.
        Information collected included parent and child-completed questionnaires and a phone interview.
        Data were collected on 24 of 52 potential participants (46%). With an average of 35-month follow-up,
        participants had a mean age of 12.6 years. Results indicated that participants at follow-up were
        invited on significantly more play dates, showed less play date conflict, improved significantly
        in parent-reported social skills and problem behaviours, and demonstrated marginally significant
        decreases in loneliness when compared to pre-children's Friendship Training.

        ### Example 2:
        A.D.A.M (Animated Dissection of Anatomy for Medicine) contains articles discussing diseases, tests,
        symptoms, injuries and surgeries. Content is reviewed by physicians;[3] the goal is to present
        evidence-based health information. It also contains a library of medical photographs and
        illustrations.[4] MedlinePlus is a free Web site that provides consumer health information for
        patients, families, and health care providers. MedlinePlus brings together information from the United
        States National Library of Medicine, the National Institutes of Health (NIH), other U.S.
        government agencies, and health-related organizations. The U.S. National Library of Medicine produces
        and maintains MedlinePlus.

        ### Example 3:
        for more examples you can visit https://pubmed.ncbi.nlm.nih.gov/36988766/

        """

    if st.button("Predict"):
        if abstract:
            prediction_and_display(abstract)
        else:
            st.error("Please enter an abstract first.")

    col1, col2, col3 = st.columns(3)

    col3.page_link("pages/Model_architecture.py", label="Next: Model_architecture ➡️")
    # col1.page_link("index.py", label="⬅️ Previous: Home ")


if __name__ == "__main__":
    main()
