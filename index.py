import os
import subprocess

import streamlit as st

st.write("# MediScan: Making Medical Texts More Readable with Deep Learning")
# st.sidebar("Welcome to MediScan")
image1 = "images/Body_3.png"
image2 = None


st.markdown(
    """
        Navigating the intricate world of medical literature can be a daunting task.
        Dense paragraphs filled with technical jargon often overwhelm healthcare professionals
        and the general public alike, making it challenging to extract crucial information quickly.
        Our project addresses this issue by transforming these complex texts into more skimmable and
        accessible formats, significantly enhancing readability and comprehension. """
)

st.image(image1)

st.link_button(label="Try it now", url="main", use_container_width=True, type="primary")
st.markdown(
    """
    ## **Introduction**
    The main motivation for this project comes from the paper  [Neural Networks for Joint Sentence Classification
    in Medical Paper Abstracts.](https://arxiv.org/pdf/1612.05251.pdf) 
    
    
    Inspired by research, including the paper "Neural Networks for Joint Sentence Classification in Medical
    Paper Abstracts" and advanced deep learning resources, our model is designed to revolutionize the way medical information
    is presented. By processing medical paragraphs and dividing them into distinct sections such as objectives, results, background, method
    and conclusions, we make vital information more approachable and easier to digest.
    
    ### Know more about
     - 📊[Model architecture](Model_architecture)
     - 🧠[Dataset](dataset)


"""
)

_, _, col3 = st.columns(3)

col3.page_link("pages/main.py", label="Next: MediScan Model ➡️")
# st.page_link("pages/dataset.py", label="Dataset", icon="🧠")

# - [Model architecture](Model_architecture)
# - [Dataset](dataset)
