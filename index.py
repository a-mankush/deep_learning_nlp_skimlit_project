import streamlit as st



st.write("# SkimLit: Making Medical Texts More Readable with Deep Learning")
# st.sidebar("Welcome to SkimLit")
image1 = "images/Body_3.png"
image2 = None


st.markdown(
    """
        In the realm of 📚**medical literature**, dense paragraphs filled with **technical 
        jargon** can be daunting for both 👩‍⚕️**healthcare professionals** the 🌐**general audience**.
        Skimming through ⏳**lengthy medical texts** to extract key information is a skill that 
        requires time and effort. However, this project/model makes those 📝dense paragraphs more skimmable and accessible
        **ultimately improving readability** """)

st.image(image1)

st.link_button(label="Try it now", url="main", use_container_width=True, type='primary')
st.markdown("""
    ## **Introduction**
    The main motivation for this project comes from the paper  [Neural Networks for Joint Sentence Classification
    in Medical Paper Abstracts.](https://arxiv.org/pdf/1612.05251.pdf) and https://github.com/mrdbourke/tensorflow-deep-learning 
    
    
    It is a **deep learning model** that designed to take paragraphs related to medicine as input and divide them into **distinct sections**,
    such as the objective, result, conclusion, and more. This makes the information more skimmable and accessible,
    **ultimately improving readability**.

    ### Know more about
    - [Model architecture](Model_architecture) 
    - [Dataset](dataset)
    - Data Pipeline 
"""
)
