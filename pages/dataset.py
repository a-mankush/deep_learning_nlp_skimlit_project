import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


@st.cache_resource()
def read_dataset(filename):
    return pd.read_csv(filename).drop(columns=["Unnamed: 0"])


train_df = read_dataset("datasets/train.csv")
st.dataframe(train_df.head())

train_df["total_char"] = train_df['text'].apply(lambda x: len(x.split()))

fig = px.bar(data_frame=train_df.target.value_counts())
st.plotly_chart(fig)

fig = px.histogram(train_df, x='line_number', title="sentences per abstract ")
st.plotly_chart(fig)

fig = px.histogram(train_df, x='total_lines', title="total lines distribution")
st.plotly_chart(fig)

fig = px.histogram(data_frame=train_df, x="total_char")
st.plotly_chart(fig)
