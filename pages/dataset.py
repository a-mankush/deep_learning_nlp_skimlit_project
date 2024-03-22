import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st


@st.cache_resource()
def read_dataset(filename) -> pd.DataFrame:
    return pd.read_csv(filename).drop(columns=["Unnamed: 0"])


"""
# Dataset

Dateset used for training the MediScan is https://github.com/Franck-Dernoncourt/pubmed-rct.git . Within it there are multiple datasets available and I have chosen **PubMed_20k_number_replace_with_at_sign**.

#### The original dataset looks like this:
```
###24293578
OBJECTIVE	To investigate the efficacy of 6 weeks of daily low-dose oral prednisolone in improving pain , mobility , and systemic low-grade inflammation in the short term and whether the effect would be sustained at 12 weeks in older adults with moderate to severe knee osteoarthritis ( OA ) .
METHODS	A total of 125 patients with primary knee OA were randomized 1:1 ; 63 received 7.5 mg/day of prednisolone and 62 received placebo for 6 weeks .
METHODS	Outcome measures included pain reduction and improvement in function scores and systemic inflammation markers .
METHODS	Pain was assessed using the visual analog pain scale ( 0-100 mm ) .
METHODS	Secondary outcome measures included the Western Ontario and McMaster Universities Osteoarthritis Index scores , patient global assessment ( PGA ) of the severity of knee OA , and 6-min walk distance ( 6MWD ) .
METHODS	Serum levels of interleukin 1 ( IL-1 ) , IL-6 , tumor necrosis factor ( TNF ) - , and high-sensitivity C-reactive protein ( hsCRP ) were measured .
RESULTS	There was a clinically relevant reduction in the intervention group compared to the placebo group for knee pain , physical function , PGA , and 6MWD at 6 weeks .
RESULTS	The mean difference between treatment arms ( 95 % CI ) was 10.9 ( 4.8-18 .0 ) , p < 0.001 ; 9.5 ( 3.7-15 .4 ) , p < 0.05 ; 15.7 ( 5.3-26 .1 ) , p < 0.001 ; and 86.9 ( 29.8-144 .1 ) , p < 0.05 , respectively .
RESULTS	Further , there was a clinically relevant reduction in the serum levels of IL-1 , IL-6 , TNF - , and hsCRP at 6 weeks in the intervention group when compared to the placebo group .
RESULTS	These differences remained significant at 12 weeks .
RESULTS	The Outcome Measures in Rheumatology Clinical Trials-Osteoarthritis Research Society International responder rate was 65 % in the intervention group and 34 % in the placebo group ( p < 0.05 ) .
CONCLUSIONS	Low-dose oral prednisolone had both a short-term and a longer sustained effect resulting in less knee pain , better physical function , and attenuation of systemic inflammation in older patients with knee OA ( ClinicalTrials.gov identifier NCT01619163 ) .

```
"""

"#####   Converting the text data into a pandas DataFrame so its easier to work with"

train_df = read_dataset("datasets/train.csv")
st.dataframe(train_df.head())

train_df["total_char"] = train_df["text"].apply(lambda x: len(x.split()))

"""
## Exploratory data analysis(EDA)
"""

fig = px.bar(data_frame=train_df.target.value_counts())
fig.update_layout(yaxis=dict(title="Number of sentences"))
st.plotly_chart(fig)

""" 
> counts the number of sentences per label: the least common label (**objective**) is approximately four times less frequent than the most
common label (**results**),
"""

# fig = px.histogram(train_df, x="line_number", title="sentences per abstract ")
# st.plotly_chart(fig)

fig = px.histogram(
    train_df,
    x="total_lines",
    title="Distribution of the number of sentences per abstract",
)
fig.update_layout(
    xaxis=dict(title="Number of sentences per abstract"),
    yaxis=dict(title="Number of occurrences"),
)
st.plotly_chart(fig)

fig = px.histogram(
    data_frame=train_df,
    x="total_char",
    title="Distribution of the number of sentences per abstract",
)
fig.update_layout(
    xaxis=dict(title="Number of tokens per sentence"),
    yaxis=dict(title="Number of occurrences"),
)
st.plotly_chart(fig)


col1, col2, col3 = st.columns(3)

col3.page_link("index.py", label="Next: Home ➡️")
