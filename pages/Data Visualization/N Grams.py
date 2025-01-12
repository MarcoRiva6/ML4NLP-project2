import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import matplotlib.pyplot as plt

data_path = './Traduction avis clients'

@st.cache_data
def compute_ngram(n_grams):
    vectorizer = CountVectorizer(ngram_range=(n_grams, n_grams))
    ngrams = vectorizer.fit_transform(df['avis'])
    ngram_counts = pd.DataFrame(ngrams.sum(axis=0), columns=vectorizer.get_feature_names_out()).T
    ngram_counts.columns = ['Frequency']
    return ngram_counts.sort_values(by='Frequency', ascending=False).head(100)

df = st.session_state.ds_cleaned

st.title("N-Gram Analysis Dashboard")
st.markdown("""
This app shows the most frequent n-grams in the dataset.

The n-gram size and the number of top results displayed can be customized.
""")

st.sidebar.header("Customize Analysis")
n_grams = st.slider('Size of n-grams', min_value=2, max_value=6, value=3, step=1)

ngram_counts = compute_ngram(n_grams)

st.subheader(f"Top {n_grams}-Grams in the Dataset")
st.dataframe(ngram_counts, use_container_width=True)

length = st.slider('Bar Plot Length', min_value=5, max_value=50, value=20, step=5)

fig, ax = plt.subplots(figsize=(10, 6))
ngram_counts.head(length).plot(kind='bar', legend=False, ax=ax, color='steelblue')
ax.set_title("Top N-Grams", fontsize=16)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_xlabel("N-Grams", fontsize=12)
plt.xticks(rotation=45, ha='right')

# Display bar plot
st.pyplot(fig)