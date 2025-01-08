import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import matplotlib.pyplot as plt

# Define the data path
data_path = './Traduction avis clients'

# Cache the loading of the dataset
@st.cache_data
def load_data():
    """Load the cleaned dataset."""
    return pd.read_pickle(os.path.join(data_path, 'dataset_cleaned.pkl'))

# Cache the computation of n-grams
@st.cache_data
def compute_ngram(n_grams):
    """Compute the top n-grams for the dataset."""
    vectorizer = CountVectorizer(ngram_range=(n_grams, n_grams))  # bigrams to n-grams
    ngrams = vectorizer.fit_transform(df['avis'])
    ngram_counts = pd.DataFrame(ngrams.sum(axis=0), columns=vectorizer.get_feature_names_out()).T
    ngram_counts.columns = ['Frequency']
    return ngram_counts.sort_values(by='Frequency', ascending=False).head(100)

# Load the data
df = load_data()

# Streamlit app layout
st.title("N-Gram Analysis Dashboard")
st.markdown("""
Welcome to the **N-Gram Analysis Dashboard**!  
Explore the most frequent n-grams (phrases of 2 or more words) in the customer reviews dataset.  
Adjust the settings below to customize the analysis.
""")

# User input for the number of grams
st.sidebar.header("Customize Analysis")
n_grams = st.sidebar.slider('Select the maximum size of n-grams', min_value=3, max_value=6, value=3, step=1)
st.sidebar.markdown(f"### Selected: {n_grams}-grams")

# Compute n-grams
ngram_counts = compute_ngram(n_grams)

# Display top 10 n-grams
st.subheader(f"Top {n_grams}-Grams in the Dataset")
st.dataframe(ngram_counts, use_container_width=True)

# User input for bar plot length
length = st.sidebar.slider('Bar Plot Length', min_value=5, max_value=50, value=10, step=5)

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))
ngram_counts.head(length).plot(kind='bar', legend=False, ax=ax, color='steelblue')
ax.set_title("Top N-Grams", fontsize=16)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_xlabel("N-Grams", fontsize=12)
plt.xticks(rotation=45, ha='right')

# Display bar plot
st.pyplot(fig)

# Add a footer
st.markdown("""
---
#### About
This app allows you to visualize the most frequent n-grams in customer reviews.  
You can customize the n-gram size and the number of top results displayed.  
**Powered by Streamlit**.
""")