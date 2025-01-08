import streamlit as st
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

data_path = './Traduction avis clients'

# Sample DataFrame with reviews
@st.cache_data
def load_data():
    """Load the cleaned dataset."""
    return pd.read_pickle(os.path.join(data_path, 'dataset_cleaned.pkl'))

# Function to compute word frequencies
def compute_word_counts(reviews):
    word_counts = Counter(" ".join(reviews).lower().split())
    return word_counts

# Generate Word Cloud
def generate_wordcloud(word_counts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').fit_words(word_counts)
    return wordcloud

# Streamlit App
st.title("Most Common Words Visualization")
st.write("This app visualizes the most frequent words in customer reviews using a table and a Word Cloud.")

# Load data
df = load_data()

# Compute word counts
word_counts = compute_word_counts(df['avis'])

# Slider to select the number of top words
top_n = st.slider("Select the number of top words to display:", min_value=5, max_value=50, value=10)

# Display most common words in a table
most_common_words = word_counts.most_common(top_n)
df_most_common_words = pd.DataFrame(most_common_words, columns=['Word', 'Occurrences'])
st.write("### Table of Most Common Words")
st.table(df_most_common_words)

# Display Word Cloud
st.write("### Word Cloud of Frequent Words")
wordcloud = generate_wordcloud(word_counts)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title("Word Cloud of Frequent Words", fontsize=16)
st.pyplot(fig)