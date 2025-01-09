import streamlit as st
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

data_path = './Traduction avis clients'

@st.cache_data
def load_data():
    return pd.read_pickle(os.path.join(data_path, 'dataset_cleaned.pkl'))

@st.cache_data
def compute_word_counts(reviews):
    word_counts = Counter(" ".join(reviews).lower().split())
    return word_counts

@st.cache_data
def generate_wordcloud(word_counts, max_words=None):
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=max_words).fit_words(word_counts)
    return wordcloud

st.title("Most Common Words Visualization")
st.write("This app visualizes the most frequent words in customer reviews using a Word Cloud. The ordered list of most common words is also displayed in a table.")

df = load_data()

word_counts = compute_word_counts(df['avis'])

# Display Word Cloud
st.write("### Word Cloud of Frequent Words")

top_n = st.slider("Number of top words in the cloud:", min_value=10, max_value=100, value=50)

most_common_words = word_counts.most_common()
wordcloud = generate_wordcloud(word_counts, max_words=top_n)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title("Word Cloud of Frequent Words", fontsize=16)
st.pyplot(fig)

st.write("### Table of Most Common Words")
df_most_common_words = pd.DataFrame(most_common_words, columns=['Word', 'Occurrences'])
st.dataframe(df_most_common_words, use_container_width=True)
