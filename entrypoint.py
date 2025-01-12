import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os

data_path = './Traduction avis clients'
models_path = './models'

@st.cache_data
def load_data(type='cleaned'):
    if type == 'cleaned':
        return pd.read_pickle(os.path.join(data_path, 'dataset_cleaned.pkl'))
    elif type == 'topics':
        return pd.read_pickle(os.path.join(data_path, 'dataset_cleaned_topics.pkl'))
    elif type == 'sentiment':
        return pd.read_pickle(os.path.join(data_path, 'dataset_cleaned_sentiment.pkl'))

# Set page configuration
st.set_page_config(
    page_title="NLP Project 2 - Streamlit Application",
    layout="wide",
    initial_sidebar_state="expanded",
)
#Introduction
introduction = st.Page("pages/Introduction.py", title="Begin Here")
#Data Visualization
embeddings = st.Page("pages/Data Visualization/Embedding for Similar Words.py", title="Embeddings", icon=":material/emoji_objects:")
wordcloud = st.Page("pages/Data Visualization/Words Cloud.py", title="Word Cloud", icon=":material/cloud_queue:")
n_grams = st.Page("pages/Data Visualization/N Grams.py", title="N-Grams", icon=":material/format_list_numbered:")
topic_heatmap = st.Page("pages/Data Visualization/Topic Heatmap.py", title="Topic Heatmap", icon=":material/dataset:")
#Modeling
semantic_search = st.Page("pages/Modeling/Semantic Search.py", title="Semantic Search")
sentiment_analysis = st.Page("pages/Modeling/Sentiment Analysis.py", title="Sentiment Analysis")
assureur_summary = st.Page("pages/Modeling/Assureur Summary.py", title="Assureur Summary")
category_prediction = st.Page("pages/Modeling/Category Prediction.py", title="Category Prediction")
review_rating = st.Page("pages/Modeling/Review Star Prediction.py", title="Review Rating")

pg = st.navigation(
    {
        "Introduction": [introduction],
        "Data Visualization": [wordcloud, n_grams, topic_heatmap, embeddings],
        "Use Cases": [sentiment_analysis, review_rating, category_prediction, semantic_search, assureur_summary],
    }
)

plt.style.use('dark_background')

ds_clenaed = load_data('cleaned')
ds_topics = load_data('topics')
ds_sentiment = load_data('sentiment')

st.session_state.ds_cleaned = ds_clenaed
st.session_state.ds_topics = ds_topics
st.session_state.ds_sentiment = ds_sentiment

pg.run()

# Links
st.sidebar.header("Links")
st.sidebar.markdown("""
- [GitHub Repository](https://github.com/MarcoRiva6/ML4NLP-project2)
- [Colab Notebook](https://colab.research.google.com/drive/1YK1_AoIyKyh4Kae5htVmt0CAHUSVIBYf)
- [Original dataset](https://drive.google.com/file/d/1_kg5JzAzntzLI6eGM3_vmUSoeWk7f8ip/view?usp=sharing)
- [This Streamlit App on Community Cloud](https://ml4nlp-project2-6qzrsnb3spgf3gqd9hekpf.streamlit.app)
""")

st.sidebar.header("Notes about running this app")
st.sidebar.markdown("""
- The figures displaying is optimized for the **dark theme**: please switch to the dark theme for the best experience.
- While running this app from Community Cloud, the **resource limits** may be easily reached: please consider running this app locally,
    (following the instructions at the beginning of the Colab notebook) or wait for us to restart the app so to reset the resource limits.
""")