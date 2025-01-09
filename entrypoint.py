import streamlit as st
import matplotlib.pyplot as plt

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

pg.run()

# Links
st.sidebar.header("Links")
st.sidebar.markdown("""
- [GitHub Repository](https://github.com/MarcoRiva6/ML4NLP-project2)
- [Colab Notebook](https://colab.research.google.com/drive/1V2qJyJfu0YqazOcWnE-c24T2hAIpUBLB)
- [Original dataset](https://drive.google.com/file/d/1_kg5JzAzntzLI6eGM3_vmUSoeWk7f8ip/view?usp=sharing)
- [This Streamlit App on Community Cloud](https://ml4nlp-project2-6qzrsnb3spgf3gqd9hekpf.streamlit.app)
""")

st.sidebar.header("Notes about running this app")
st.sidebar.markdown("""
- The figures displaying is optimized for the **dark theme**: please switch to the dark theme for the best experience.
- While running this app from Community Cloud, the **resource limits** may be easily reached: please consider running this app locally,
    (following the instructions at the beginning of the Colab notebook) or wait for us to restart the app so to reset the resource limits.
""")