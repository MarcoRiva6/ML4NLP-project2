import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="NLP Project 2 - Streamlit Application",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)
#Introduction
introduction = st.Page("pages/Introduction.py", title="Introduction")
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
        "Use Cases": [semantic_search, sentiment_analysis, assureur_summary, category_prediction, review_rating],
    }
)

pg.run()

# Sidebar configuration
# Sidebar configuration
st.sidebar.title("Navigation")

# Section 1: Main Navigation
st.sidebar.header("Main Navigation")
st.sidebar.markdown("Use the links below to navigate to specific applications:")

# Add buttons for each app
if st.sidebar.button("Prediction App"):
    st.experimental_set_query_params(page="prediction_app")
    st.write("Redirecting to Prediction App...")

if st.sidebar.button("Insurer Analysis App"):
    st.experimental_set_query_params(page="insurer_analysis_app")
    st.write("Redirecting to Insurer Analysis App...")

# Section 2: About the Project
st.sidebar.write("---")  # Divider
st.sidebar.header("About the Project")
st.sidebar.markdown("""
- **Objective**: Demonstrate the application of NLP techniques.
- **Focus Areas**:
    - Data Exploration
    - Supervised Learning
    - Sentiment Analysis
    - Interactive Dashboards
""")

# Section 3: Resources
st.sidebar.write("---")  # Divider
st.sidebar.header("Resources")
with st.sidebar.expander("Expand Resources"):
    st.markdown("""
    - [Dataset](https://drive.google.com/file/d/1_kg5JzAzntzLI6eGM3_vmUSoeWk7f8ip/view?usp=sharing)
    - [English Template](https://docs.google.com/presentation/d/1hyaVKY31U0wP4kensljOgIiudkRC5N1OxZMWqZ07Y5Q/edit)
    - [Contact: Kezhan Shi](mailto:shikezhan@gmail.com)
    """)

# Section 4: Additional Links
st.sidebar.write("---")  # Divider
st.sidebar.header("Additional Links")
st.sidebar.markdown("""
- **GitHub Repository**: [NLP Project Repo](#)
- **Documentation**: [Project Docs](#)
""")