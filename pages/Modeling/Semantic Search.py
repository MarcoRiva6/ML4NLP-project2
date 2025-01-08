import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import os

data_path = './Traduction avis clients'
models_path = './models'

# Load precomputed data
@st.cache_data
def load_precomputed_data():
    # Load the precomputed review embeddings
    review_embeddings = np.load(os.path.join(models_path, "review_embeddings.npy"))  # Update with the actual file path
    # Load the DataFrame containing the reviews, assured, and product details
    df = pd.read_pickle(os.path.join(data_path, 'dataset_cleaned.pkl'))  # Update with the actual file path
    return review_embeddings, df

review_embeddings, df = load_precomputed_data()

# Load the Universal Sentence Encoder model
@st.cache_resource
def load_use_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

embed = load_use_model()

# Semantic search function
def semantic_search(query, embeddings, top_k=3):
    query_embedding = embed([query])
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [
        {
            "avis": df["avis"].iloc[i],
            "assureur": df["assureur"].iloc[i],
            "produit": df["produit"].iloc[i],
            "score": similarities[i]
        }
        for i in top_indices
    ]
    return results

# Streamlit App Layout
st.title("Semantic Search for Reviews")
st.write("This app allows you to search for relevant reviews based on a query. Use the slider to set the number of top results and click the button to run your query.")

# Query Input
query = st.text_input("Enter your query:", value="Quel est le service client le plus rapide ?")

# Top-K Slider
top_k = st.slider("Select the number of top results to display:", min_value=1, max_value=10, value=3)

# Button to Run the Query
if st.button("Run Query"):
    if query:
        st.write(f"### Query: {query}")
        results = semantic_search(query, review_embeddings, top_k=top_k)

        if results:
            st.write("### Top Relevant Reviews:")
            result_df = pd.DataFrame(results)
            st.table(result_df)
        else:
            st.write("No results found.")