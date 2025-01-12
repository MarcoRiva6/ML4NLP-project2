import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import os

data_path = './Traduction avis clients'
models_path = './models'

@st.cache_data
def load_precomputed_data():
    review_embeddings = np.load(os.path.join(models_path, "review_embeddings.npy"))
    df = st.session_state.ds_cleaned
    return review_embeddings, df

review_embeddings, df = load_precomputed_data()

@st.cache_resource
def load_use_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

embed = load_use_model()

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

st.title("Semantic Search for Reviews")
st.markdown("""
This app uses Universal Sentence Encoder (USE) to encode both reviews and search queries into dense vectors.
By calculating cosine similarity between the query and the reviews, it retrieves and ranks the most relevant reviews
based on the provided query.
""")

st.write("### Query")

query = st.text_input("Enter a query:", value="Quel est le service client le plus rapide ?")

top_k = st.slider("Top results to display:", min_value=1, max_value=10, value=3)

if st.button("Run Query"):
    if query:
        results = semantic_search(query, review_embeddings, top_k=top_k)

        if results:
            st.write("### Top Relevant Reviews:")
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True)
        else:
            st.write("No results found.")