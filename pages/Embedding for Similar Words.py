import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

data_path = './Traduction avis clients'
models_path = './models'


# Sample DataFrame with reviews
@st.cache_data
def load_data():
    """Load the cleaned dataset."""
    return pd.read_pickle(os.path.join(data_path, 'dataset_cleaned.pkl'))


# Train Word2Vec model
@st.cache_resource
def train_word2vec():
    return pd.read_pickle(os.path.join(models_path, 'word2vec_model'))


# Compute Euclidean distance
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


# Compute cosine similarity
def cosine_similarity_manual(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Find similar words
def find_similar_words_custom(word, model, topn=10, metric="cosine"):
    if word not in model.wv:
        return f"Word '{word}' not in vocabulary."

    word_vector = model.wv[word]
    similarities = []

    for other_word in model.wv.index_to_key:
        if other_word == word:
            continue
        other_vector = model.wv[other_word]

        # Compute the specified metric
        if metric == "cosine":
            similarity = cosine_similarity_manual(word_vector, other_vector)
        elif metric == "euclidean":
            similarity = -euclidean_distance(word_vector, other_vector)  # Negate for ranking
        else:
            raise ValueError("Invalid metric. Use 'cosine' or 'euclidean'.")

        similarities.append((other_word, similarity))

    # Sort and return top-n similar words
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:topn]


# t-SNE visualization
@st.cache_data
def tsne_visualization(_model):
    words = list(model.wv.index_to_key)[:200]  # Limit to first 100 words for clarity
    word_vectors = model.wv[words]
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(word_vectors)
    return reduced_vectors, words


# Streamlit App
st.title("Word Embedding Analysis")
st.write(
    "This app allows you to explore word embeddings trained on French customer reviews. Find similar words using cosine similarity or Euclidean distance, and view a t-SNE visualization of the word embeddings.")

# Load data and train model
model = train_word2vec()

# Input word and similarity metric
word = st.text_input("Enter a word to find similar words:", value="prix")
metric = st.selectbox("Select similarity measure:", ["cosine", "euclidean"])
topn = st.slider("Number of similar words to display:", min_value=1, max_value=20, value=10)

# Find similar words
if st.button("Find Similar Words"):
    similar_words = find_similar_words_custom(word, model, topn=topn, metric=metric)
    if isinstance(similar_words, str):  # Word not in vocabulary
        st.error(similar_words)
    else:
        st.write(f"### Words similar to '{word}' using {metric} similarity:")
        result_df = pd.DataFrame(similar_words, columns=["Word", "Similarity"])
        result_df["Similarity"] = result_df["Similarity"].apply(lambda x: f"{x:.4f}")
        st.table(result_df)

# t-SNE Visualization
st.write("### t-SNE Visualization of Word Embeddings")
reduced_vectors, words = tsne_visualization(model)

# Plot with Matplotlib
fig, ax = plt.subplots(figsize=(12, 8))
for i, word in enumerate(words):
    ax.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
    ax.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9)
ax.set_title("2D Visualization of Word Embeddings")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
st.pyplot(fig)
