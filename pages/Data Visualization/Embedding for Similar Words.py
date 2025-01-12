import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

data_path = './Traduction avis clients'
models_path = './models'


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
def tsne_visualization(_model, num_words=200):
    words = list(model.wv.index_to_key)[:num_words]  # Limit to first 100 words for clarity
    word_vectors = model.wv[words]
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(word_vectors)
    return reduced_vectors, words


# Streamlit App
st.title("Word Embedding Analysis")
st.write(
    "This app allows to explore word embeddings trained on the dataset. Find similar words using cosine similarity or Euclidean distance, and view a t-SNE visualization of the word embeddings.")
st.markdown("""
This app allows to explore word embeddings trained on the dataset.

In order to implement the embedding to identify similar words, a **Word2Vec** model is used.
This is extended by allowing to find similar words based on cosine or euclidean similarity.
Finally, **t-SNE** is used to visualize the found word embeddings in a 2D space.

Here the following parameters can be chosen:
- The word to find similar words for
- The similarity metric to use (cosine or euclidean)
- The number of similar words to display

The t-SNE visualization of the word embeddings is also integrated in the app.
""")

# Load data and train model
model = train_word2vec()

st.write(f"### Words similarity")
# Input word and similarity metric
word = st.text_input("Enter a word to find similar words:", value="prix")
metric = st.selectbox("Similarity measure:", ["cosine", "euclidean"])
topn = st.slider("Similar words to display:", min_value=1, max_value=20, value=10)

# Find similar words
if st.button("Find Similar Words"):
    similar_words = find_similar_words_custom(word, model, topn=topn, metric=metric)
    if isinstance(similar_words, str):  # Word not in vocabulary
        st.error(similar_words)
    else:
        result_df = pd.DataFrame(similar_words, columns=["Word", "Similarity"])
        result_df["Similarity"] = result_df["Similarity"].apply(lambda x: f"{x:.4f}")
        st.dataframe(result_df, use_container_width=True)

# t-SNE Visualization
st.markdown("""
### t-SNE Visualization of Word Embeddings

A 2D visualization of word embeddings is plotted, using the **t-SNE** algorithm.
Dimensionality reduction is applied in order to be able to observe relationships between words in a simplified two-dimensional space.

The number of words to visualize can be customized.
""")

words_in_graph = st.slider("Words to display:", min_value=20, max_value=300, value=200, step=20)

reduced_vectors, words = tsne_visualization(model, words_in_graph)

# Plot with Matplotlib
fig, ax = plt.subplots(figsize=(12, 8))
for i, word in enumerate(words):
    ax.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
    ax.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9)
ax.set_title("2D Visualization of Word Embeddings")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
st.pyplot(fig)

st.markdown("""
We can notice that words with similar meanings or contexts are grouped together, demonstrating the semantic structure captured by the Word2Vec model.

For example "satisfaite", "simple", "rapide" are clustered, indicating customer satisfaction and ease of service; "tarif", "rapport", "devis" are grouped, reflecting pricing-related discussions.

We can also find semantic relationships: words related to service experience (like "client", "service", "téléphonique") form a distinct region, just like words about documents and claims ("dossier", "contrat", "remboursement") which appear near each other.
""")
