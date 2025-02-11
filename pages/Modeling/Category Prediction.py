import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling1D

data_path = './Traduction avis clients'
models_path = './models'

max_words = 10000  # Vocabulary size
max_len = 50  # Maximum length of sequences


@st.cache_resource
def load_my_model():
    return load_model(os.path.join(models_path, 'topic_classification_model.keras'))


@st.cache_resource
def load_tokenizer(X_train):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)

    return tokenizer


@st.cache_data
def load_data():
    df = st.session_state.ds_topics

    reviews = df[['avis', 'topic']]

    X = reviews['avis']
    y = reviews['topic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X, y, X_train, X_test, y_train, y_test


st.title("Category Prediction")
st.markdown("""
The app classifies reviews into the the found categories using pre-trained embeddings.

The previously found categories are:
- Policy Claims
- Satisfaction and Pricing
- Service Experience
- Contracts and Policy Management
""")

model = load_my_model()
X, y, X_train, X_test, y_train, y_test = load_data()
tokenizer = load_tokenizer(X_train)

st.write("### Predict Category")
example_review = st.text_input("Enter a review:", "Prix raisonnables pour un bon service.")
if st.button("Predict"):
    example_seq = tokenizer.texts_to_sequences([example_review])
    example_padded = pad_sequences(example_seq, maxlen=max_len, padding='post')
    predicted_category = model.predict(example_padded)
    category_labels = y.unique()
    predicted_label = category_labels[np.argmax(predicted_category)]
    st.write(f"Predicted Category: {predicted_label}")
