import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

data_path = './Traduction avis clients'
models_path = './models'

max_words = 10000  # Vocabulary size
max_len = 50  # Maximum length of sequences


# Cache the loading of the dataset
@st.cache_resource
def load_my_model():
    """Load the trained model."""
    model_path = os.path.join(models_path, 'sentiment_analysis_model.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


@st.cache_resource
def load_vecktorizer(X_train, X_test):
    """Load the tokenizer."""
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return vectorizer


@st.cache_data
def load_data():
    """Load the cleaned dataset."""
    df = pd.read_pickle(os.path.join(data_path, 'dataset_cleaned_sentiment.pkl'))

    reviews = df[['avis', 'sentiment']]

    X = reviews['avis']
    y = reviews['sentiment']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X, y, X_train, X_test, y_train, y_test


# Initialize the Streamlit app
st.title("Sentiment Analysis")
st.write("Description.")

# Load the trained model
model = load_my_model()
X, y, X_train, X_test, y_train, y_test = load_data()
vectorizer = load_vecktorizer(X_train, X_test)

# Input for prediction
st.write("### Sentiment Anaylsis")
example_review = st.text_input("Enter a review:", "Service client va bien.")
if st.button("Predict"):
    example_tfidf = vectorizer.transform([example_review])
    predicted_sentiment = model.predict(example_tfidf)
    st.write(f"Predicted Sentiment: {predicted_sentiment[0]}")