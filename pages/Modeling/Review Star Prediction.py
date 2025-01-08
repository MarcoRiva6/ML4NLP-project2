import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

data_path = './Traduction avis clients'
models_path = './models'

max_words = 10000
max_len = 100

# Cache the loading of the dataset
@st.cache_resource
def load_my_model():
    """Load the trained model."""
    return load_model(os.path.join(models_path, 'star_rating_prediction_model.keras'))

@st.cache_resource
def load_tokenizer(X_train):
    """Load the tokenizer."""
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)

    return tokenizer

@st.cache_data
def load_data():
    """Load the cleaned dataset."""
    df = pd.read_pickle(os.path.join(data_path, 'dataset_cleaned.pkl'))

    reviews = df[['avis', 'note']]

    # Prepare data: Input (reviews) and target (star ratings)
    X = reviews['avis']  # Preprocessed reviews
    y = reviews['note']  # Star ratings (1-5)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

@st.cache_resource
def load_scaler():

    X_train, X_test, y_train, y_test = load_data()

    # Scale target values to the range [0, 1]
    scaler = MinMaxScaler()
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

    return scaler

# Initialize the Streamlit app
st.title("Star Rating Prediction App")
st.write("Enter a review, and the model will predict its star rating (1 to 5).")

# Load the trained model
model = load_my_model()
X_train, X_test, y_train, y_test = load_data()
tokenizer = load_tokenizer(X_train)
scaler = load_scaler()

# Input for prediction
st.write("### Predict Star Rating")
example_review = st.text_input("Enter a review:", "Service rapide et excellent.")
if st.button("Predict"):
    example_seq = tokenizer.texts_to_sequences([example_review])
    example_padded = pad_sequences(example_seq, maxlen=max_len, padding='post')
    predicted_rating_scaled = model.predict(example_padded)
    predicted_rating = scaler.inverse_transform(predicted_rating_scaled)
    st.write(f"Predicted Rating for '{example_review}': {predicted_rating[0][0]:.2f}")