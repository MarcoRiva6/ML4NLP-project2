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

@st.cache_resource
def load_my_model():
    return load_model(os.path.join(models_path, 'star_rating_prediction_model.keras'))

@st.cache_resource
def load_tokenizer(X_train):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)

    return tokenizer

@st.cache_data
def load_data():
    df = st.session_state.ds_cleaned

    reviews = df[['avis', 'note']]

    X = reviews['avis']
    y = reviews['note']

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

st.title("Star Rating Prediction App")
st.markdown("""
The app predicts the star rating of a review using the model resulting from the training of a basic model with an embedding layer on the notebook.
""")

model = load_my_model()
X_train, X_test, y_train, y_test = load_data()
tokenizer = load_tokenizer(X_train)
scaler = load_scaler()

st.write("### Predict Star Rating")
example_review = st.text_input("Enter a review:", "Service rapide et excellent.")
if st.button("Predict"):
    example_seq = tokenizer.texts_to_sequences([example_review])
    example_padded = pad_sequences(example_seq, maxlen=max_len, padding='post')
    predicted_rating_scaled = model.predict(example_padded)
    predicted_rating = scaler.inverse_transform(predicted_rating_scaled)
    st.write(f"Predicted Rating: {predicted_rating[0][0]:.2f}")