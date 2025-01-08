import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

data_path = './Traduction avis clients'

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "plguillou/t5-base-fr-sum-cnndm"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()


# Function to get the first sentence of the summary
def get_first_sentence(text):
    period_pos = text.find('.')
    if period_pos != -1:
        return text[:period_pos + 1]
    return text


# Summarization function
def generate_insurer_summary(insurer_name, df):
    reviews = df[df['assureur'] == insurer_name]['avis'].tolist()
    reviews = list(set(reviews))
    aggregated_reviews = "\n".join([f"- {review}" for review in reviews[:50]])

    prompt = (
            f"Voici des avis des clients pour l'assureur {insurer_name}. Résumez ces avis en français en une ou deux phrases, "
            f"en mettant en évidence les points positifs et les points négatifs:\n"
            + aggregated_reviews
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=80, min_length=10, length_penalty=3, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return get_first_sentence(summary)


# Sample DataFrame for reviews
@st.cache_data
def load_data():
    """Load the cleaned dataset."""
    return pd.read_pickle(os.path.join(data_path, 'dataset_cleaned.pkl'))


df = load_data()

# Streamlit App Layout
st.title("Insurer Review Summarization")
st.write("This app summarizes customer reviews for a selected insurer in one or two sentences.")

# Insurer Selection
insurer_name = st.selectbox("Select an Insurer:", df['assureur'].unique(), index=52)

# Generate Summary
if st.button("Generate Summary"):
    with st.spinner("Generating summary..."):
        summary = generate_insurer_summary(insurer_name, df)
    st.success("Summary generated!")
    st.write(f"### Summary for {insurer_name}")
    st.write(summary)

# Display Reviews
if st.checkbox("Show Reviews"):
    st.write(f"### Reviews for {insurer_name}")
    insurer_reviews = df[df['assureur'] == insurer_name]
    st.table(insurer_reviews[['avis']][:10])
