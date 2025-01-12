import streamlit as st
import pandas as pd
import os

data_path = './Traduction avis clients'

# Main title
st.title("NLP Project 2: Supervised Learning and Applications")
st.markdown("""
This project showcases the application of Natural Language Processing (NLP) techniques to tackle supervised learning challenges. 

This Streamlit app is a demonstration of the project's key components. For the full project details, analysis, evaluations and for the full code, refer to the Notebook.
""")

# Project Overview
st.markdown("""
# Project Overview
This project involves the following key stages:
1. **Data Preprocessing**: Cleaning and preprocessing of review data.
2. **Data Exploration**: Exploratory data analysis to understand the dataset.
3. **Supervised Learning**: Various model implementations for different use cases.

These steps are briefly described below. The full project is available in the Notebook.
""")

st.markdown("""
## Data Preprocessing
The project is based on a large dataset containing reviews of insurance products. The dataset is available on
[Google Drive](https://drive.google.com/file/d/1_kg5JzAzntzLI6eGM3_vmUSoeWk7f8ip/view?usp=sharing).

Different preprocessing steps were applied to clean the data and prepare it for analysis and modeling. These steps include:
- Check for empty reviews
- Translation missing reviews
- Removal of non-text characters
- Spelling correction
- Lowering of text
- Removal of stopwords and punctualization

The cleaned dataset is shown below:
""")
# Data Preprocessing Table
cleaned_dataset = st.session_state.ds_cleaned
st.dataframe(cleaned_dataset)

st.markdown("""
## Data Exploration
Exploratory Data Analysis (EDA) was performed to understand the dataset better. This includes:
1. **Most common words**: finding and visualization with a **Word Cloud** of the most common words in the reviews.
2. **Most common n-grams**: identification and visualization of the most common n-grams.
3. **Topic Modeling**: using **Latent Dirichlet Allocation** (LDA) to identify topics in the reviews.
4. **Similar words**: Finding similar words using embeddings and visualizing them with **t-SNE**.
### Data Visualization
For each of these steps, interactive visualizations with dedicated pages have been created.
These are available in the sidebar under the **Data Visualization** section.
For more details, refer to the respective pages in the sidebar.

The full EDA is available in the Notebook.
""")

st.markdown("""
## Modeling
The project involves several supervised learning tasks, developing different use cases. These are:
- **Sentiment Analysis**: predicting the sentiment of reviews using **TF-IDF** and **classic ML**
- **Review Rating Prediction**: predicting the star rating of reviews using a basic model with **Embedding Layer**
- **Review Category Prediction**: predicting the category of reviews using **pre-trained embeddings**
- **Semantic Search**: Finding similar reviews based on a query using **Universal Sentence Encoder**

Each of these use cases have been developed within the Notebook, in which the full code and analysis are available.
Additionally for each of these use cases a dedicated interactive page with their implementation has been created in the
sidebar under the **Use Cases** section.
""")