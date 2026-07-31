import os
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

# ==========================
# Download files from Hugging Face
# ==========================

model_path = hf_hub_download(
    repo_id="bashirkhan0329/movie_recomendation_system",
    filename="knn_model.pkl",
    repo_type="model"
)

vectors_path = hf_hub_download(
    repo_id="bashirkhan0329/movie_recommendation_system",
    filename="vectors.pkl",
    repo_type="model"
)

# ==========================
# Load files
# ==========================

model = joblib.load(model_path)
vectors = joblib.load(vectors_path)
tfidf = joblib.load("tfidf.pkl")
new_df = joblib.load("movies_df.pkl")
