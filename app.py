import os
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

# ==========================
# Download files from Hugging Face
# ==========================

model_path = hf_hub_download(
    repo_id="b9a8d6a8678ff0f359f2374102f212672dbd98c93a7602d23d5e15b60707266c",
    filename="knn_model.pkl",
    repo_type="model"
)

vectors_path = hf_hub_download(
    repo_id="c980362cbe44f737ccd1efe102f18d60dfcca27b12bcdf4437f033b8258c718c",
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
