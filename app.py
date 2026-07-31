# app.py -- Streamlit web app for movie recommendation

import os
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

# ==============================
# Download large files from Hugging Face
# ==============================

model_path = hf_hub_download(
    repo_id="bashirkhan0329/movie_recommendation_system",
    filename="knn_model.pkl"
)

vectors_path = hf_hub_download(
    repo_id="bashirkhan0329/movie_recommendation_system",
    filename="vectors.pkl"
)

# ==============================
# Load files
# ==============================

model = joblib.load(model_path)
vectors = joblib.load(vectors_path)

# Small files loaded from GitHub
tfidf = joblib.load("tfidf.pkl")
new_df = joblib.load("movies_df.pkl")

# ==============================
# Streamlit UI
# ==============================

st.title("🎬 Movie Recommendation System")

# Model performance metrics
metrics_path = "model_metrics.pkl"

if os.path.exists(metrics_path):
    metrics = joblib.load(metrics_path)

    st.sidebar.header("Model Performance")
    st.sidebar.metric(
        "Accuracy (Hit Rate@5)",
        f"{metrics['accuracy']:.2%}"
    )

    st.sidebar.metric(
        "Precision@5",
        f"{metrics['precision_at_5']:.2%}"
    )

else:
    st.sidebar.warning(
        "Run training script to generate model_metrics.pkl"
    )

# Movie dropdown
movie_name = st.selectbox(
    "Select a Movie:",
    new_df["title"].values
)


# ==============================
# Recommendation Function
# ==============================

def recommend(movie_name):

    if movie_name not in new_df["title"].values:
        return []

    movie_index = new_df[new_df["title"] == movie_name].index[0]

    distances, indices = model.kneighbors([vectors[movie_index]])

    recommended_movies = []

    for i in indices[0][1:]:
        recommended_movies.append(new_df.iloc[i]["title"])

    return recommended_movies


# ==============================
# Recommend Button
# ==============================

if st.button("Recommend"):

    recommendations = recommend(movie_name)

    st.subheader(f"Movies similar to '{movie_name}'")

    for movie in recommendations:
        st.write("👉", movie)
