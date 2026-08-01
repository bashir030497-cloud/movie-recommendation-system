import os
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

# =====================================
# Download large files from Hugging Face
# =====================================

model_path = hf_hub_download(
    repo_id="bashirkhan0329/movie_recomendation_system",
    filename="knn_model.pkl",
    repo_type="model"
)

vectors_path = hf_hub_download(
    repo_id="bashirkhan0329/movie_recomendation_system",
    filename="vectors.pkl",
    repo_type="model"
)

# =====================================
# Load model files
# =====================================

model = joblib.load(model_path)
vectors = joblib.load(vectors_path)

# Load small files from GitHub
tfidf = joblib.load("tfidf.pkl")
new_df = joblib.load("movies_df.pkl")

# =====================================
# Streamlit Page
# =====================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Content-Based Movie Recommendation using KNN + TF-IDF")

# =====================================
# Sidebar Metrics
# =====================================

metrics_path = "model_metrics.pkl"

if os.path.exists(metrics_path):
    metrics = joblib.load(metrics_path)

    st.sidebar.header("📊 Model Performance")

    st.sidebar.metric(
        "Hit Rate@5",
        f"{metrics['accuracy']:.2%}"
    )

    st.sidebar.metric(
        "Precision@5",
        f"{metrics['precision_at_5']:.2%}"
    )
else:
    st.sidebar.warning("model_metrics.pkl not found.")

# =====================================
# Movie Selection
# =====================================

movie_name = st.selectbox(
    "Select a Movie",
    new_df["title"].values
)

# =====================================
# Recommendation Function
# =====================================

def recommend(movie):

    if movie not in new_df["title"].values:
        return []

    movie_index = new_df[new_df["title"] == movie].index[0]

    distances, indices = model.kneighbors(
        [vectors[movie_index]],
        n_neighbors=6
    )

    recommendations = []

    for i in indices[0][1:]:
        recommendations.append(
            new_df.iloc[i]["title"]
        )

    return recommendations

# =====================================
# Recommend Button
# =====================================

if st.button("Recommend"):

    movies = recommend(movie_name)

    st.subheader("Recommended Movies")

    for movie in movies:
        st.write("👉", movie)
