# app.py -- Streamlit web app for movie recommendation

import os
import streamlit as st
import joblib

# saved files load kiye
model = joblib.load('Knn_model.pkl')
vectors = joblib.load('vectors.pkl')
tfidf = joblib.load('tfidf.pkl')
new_df = joblib.load('movies_df.pkl')

# page title set kiya
st.title("🎬 Movie Recommendation System")

# model accuracy metrics sidebar mein show kiye
metrics_path = "model_metrics.pkl"
if os.path.exists(metrics_path):
    metrics = joblib.load(metrics_path)
    st.sidebar.header("Model Performance")
    st.sidebar.metric("Accuracy (Hit Rate@5)", f"{metrics['accuracy']:.2%}")
    st.sidebar.metric("Precision@5", f"{metrics['precision_at_5']:.2%}")
else:
    st.sidebar.warning("Run training script to generate model_metrics.pkl")

# dropdown banaya sab movie titles ke saath
movie_name = st.selectbox("Movie select karein:", new_df['title'].values)

# recommend function banaya
def recommend(movie_name):
    # movie exist check kiya
    if movie_name not in new_df['title'].values:
        return []

    # movie ka index nikala
    movie_index = new_df[new_df['title'] == movie_name].index[0]

    # 6 nearest neighbors nikale (khud + 5 similar)
    distances, indices = model.kneighbors([vectors[movie_index]])

    # recommended movies list banayi
    recommended_movies = []
    for i in indices[0][1:]:
        recommended_movies.append(new_df.iloc[i]['title'])

    return recommended_movies

# button banaya
if st.button("Recommend"):
    # recommendations nikale
    recommendations = recommend(movie_name)

    # heading print kiya
    st.subheader(f"Movies similar to '{movie_name}':")

    # sab recommendations show kiye
    for movie in recommendations:
        st.write("👉", movie)