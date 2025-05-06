import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = "IMDB-Movie-Dataset(2023-1951).csv"
movies_df = pd.read_csv(data)

new_df = movies_df[['movie_id', 'movie_name', 'overview']].dropna()
new_df['overview'] = new_df['overview'].apply(lambda x: x.lower())

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(new_df['overview'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# TMDb API Key
API_KEY = "ff2de638f2d96a0291a0f19086da8839"

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/images?api_key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    posters = data.get("posters")
    if posters:
        poster_path = posters[0].get("file_path")
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/300x450?text=No+Image"

def recommend_movies(movie_name, n=5):
    idx = new_df[new_df['movie_name'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
    return [(new_df.iloc[i]['movie_name'], new_df.iloc[i]['movie_id']) for i, _ in scores]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
movie_titles = new_df['movie_name'].tolist()
selected_movie = st.selectbox("Select a movie", movie_titles)

if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie)
    if not recommendations:
        st.error("No recommendations found. Try another movie.")
    else:
        st.subheader("Recommended Movies:")
        cols = st.columns(5)
        for idx, (title, movie_id) in enumerate(recommendations):
            with cols[idx]:
                st.image(fetch_poster(movie_id), use_column_width=True)
                st.write(title)
