# Step 1 => Importing necessary libraries
import streamlit as st
import pickle
import pandas as pd
import requests

# Step 2 => Defining the function to fetch movie posters using OMDb API
def fetch_poster(title):
    api_key = "73e9a009"  # Replace with your OMDb API key
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    return data.get('Poster', None)

# Step 3 => Defining the movie recommendation function
def recommendation(movie):
    movie_idx = movies_title[movies_title['title'] == movie].index[0]
    distances = similar[movie_idx]
    movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:10]

    recommended_movies = []
    recommended_posters = []
    for i in movie_indices:
        movie_id = i[0]
        movie_title = movies_title.iloc[movie_id].title
        recommended_movies.append(movie_title)
        poster_url = fetch_poster(movie_title)
        recommended_posters.append(poster_url)
    return recommended_movies, recommended_posters

# Step 4 => Loading the movie data and similarity matrix from pickle files
movies_d = pickle.load(open('movies_d.pkl', 'rb'))
movies_title = pd.DataFrame(movies_d)
similar = pickle.load(open('similar.pkl', 'rb'))

# Step 5 => Setting up the Streamlit app UI
st.title('MOVIE RECOMMENDATION SYSTEM')
st.markdown('<h2 style="font-size: 24px; text-align: center;">Choose a movie to get recommendations:</h2>', unsafe_allow_html=True)

select_movie = st.selectbox('', movies_title['title'].values)

# Step 6 => Handling movie recommendations and display when the button is clicked
if st.button('Recommend'):
    recommended_movies, recommended_posters = recommendation(select_movie)

    # Step 7 => Displaying recommended movies and posters in a grid layout
    cols = st.columns(3, gap="large")  # Create 3 columns for display
    for i in range(len(recommended_movies)):
        movie = recommended_movies[i]
        poster = recommended_posters[i]

        with cols[i % 3]:  # Ensure correct column for each movie
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    border: 2px solid #d3d3d3;
                    padding: 10px;
                    margin: 10px;
                    border-radius: 10px;
                    background-color: #f8f9fa;
                    box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
                    text-align: center;
                ">
                    <img src="{poster}" style="width: 150px; height: auto; border-radius: 8px; margin-bottom: 10px;">
                    <h4 style="font-size: 20px; color: white;">{movie}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

        if (i + 1) % 3 == 0:  # Ensure new row every 3 movies
            cols = st.columns(3, gap="large")
