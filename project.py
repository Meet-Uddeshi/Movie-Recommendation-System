# Step 1 => Importing required libraries
import numpy as np 
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Step 2 => Loading and merging the datasets
films = pd.read_csv('tmdb_5000_movies.csv')
movie_credits = pd.read_csv('tmdb_5000_credits.csv') 
films = films.merge(movie_credits,on='title')

# Step 3 => Selecting relevant columns
films = films[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Step 4 => Defining and applying a function to extract names from stringified lists of dictionaries for 'genres' and 'keywords'
def modification(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name']) 
    return l 
films['genres'] = films['genres'].apply(modification)
films['keywords'] = films['keywords'].apply(modification)

# Step 5 => Defining and applying a function to extract the first three names from stringified lists of dictionaries for 'cast'
def new_modification(text):
    l = []
    c = 0
    for i in ast.literal_eval(text):
        if c < 3:
            l.append(i['name'])
        c+=1
    return l
films['cast'] = films['cast'].apply(new_modification)
films['cast'] = films['cast'].apply(lambda x:x[0:3])

# Step 6 => Defining and applying a function to extract the director's name
def director_name(text):
    l = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            l.append(i['name'])
    return l
films['crew'] = films['crew'].apply(director_name)

# Step 7 => Defining and applying a function to remove spaces from lists of strings for 'cast', 'crew', 'genres', and 'keywords'
def collapse(L):
    l = []
    for i in L:
        l.append(i.replace(" ",""))
    return l
films['cast'] = films['cast'].apply(collapse)
films['crew'] = films['crew'].apply(collapse)
films['genres'] = films['genres'].apply(collapse)
films['keywords'] = films['keywords'].apply(collapse)

# Step 8 => Splitting the overview into a list of words and creating the 'tags' column
films['overview'] = films['overview'].apply(lambda x:x.split())
films['tags'] = films['overview'] + films['genres'] + films['keywords'] + films['cast'] + films['crew']

# Step 9 => Dropping the original columns and joining the list of words in 'tags' into a single string
new_films = films.drop(columns=['overview','genres','keywords','cast','crew'])
new_films['tags'] = new_films['tags'].apply(lambda x: " ".join(x))

# Step 10 => Converting the text data into numerical vectors using CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(new_films['tags']).toarray()

# Step 11 => Calculating the cosine similarity between the vectors
similarity = cosine_similarity(vector)

# Step 12 => Defining a function to recommend movies based on cosine similarity
def recommendation(movie):
    index = new_films[new_films['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:11]:
        print(new_films.iloc[i[0]].title)

# Step 13 => Saving the data frame and similarity matrix to pickle files
pickle.dump(new_films,open('movie_d.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
