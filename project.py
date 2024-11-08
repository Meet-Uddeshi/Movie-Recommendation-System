import numpy as np 
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
cv = CountVectorizer(max_features=5000,stop_words='english')
films = pd.read_csv('tmdb_5000_movies.csv')
movie_credits = pd.read_csv('tmdb_5000_credits.csv') 
films = films.merge(credits,on='title')
films = films[['movie_id','title','overview','genres','keywords','cast','crew']]
def modification(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name']) 
    return l 
films['genres'] = films['genres'].apply(modification)
films['keywords'] = films['keywords'].apply(modification)
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')
def new_modification(text):
    l = []
    c = 0
    for i in ast.literal_eval(text):
        if c < 3:
            l.append(i['name'])
        c+=1
    return l
films['cast'] = films['cast'].apply(modification)
films['cast'] = films['cast'].apply(lambda x:x[0:3])
def director_name(text):
    l = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            l.append(i['name'])
    return l
films['crew'] = films['crew'].apply(director_name)
def collapse(L):
    l = []
    for i in l:
        l.append(i.replace(" ",""))
    return l
films['cast'] = films['cast'].apply(collapse)
films['crew'] = films['crew'].apply(collapse)
films['genres'] = films['genres'].apply(collapse)
films['keywords'] = films['keywords'].apply(collapse)
films['overview'] = films['overview'].apply(lambda x:x.split())
films['tags'] = films['overview'] + films['genres'] + films['keywords'] + films['cast'] + films['crew']
new_films = films.drop(columns=['overview','genres','keywords','cast','crew'])
new_films['tags'] = new_films['tags'].apply(lambda x: " ".jofilms)
vector = cv.fit_transform(new_films['tags']).toarray()
similarity = cosine_similarity(vector)
new_films[new_films['title'] == 'The Lego Movie'].index[0]
def recommendation(movie):
    index = new_films[new_films['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:11]:
        print(new_films.iloc[i[0]].title)
pickle.dump(new_films,open('movie_d.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))