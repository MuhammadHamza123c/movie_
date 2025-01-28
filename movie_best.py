import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

data=pd.read_csv("imdb_movies.csv")

data=pd.DataFrame({
    'Title':data['Name'],
    'Genre':data['Genre'],
    'Overview':data['Overview'],
    'Caste':data['Caste']
})
data['Genre']=data['Genre'].fillna(data['Genre'].mode)


def movie_poster(user):
  url=f'https://api.themoviedb.org/3/search/movie?api_key=fcd82d6941bdac53e63708ffcfbb955a&query={user}'
  response = requests.get(url)
  data_poster = response.json()
  poster_url = data_poster['results'][0]['poster_path']
  popularity=data_poster['results'][0]['popularity']
  poster_real_url = f'https://image.tmdb.org/t/p/w500{poster_url}'
  return poster_real_url,popularity
data['Caste']=data['Caste'].fillna(data['Caste'].mode)
change_data=data['Overview'].astype(str)+ '' +data['Genre'].astype(str)+ '' +data['Caste'].astype(str)
vector=TfidfVectorizer(max_features=10000)
change_main_data=vector.fit_transform(change_data)
matrix=change_main_data.toarray()
similar=cosine_similarity(matrix)
name=st.text_input("Write down movie name here: ")
if name:
 user_name = data[data['Title'] == name].index[0]
 similarities = similar[user_name]
 indexed_arr = list(enumerate(similarities))
 sorted_indexed_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
 top_2_indices = [index for index, value in sorted_indexed_arr[:5]]
 for i in top_2_indices:
    if data['Title'].iloc[i]==name:
        continue
    st.write(f"{data['Title'].iloc[i]}")
    poster,pop=movie_poster(data['Title'].iloc[i])
    st.write(pop)
    st.image(poster)
    


