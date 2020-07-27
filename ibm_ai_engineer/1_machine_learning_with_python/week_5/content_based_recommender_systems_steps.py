
# Import libraries
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

!wget -O moviedataset.zip https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
print('unziping ...')
!unzip -o -j moviedataset.zip

# Load the csv data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Remove the year from the title column and store it in a separate column
movies_df['year'] = movies_df.str.extract('(\(\d\d\d\d\))', expand = False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand = False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

# Split the genres column into a list of genres
movies_df['genres'] = movies_df.genres.str.split('|')

# One-hot encoding for genres
movie_genres_df = movies_df.copy()
for i, row in movies_df.iterrows():
    for genre in row['genres']:
        movie_genres_df.at[index, genre] = 1
movie_genres_df = movie_genres_df.fillna(0)

# Dropping the unnecessary timestamp column
ratings_df = ratings_df.drop('timestamp', 1)

# Implementing content-based recommendation systems
user_input = [
    {'title': 'Breakfast Club, The', 'rating': 5},
    {'title': 'Toy Story', 'rating': 3.5},
    {'title': 'Jumanji', 'rating': 2},
    {'title': 'Pulp Fiction', 'rating': 5},
    {'title': 'Akira', 'rating': 4.5}
]
input_movies = pd.DataFrame(user_input)

# Add the movie id to input user
input_id = movies_df[movies_df['title'].isin(input_movies['title'].tolist())]
input_movies = pd.merge(input_id, input_movies)
input_movies = input_movies.drop('genres', 1).drop('year', 1)

# Filtering out the movies from the input
user_movies = movie_genres_df[movie_genres_df['movieId'].isin(input_movies['movieId'].tolist())]

# Reset the index and drop the movieId, title, genre and year columns
user_movies = user_movies.reset_index(drop = True)
user_genre_table = user_movies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

# Turn each genre into weights
user_profile = user_genre_table.transpose().dot(input_movies['rating'])

# Extracting the genre table from the original data frame
genre_table = movie_genres_df.set_index(movie_genres_df['movieId'])
genre_table = genre_table.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

recommendation_table_df = ((genre_table * user_profile).sum(axis = 1)) / (user_profile.sum())

# Save the recommendation table
recommended = movies_df['movieId'].isin(recommendation_table_df.head(20).keys())

