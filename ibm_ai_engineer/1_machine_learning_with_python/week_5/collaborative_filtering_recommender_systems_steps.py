
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# Download the data set
!wget -O moviedataset.zip https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
print('unziping ...')
!unzip -o -j moviedataset.zip 

# Load csv files
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Remove the year from the title column and move it to it's own year column
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand = False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand = False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

# Drop the genres column as we won't need it for this particular recommendation system
movies_df = movies_df.drop('genres', 1)

# Creating the test user input
user_input = [
    {'title': 'Breakfast Club, The', 'rating': 5},
    {'title': 'Toy Story', 'rating': 3.5},
    {'title': 'Jumanji', 'rating': 2},
    {'title': 'Pulp Fiction', 'rating': 5},
    {'title': 'Akira', 'rating': 4.5}
]
input_movies = pd.DataFrame(user_input)

# Extracting the movie ids for the user input
input_id = movies_df[movies_df['title'].isin(input_movies['title'].tolist())]
input_movies = pd.merge(input_id, input_movies)
input_movies = input_movies.drop('year', 1)

# Create test users to serve as neighbors for user
user_subset = ratings_df[ratings_df['movieId'].isin(input_movies['movieId'].tolist())]

# Group the rows by user id
user_subset_group = user_subset.groupby(['userId'])

# Sort the groups so the users that share the most common movies are of a higher priority
user_subset_group = sorted(user_subset_group, key = lambda x: len(x[1]), reverse = True)

# Calculate the pearson correlation between input_user and the subset_group and store it in a dictionary {user_id: coefficient}
pearson_correlation_dict = {}
for name, group in user_subset_group:
    group = group.sort_values(by = 'movieId')
    input_movies = input_movies.sort_values(by = 'movieId')
    n_ratings = len(group)
    temp_df = input_movies[input_movies['movieId'].isin(group['movieId'].tolist())]
    temp_ratings_list = temp_df['rating'].tolist()
    temp_group_list = group['rating'].tolist()
    sxx = sum([i**2 for i in temp_ratings_list]) - pow(sum(temp_ratings_list), 2) / float(n_ratings)
    sxx = sum([i**2 for i in temp_group_list]) - pow(sum(temp_group_list), 2) / float(n_ratings)
    sxx = sum([i * j for i, j in zip(temp_ratings_list, temp_group_list)]) - sum(temp_ratings_list) * sum(temp_group_list) / float(n_ratings)
    if sxx != 0 and syy != 0:
        pearson_correlation_dict[name] = sxy / sqrt(sxx * syy)
    else:
        pearson_correlation_dict[name] = 0

# Convert person_correlation_dict to a dataframe
pearson_df = pd.DataFrame.from_dict(pearson_correlation_dict, orient = 'index')
pearson_df.columns = ['similarity_index']
pearson_df['userId'] = pearson_df.index
pearson.index = range(len(pearson_df))

# Top users
top_users = pearson_df.sort_values(by = 'similarity_index', ascending = False)[0:50]

# Take the weighted average of the ratings of the movies using th pearson correlation as the weight.
# By first getting the movies watched by the users in pearson_df from the ratings dataframe and then storing
# their correlation in a new column called similarity_index_, by merging the following tables
top_users_rating = top_users.merge(ratings_df, left_on = 'userId', right_on = 'userId', how = 'inner')

# Multiply the new rating by its weight, sum up the new ratings and divide the sum of the weights
top_users_rating['weighted_rating'] = top_users_rating['similarity_index'] * top_users_rating['rating']
temp_top_users_rating = top_users_rating.groupby('movieId').sum()[['similarity_index', 'weighted_rating']]
temp_top_users_rating.columns = ['sum_similarity_index', 'sum_weighted_rating']
recommendation_df = pd.DataFrame()
recommendation_df['weighted_average_recommendation_score'] = temp_top_users_rating['sum_weighted_rating'] / temp_top_users_rating['sum_similarity_index']
recommendation_df['movieId'] = temp_top_users_rating.index

# Top 20 recommendations saved in a variable
recommendation_df = recommendation_df.sort_values(by = 'weighted_average_recommendation_score', ascending = False)
top_20_recommendations = recommendation_df.head(20)
movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(20)['movieId'].tolist())]