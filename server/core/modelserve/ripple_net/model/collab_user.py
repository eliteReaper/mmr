import sys
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors

from consts import MOVIES_CSV, RATINGS_CSV, MediaType

import utils.Timer as Timer

# TODO: Learn collaborative filtering and its various types

def get_recommendations(user, num_recommended_movies, df):

#   print('The list of the Movies User with userId {} Has Watched \n'.format(user))

#   for i, m in enumerate(df[df[user] > 0][user].index.tolist()):
#     print("{:4d}. {}".format(i + 1, m))
  
  print('\n')

  recommended_movies = []

  for m in df[df[user] == 0].index.tolist():
    index_df = df.index.tolist().index(m)
    predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(user)]
    recommended_movies.append((m, predicted_rating))

  sorted_rm = sorted(recommended_movies, key=lambda x:x[1], reverse=True)
  
  print('The list of the Recommended Movies for user with userId {} \n'.format(user))
  rank = 1
  for recommended_movie in sorted_rm[:num_recommended_movies]:
    print('{}: {} - predicted rating:{:.3f}'.format(rank, recommended_movie[0], recommended_movie[1]))
    rank = rank + 1

def recommender(user, num_neighbors, num_recommendation, df, df1):
  
  number_neighbors = num_neighbors

  knn = NearestNeighbors(metric='cosine', algorithm='brute')
  knn.fit(df.values)
  distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)

  user_index = df.columns.tolist().index(user)

  for m,t in list(enumerate(df.index)):
    if df.iloc[m, user_index] == 0:
      similar_movies = indices[m].tolist()
      movie_distances = distances[m].tolist()
    
      if m in similar_movies:
        id_movie = similar_movies.index(m)
        similar_movies.remove(m)
        movie_distances.pop(id_movie) 

      else:
        similar_movies = similar_movies[:num_neighbors-1]
        movie_distances = movie_distances[:num_neighbors-1]
           
      movie_similarity = [1-x for x in movie_distances]
      movie_similarity_copy = movie_similarity.copy()
      nominator = 0

      for s in range(0, len(movie_similarity)):
        if df.iloc[similar_movies[s], user_index] == 0:
          if len(movie_similarity_copy) == (number_neighbors - 1):
            movie_similarity_copy.pop(s)
          
          else:
            movie_similarity_copy.pop(s-(len(movie_similarity)-len(movie_similarity_copy)))
            
        else:
          nominator = nominator + movie_similarity[s]*df.iloc[similar_movies[s],user_index]
          
      if len(movie_similarity_copy) > 0:
        if sum(movie_similarity_copy) > 0:
          predicted_r = nominator/sum(movie_similarity_copy)
        
        else:
          predicted_r = 0

      else:
        predicted_r = 0
        
      df1.iloc[m,user_index] = predicted_r
  get_recommendations(user,num_recommendation, df)

'''
    Argv:
        0: main.py
        1: Type of media to be recommended
'''
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Not enough arguments provided")
    elif sys.argv[1] == MediaType.MOVIES.name:
        print("Movies Recommendation Engine, starting...")
        print("Loading datasets...")

        timer = Timer.Timer()

        timer.start()

        print("\n1. Movies...")
        movies = pd.read_csv(MOVIES_CSV)
        timer.lap()

        print("2. Ratings...")
        timer.start()
        ratings = pd.read_csv(RATINGS_CSV)
        timer.lap()

        # Merging the df's
        print("\nMerging dfs and pivoting...")
        ratings_final_df = pd.merge(ratings, movies, how='inner', on='movieId')
        df = ratings_final_df.pivot_table(index='title',columns='userId',values='rating').fillna(0)
        df1 = df.copy()
        timer.lap()

        
        recommender(10, 10, 10, df, df1)
        timer.lap()

        timer.stop()
    else:
        print("Something went wrong")
    pass