import pandas as pd
import os

class DAO:
    def __init__(self):
        self.base_path = os.getcwd()
        pass

    def get_all_movies(self):
        df = pd.read_csv(self.base_path + "/data/movies2.csv", sep="::")
        movie_dict = {"movies": []}
        iter_dict = df.to_dict('records')
        for row in iter_dict:
            movieId, title, genres = str(row['movieId']), str(row['title']), str(row['genres']).split("|")
            movie_dict["movies"].append({"movieId": movieId, "title": title, "genres": genres})
        return movie_dict
    
    def get_ratings_for_user(self, userId):
        df = pd.read_csv(self.base_path + "/data/ratings2.csv", sep="::")
        df = df[df['userId'] == userId]
        ratings_dict = {"movies": []}
        iter_dict = df.to_dict('records')
        for row in iter_dict:
            movieId,rating,timestamp = str(row['movieId']), str(row['rating']), str(row['timestamp'])
            ratings_dict["movies"].append({"movieId": movieId, "title": rating, "timestamp": timestamp})
        return ratings_dict