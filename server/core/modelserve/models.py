from django.db import models

# Create your models here.
class Movie():
    '''
        id: Unique id of the movie
        title: Name of the movie
        genres: The genres the movie falls under
        cast: Cast for the movie
        production: People behind production of the movie
    '''
    def __init__(self, id, title, genres, cast = [], production = []):
        self._id = id
        self.title = title
        self.genres = genres
        self.cast = cast
        self.production = production

class User:
    '''
        id: Unique user id
        name: Name of the user
        rating: Rating of the user
        ratings: Array to store rating interaction id of the user
    '''
    def __init__(self, id, name, rating, ratings):
        self._id = None
        self.name = None
        self.userRating = None
        self.ratings = ratings
