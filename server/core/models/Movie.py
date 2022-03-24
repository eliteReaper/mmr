from numpy import product


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