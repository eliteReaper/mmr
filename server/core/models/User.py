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