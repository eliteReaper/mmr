from enum import Enum

# Data Filepaths
MOVIES_CSV="./data/movies.csv"
RATINGS_CSV="./data/ratings.csv"
TAGS_CSV="./data/tags/csv"
LINKS_CSV="./data/links.csv"

# Media types supported
class MediaType(Enum):
    MOVIES = 1
    SONGS = 1