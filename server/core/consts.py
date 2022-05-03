from enum import Enum

# Data Filepaths
MOVIES_CSV="./data/movies2.csv"
RATINGS_CSV="./data/ratings2.csv"
TAGS_CSV="./data/tags/csv"
LINKS_CSV="./data/links.csv"

# Media types supported
class MediaType(Enum):
    MOVIES = 1
    SONGS = 1