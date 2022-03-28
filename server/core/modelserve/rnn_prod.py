
from tkinter import TOP
import pandas as pd
from collections import defaultdict, deque
import itertools
import numpy as np
# import dask.dataframe as dd
from tqdm import tqdm


MOVIES_DATASET = "./data/movies.csv"
RATINGS_DATASET = "./data/ratings.csv"
NUMBER_OF_USER_WATCHED_MIN = 50
NUMBER_OF_MOVIES_WATCHED_MIN = 20
TRAIN_TEST_SPLIT = 0.8
SEQ_LEN = 10
TOP_N = 10

# # Preprocessing Movies Dataset

# Mapping movieId to indexes for later use in one-hot encoding  
movie_mapper = defaultdict(tuple)
genre_mapper = defaultdict(int)
all_genres = defaultdict(int)
def map_movie_to_idx():
    movies_df = pd.read_csv(MOVIES_DATASET)
    
    ratings_df = pd.read_csv(RATINGS_DATASET)
    movie_ids = ratings_df['movieId']
    movie_dict = defaultdict(int)
    for movie in movie_ids:
        movie_dict[movie] += 1
    
    movies_df.dropna(inplace=True)
    counter = 0
    
    # Fastest way to iterate a df
    iter_dict = movies_df.to_dict('records')
    for row in tqdm(iter_dict):
        genres = str(row['genres']).split('|')
        for genre in genres: all_genres[genre] += 1

        if (row['movieId'] not in movie_dict) or (movie_dict[row['movieId']] < NUMBER_OF_USER_WATCHED_MIN): 
            continue

        movie_mapper[row['movieId']] = (counter, row['title'], genres)
        counter += 1
    # print("Number of movies {}".format(counter))

def map_genres_to_idx():
    counter = 0
    for genre in all_genres:
        genre_mapper[genre] = counter
        counter += 1


map_movie_to_idx()
map_genres_to_idx()

# # Preprocessing Ratings Dataset


def one_hot_encode_movie(movieId):
    num_movies = len(movie_mapper)
    encoded_movie = np.zeros(num_movies, dtype=np.float32) 
    # encoded_movie = [0] * num_movies
    encoded_movie[movie_mapper[movieId][0]] = 1
    return encoded_movie

def one_hot_encode_user_seq(user_seq):
    encoded = []
    for movie in user_seq:
        encoded.append(one_hot_encode_movie(movie))
    return np.array(encoded, dtype=object)

def encode_movie_with_genre(movieId):
    genres = movie_mapper[movieId][2]
    sz = len(all_genres)
    encoded = np.zeros(sz)
    for genre in genres:
        encoded[genre_mapper[genre]] = 1
    return encoded / np.linalg.norm(encoded)

def one_hot_encode_movie_genre(movieId):
    seq1 = one_hot_encode_movie(movieId)
    seq2 = encode_movie_with_genre(movieId)
    return np.concatenate((seq1,seq2))


def sequentialize(ratings_df):
    ratings_df = ratings_df.dropna()
    ratings_df = ratings_df.sort_values(by=['userId', 'timestamp'])
    
    ratings_df = ratings_df.dropna()
    userIds = ratings_df['userId'].unique()
    
    X = []
    Y = []
    for user in tqdm(userIds):
        user_seq = list(ratings_df[(ratings_df['userId'] == user)]['movieId'])
        if len(user_seq) >= NUMBER_OF_MOVIES_WATCHED_MIN:
            sequence = deque(maxlen=SEQ_LEN)
            for movieId in user_seq:
                if movieId not in movie_mapper: continue
                # sequence.append(encode_movie_with_genre(movieId))
                sequence.append(one_hot_encode_movie(movieId))
                if len(sequence) == SEQ_LEN:
                    X.append(list(itertools.islice(sequence, 0, SEQ_LEN - 1)))
                    Y.append(one_hot_encode_movie(movieId))
                    # Y.append(sequence[-1])
    return np.array(X), np.array(Y) # Required

def get_datasets():
    ratings_df = pd.read_csv(RATINGS_DATASET)
    ratings_df = ratings_df.dropna()
    
    sz = len(ratings_df)
    train_df = ratings_df[:int(TRAIN_TEST_SPLIT * sz)]
    test_df = ratings_df[int(TRAIN_TEST_SPLIT * sz):]
    
    X_train, Y_train = sequentialize(train_df)
    X_test, Y_test = sequentialize(test_df)
    
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = get_datasets()


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM # CuDNNLSTM
from keras import backend as K

model = Sequential()

model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(256))
model.add(Dropout(0.1))

# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(len(movie_mapper), activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-6)

def sps(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=TOP_N)


# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=[sps]
)


model.fit(X_train,Y_train,epochs=100,validation_data=(X_test, Y_test), batch_size=512)
