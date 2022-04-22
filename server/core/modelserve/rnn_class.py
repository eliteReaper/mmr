
from gc import callbacks
from tabnanny import verbose
import pandas as pd
from collections import defaultdict, deque
import itertools
import numpy as np
# import dask.dataframe as dd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM # CuDNNLSTM
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
# from keras import backend as K


MOVIES_DATASET = "../data/movies2.csv"
RATINGS_DATASET = "../data/ratings2.csv"


class RNN:
    def __init__(self):
        self.NUMBER_OF_USER_WATCHED_MIN = 500
        self.NUMBER_OF_MOVIES_WATCHED_MIN = 20
        self.TRAIN_TEST_SPLIT = 0.8
        self.SEQ_LEN = 5
        self.TOP_N = 250
        self.TRAINED = False
        self.SEQUENCE_MAX_LEN = 100

        self.model = None
        self.history = None
        self.samples = 0
        self.timesteps = 0
        self.features = 0
        self.movie_mapper = defaultdict(tuple)
        self.genre_mapper = defaultdict(int)
        self.all_genres = defaultdict(int)
        self.tf_batch_size = 128
        self.epochs = 100
        self.patience = 20
    
    def map_movie_to_idx(self):
        movies_df = pd.read_csv(MOVIES_DATASET, sep='::')
        
        ratings_df = pd.read_csv(RATINGS_DATASET, sep='::')
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
            for genre in genres: self.all_genres[genre] += 1

            if (row['movieId'] not in movie_dict) or (movie_dict[row['movieId']] < self.NUMBER_OF_USER_WATCHED_MIN): 
                continue

            self.movie_mapper[row['movieId']] = (counter, row['title'], genres)
            counter += 1
        # print("Number of movies {}".format(counter))

    def map_genres_to_idx(self):
        counter = 0
        for genre in self.all_genres:
            self.genre_mapper[genre] = counter
            counter += 1
    
    def one_hot_encode_movie(self, movieId):
        num_movies = len(self.movie_mapper)
        encoded_movie = np.zeros(num_movies, dtype=np.float32) 
        # encoded_movie = [0] * num_movies
        encoded_movie[self.movie_mapper[movieId][0]] = 1
        return encoded_movie

    def one_hot_encode_user_seq(self, user_seq):
        encoded = []
        for movie in user_seq:
            encoded.append(self.one_hot_encode_movie(movie))
        return np.array(encoded, dtype=object)

    def encode_movie_with_genre(self, movieId):
        genres = self.movie_mapper[movieId][2]
        sz = len(self.all_genres)
        encoded = np.zeros(sz)
        for genre in genres:
            encoded[self.genre_mapper[genre]] = 1
        # encoded = encoded / np.linalg.norm(encoded)
        return encoded

    def one_hot_encode_movie_genre(self, movieId):
        seq1 = self.one_hot_encode_movie(movieId)
        seq2 = self.encode_movie_with_genre(movieId)
        encoded = np.concatenate((seq1,seq2))
        # encoded = encoded / np.linalg.norm(encoded)
        # np.random.shuffle(encoded)
        return encoded

    def encode_rating(self, rating):
        encoded = np.zeros(5, dtype=np.float32)
        encoded[rating - 1] = 1
        return encoded

    def one_hot_encode_movie_genre_rating(self, movieId, rating):
        # print(movieId, rating)
        genre_encode = self.one_hot_encode_movie_genre(movieId)
        rating_encode = self.encode_rating(rating)
        encoded = np.concatenate((genre_encode,rating_encode))
        return encoded

    def sequentialize(self, ratings_df):
        ratings_df = ratings_df.dropna()
        ratings_df = ratings_df.sort_values(by=['userId', 'timestamp'])
        
        ratings_df = ratings_df.dropna()
        userIds = ratings_df['userId'].unique()
        
        X = []
        Y = []
        for user in tqdm(userIds):
            user_seq = ratings_df[(ratings_df['userId'] == user)]
            if len(user_seq) >= self.NUMBER_OF_MOVIES_WATCHED_MIN:
                sequence = deque(maxlen=self.SEQ_LEN)
                iter_dict = user_seq.to_dict('records')
                for row in iter_dict:
                    movieId = row['movieId']
                    rating = int(row['rating'])
                    if movieId not in self.movie_mapper: continue
                    # sequence.append(movieId)
                    sequence.append(self.one_hot_encode_movie_genre_rating(movieId, rating))
                    # sequence.append(one_hot_encode_movie_genre(movieId))
                    # sequence.append(one_hot_encode_movie(movieId))
                    if len(sequence) == self.SEQ_LEN:
                        X.append(list(itertools.islice(sequence, 0,self.SEQ_LEN - 1)))
                        Y.append(self.one_hot_encode_movie(movieId))
                        # Y.append(sequence[-1])
        return np.array(X), np.array(Y) # Required
    
    def sequentialize_v2(self, ratings_df):
        ratings_df = ratings_df.dropna()
        ratings_df = ratings_df.sort_values(by=['userId', 'timestamp'])
        
        ratings_df = ratings_df.dropna()
        userIds = ratings_df['userId'].unique()
        
        X = []
        Y = []
        for user in tqdm(userIds):
            user_seq = ratings_df[(ratings_df['userId'] == user)]
            if len(user_seq) >= self.NUMBER_OF_MOVIES_WATCHED_MIN:
                sequence = deque(maxlen=self.SEQUENCE_MAX_LEN)
                iter_dict = user_seq.to_dict('records')
                lastMovieId = -1
                for row in iter_dict:
                    movieId = row['movieId']
                    rating = int(row['rating'])
                    if movieId not in self.movie_mapper: continue
                    sequence.append(self.one_hot_encode_movie_genre_rating(movieId, rating))
                    lastMovieId = movieId
                
                if lastMovieId != -1:
                    while len(sequence) < self.SEQUENCE_MAX_LEN:
                        sequence.append([-1] * (len(self.movie_mapper) + len(self.all_genres) + 5))
                    X.append(list(itertools.islice(sequence, 0, self.SEQUENCE_MAX_LEN- 1)))
                    Y.append(self.one_hot_encode_movie(lastMovieId))
        return np.array(X), np.array(Y) # Required

    def get_datasets(self):
        ratings_df = pd.read_csv(RATINGS_DATASET, sep='::')
        ratings_df = ratings_df.dropna()
        train_df, test_df = train_test_split(ratings_df, train_size=self.TRAIN_TEST_SPLIT)
        X_train, Y_train = self.sequentialize_v2(train_df)
        X_test, Y_test = self.sequentialize_v2(test_df)
        
        return X_train, Y_train, X_test, Y_test
    
    def map_x_to_idx(self):
        self.map_movie_to_idx()
        self.map_genres_to_idx()
    
    def sps(self, y_true, y_pred):
        return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=self.TOP_N)
    
    def setup_model(self):
        self.model = Sequential()
        self.model.add(tf.keras.layers.Masking(mask_value=-1.0, input_shape=(self.timesteps, self.features)))

        # self.model.add(LSTM(64, return_sequences=True))
        # self.model.add(Dropout(0.2))

        self.model.add(LSTM(128))
        # self.model.add(Dropout(0.1))
        self.model.add(Dropout(0.2))

        # self.model.add(Dense(64, activation='relu'))
        # self.model.add(Dropout(0.2))

        self.model.add(Dense(len(self.movie_mapper), activation='softmax'))

        opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-6)

        # Compile model
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=[self.sps]
        )

    def train_model(self):
        self.map_x_to_idx()
        X_train, Y_train, X_test, Y_test = self.get_datasets()
        self.samples, self.timesteps, self.features = X_train.shape
        self.setup_model()
        es = tf.keras.callbacks.EarlyStopping(monitor='val_sps', mode="max", patience=self.patience)
        mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        self.history = self.model.fit(X_train,Y_train,epochs=self.epochs,validation_data=(X_test, Y_test), batch_size=self.tf_batch_size, callbacks=[es, mc], verbose=False)
        self.TRAINED = True

        saved_model = tf.keras.models.load_model('best_model.h5', custom_objects={"sps": self.sps})
        # evaluate the model
        _, train_acc = saved_model.evaluate(X_train, Y_train, verbose=0)
        _, test_acc = saved_model.evaluate(X_test, Y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

        # pyplot.plot(self.history.history['loss'], label='train')
        # pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.plot(self.history.history['sps'], label='train')
        pyplot.plot(self.history.history['val_sps'], label='test')
        pyplot.legend()
        pyplot.show()

        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

if __name__ == '__main__':
    rnn = RNN()
    rnn.train_model()
