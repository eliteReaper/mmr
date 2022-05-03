from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from modelserve.ripple_net.tools.load_data import LoadData
import numpy as np
import datetime
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import math
import pandas as pd
from collections import defaultdict


class BuildModel:
    def __init__(self, args):
        self.args = args
        self._parse_args()
        self.data_info = LoadData(args)
        self.train_data, self.test_data, self.n_entity, self.n_relation, self.ripple_set = self.data_info.load_data()
        self.model = self.build_model()

    def _parse_args(self):
        self.batch_size = self.args["batch_size"]
        self.epochs = self.args["n_epoch"]
        self.patience = self.args["patience"]
        self.dim = self.args["dim"]
        self.n_hop = self.args["n_hop"]
        self.kge_weight = self.args["kge_weight"]
        self.l2_weight = self.args["l2_weight"]
        self.lr = self.args["lr"]
        self.n_memory = self.args["n_memory"]
        self.item_update_mode = self.args["item_update_mode"]
        self.using_all_hops = self.args["using_all_hops"]
        self.save_path = self.args["base_path"] + "/data/" + self.args["dataset"]
        self.save_path += "/ripple_net_{}_model.h5".format(self.args["dataset"])
        current_time = datetime.datetime.now().strftime(('%Y%m%d-%H%M%S'))
        self.log_path = self.args["base_path"] + "/logs/{}_{}".format(self.args["dataset"], current_time)
        self.base_path = self.args["base_path"]

    def step_decay(self, epoch):
        # learning rate step decay
        initial_l_rate = self.lr
        drop = 0.5
        epochs_drop = 10.0
        l_rate = initial_l_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        print("learning_rate", l_rate)
        return l_rate

    def build_model(self):
        pass

    def data_parse(self, data):
        # build X, y from data
        np.random.shuffle(data)
        items = data[:, 1]
        labels = data[:, 2]
        memories_h = list(range(self.n_hop))
        memories_r = list(range(self.n_hop))
        memories_t = list(range(self.n_hop))
        for hop in range(self.n_hop):
            memories_h[hop] = np.array([self.ripple_set[user][hop][0] for user in data[:, 0]])
            memories_r[hop] = np.array([self.ripple_set[user][hop][1] for user in data[:, 0]])
            memories_t[hop] = np.array([self.ripple_set[user][hop][2] for user in data[:, 0]])
        return [items, labels] + memories_h + memories_r + memories_t, labels

    def train(self):
        print("train model ...")
        self.model.summary()
        X, y = self.data_parse(self.train_data)
        tensorboard = TensorBoard(log_dir=self.log_path, histogram_freq=1)
        early_stopper = EarlyStopping(patience=self.patience, verbose=1)
        model_checkpoint = ModelCheckpoint(self.save_path, verbose=1, save_best_only=True)
        learning_rate_scheduler = LearningRateScheduler(self.step_decay)
        self.model.fit(x=X,
                       y=y,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=0.2,
                       callbacks=[early_stopper, model_checkpoint, learning_rate_scheduler, tensorboard])

    def evaluate(self):
        model = self.build_model()
        model.load_weights(self.save_path)
        print("evaluate model ...")
        X, y = self.data_parse(self.test_data)
        score = model.evaluate(X, y, batch_size=self.batch_size)
        print("- loss: {} "
              "- binary_accuracy: {} "
              "- auc: {} "
              "- f1: {} "
              "- precision: {} "
              "- recall: {}".format(*score))

    
    def user_user(self, df, userId):
        user_movie_table_matrix = csr_matrix(df.values)
        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        model_knn.fit(user_movie_table_matrix)

        query_index = [i for i, e in enumerate(list(df.index)) if e == userId][0]
        distances, indices = model_knn.kneighbors(df.iloc[query_index,:].values.reshape(1,-1), n_neighbors = 50)
        

        user = []
        distance = []

        for i in range(0, len(distances.flatten())):
            if i != 0: # Not adding the point itself
                user.append(df.index[indices.flatten()[i]])
                distance.append(distances.flatten()[i])    

        m=pd.Series(user,name='user')
        d=pd.Series(distance,name='distance')
        recommend = pd.concat([m,d], axis=1)
        recommend = recommend.sort_values('distance',ascending=False)

        # print('Recommendations for {0}:\n'.format(df.index[query_index]))
        # for i in range(0,recommend.shape[0]):
        #     print('{0}: {1}, with distance of {2}'.format(i, recommend["user"].iloc[i], recommend["distance"].iloc[i]))

        return recommend
    
    def get_similar_users(self, userId):
        movies = pd.read_csv(self.base_path + "/data/movies2.csv", sep="::")
        ratings = pd.read_csv(self.base_path + "/data/ratings2.csv", sep="::")

        # Merging the df's
        ratings_final_df = pd.merge(ratings, movies, how='inner', on='movieId')
        df = ratings_final_df.pivot_table(index='userId',columns='title',values='rating').fillna(0)
        return self.user_user(df, userId)

    def predict(self, userId, movie_list):
        model = self.build_model()
        model.load_weights(self.save_path)
        sim_users = self.get_similar_users(userId)

        X = []
        for movie in movie_list:
            for i in range(0,sim_users.shape[0]):
                X.append([sim_users["user"].iloc[i], movie[0], 1])
        
        X = np.array(X)
        prev_X = np.copy(X)
        X, _ = self.data_parse(X)
        pred = model.predict(X)
        result = [prev_X[i][1] if x > 0.5 else -1 for i, x in enumerate(pred)]
        
        freq = defaultdict(int)
        for id in result:
            if id != -1:
                freq[id] += 1
        
        arr = []
        for k, v in freq.items():
            arr.append((k, v))
        
        arr = sorted(arr, key= lambda x: x[1], reverse=True)
        return arr
    
    def get_all(self):
        model = self.build_model()
        model.load_weights(self.save_path)
        movies = pd.read_csv(self.base_path + "/data/movies2.csv", sep="::")
        ratings = pd.read_csv(self.base_path + "/data/ratings2.csv", sep="::")
        return model, movies, ratings

    def test_get_similar_users(self, userId, movies, ratings):
        # Merging the df's
        ratings_final_df = pd.merge(ratings, movies, how='inner', on='movieId')
        df = ratings_final_df.pivot_table(index='userId',columns='title',values='rating').fillna(0)
        return self.user_user(df, userId)
    
    def test_prediction(self, userId, movie_list, model, movies, ratings):
        sim_users = self.test_get_similar_users(userId, movies, ratings)
        X = []
        for movie in movie_list:
            for i in range(0,sim_users.shape[0]):
                X.append([sim_users["user"].iloc[i], movie[0], 1])
        
        X = np.array(X)
        prev_X = np.copy(X)
        X, _ = self.data_parse(X)
        pred = model.predict(X)
        result = [prev_X[i][1] if x > 0.5 else -1 for i, x in enumerate(pred)]
        
        freq = defaultdict(int)
        for id in result:
            if id != -1:
                freq[id] += 1
        
        arr = []
        for k, v in freq.items():
            arr.append((k, v))
        
        arr = sorted(arr, key= lambda x: x[1], reverse=True)
        return arr
