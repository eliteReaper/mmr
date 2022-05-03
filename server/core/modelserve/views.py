import profile
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from modelserve.rnn_class import RNN
from modelserve.ripple_net.model.ripple_net import RippleNet
import numpy as np
import os
import gc
import json
from memory_profiler import profile
import pandas as pd
import tqdm
from collections import deque

def predict_rnn(request):
    rnn = RNN()

    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    print(body["sequence"])
    res = rnn.predict(body["sequence"])

    del rnn
    gc.collect()
    res_json = {"recommendations": []}
    for i in res:
        res_json["recommendations"].append({"movieId" : i[0], "title": i[1]})
    return JsonResponse(res_json)


def predict_ripple_net(request):
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    args = load_args(body)
    rnn = RNN()

    res = rnn.predict(body["sequence"])

    del rnn
    gc.collect()

    ripple_net = RippleNet(args)

    res = ripple_net.predict(body["userId"], res)

    del ripple_net
    gc.collect()
    res_json = {"recommendations": []}
    for i in res:
        res_json["recommendations"].append({"movieId": int(i[0]), "movieFreq": int(i[1])})

    return JsonResponse(res_json)

def get_accuracy(request):
    if request.method == "POST":
        acc = 0
        rnn = RNN()
        X, Y, users = rnn.get_sequence_for_accuracy()

        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        args = load_args(body)

        ripple_net = RippleNet(args)
        n = 0
        
        saved_model, movie_mapper, all_genres, movie_counter_to_id = rnn.get_all()
        model, movies, ratings = ripple_net.get_all()
        with open("res.txt", "a+") as wr:
            for i in range(0, min(1000, len(X))):
                try:
                    res = rnn.test_prediction(X[i], saved_model, movie_mapper, all_genres, movie_counter_to_id)
                    # for rr in res:
                    #     print("1st Res: ", rr)

                    res = ripple_net.test_prediction(users[i], res, model, movies, ratings)
                    done = False
                    for movie_tuple in res:
                        for yy in Y[i][:min(10, len(Y[i]))]: # TODO Adjust
                            if yy["movieId"] == movie_tuple[0]:
                                done = True
                                break
                        if done:
                            break

                    if done: acc += 1
                    n += 1
                    acc_str = "{},{},{}\n".format(acc / n, acc, n)
                    wr.write(acc_str)
                    print(acc_str)
                except Exception:
                    continue
                # for i, rr in enumerate(res):
                #     print("2nd Res, {}. {}".format(i, rr))
                
                # print(len(Y[i]))
            acc /= n
        return JsonResponse({"accuracy": acc})
    return JsonResponse({"accuracy": "-inf"})

@profile
def train_rnn(request):
    rnn = RNN()
    rnn.train_model()
    del rnn
    gc.collect()
    return HttpResponse('Training Rnn Model Complete\n')
    

def load_args(body):
    base_path = os.getcwd()
    args = {
        "dataset": "movie" if "movie" not in body else body["movie"], 
        "dim": 16 if "dim" not in body  else body["dim"], 
        "n_hop": 2 if "n_hop" not in body  else body["n_hop"], 
        "kge_weight": 0.01 if "kge_weight" not in body  else body["kge_weight"], 
        "l2_weight": 1e-7 if "l2_weight" not in body  else body["l2_weight"], 
        "lr": 0.02 if "lr" not in body  else body["lr"], 
        "batch_size": 1024 if "batch_size" not in body  else body["batch_size"], 
        "n_epoch": 1 if "n_epoch" not in body  else body["n_epoch"], 
        "n_memory": 32 if "n_memory" not in body  else body["n_memory"], 
        "patience": 10 if "patience" not in body  else body["patience"], 
        "item_update_mode": "plus_transform" if "item_update_mode" not in body  else body["item_update_mode"],
        "using_all_hops": True if "using_all_hops" not in body  else body["using_all_hops"],
        "base_path": base_path if "base_path" not in body  else body["base_path"]
    }
    return args

@profile
def train_ripple_net(request):
    if request.method == "POST":
        np.random.seed(555)
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        args = load_args(body)
        ripple_net = RippleNet(args)
        ripple_net.train()
        ripple_net.evaluate()

        del ripple_net
        gc.collect()
    return HttpResponse('Training Ripple Net Complete\n')
