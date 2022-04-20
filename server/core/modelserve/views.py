from django.shortcuts import render
from django.http import HttpResponse
import modelserve.rnn_prod2 as rnn_prod2

def predict(request):
    return HttpResponse('Prediction here')

def train_model(request):
    rnn_prod2.fit_model()
    return HttpResponse('Training Model Complete')
