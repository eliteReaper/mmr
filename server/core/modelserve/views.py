from django.shortcuts import render
from django.http import HttpResponse

def predict(request):
    return HttpResponse('Prediction here')
