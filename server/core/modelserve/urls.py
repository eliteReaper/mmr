from django.urls import path
from . import views

urlpatterns = [
    path('predict_rnn/', views.predict_rnn),
    path('predict_ripple_net/', views.predict_ripple_net),
    path('train_rnn/', views.train_rnn),
    path('train_ripple_net/', views.train_ripple_net),
    path('get_accuracy/', views.get_accuracy),
    path('get_all_movies', views.get_all_movies),
    path('get_user_rated_movies', views.get_user_rated_movies)
]