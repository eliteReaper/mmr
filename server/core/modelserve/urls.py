from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict),
    path('train_model/', views.train_model)
]