from django.urls import path
from . import views

urlpatterns = [
    path('predict_npk', views.predict_npk, name='predict_npk'),
    path('recommend_fertilizer', views.recommend_fertilizer, name='recommend_fertilizer'),
]
