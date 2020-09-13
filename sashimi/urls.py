from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="home"),
    path("post/five", views.five, name="five"),
    path("post/shiromi", views.shiromi, name="shiromi"),
    path("post/ao", views.ao, name="ao"),
    path("result", views.result, name="result"),
]
