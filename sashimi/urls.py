from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="home"),
    path("post/five", views.five, name="five"),
    path("post/shiromi", views.shiromi, name="shiromi"),
    path("post/ao", views.ao, name="ao"),
    path("five_result", views.five_result, name="five_result"),
    path("shiromi_result", views.shiromi_result, name="shiromi_result"),
    path("ao_result", views.ao_result, name="ao_result"),
]
