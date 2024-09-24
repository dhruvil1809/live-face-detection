from django.urls import path
from . import views 

urlpatterns = [
    path('', views.home, name='home'),
    path('activate_webcam/', views.activate_webcam, name='activate_webcam'),
]
