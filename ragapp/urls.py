from django.urls import path
from . import views

urlpatterns = [
    path('', views.load_manual_view, name='load_manual'),
    path('chat/', views.chat_view, name='chat'),
    path('api/chat/', views.chat_api_view, name='chat_api'),
    path('upload/', views.upload_manual_view, name='upload_manual'),
]