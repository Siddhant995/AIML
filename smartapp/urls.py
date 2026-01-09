from django.urls import path
from core import views
urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_file, name='upload'),
    path('overview/', views.overview, name='overview'),
    path('column/<str:col>/', views.column_view, name='column'),
    path('predict/', views.predict_select, name='predict_select'),
    path('predict/run/', views.predict_run, name='predict_run'),
    path('visualize/', views.visualize, name='visualize'),
]
