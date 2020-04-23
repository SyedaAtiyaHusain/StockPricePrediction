from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='stockhome'),
    path('features/',views.features,name='features'),
    path('contact/',views.contact,name='contact'),
]
