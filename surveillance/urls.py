from django.urls import path, include
from surveillance import views as surveillance_views

app_name = 'surveillance'

"""Contains all the urls to access surveillance views."""

urlpatterns = [
    path('', surveillance_views.home, name="home_page"),
]
