from django.urls import path

from surveillance import views as surveillance_views

app_name = 'surveillance'

"""Contains all the urls to access surveillance views."""

urlpatterns = [
    path('', surveillance_views.home, name="home_page"),
    path('video_feed', surveillance_views.video_feed, name="video_feed"),
    path('violence_video_feed', surveillance_views.violence_video_feed, name="violence_video_feed"),
    path('violence_list', surveillance_views.ViolenceListView.as_view(), name="violence_list_view"),
]
