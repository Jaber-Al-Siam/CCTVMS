from django.urls import path

from surveillance import views as surveillance_views

app_name = 'surveillance'

"""Contains all the urls to access surveillance views."""

urlpatterns = [
    path('', surveillance_views.home, name="home_page"),
    path('cameras', surveillance_views.CameraListView.as_view(), name="camera_list_view"),
    path('cameras/<int:pk>', surveillance_views.CameraDetailsView.as_view(), name="camera_detail_view"),
    path('video_feed/<int:cid>', surveillance_views.video_feed, name="video_feed"),
    path('violence_video_feed/<int:cid>', surveillance_views.violence_video_feed, name="violence_video_feed"),
]
