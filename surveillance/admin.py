import os

from django.contrib import admin

from CCTVMS import settings
from surveillance.models import Camera, Violence, Person
from surveillance.face_recognizer import save_face_from_video_path


# Register your models here.


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ['pk', 'name', 'age', 'gender', 'image_file', ]

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        save_face_from_video_path(os.path.join(settings.BASE_DIR, 'media/{0}'.format(str(obj.image_file.name))), obj.id)


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['pk', 'place', 'ip', 'host_name', ]

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)


@admin.register(Violence)
class ViolenceAdmin(admin.ModelAdmin):
    list_display = ['time', 'camera', 'video', ]

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
