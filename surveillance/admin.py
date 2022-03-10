import os
import threading

from django.contrib import admin

from CCTVMS import settings
from surveillance.models import Camera, Violence, Person
from surveillance.face_recognizer import save_face_from_video_path


# Register your models here.

class InvolvedPersonInline(admin.TabularInline):
    model = Violence.involved_persons.through


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ['pk', 'name', 'age', 'gender', 'image_file', ]
    readonly_fields = ['pk', ]
    inlines = [InvolvedPersonInline, ]

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        threading.Thread(target=save_face_from_video_path, args=(os.path.join(settings.BASE_DIR, 'media/{0}'.format(str(obj.image_file.name))), obj.id)).start()


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['pk', 'place', 'ip', 'host_name', ]
    readonly_fields = ['pk', ]

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)


@admin.register(Violence)
class ViolenceAdmin(admin.ModelAdmin):
    list_display = ['pk', 'time', 'camera', 'video', ]
    readonly_fields = ['pk', ]
    inlines = [InvolvedPersonInline, ]
    exclude = ['involved_persons', ]

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
