from django.contrib import admin
from surveillance.models import Camera, Violence, Person


# Register your models here.


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ['pk', 'name', 'age', 'gender', 'image_file', ]

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)


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
