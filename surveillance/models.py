from django.db import models


# Create your models here.


class Person(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    image_file = models.FileField(upload_to='persons/%Y/%m/%d/', blank=True)

    def save(self, *args, **kwargs):
        """
        This funtcion will be called on Person instance save.
        :param args: Arguments data.
        :param kwargs: Kwargs data.
        :return: None
        """
        super(Person, self).save(*args, **kwargs)

class Camera(models.Model):
    place = models.CharField(max_length=100)
    ip = models.CharField(max_length=100, unique=True)
    host_name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return str(self.host_name)

    # def get_absolute_url(self):
    #     """
    #     Function to get absolute url of parcel objects.
    #     :return: Absolute url of the instance.
    #     :rtype: django.shortcuts.reverse
    #     """
    #     return reverse('parcels:detail', kwargs={'pk': self.pk})

    def save(self, *args, **kwargs):
        """
        This funtcion will be called on Camera instance save.
        :param args: Arguments data.
        :param kwargs: Kwargs data.
        :return: None
        """
        super(Camera, self).save(*args, **kwargs)


class Violence(models.Model):
    time = models.DateTimeField(auto_now_add=True)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    video = models.FileField(upload_to='violences')
    involved_persons = models.ManyToManyField(Person, blank=True)

    def save(self, *args, **kwargs):
        """
        This funtcion will be called on Violence instance save.
        :param args: Arguments data.
        :param kwargs: Kwargs data.
        :return: None
        """
        super(Violence, self).save(*args, **kwargs)
