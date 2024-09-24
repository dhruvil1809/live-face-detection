from django.db import models

class UploadedMedia(models.Model):
    media = models.FileField(upload_to='media/', default=None)