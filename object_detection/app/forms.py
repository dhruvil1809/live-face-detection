from django import forms
from .models import UploadedMedia
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

class MediaUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedMedia
        fields = ['media']

    def clean_media(self):
        media = self.cleaned_data['media']
        if media:
            file_extension = media.name.split('.')[-1].lower()
            if file_extension not in ['jpg', 'jpeg', 'mp4']:
                raise ValidationError(_('File type not supported.'))
        return media
