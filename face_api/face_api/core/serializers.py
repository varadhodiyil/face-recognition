from rest_framework import serializers
from face_api.core import models

class EnrollSerializer(serializers.ModelSerializer):
    data = serializers.FileField()

    class Meta:
        model = models.Users
        fields = ('username','first_name','last_name','email','file_path','data')

class VerifySerializer(serializers.Serializer):
    data = serializers.FileField()

    class Meta:
        fields = '__all__'


class FundTransferSerializer(serializers.ModelSerializer):
    data = serializers.FileField()
    class Meta:
        fields = '__all__'
        model = models.FundTransfer