# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.generics import GenericAPIView
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from face_api.core import models, serializers
from face_api.core.model.DataSetGenerator import DataSetGenerator
from face_api.core.model.predictor import Predictor
from face_api.core.model.verify_user import VerifyUser

MEDIA_ROOT = getattr(settings,'MEDIA_ROOT','')
# Create your views here.
path = os.path.dirname(os.path.abspath(__file__))
dg = DataSetGenerator(os.path.join(path,"model" , "training_face_rec"))
MAX_LABELS  = len(dg.data_labels)
# p = Predictor(dg,num_labels=MAX_LABELS)

class Enroll(GenericAPIView):
    parser_classes = (MultiPartParser,)
    serializer_class = serializers.EnrollSerializer

    def post(self, request,*args,**kwargs):
        data = request.data
        s = serializers.EnrollSerializer(data=data)
        result = dict()
        if s.is_valid():
            file_obj = s.validated_data['data']
            file_type = file_obj.content_type.split('/')[0]
            if file_type == "video" or file_type == "application":
                fs = FileSystemStorage()
                s.validated_data.pop('data')
                instance = s.save()
                name = instance.id.__str__()
                path = os.path.join(name,file_obj.name)
                filename = fs.save(path, file_obj)
                uploaded_file_url = fs.url(filename)
                
                instance.file_path = uploaded_file_url
                instance.save()
                result['status'] = True
                return Response(result)
            else:
                result['status'] = False
                result['error'] = "Please Select a valid video file"
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
        else:
            result['status'] = False
            result['errors'] = s.errors
            return Response(result, status=status.HTTP_400_BAD_REQUEST)


class Login(GenericAPIView):
    parser_classes = (MultiPartParser,)
    serializer_class = serializers.VerifySerializer
    def post(self, request,*args,**kwargs):
        data = request.data
        s = serializers.VerifySerializer(data=data)
        result = dict()
        if s.is_valid():
            file_obj = s.validated_data['data']
            file_type = file_obj.content_type.split('/')[0]
            if file_type == "video" or file_type == "application":
                fs = FileSystemStorage()
                path = os.path.join('verify',file_obj.name)
                filename = fs.save(path, file_obj)
                _path =  os.path.join(MEDIA_ROOT,filename)

                
                v_user  = VerifyUser()

                user , status_ = v_user.get_results(_path)
                print user , status_
                if user is None:
                    result['status'] = False
                    result['errors']  = status_
                if user:
                    result['status'] = True
                    user = models.Users.objects.filter(id=user)
                    user = get_object_or_404(user)
                    token = models.Token(user=user).save()
                    result['token'] = token.key
                    result['status'] = status_
                # result['status'] = False
                return Response(result)
            else:
                result['status'] = False
                result['error'] = "Please Select a valid video file"
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
        else:
            result['status'] = False
            result['errors'] = s.errors
            return Response(result, status=status.HTTP_400_BAD_REQUEST)


class FundTransfer(GenericAPIView):
    # permission_classes = ((IsAuthenticated,))
    parser_classes = (MultiPartParser,)
    serializer_class = serializers.FundTransferSerializer

    def post(self,request,*args,**kwargs):
        user = request.user
        result = dict()
        if user.is_authenticated():
            user = request.user.id
            data = request.data.copy()
            data['user'] = user
            s = serializers.FundTransferSerializer(data=data)
            
            if s.is_valid():

                file_obj = s.validated_data['data']
                file_type = file_obj.content_type.split('/')[0]
                print file_type
                if file_type == "video" or file_type == "application":
                    fs = FileSystemStorage()
                    path = os.path.join('verify',file_obj.name)
                    filename = fs.save(path, file_obj)
                    _path =  os.path.join(MEDIA_ROOT,filename)
                    print _path
                    result['status'] = True
                    v_user = VerifyUser()
                    r_user , status_ = v_user.get_results(_path)
                    print r_user , user.__str__()
                    if r_user == user.__str__():  
                        s.validated_data.pop("data")   
                        s.save()
                        result['status'] = True
                        result['result'] = status_
                        return Response(result)
                    else:
                        result['status'] = False
                        result['error'] = "Invalid User"
                        result['result'] = status_
                        return Response(result)
            else:
                result['status'] = False
                result['erros'] = s.errors
                return Response(result,status=status.HTTP_400_BAD_REQUEST)
        else:
            result['status'] = False
            result['erros'] = "Login Required"
            return Response(result,status=status.HTTP_401_UNAUTHORIZED)


class Verify(GenericAPIView):
    parser_classes = (MultiPartParser,)
    serializer_class = serializers.VerifySerializer
    def post(self, request,*args,**kwargs):
        data = request.data
        s = serializers.VerifySerializer(data=data)
        result = dict()
        if s.is_valid():
            file_obj = s.validated_data['data']
            file_type = file_obj.content_type.split('/')[0]
            print file_type
            if file_type == "video" or file_type == "application":
                fs = FileSystemStorage()
                path = os.path.join('verify',file_obj.name)
                filename = fs.save(path, file_obj)
                _path =  os.path.join(MEDIA_ROOT,filename)
                print _path
                result['status'] = True
                v_user = VerifyUser()

                user , status_ = v_user.get_results(_path)
                user = models.Users.objects.filter(id=user)
                user = get_object_or_404(user)
                
                token = models.Token(user=user).save()
                result['token'] = token.key
                result['result'] = status_
                return Response(result)
            else:
                result['status'] = False
                result['error'] = "Please Select a valid video file"
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
        else:
            result['status'] = False
            result['errors'] = s.errors
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
