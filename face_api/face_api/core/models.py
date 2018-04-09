# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import datetime
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
from hashlib import sha1
TOKEN_EXPIRE_TIME = datetime.timedelta(days=30)
# Create your models here.

class Users(AbstractUser):
    file_path = models.CharField(max_length=250,default=None,null=True)

class FundTransfer(models.Model):
	id = models.AutoField(primary_key=True)
	user = models.ForeignKey(Users)
	amount = models.FloatField()
	to_user = models.CharField(max_length=100)
	transaction_time = models.DateTimeField(auto_now=True)

	class Meta:
		db_table = 'fund_transfer'

class TokenManager(models.Manager):
	def delete_session(self, *args, **kwargs):
		if "key" in kwargs:
			tokent_str = kwargs['key']
			Token.objects.filter(key=tokent_str).delete()
		return True

	def is_valid(self, *args, **kwargs):

		if "key" in kwargs:
			tokent_str = kwargs['key']
			Token.objects.filter(
			    key=tokent_str, expires_at__lte=timezone.now()).delete()
			tokens = Token.objects.filter(
			    key=tokent_str, expires_at__gte=timezone.now()).get()
		return tokens


class Token(models.Model):
	user = models.ForeignKey(Users)
	key = models.CharField(primary_key=True, max_length=60)
	created_at = models.DateTimeField(auto_now_add=True)
	expires_at = models.DateTimeField(default=None)

	objects = TokenManager()

	class Meta:
		db_table = 'access_tokens'
		db_tablespace = 'default'

	def save(self, *args, **kwargs):
		user_id_str = self.user.id
		exists, self.key = self.generate_key(user_id_str)
		self.expires_at = timezone.now() + TOKEN_EXPIRE_TIME
		if not exists:
			super(Token, self).save(*args, **kwargs)

		return self

	def generate_key(self, id_):
		existing_token = None
		exists = True
		tokens = Token.objects.filter(
		    user_id=id_, expires_at__gte=timezone.now()).values("key")
		for token in tokens:
			existing_token = token['key']
			token = existing_token
		if existing_token is None:
			exists = False
			token = sha1(id_.__str__() + str(timezone.now())).hexdigest()
			Token.objects.filter(user=id_, expires_at__lte=timezone.now()).delete()
		return exists, token