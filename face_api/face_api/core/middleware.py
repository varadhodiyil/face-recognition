from rest_framework.authentication import TokenAuthentication
from face_api.core.models import Token
class CustomTokenAuthentication(TokenAuthentication):
    model = Token
    keyword = "Bearer"
