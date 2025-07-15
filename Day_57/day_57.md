# Day 57: Django API Authentication & Permissions

## Learning Objective
By the end of this lesson, you will understand how to implement secure Django REST API authentication using tokens and JWTs, manage user permissions, and protect your APIs from abuse through rate limiting - just like a master chef protects their kitchen with proper access controls and portion management.

---

## Introduction

Imagine that you're the head chef of an exclusive restaurant. You wouldn't let just anyone walk into your kitchen, start cooking, or access your secret recipes. Similarly, your Django API is like your kitchen - it needs proper security measures to control who can access what resources and how often they can use them.

Just as a restaurant has different levels of access (customers can only order from the menu, waiters can access the dining area, sous chefs can use basic equipment, but only the head chef can access the wine cellar), your Django API needs authentication and authorization systems to manage user permissions effectively.

---

## Lesson 1: Django Token Authentication

### The Kitchen Pass System

Think of Django's token authentication like a kitchen pass system. When someone wants to enter your kitchen, they must present a valid pass (token) that proves they're authorized to be there.

### What is Django Token Authentication?

Django REST Framework provides built-in token authentication where users exchange their credentials for a unique token that grants them access to protected resources. This token acts like a temporary kitchen pass.

### Code Example: Basic Token Authentication

**1. First, let's set up our Django project structure:**

```python
# settings.py
import os
from datetime import timedelta

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',  # For token authentication
    'kitchen_api',  # Our main app
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}

SECRET_KEY = 'kitchen_master_key_2024'
```

**2. Create our kitchen staff models:**

```python
# kitchen_api/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone
from datetime import timedelta

class KitchenStaff(AbstractUser):
    """Custom user model for kitchen staff"""
    ROLE_CHOICES = [
        ('head_chef', 'Head Chef'),
        ('sous_chef', 'Sous Chef'),
        ('line_cook', 'Line Cook'),
        ('prep_cook', 'Prep Cook'),
        ('waiter', 'Waiter'),
        ('customer', 'Customer'),
    ]
    
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='customer')
    hired_date = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"
    
    @property
    def is_chef(self):
        return self.role in ['head_chef', 'sous_chef']
    
    @property
    def can_access_recipes(self):
        return self.role in ['head_chef', 'sous_chef', 'line_cook', 'prep_cook']

class KitchenPass(models.Model):
    """Extended token model for kitchen passes"""
    user = models.OneToOneField(KitchenStaff, on_delete=models.CASCADE)
    key = models.CharField(max_length=40, unique=True)
    created = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    
    def save(self, *args, **kwargs):
        if not self.key:
            self.key = self.generate_key()
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(hours=24)
        super().save(*args, **kwargs)
    
    def generate_key(self):
        import secrets
        return secrets.token_urlsafe(32)
    
    def is_valid(self):
        """Check if kitchen pass is still valid"""
        return timezone.now() < self.expires_at
    
    def __str__(self):
        return f"Kitchen Pass for {self.user.username}"
```

**3. Create our kitchen pass manager:**

```python
# kitchen_api/authentication.py
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from django.contrib.auth import get_user_model
from .models import KitchenPass

User = get_user_model()

class KitchenPassAuthentication(BaseAuthentication):
    """Custom authentication for kitchen passes"""
    
    def authenticate(self, request):
        """Authenticate using kitchen pass from header"""
        kitchen_pass = request.META.get('HTTP_KITCHEN_PASS')
        
        if not kitchen_pass:
            return None
        
        try:
            pass_obj = KitchenPass.objects.get(key=kitchen_pass)
            
            if not pass_obj.is_valid():
                # Remove expired pass
                pass_obj.delete()
                raise AuthenticationFailed('Kitchen pass has expired')
            
            return (pass_obj.user, pass_obj)
            
        except KitchenPass.DoesNotExist:
            raise AuthenticationFailed('Invalid kitchen pass')
    
    def authenticate_header(self, request):
        return 'Kitchen-Pass'
```

**4. Create our API views:**

```python
# kitchen_api/views.py
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from django.contrib.auth import authenticate
from django.utils import timezone
from datetime import timedelta
from .models import KitchenStaff, KitchenPass
from .authentication import KitchenPassAuthentication

@api_view(['POST'])
@permission_classes([AllowAny])
def enter_kitchen(request):
    """Staff entrance - exchange credentials for kitchen pass"""
    username = request.data.get('username')
    password = request.data.get('password')
    
    if not username or not password:
        return Response({
            'error': 'Both username and password are required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Authenticate staff member
    user = authenticate(username=username, password=password)
    
    if user and isinstance(user, KitchenStaff):
        # Create or update kitchen pass
        kitchen_pass, created = KitchenPass.objects.get_or_create(
            user=user,
            defaults={
                'expires_at': timezone.now() + timedelta(hours=24)
            }
        )
        
        if not created:
            # Update expiration for existing pass
            kitchen_pass.expires_at = timezone.now() + timedelta(hours=24)
            kitchen_pass.save()
        
        return Response({
            'message': f'Welcome to the kitchen, {user.username}!',
            'kitchen_pass': kitchen_pass.key,
            'role': user.get_role_display(),
            'expires_at': kitchen_pass.expires_at.isoformat()
        })
    
    return Response({
        'error': 'Invalid credentials - access denied'
    }, status=status.HTTP_401_UNAUTHORIZED)

@api_view(['GET'])
def kitchen_status(request):
    """Check kitchen status - requires valid pass"""
    # This view uses our custom authentication
    user = request.user
    
    return Response({
        'message': 'Kitchen is operational',
        'your_role': user.get_role_display(),
        'username': user.username,
        'permissions': get_user_permissions(user)
    })

@api_view(['POST'])
def leave_kitchen(request):
    """Staff exit - revoke kitchen pass (logout)"""
    try:
        kitchen_pass = KitchenPass.objects.get(user=request.user)
        kitchen_pass.delete()
        return Response({
            'message': 'Kitchen pass revoked. Thank you for your service!'
        })
    except KitchenPass.DoesNotExist:
        return Response({
            'error': 'No active kitchen pass found'
        }, status=status.HTTP_400_BAD_REQUEST)

def get_user_permissions(user):
    """Get permissions for a user based on their role"""
    permissions = {
        'head_chef': ['read_recipes', 'write_recipes', 'manage_staff', 'access_wine_cellar'],
        'sous_chef': ['read_recipes', 'write_recipes', 'manage_inventory'],
        'line_cook': ['read_recipes', 'manage_inventory'],
        'prep_cook': ['read_recipes'],
        'waiter': ['read_menu', 'take_orders'],
        'customer': ['read_menu']
    }
    return permissions.get(user.role, [])
```

**5. Update settings for custom authentication:**

```python
# settings.py (update REST_FRAMEWORK)
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'kitchen_api.authentication.KitchenPassAuthentication',
        'rest_framework.authentication.TokenAuthentication',  # Fallback
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}

# Don't forget to set AUTH_USER_MODEL
AUTH_USER_MODEL = 'kitchen_api.KitchenStaff'
```

### Syntax Explanation:
- `AbstractUser`: Django's built-in user model that we extend with custom fields
- `OneToOneField`: Creates a one-to-one relationship between User and KitchenPass
- `@api_view`: Django REST Framework decorator for function-based views
- `authenticate()`: Django's built-in authentication function
- `timezone.now()`: Django's timezone-aware datetime function

---

## Lesson 2: Django JWT Implementation

### The Smart Kitchen Badge System

JWT (JSON Web Token) with Django is like upgrading from simple kitchen passes to smart badges that contain encoded information about the staff member's role, permissions, and badge expiration - all in one compact format.

### What is Django JWT?

Django JWT creates self-contained tokens that include user information and permissions encoded within them. Like a smart badge that contains all necessary information without needing to check a central database.

### Code Example: Django JWT Implementation

**1. Install and configure django-rest-framework-simplejwt:**

```bash
pip install djangorestframework-simplejwt
```

```python
# settings.py
from datetime import timedelta

INSTALLED_APPS = [
    # ... other apps
    'rest_framework_simplejwt',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=24),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
}
```

**2. Create custom JWT serializers:**

```python
# kitchen_api/serializers.py
from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.contrib.auth import get_user_model

User = get_user_model()

class SmartBadgeSerializer(TokenObtainPairSerializer):
    """Custom JWT serializer that includes role and permissions in token"""
    
    @classmethod
    def get_token(cls, user):
        """Create a smart badge (JWT) with embedded user information"""
        token = super().get_token(user)
        
        # Add custom claims to the token (like badge information)
        token['role'] = user.role
        token['role_display'] = user.get_role_display()
        token['permissions'] = get_user_permissions(user)
        token['is_chef'] = user.is_chef
        token['hired_date'] = user.hired_date.isoformat()
        
        return token

class KitchenStaffSerializer(serializers.ModelSerializer):
    """Serializer for kitchen staff information"""
    role_display = serializers.CharField(source='get_role_display', read_only=True)
    permissions = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'role', 'role_display', 'permissions', 'hired_date']
    
    def get_permissions(self, obj):
        return get_user_permissions(obj)

def get_user_permissions(user):
    """Get permissions for a user based on their role"""
    role_permissions = {
        'head_chef': ['read_recipes', 'write_recipes', 'manage_staff', 'access_wine_cellar'],
        'sous_chef': ['read_recipes', 'write_recipes', 'manage_inventory'],
        'line_cook': ['read_recipes', 'manage_inventory'],
        'prep_cook': ['read_recipes'],
        'waiter': ['read_menu', 'take_orders'],
        'customer': ['read_menu']
    }
    return role_permissions.get(user.role, [])
```

**3. Create JWT-based views:**

```python
# kitchen_api/jwt_views.py
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model
from .serializers import SmartBadgeSerializer, KitchenStaffSerializer

User = get_user_model()

class GetSmartBadgeView(TokenObtainPairView):
    """Custom JWT view for getting smart badges"""
    serializer_class = SmartBadgeSerializer
    
    def post(self, request, *args, **kwargs):
        """Issue a smart badge with custom response format"""
        response = super().post(request, *args, **kwargs)
        
        if response.status_code == 200:
            # Get user for additional information
            username = request.data.get('username')
            try:
                user = User.objects.get(username=username)
                response.data.update({
                    'message': f'Smart badge issued to {user.username}',
                    'role': user.get_role_display(),
                    'permissions': get_user_permissions(user)
                })
            except User.DoesNotExist:
                pass
        
        return response

@api_view(['GET'])
def read_smart_badge(request):
    """Read information from current user's smart badge"""
    user = request.user
    
    return Response({
        'message': f'Smart badge verified for {user.username}',
        'badge_info': {
            'username': user.username,
            'role': user.role,
            'role_display': user.get_role_display(),
            'permissions': get_user_permissions(user),
            'is_chef': user.is_chef,
            'hired_date': user.hired_date.isoformat()
        }
    })

@api_view(['POST'])
def revoke_smart_badge(request):
    """Revoke smart badge (logout)"""
    try:
        refresh_token = request.data.get('refresh_token')
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({
                'message': 'Smart badge revoked successfully'
            })
        else:
            return Response({
                'error': 'Refresh token is required'
            }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({
            'error': 'Failed to revoke smart badge'
        }, status=status.HTTP_400_BAD_REQUEST)

def get_user_permissions(user):
    """Get permissions for a user based on their role"""
    role_permissions = {
        'head_chef': ['read_recipes', 'write_recipes', 'manage_staff', 'access_wine_cellar'],
        'sous_chef': ['read_recipes', 'write_recipes', 'manage_inventory'],
        'line_cook': ['read_recipes', 'manage_inventory'],
        'prep_cook': ['read_recipes'],
        'waiter': ['read_menu', 'take_orders'],
        'customer': ['read_menu']
    }
    return role_permissions.get(user.role, [])
```

### Syntax Explanation:
- `TokenObtainPairSerializer`: Base class for JWT token generation
- `get_token()`: Class method that creates and customizes JWT tokens
- `SerializerMethodField()`: Creates computed fields in serializers
- `RefreshToken.blacklist()`: Invalidates a refresh token for logout
- `super().post()`: Calls parent class method while extending functionality

---

## Lesson 3: Django API Permissions

### The Kitchen Hierarchy System

Just like in a professional kitchen where different staff have different responsibilities and access levels, Django's permission system controls what actions users can perform based on their roles.

### Code Example: Django Role-Based Access Control

**1. Create custom permission classes:**

```python
# kitchen_api/permissions.py
from rest_framework import permissions
from django.contrib.auth import get_user_model

User = get_user_model()

class KitchenPermission(permissions.BasePermission):
    """Base permission class for kitchen operations"""
    
    def has_permission(self, request, view):
        """Check if user has permission to access the view"""
        if not request.user.is_authenticated:
            return False
        
        # Get required permission from view
        required_permission = getattr(view, 'required_permission', None)
        
        if not required_permission:
            return True  # No specific permission required
        
        return self.user_has_permission(request.user, required_permission)
    
    def user_has_permission(self, user, permission):
        """Check if user has a specific permission based on role"""
        role_permissions = {
            'head_chef': [
                'read_recipes', 'write_recipes', 'manage_inventory',
                'access_wine_cellar', 'manage_staff', 'modify_menu'
            ],
            'sous_chef': [
                'read_recipes', 'write_recipes', 'manage_inventory', 'modify_menu'
            ],
            'line_cook': [
                'read_recipes', 'manage_inventory'
            ],
            'prep_cook': [
                'read_recipes'
            ],
            'waiter': [
                'read_menu', 'take_orders'
            ],
            'customer': [
                'read_menu'
            ]
        }
        
        user_permissions = role_permissions.get(user.role, [])
        return permission in user_permissions

class CanReadRecipes(permissions.BasePermission):
    """Permission for reading recipes"""
    
    def has_permission(self, request, view):
        return (request.user.is_authenticated and 
                request.user.can_access_recipes)

class IsChefOrReadOnly(permissions.BasePermission):
    """Permission that allows chefs to modify, others to read only"""
    
    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False
        
        if request.method in permissions.SAFE_METHODS:
            return True
        
        return request.user.is_chef

class IsHeadChef(permissions.BasePermission):
    """Permission for head chef only operations"""
    
    def has_permission(self, request, view):
        return (request.user.is_authenticated and 
                request.user.role == 'head_chef')
```

**2. Create permission-based views:**

```python
# kitchen_api/protected_views.py
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import get_user_model
from .permissions import KitchenPermission, CanReadRecipes, IsChefOrReadOnly, IsHeadChef

User = get_user_model()

class RecipeView(APIView):
    """Recipe management with role-based permissions"""
    permission_classes = [IsAuthenticated, IsChefOrReadOnly]
    
    def get(self, request):
        """Get recipes - requires recipe reading permission"""
        if not request.user.can_access_recipes:
            return Response({
                'error': f'Access denied - {request.user.get_role_display()} cannot access recipes'
            }, status=status.HTTP_403_FORBIDDEN)
        
        recipes = [
            {
                'id': 1,
                'name': 'Marinara Sauce',
                'difficulty': 'Easy',
                'chef': 'Mario',
                'ingredients': ['tomatoes', 'garlic', 'basil']
            },
            {
                'id': 2,
                'name': 'Beef Wellington',
                'difficulty': 'Hard',
                'chef': 'Gordon',
                'ingredients': ['beef', 'puff pastry', 'mushrooms']
            },
            {
                'id': 3,
                'name': 'Chocolate Soufflé',
                'difficulty': 'Expert',
                'chef': 'Julia',
                'ingredients': ['chocolate', 'eggs', 'sugar']
            }
        ]
        
        return Response({
            'message': f'Recipes accessed by {request.user.get_role_display()}',
            'recipes': recipes
        })
    
    def post(self, request):
        """Create new recipe - requires chef status"""
        recipe_data = request.data
        
        return Response({
            'message': f'Recipe created by {request.user.get_role_display()}!',
            'recipe': {
                'name': recipe_data.get('name'),
                'difficulty': recipe_data.get('difficulty'),
                'chef': request.user.username,
                'ingredients': recipe_data.get('ingredients', [])
            }
        }, status=status.HTTP_201_CREATED)

@api_view(['GET'])
@permission_classes([IsAuthenticated, IsHeadChef])
def wine_cellar(request):
    """Access wine cellar - head chef only"""
    return Response({
        'message': f'Welcome to the wine cellar, {request.user.get_role_display()}!',
        'wines': [
            {'name': 'Vintage Chianti', 'year': 2018, 'price': 150},
            {'name': 'Aged Barolo', 'year': 2015, 'price': 200},
            {'name': 'Special Reserve Pinot', 'year': 2019, 'price': 180}
        ]
    })

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated, IsHeadChef])
def manage_staff(request):
    """Manage staff - head chef only"""
    if request.method == 'GET':
        staff = User.objects.exclude(role='customer').values(
            'username', 'role', 'hired_date'
        )
        return Response({
            'message': 'Staff management accessed',
            'staff': list(staff)
        })
    
    elif request.method == 'POST':
        # Add new staff member logic here
        return Response({
            'message': 'Staff member added successfully'
        }, status=status.HTTP_201_CREATED)

class MenuView(APIView):
    """Menu management with different permissions for different roles"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Read menu - everyone can read"""
        menu = [
            {'item': 'Pasta Carbonara', 'price': 18, 'category': 'Main'},
            {'item': 'Caesar Salad', 'price': 12, 'category': 'Appetizer'},
            {'item': 'Tiramisu', 'price': 8, 'category': 'Dessert'}
        ]
        
        return Response({
            'message': f'Menu viewed by {request.user.get_role_display()}',
            'menu': menu
        })
    
    def post(self, request):
        """Modify menu - chefs only"""
        if not request.user.is_chef:
            return Response({
                'error': 'Only chefs can modify the menu'
            }, status=status.HTTP_403_FORBIDDEN)
        
        return Response({
            'message': f'Menu updated by {request.user.get_role_display()}',
            'new_item': request.data
        }, status=status.HTTP_201_CREATED)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_permissions(request):
    """Get current user's permissions"""
    role_permissions = {
        'head_chef': ['read_recipes', 'write_recipes', 'manage_staff', 'access_wine_cellar'],
        'sous_chef': ['read_recipes', 'write_recipes', 'manage_inventory'],
        'line_cook': ['read_recipes', 'manage_inventory'],
        'prep_cook': ['read_recipes'],
        'waiter': ['read_menu', 'take_orders'],
        'customer': ['read_menu']
    }
    
    user_permissions = role_permissions.get(request.user.role, [])
    
    return Response({
        'username': request.user.username,
        'role': request.user.get_role_display(),
        'permissions': user_permissions,
        'is_chef': request.user.is_chef
    })
```

**3. Create URL patterns:**

```python
# kitchen_api/urls.py
from django.urls import path
from . import views, jwt_views, protected_views

urlpatterns = [
    # Token authentication endpoints
    path('enter-kitchen/', views.enter_kitchen, name='enter_kitchen'),
    path('kitchen-status/', views.kitchen_status, name='kitchen_status'),
    path('leave-kitchen/', views.leave_kitchen, name='leave_kitchen'),
    
    # JWT authentication endpoints
    path('get-badge/', jwt_views.GetSmartBadgeView.as_view(), name='get_badge'),
    path('read-badge/', jwt_views.read_smart_badge, name='read_badge'),
    path('revoke-badge/', jwt_views.revoke_smart_badge, name='revoke_badge'),
    
    # Protected resource endpoints
    path('recipes/', protected_views.RecipeView.as_view(), name='recipes'),
    path('wine-cellar/', protected_views.wine_cellar, name='wine_cellar'),
    path('staff/', protected_views.manage_staff, name='manage_staff'),
    path('menu/', protected_views.MenuView.as_view(), name='menu'),
    path('my-permissions/', protected_views.my_permissions, name='my_permissions'),
]
```

**4. Main URL configuration:**

```python
# urls.py (main project)
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('kitchen_api.urls')),
]
```

### Syntax Explanation:
- `BasePermission`: Django REST Framework's base class for custom permissions
- `SAFE_METHODS`: HTTP methods that don't modify data (GET, HEAD, OPTIONS)
- `APIView`: Class-based view for handling different HTTP methods
- `@permission_classes`: Decorator to specify required permissions for views
- `has_permission()`: Method that determines if user has access to a view

---

## Testing Your Kitchen API

### Example API Usage:

```python
# Test script to demonstrate API usage
import requests
import json

BASE_URL = 'http://localhost:8000/api'

# 1. Get a kitchen pass (token authentication)
login_data = {
    'username': 'chef_mario',
    'password': 'pasta123'
}

response = requests.post(f'{BASE_URL}/enter-kitchen/', json=login_data)
kitchen_pass = response.json()['kitchen_pass']

# 2. Access kitchen status with pass
headers = {'Kitchen-Pass': kitchen_pass}
response = requests.get(f'{BASE_URL}/kitchen-status/', headers=headers)
print("Kitchen Status:", response.json())

# 3. Get a smart badge (JWT)
response = requests.post(f'{BASE_URL}/get-badge/', json=login_data)
tokens = response.json()
access_token = tokens['access']

# 4. Access recipes with JWT
jwt_headers = {'Authorization': f'Bearer {access_token}'}
response = requests.get(f'{BASE_URL}/recipes/', headers=jwt_headers)
print("Recipes:", response.json())

# 5. Try to access wine cellar (head chef only)
response = requests.get(f'{BASE_URL}/wine-cellar/', headers=jwt_headers)
print("Wine Cellar:", response.json())
```

---

## Summary

You've now learned how to implement secure Django API authentication and permissions using:

1. **Token Authentication**: Like kitchen passes that grant temporary access
2. **JWT Implementation**: Smart badges with embedded user information
3. **Role-Based Permissions**: Kitchen hierarchy system controlling access levels

Your Django API is now as secure as a professional kitchen, with proper authentication, authorization, and permission controls ensuring only authorized staff can access appropriate resources based on their roles and responsibilities.

Remember, just as a head chef carefully manages who enters their kitchen and what they can do there, your API security should be thoughtfully designed to protect your valuable resources while providing the right level of access to legitimate users.

----
Django Secure Restaurant API - Complete Project
Project Structure
secure_restaurant/
├── manage.py
├── secure_restaurant/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── restaurant/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── management/
│   │   ├── __init__.py
│   │   └── commands/
│   │       ├── __init__.py
│   │       └── create_kitchen_staff.py
│   ├── migrations/
│   │   ├── __init__.py
│   │   └── (migration files)
│   ├── models.py
│   ├── permissions.py
│   ├── serializers.py
│   ├── throttling.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
└── requirements.txt

Requirements
# requirements.txt
Django==4.2.7
djangorestframework==3.14.0
djangorestframework-simplejwt==5.3.0
django-cors-headers==4.3.1
redis==5.0.1
django-redis==5.4.0
django-ratelimit==4.1.0
python-decouple==3.8

Settings
# secure_restaurant/settings.py
import os
from pathlib import Path
from datetime import timedelta
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config('SECRET_KEY', default='your-secret-key-here-change-in-production')

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework_simplejwt',
    'corsheaders',
    'restaurant.apps.RestaurantConfig',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'secure_restaurant.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '5/hour',
        'user': '20/hour',
        'head_chef': '500/hour',
        'sous_chef': '100/hour',
        'master_chef': '1000/hour',
    }
}

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=24),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'VERIFYING_KEY': None,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
}

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Additional Security Settings
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'static'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs/kitchen.log',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'restaurant': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

App Configuration
# restaurant/apps.py
from django.apps import AppConfig

class RestaurantConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'restaurant'

Models
# restaurant/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone

class User(AbstractUser):
    ROLE_CHOICES = [
        ('master', 'Master Chef'),
        ('head', 'Head Chef'),
        ('sous', 'Sous Chef'),
        ('waiter', 'Waiter'),
        ('customer', 'Customer'),
    ]
    
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='customer')
    kitchen_access_level = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.username} - {self.get_role_display()}"
    
    @property
    def permissions(self):
        permission_map = {
            'master': ['read', 'write', 'delete', 'secret_access', 'manage_staff'],
            'head': ['read', 'write', 'manage_orders'],
            'sous': ['read', 'basic_write'],
            'waiter': ['read', 'take_orders'],
            'customer': ['read', 'place_orders'],
        }
        return permission_map.get(self.role, ['read'])

class Recipe(models.Model):
    SECRET_LEVELS = [
        ('basic', 'Basic'),
        ('advanced', 'Advanced'),
        ('top_secret', 'Top Secret'),
    ]
    
    name = models.CharField(max_length=200)
    ingredients = models.JSONField(default=list)
    instructions = models.TextField(blank=True)
    secret_level = models.CharField(max_length=20, choices=SECRET_LEVELS, default='basic')
    chef = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recipes')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.get_secret_level_display()})"

class MenuItem(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.CharField(max_length=100)
    is_available = models.BooleanField(default=True)
    recipe = models.ForeignKey(Recipe, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['category', 'name']
    
    def __str__(self):
        return f"{self.name} - ${self.price}"

class Order(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('preparing', 'Preparing'),
        ('ready', 'Ready'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]
    
    customer = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders')
    items = models.JSONField(default=list)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    special_instructions = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Order #{self.id} - {self.customer.username}"

class KitchenLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    action = models.CharField(max_length=200)
    details = models.JSONField(default=dict)
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField()
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.user.username} - {self.action}"

Serializers
# restaurant/serializers.py
from rest_framework import serializers
from django.contrib.auth import authenticate
from .models import User, Recipe, MenuItem, Order, KitchenLog

class UserSerializer(serializers.ModelSerializer):
    permissions = serializers.ReadOnlyField()
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'role', 'permissions', 'kitchen_access_level', 'created_at']
        read_only_fields = ['created_at']

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)
    
    def validate(self, data):
        username = data.get('username')
        password = data.get('password')
        
        if username and password:
            user = authenticate(username=username, password=password)
            if user:
                if user.is_active:
                    data['user'] = user
                else:
                    raise serializers.ValidationError("Chef account is deactivated!")
            else:
                raise serializers.ValidationError("Invalid credentials! Kitchen access denied.")
        else:
            raise serializers.ValidationError("Must provide username and password!")
        
        return data

class RecipeSerializer(serializers.ModelSerializer):
    chef_name = serializers.CharField(source='chef.username', read_only=True)
    
    class Meta:
        model = Recipe
        fields = ['id', 'name', 'ingredients', 'instructions', 'secret_level', 
                 'chef', 'chef_name', 'created_at', 'updated_at', 'is_active']
        read_only_fields = ['chef', 'created_at', 'updated_at']

class MenuItemSerializer(serializers.ModelSerializer):
    recipe_name = serializers.CharField(source='recipe.name', read_only=True)
    
    class Meta:
        model = MenuItem
        fields = ['id', 'name', 'description', 'price', 'category', 
                 'is_available', 'recipe', 'recipe_name', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']

class OrderSerializer(serializers.ModelSerializer):
    customer_name = serializers.CharField(source='customer.username', read_only=True)
    
    class Meta:
        model = Order
        fields = ['id', 'customer', 'customer_name', 'items', 'total_amount', 
                 'status', 'special_instructions', 'created_at', 'updated_at']
        read_only_fields = ['customer', 'created_at', 'updated_at']

class KitchenLogSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = KitchenLog
        fields = ['id', 'user', 'user_name', 'action', 'details', 'timestamp', 'ip_address']
        read_only_fields = ['user', 'timestamp', 'ip_address']

Permissions
# restaurant/permissions.py
from rest_framework import permissions

class IsChefOrReadOnly(permissions.BasePermission):
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user.is_authenticated and request.user.role in ['master', 'head', 'sous']

class IsMasterChef(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == 'master'

class IsHeadChefOrAbove(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role in ['master', 'head']

class CanAccessSecretRecipes(permissions.BasePermission):
    def has_permission(self, request, view):
        return (request.user.is_authenticated and 
                request.user.role == 'master' and 
                'secret_access' in request.user.permissions)

class CanModifyOwnRecipes(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.chef == request.user or request.user.role == 'master'

Throttling
# restaurant/throttling.py
from rest_framework.throttling import UserRateThrottle
from django.core.cache import cache

class KitchenRoleThrottle(UserRateThrottle):
    def get_cache_key(self, request, view):
        if request.user.is_authenticated:
            ident = request.user.pk
        else:
            ident = self.get_ident(request)
        return self.cache_format % {
            'scope': self.scope,
            'ident': ident
        }
    
    def get_rate(self):
        if hasattr(self, 'request') and self.request.user.is_authenticated:
            role = self.request.user.role
            role_rates = {
                'master': '1000/hour',
                'head': '500/hour',
                'sous': '100/hour',
                'waiter': '50/hour',
                'customer': '20/hour',
            }
            return role_rates.get(role, '5/hour')
        return '5/hour'

class LoginThrottle(UserRateThrottle):
    scope = 'login'
    rate = '5/hour'

Views
# restaurant/views.py
from rest_framework import generics, status, viewsets
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from django.utils import timezone
from django.db.models import Q
from .models import User, Recipe, MenuItem, Order, KitchenLog
from .serializers import (
    UserSerializer, LoginSerializer, RecipeSerializer, 
    MenuItemSerializer, OrderSerializer, KitchenLogSerializer
)
from .permissions import (
    IsChefOrReadOnly, IsMasterChef, IsHeadChefOrAbove, 
    CanAccessSecretRecipes, CanModifyOwnRecipes
)
from .throttling import KitchenRoleThrottle, LoginThrottle

def log_kitchen_activity(user, action, details, ip_address):
    KitchenLog.objects.create(
        user=user,
        action=action,
        details=details,
        ip_address=ip_address
    )

@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([LoginThrottle])
def kitchen_login(request):
    serializer = LoginSerializer(data=request.data)
    
    if serializer.is_valid():
        user = serializer.validated_data['user']
        refresh = RefreshToken.for_user(user)
        
        log_kitchen_activity(
            user=user,
            action='LOGIN',
            details={
                'role': user.role,
                'timestamp': timezone.now().isoformat(),
                'success': True
            },
            ip_address=request.META.get('REMOTE_ADDR', '127.0.0.1')
        )
        
        return Response({
            'message': f'Welcome to the kitchen, Chef {user.username}!',
            'kitchen_status': 'access_granted',
            'access_token': str(refresh.access_token),
            'refresh_token': str(refresh),
            'chef_role': user.role,
            'permissions': user.permissions,
            'shift_expires': '24 hours'
        }, status=status.HTTP_200_OK)
    
    return Response({
        'message': 'Invalid credentials! Kitchen access denied.',
        'kitchen_status': 'authentication_failed',
        'errors': serializer.errors
    }, status=status.HTTP_401_UNAUTHORIZED)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def kitchen_logout(request):
    try:
        refresh_token = request.data.get('refresh_token')
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        
        log_kitchen_activity(
            user=request.user,
            action='LOGOUT',
            details={
                'timestamp': timezone.now().isoformat(),
                'success': True
            },
            ip_address=request.META.get('REMOTE_ADDR', '127.0.0.1')
        )
        
        return Response({
            'message': 'Successfully logged out from kitchen!',
            'kitchen_status': 'logged_out'
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response({
            'message': 'Error during logout',
            'kitchen_status': 'logout_error',
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)

class RecipeViewSet(viewsets.ModelViewSet):
    serializer_class = RecipeSerializer
    permission_classes = [IsAuthenticated, IsChefOrReadOnly]
    throttle_classes = [KitchenRoleThrottle]
    
    def get_queryset(self):
        user = self.request.user
        
        if user.role == 'master':
            return Recipe.objects.filter(is_active=True)
        elif user.role == 'head':
            return Recipe.objects.filter(
                secret_level__in=['basic', 'advanced'],
                is_active=True
            )
        elif user.role == 'sous':
            return Recipe.objects.filter(
                secret_level='basic',
                is_active=True
            )
        else:
            return Recipe.objects.filter(
                secret_level='basic',
                is_active=True
            )
    
    def perform_create(self, serializer):
        recipe = serializer.save(chef=self.request.user)
        log_kitchen_activity(
            user=self.request.user,
            action='RECIPE_CREATED',
            details={
                'recipe_id': recipe.id,
                'recipe_name': recipe.name,
                'secret_level': recipe.secret_level
            },
            ip_address=self.request.META.get('REMOTE_ADDR', '127.0.0.1')
        )
    
    def perform_update(self, serializer):
        recipe = serializer.save()
        log_kitchen_activity(
            user=self.request.user,
            action='RECIPE_UPDATED',
            details={
                'recipe_id': recipe.id,
                'recipe_name': recipe.name
            },
            ip_address=self.request.META.get('REMOTE_ADDR', '127.0.0.1')
        )
    
    def perform_destroy(self, instance):
        log_kitchen_activity(
            user=self.request.user,
            action='RECIPE_DELETED',
            details={
                'recipe_id': instance.id,
                'recipe_name': instance.name
            },
            ip_address=self.request.META.get('REMOTE_ADDR', '127.0.0.1')
        )
        instance.is_active = False
        instance.save()

@api_view(['GET'])
@permission_classes([IsAuthenticated, CanAccessSecretRecipes])
@throttle_classes([KitchenRoleThrottle])
def secret_vault(request):
    secret_recipes = Recipe.objects.filter(
        secret_level='top_secret',
        is_active=True
    )
    serializer = RecipeSerializer(secret_recipes, many=True)
    
    log_kitchen_activity(
        user=request.user,
        action='SECRET_VAULT_ACCESS',
        details={
            'timestamp': timezone.now().isoformat(),
            'recipes_count': secret_recipes.count()
        },
        ip_address=request.META.get('REMOTE_ADDR', '127.0.0.1')
    )
    
    return Response({
        'message': 'Welcome to the secret vault, Master Chef!',
        'kitchen_status': 'secret_access_granted',
        'secret_recipes': serializer.data,
        'vault_warning': 'These recipes are classified - handle with care!'
    }, status=status.HTTP_200_OK)

class MenuItemViewSet(viewsets.ModelViewSet):
    queryset = MenuItem.objects.filter(is_available=True)
    serializer_class = MenuItemSerializer
    throttle_classes = [KitchenRoleThrottle]
    
    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            permission_classes = [AllowAny]
        else:
            permission_classes = [IsAuthenticated, IsHeadChefOrAbove]
        return [permission() for permission in permission_classes]

class OrderViewSet(viewsets.ModelViewSet):
    serializer_class = OrderSerializer
    permission_classes = [IsAuthenticated]
    throttle_classes = [KitchenRoleThrottle]
    
    def get_queryset(self):
        user = self.request.user
        if user.role in ['master', 'head']:
            return Order.objects.all()
        elif user.role in ['sous', 'waiter']:
            return Order.objects.filter(status__in=['pending', 'preparing'])
        else:
            return Order.objects.filter(customer=user)
    
    def perform_create(self, serializer):
        order = serializer.save(customer=self.request.user)
        log_kitchen_activity(
            user=self.request.user,
            action='ORDER_PLACED',
            details={
                'order_id': order.id,
                'total_amount': float(order.total_amount),
                'items_count': len(order.items)
            },
            ip_address=self.request.META.get('REMOTE_ADDR', '127.0.0.1')
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([KitchenRoleThrottle])
def kitchen_status(request):
    cache_key = f"throttle_user_{request.user.pk}"
    request_count = len(cache.get(cache_key, []))
    
    return Response({
        'message': f'Kitchen status check for Chef {request.user.username}',
        'kitchen_status': 'operational',
        'your_role': request.user.role,
        'your_permissions': request.user.permissions,
        'kitchen_stats': {
            'total_recipes': Recipe.objects.filter(is_active=True).count(),
            'total_menu_items': MenuItem.objects.filter(is_available=True).count(),
            'pending_orders': Order.objects.filter(status='pending').count(),
        },
        'kitchen_time': timezone.now().isoformat(),
        'rate_limit_info': {
            'your_recent_requests': request_count,
            'role_based_limit': '1000/hour' if request.user.role == 'master' else '500/hour'
        }
    }, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated, IsMasterChef])
@throttle_classes([KitchenRoleThrottle])
def kitchen_logs(request):
    logs = KitchenLog.objects.all()[:100]
    serializer = KitchenLogSerializer(logs, many=True)
    
    return Response({
        'message': 'Kitchen activity logs',
        'kitchen_status': 'logs_retrieved',
        'logs': serializer.data,
        'total_logs': logs.count()
    }, status=status.HTTP_200_OK)

URLs
# restaurant/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

router = DefaultRouter()
router.register(r'recipes', views.RecipeViewSet, basename='recipe')
router.register(r'menu', views.MenuItemViewSet, basename='menuitem')
router.register(r'orders', views.OrderViewSet, basename='order')

urlpatterns = [
    path('auth/login/', views.kitchen_login, name='kitchen_login'),
    path('auth/logout/', views.kitchen_logout, name='kitchen_logout'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('kitchen/status/', views.kitchen_status, name='kitchen_status'),
    path('kitchen/secret-vault/', views.secret_vault, name='secret_vault'),
    path('kitchen/logs/', views.kitchen_logs, name='kitchen_logs'),
    path('api/', include(router.urls)),
]

# secure_restaurant/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('restaurant.urls')),
]

Admin Configuration
# restaurant/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, Recipe, MenuItem, Order, KitchenLog

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ['username', 'email', 'role', 'kitchen_access_level', 'is_active']
    list_filter = ['role', 'is_active', 'created_at']
    search_fields = ['username', 'email']
    fieldsets = UserAdmin.fieldsets + (
        ('Kitchen Info', {'fields': ('role', 'kitchen_access_level')}),
    )

@admin.register(Recipe)
class RecipeAdmin(admin.ModelAdmin):
    list_display = ['name', 'chef', 'secret_level', 'is_active', 'created_at']
    list_filter = ['secret_level', 'is_active', 'created_at']
    search_fields = ['name', 'chef__username']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(MenuItem)
class MenuItemAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'price', 'is_available', 'created_at']
    list_filter = ['category', 'is_available', 'created_at']
    search_fields = ['name', 'category']

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ['id', 'customer', 'total_amount', 'status', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['customer__username', 'id']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(KitchenLog)
class KitchenLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'action', 'timestamp', 'ip_address']
    list_filter = ['action', 'timestamp']
    search_fields = ['user__username', 'action']
    readonly_fields = ['timestamp']

Management Commands
# restaurant/management/commands/create_kitchen_staff.py
from django.core.management.base import BaseCommand
from django.contrib.auth.hashers import make_password
from restaurant.models import User

class Command(BaseCommand):
    help = 'Create initial kitchen staff users'
    
    def handle(self, *args, **options):
        staff_data = [
            {
                'username': 'masterchef',
                'email': 'master@restaurant.com',
                'password': 'masterpass123',
                'role': 'master',
                'kitchen_access_level': 5
            },
            {
                'username': 'headchef',
                'email': 'head@restaurant.com',
                'password': 'headpass123',
                'role': 'head',
                'kitchen_access_level': 4
            },
            {
                'username': 'souschef',
                'email': 'sous@restaurant.com',
                'password': 'souspass123',
                'role': 'sous',
                'kitchen_access_level': 3
            },
            {
                'username': 'waiter1',
                'email': 'waiter1@restaurant.com',
                'password': 'waiterpass123',
                'role': 'waiter',
                'kitchen_access_level': 2
            },
            {
                'username': 'customer1',
                'email': 'customer1@restaurant.com',
                'password': 'customerpass123',
                'role': 'customer',
                'kitchen_access_level': 1
            },
        ]

        for staff in staff_data:
            if not User.objects.filter(username=staff['username']).exists():
                user = User.objects.create(
                    username=staff['username'],
                    email=staff['email'],
                    password=make_password(staff['password']),
                    role=staff['role'],
                    kitchen_access_level=staff['kitchen_access_level'],
                    is_active=True
                )
                self.stdout.write(self.style.SUCCESS(
                    f"Successfully created {staff['role']} user: {staff['username']}"
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    f"User {staff['username']} already exists"
                ))

Tests
# restaurant/tests.py
from django.test import TestCase
from rest_framework.test import APIClient
from django.urls import reverse
from rest_framework import status
from .models import User

class KitchenAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user_data = {
            'username': 'testchef',
            'email': 'test@restaurant.com',
            'password': 'testpass123',
            'role': 'sous',
            'kitchen_access_level': 3
        }
        self.user = User.objects.create_user(**self.user_data)

    def test_login(self):
        response = self.client.post(
            reverse('kitchen_login'),
            {'username': 'testchef', 'password': 'testpass123'},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access_token', response.data)

Setup Instructions

Install dependencies:pip install -r requirements.txt


Create log directory:mkdir -p secure_restaurant/logs
touch secure_restaurant/logs/kitchen.log


Apply migrations:python manage.py makemigrations
python manage.py migrate


Create initial staff:python manage.py create_kitchen_staff


Create superuser:python manage.py createsuperuser


Run server:python manage.py runserver


Run tests:python manage.py test restaurant



Notes

Ensure Redis is running (redis-server) for rate limiting.
Update SECRET_KEY and CORS_ALLOWED_ORIGINS for production.
Use HTTPS in production for secure JWT transmission.
Monitor KitchenLog for suspicious activities.