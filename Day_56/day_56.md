# Day 56: RESTful APIs with Django REST Framework

## Learning Objective
By the end of this lesson, you will be able to create a fully functional RESTful API using Django REST Framework, including proper serialization, authentication, and data filtering - transforming your Django applications from simple web pages into powerful backend services that can serve multiple clients.

---

## Introduction: The Kitchen Revolution

Imagine that you're running a successful restaurant, and until now, customers could only dine in your establishment. But suddenly, the world changes - people want delivery, takeout, drive-through, and even meal kits they can prepare at home. Your kitchen (Django backend) is amazing, but you need a way to serve your delicious food (data) in different formats to different types of customers (mobile apps, web apps, other services).

This is exactly what happened in the web development world. Django REST Framework (DRF) is like hiring a master chef coordinator who can take your existing kitchen's recipes (Django models) and prepare them for any type of service - whether it's a fancy sit-down meal (web interface), a quick takeout box (mobile app), or ingredients for a meal kit (raw JSON data for other developers).

Just as a chef coordinator standardizes how orders are taken, prepared, and served regardless of the service type, DRF standardizes how your data is accessed, manipulated, and delivered through APIs.

---

## Lesson 1: DRF Installation and Setup

### The Kitchen Infrastructure

Think of DRF installation like setting up a modern kitchen dispatch system. Before you can serve different types of customers, you need the right equipment and organization.

#### Installation

```bash
# Install Django REST Framework
pip install djangorestframework

# Install additional useful packages
pip install djangorestframework-simplejwt  # For JWT authentication
pip install django-filter  # For filtering capabilities
```

#### Settings Configuration

In your `settings.py`, add the new equipment to your kitchen:

```python
# settings.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',  # Our new dispatch system
    'rest_framework.authtoken',  # Authentication system
    'django_filters',  # Filtering system
    'your_app_name',  # Your existing app
]

# REST Framework configuration - like setting kitchen standards
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
}
```

#### URL Configuration

Set up your API routing like organizing your kitchen's order flow:

```python
# urls.py (main project)
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('your_app_name.urls')),  # All API routes under /api/
]
```

**Syntax Explanation:**
- `INSTALLED_APPS`: A list that tells Django which applications to load
- `REST_FRAMEWORK`: A dictionary containing DRF-specific settings
- `DEFAULT_AUTHENTICATION_CLASSES`: A list of authentication methods DRF will try
- `include()`: Function that includes URL patterns from another module

---

## Lesson 2: Serializers and ViewSets

### The Recipe Cards and Kitchen Stations

Imagine serializers as detailed recipe cards that tell your kitchen staff exactly how to prepare a dish for different service types. A burger might be plated differently for dine-in versus packaged for takeout, but it's the same burger. Similarly, serializers define how your Django models should be "plated" (formatted) when served as JSON.

ViewSets are like specialized kitchen stations where different types of orders are processed efficiently.

#### Creating Models (Your Ingredients)

First, let's create a simple model - think of this as your raw ingredients:

```python
# models.py
from django.db import models
from django.contrib.auth.models import User

class Recipe(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    ingredients = models.TextField()
    instructions = models.TextField()
    prep_time = models.IntegerField()  # in minutes
    cook_time = models.IntegerField()  # in minutes
    servings = models.IntegerField()
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-created_at']
```

#### Creating Serializers (Recipe Cards)

Now create your serializers - these are like standardized recipe cards:

```python
# serializers.py
from rest_framework import serializers
from .models import Recipe
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    """Serializer for user information - like a chef's name tag"""
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        extra_kwargs = {'email': {'write_only': True}}  # Email only for creation

class RecipeSerializer(serializers.ModelSerializer):
    """Main recipe serializer - like a complete recipe card"""
    created_by = UserSerializer(read_only=True)  # Nested serializer
    total_time = serializers.SerializerMethodField()  # Calculated field
    
    class Meta:
        model = Recipe
        fields = [
            'id', 'title', 'description', 'ingredients', 
            'instructions', 'prep_time', 'cook_time', 
            'servings', 'created_by', 'created_at', 
            'updated_at', 'total_time'
        ]
        read_only_fields = ['created_by', 'created_at', 'updated_at']
    
    def get_total_time(self, obj):
        """Calculate total cooking time - like a kitchen timer"""
        return obj.prep_time + obj.cook_time
    
    def validate_prep_time(self, value):
        """Validate prep time - like checking if prep time is reasonable"""
        if value < 0:
            raise serializers.ValidationError("Prep time cannot be negative")
        if value > 1440:  # 24 hours
            raise serializers.ValidationError("Prep time seems too long")
        return value

class RecipeListSerializer(serializers.ModelSerializer):
    """Simplified serializer for listing recipes - like a menu summary"""
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = Recipe
        fields = ['id', 'title', 'prep_time', 'cook_time', 'servings', 'created_by_name']
```

#### Creating ViewSets (Kitchen Stations)

ViewSets are like specialized kitchen stations that handle different types of orders:

```python
# views.py
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Recipe
from .serializers import RecipeSerializer, RecipeListSerializer

class RecipeViewSet(viewsets.ModelViewSet):
    """
    A kitchen station that handles all recipe operations
    - Like a versatile chef who can prepare, modify, and serve recipes
    """
    queryset = Recipe.objects.all()
    serializer_class = RecipeSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        """Choose the right recipe card based on the order type"""
        if self.action == 'list':
            return RecipeListSerializer  # Summary for menu browsing
        return RecipeSerializer  # Full details for individual recipes
    
    def perform_create(self, serializer):
        """Assign the chef when creating a new recipe"""
        serializer.save(created_by=self.request.user)
    
    def get_permissions(self):
        """Set kitchen access rules"""
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            permission_classes = [permissions.IsAuthenticated]
        else:
            permission_classes = [permissions.AllowAny]  # Anyone can view recipes
        return [permission() for permission in permission_classes]
    
    @action(detail=True, methods=['post'])
    def favorite(self, request, pk=None):
        """Custom action - like adding a special 'favorite' sticker"""
        recipe = self.get_object()
        # Here you would implement favorite logic
        return Response({'message': f'Recipe {recipe.title} added to favorites'})
    
    @action(detail=False)
    def my_recipes(self, request):
        """Custom action - show only recipes by the current chef"""
        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required'}, 
                          status=status.HTTP_401_UNAUTHORIZED)
        
        my_recipes = Recipe.objects.filter(created_by=request.user)
        serializer = self.get_serializer(my_recipes, many=True)
        return Response(serializer.data)
```

#### URL Configuration

Set up your API routes like organizing your kitchen's order stations:

```python
# urls.py (app level)
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create a router - like a kitchen dispatcher
router = DefaultRouter()
router.register(r'recipes', views.RecipeViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

**Syntax Explanation:**
- `ModelSerializer`: Automatically creates serializer fields based on model fields
- `SerializerMethodField`: Creates a read-only field that gets its value from a method
- `read_only_fields`: Fields that can't be modified through the API
- `source`: Tells the serializer where to get the data from (useful for nested relationships)
- `ViewSet`: A class-based view that handles multiple related actions
- `@action`: Decorator to create custom endpoints beyond the standard CRUD operations
- `DefaultRouter`: Automatically creates URL patterns for ViewSets

---

## Lesson 3: API Authentication

### The Kitchen Security System

Think of API authentication like a modern restaurant's security system. Just as you need different levels of access (customers can order, staff can access the kitchen, managers can access inventory), your API needs different authentication levels.

#### Token Authentication Setup

First, create authentication tokens - like giving each staff member a digital key card:

```python
# In your Django shell or management command
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

# Create tokens for existing users
for user in User.objects.all():
    Token.objects.get_or_create(user=user)
```

#### Authentication Views

Create login/logout endpoints - like digital turnstiles for your kitchen:

```python
# authentication.py
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

class CustomAuthToken(ObtainAuthToken):
    """
    Custom login - like a smart turnstile that gives you a key card
    and tells you about your access level
    """
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data,
                                           context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        
        return Response({
            'token': token.key,
            'user_id': user.pk,
            'username': user.username,
            'email': user.email,
            'message': 'Login successful' if not created else 'New session started'
        })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """
    Logout - like returning your key card when leaving
    """
    try:
        # Delete the user's token to log them out
        token = Token.objects.get(user=request.user)
        token.delete()
        return Response({'message': 'Successfully logged out'}, 
                       status=status.HTTP_200_OK)
    except Token.DoesNotExist:
        return Response({'error': 'No active session found'}, 
                       status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile_view(request):
    """
    Get current user profile - like checking your own employee badge
    """
    user = request.user
    return Response({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'is_staff': user.is_staff,
        'date_joined': user.date_joined
    })
```

#### Custom Permission Classes

Create custom permissions - like specific kitchen access rules:

```python
# permissions.py
from rest_framework import permissions

class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Custom permission - like "only the chef who created a recipe can modify it"
    but anyone can view recipes
    """
    def has_object_permission(self, request, view, obj):
        # Read permissions for anyone
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions only for the owner
        return obj.created_by == request.user

class IsStaffOrReadOnly(permissions.BasePermission):
    """
    Custom permission - like "only staff can create/edit, but anyone can view"
    """
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user.is_staff

class IsOwnerOrStaff(permissions.BasePermission):
    """
    Custom permission - like "only the recipe creator or kitchen staff can modify"
    """
    def has_object_permission(self, request, view, obj):
        return obj.created_by == request.user or request.user.is_staff
```

#### Updated ViewSet with Authentication

Now let's update our RecipeViewSet to use proper authentication:

```python
# Updated views.py
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Recipe
from .serializers import RecipeSerializer, RecipeListSerializer
from .permissions import IsOwnerOrReadOnly

class RecipeViewSet(viewsets.ModelViewSet):
    """
    Recipe management with proper security - like a well-organized kitchen
    with clear access rules
    """
    queryset = Recipe.objects.all()
    serializer_class = RecipeSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return RecipeListSerializer
        return RecipeSerializer
    
    def perform_create(self, serializer):
        """Automatically assign the current user as the recipe creator"""
        serializer.save(created_by=self.request.user)
    
    @action(detail=False, methods=['get'], permission_classes=[permissions.IsAuthenticated])
    def my_recipes(self, request):
        """Get recipes created by the current user - like checking your own recipe book"""
        my_recipes = Recipe.objects.filter(created_by=request.user)
        serializer = self.get_serializer(my_recipes, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def favorite(self, request, pk=None):
        """Add to favorites - requires authentication like a personal cookbook"""
        recipe = self.get_object()
        # Implementation would go here
        return Response({
            'message': f'Recipe "{recipe.title}" added to your favorites',
            'recipe_id': recipe.id
        })
```

#### Authentication URLs

Add authentication endpoints to your URLs:

```python
# urls.py (app level)
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .authentication import CustomAuthToken, logout_view, profile_view

router = DefaultRouter()
router.register(r'recipes', views.RecipeViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('auth/login/', CustomAuthToken.as_view(), name='api_login'),
    path('auth/logout/', logout_view, name='api_logout'),
    path('auth/profile/', profile_view, name='api_profile'),
]
```

**Syntax Explanation:**
- `@permission_classes`: Decorator that applies permission classes to function-based views
- `IsAuthenticatedOrReadOnly`: Built-in permission that allows read access to anyone but write access only to authenticated users
- `BasePermission`: Base class for creating custom permissions
- `has_object_permission`: Method called to check permissions on individual objects
- `SAFE_METHODS`: HTTP methods that don't modify data (GET, HEAD, OPTIONS)
- `Token.objects.get_or_create()`: Gets existing token or creates new one

---

## Lesson 4: Pagination and Filtering

### The Organized Kitchen Service

Think of pagination and filtering like an organized restaurant service. When you have hundreds of recipes (like a massive menu), you don't want to overwhelm customers by showing everything at once. Instead, you organize them into manageable pages (like menu sections) and provide ways to find exactly what they're looking for (like search filters).

#### Setting Up Filtering

First, let's create comprehensive filtering - like having a smart menu system:

```python
# filters.py
import django_filters
from django_filters import rest_framework as filters
from .models import Recipe

class RecipeFilter(django_filters.FilterSet):
    """
    Recipe filtering system - like a smart menu that can sort and filter
    dishes based on customer preferences
    """
    
    # Text search in title and description
    search = django_filters.CharFilter(method='filter_search', label='Search')
    
    # Filter by prep time range
    prep_time_min = django_filters.NumberFilter(field_name='prep_time', lookup_expr='gte')
    prep_time_max = django_filters.NumberFilter(field_name='prep_time', lookup_expr='lte')
    
    # Filter by cook time range
    cook_time_min = django_filters.NumberFilter(field_name='cook_time', lookup_expr='gte')
    cook_time_max = django_filters.NumberFilter(field_name='cook_time', lookup_expr='lte')
    
    # Filter by servings
    servings_min = django_filters.NumberFilter(field_name='servings', lookup_expr='gte')
    servings_max = django_filters.NumberFilter(field_name='servings', lookup_expr='lte')
    
    # Filter by creation date
    created_after = django_filters.DateTimeFilter(field_name='created_at', lookup_expr='gte')
    created_before = django_filters.DateTimeFilter(field_name='created_at', lookup_expr='lte')
    
    # Filter by specific chef
    chef = django_filters.CharFilter(field_name='created_by__username', lookup_expr='icontains')
    
    class Meta:
        model = Recipe
        fields = []  # We define custom fields above
    
    def filter_search(self, queryset, name, value):
        """
        Custom search method - like a smart waiter who can find dishes
        by searching in multiple places
        """
        if value:
            queryset = queryset.filter(
                models.Q(title__icontains=value) |
                models.Q(description__icontains=value) |
                models.Q(ingredients__icontains=value)
            )
        return queryset
```

#### Custom Pagination Classes

Create custom pagination - like organizing your menu into manageable sections:

```python
# pagination.py
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

class RecipePagination(PageNumberPagination):
    """
    Custom pagination - like a smart menu system that shows
    a reasonable number of items per page
    """
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100
    
    def get_paginated_response(self, data):
        """
        Custom pagination response - like a helpful waiter telling you
        about the menu structure
        """
        return Response({
            'links': {
                'next': self.get_next_link(),
                'previous': self.get_previous_link()
            },
            'pagination': {
                'count': self.page.paginator.count,
                'current_page': self.page.number,
                'total_pages': self.page.paginator.num_pages,
                'page_size': self.page_size,
                'has_next': self.page.has_next(),
                'has_previous': self.page.has_previous(),
            },
            'results': data
        })

class LargeResultsPagination(PageNumberPagination):
    """
    For endpoints that might return many results - like a buffet-style service
    """
    page_size = 50
    page_size_query_param = 'page_size'
    max_page_size = 200
```

#### Updated ViewSet with Filtering and Pagination

Now let's update our ViewSet to include filtering and pagination:

```python
# Updated views.py
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from .models import Recipe
from .serializers import RecipeSerializer, RecipeListSerializer
from .permissions import IsOwnerOrReadOnly
from .filters import RecipeFilter
from .pagination import RecipePagination

class RecipeViewSet(viewsets.ModelViewSet):
    """
    Complete recipe management system - like a fully equipped kitchen
    with organization, search, and efficient service
    """
    queryset = Recipe.objects.all()
    serializer_class = RecipeSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]
    pagination_class = RecipePagination
    
    # Filter backends - like different ways to organize your kitchen
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_class = RecipeFilter
    
    # Search fields - like quick search options
    search_fields = ['title', 'description', 'ingredients']
    
    # Ordering fields - like different ways to sort your menu
    ordering_fields = ['created_at', 'title', 'prep_time', 'cook_time', 'servings']
    ordering = ['-created_at']  # Default ordering
    
    def get_serializer_class(self):
        if self.action == 'list':
            return RecipeListSerializer
        return RecipeSerializer
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
    
    @action(detail=False, methods=['get'], permission_classes=[permissions.IsAuthenticated])
    def my_recipes(self, request):
        """Get current user's recipes with filtering and pagination"""
        queryset = Recipe.objects.filter(created_by=request.user)
        
        # Apply the same filtering to personal recipes
        filtered_queryset = self.filter_queryset(queryset)
        
        # Paginate the results
        page = self.paginate_queryset(filtered_queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = self.get_serializer(filtered_queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def quick_recipes(self, request):
        """Get recipes that can be made quickly - like a fast food menu"""
        queryset = Recipe.objects.filter(
            prep_time__lte=15,  # 15 minutes or less prep
            cook_time__lte=30   # 30 minutes or less cooking
        ).order_by('prep_time', 'cook_time')
        
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = RecipeListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = RecipeListSerializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get recipe statistics - like kitchen analytics"""
        from django.db.models import Avg, Count, Min, Max
        
        stats = Recipe.objects.aggregate(
            total_recipes=Count('id'),
            avg_prep_time=Avg('prep_time'),
            avg_cook_time=Avg('cook_time'),
            avg_servings=Avg('servings'),
            quickest_prep=Min('prep_time'),
            longest_prep=Max('prep_time'),
            quickest_cook=Min('cook_time'),
            longest_cook=Max('cook_time'),
        )
        
        return Response({
            'recipe_statistics': stats,
            'message': 'Kitchen analytics ready!'
        })
```

#### Testing Your Filters and Pagination

Here are example API calls to test your filtering system:

```python
# Example API calls (you would make these from a client application)

# Basic pagination
GET /api/recipes/?page=1&page_size=5

# Search for recipes
GET /api/recipes/?search=chicken

# Filter by prep time
GET /api/recipes/?prep_time_min=10&prep_time_max=30

# Filter by servings and order by prep time
GET /api/recipes/?servings_min=4&ordering=prep_time

# Complex filtering with pagination
GET /api/recipes/?search=pasta&prep_time_max=20&page=1&page_size=10&ordering=-created_at

# Get quick recipes
GET /api/recipes/quick_recipes/

# Get statistics
GET /api/recipes/stats/
```

**Syntax Explanation:**
- `FilterSet`: Django-filter class that defines how to filter a queryset
- `lookup_expr`: Defines the type of lookup (gte=greater than or equal, lte=less than or equal, icontains=case-insensitive contains)
- `method`: Points to a custom filtering method
- `Q objects`: Django's way of building complex queries with AND/OR logic
- `aggregate()`: Django method for performing calculations across multiple records
- `filter_backends`: List of classes that process query parameters for filtering/searching/ordering
- `filterset_class`: Points to your custom FilterSet class
- `search_fields`: Fields that the SearchFilter will search in
- `ordering_fields`: Fields that can be used for ordering results

---
# Project: Building an Expense Tracker API

## Project Objective
By the end of this project, you will have created a complete RESTful API for an expense tracking application using Django REST Framework, demonstrating mastery of serializers, viewsets, authentication, pagination, and filtering.

## Imagine...
Imagine that you're the head chef of a bustling restaurant's financial kitchen. Your expense tracker API is like a sophisticated ordering system that helps the restaurant manage all its financial ingredients - from daily expenses to monthly budgets. Just as a chef needs to track ingredient costs, supplier payments, and kitchen equipment expenses, your API will organize and serve financial data to hungry client applications.

## Project: Expense Tracker API

### Project Structure
```
expense_tracker/
├── manage.py
├── expense_tracker/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── expenses/
    ├── __init__.py
    ├── models.py
    ├── serializers.py
    ├── views.py
    ├── urls.py
    └── migrations/
```

### Step 1: Create the Models (The Recipe Foundation)

```python
# expenses/models.py
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator
from decimal import Decimal

class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    color = models.CharField(max_length=7, default='#3498db')  # Hex color
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Categories"
        ordering = ['name']
    
    def __str__(self):
        return self.name

class Expense(models.Model):
    PAYMENT_METHODS = [
        ('cash', 'Cash'),
        ('card', 'Credit/Debit Card'),
        ('bank_transfer', 'Bank Transfer'),
        ('digital_wallet', 'Digital Wallet'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='expenses')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    amount = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0.01'))]
    )
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='expenses')
    payment_method = models.CharField(max_length=20, choices=PAYMENT_METHODS, default='cash')
    date = models.DateField()
    receipt_image = models.ImageField(upload_to='receipts/', blank=True, null=True)
    is_recurring = models.BooleanField(default=False)
    tags = models.CharField(max_length=200, blank=True, help_text="Comma-separated tags")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date', '-created_at']
    
    def __str__(self):
        return f"{self.title} - ${self.amount}"
    
    @property
    def tag_list(self):
        return [tag.strip() for tag in self.tags.split(',') if tag.strip()]

class Budget(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='budgets')
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='budgets')
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    month = models.DateField()  # First day of the month
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user', 'category', 'month']
        ordering = ['-month']
    
    def __str__(self):
        return f"{self.user.username} - {self.category.name} - {self.month.strftime('%B %Y')}"
```

### Step 2: Create the Serializers (The Recipe Cards)

```python
# expenses/serializers.py
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Category, Expense, Budget

class CategorySerializer(serializers.ModelSerializer):
    expense_count = serializers.SerializerMethodField()
    total_spent = serializers.SerializerMethodField()
    
    class Meta:
        model = Category
        fields = ['id', 'name', 'description', 'color', 'expense_count', 'total_spent', 'created_at']
    
    def get_expense_count(self, obj):
        # Get current user from context
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.expenses.filter(user=request.user).count()
        return 0
    
    def get_total_spent(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            total = obj.expenses.filter(user=request.user).aggregate(
                total=models.Sum('amount')
            )['total']
            return total or 0
        return 0

class ExpenseSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    category_color = serializers.CharField(source='category.color', read_only=True)
    tag_list = serializers.ListField(read_only=True)
    
    class Meta:
        model = Expense
        fields = [
            'id', 'title', 'description', 'amount', 'category', 'category_name', 
            'category_color', 'payment_method', 'date', 'receipt_image', 
            'is_recurring', 'tags', 'tag_list', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']
    
    def validate_amount(self, value):
        if value <= 0:
            raise serializers.ValidationError("Amount must be greater than zero.")
        return value
    
    def validate_tags(self, value):
        if value:
            tags = [tag.strip() for tag in value.split(',')]
            if len(tags) > 10:
                raise serializers.ValidationError("Maximum 10 tags allowed.")
        return value

class BudgetSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    spent_amount = serializers.SerializerMethodField()
    remaining_amount = serializers.SerializerMethodField()
    percentage_used = serializers.SerializerMethodField()
    
    class Meta:
        model = Budget
        fields = [
            'id', 'category', 'category_name', 'amount', 'month', 
            'spent_amount', 'remaining_amount', 'percentage_used', 'created_at'
        ]
        read_only_fields = ['created_at']
    
    def get_spent_amount(self, obj):
        # Calculate spent amount for this category in this month
        from django.db.models import Sum
        spent = Expense.objects.filter(
            user=obj.user,
            category=obj.category,
            date__year=obj.month.year,
            date__month=obj.month.month
        ).aggregate(total=Sum('amount'))['total']
        return spent or 0
    
    def get_remaining_amount(self, obj):
        spent = self.get_spent_amount(obj)
        return obj.amount - spent
    
    def get_percentage_used(self, obj):
        spent = self.get_spent_amount(obj)
        if obj.amount > 0:
            return round((spent / obj.amount) * 100, 2)
        return 0

class UserSerializer(serializers.ModelSerializer):
    total_expenses = serializers.SerializerMethodField()
    monthly_spending = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 
                 'total_expenses', 'monthly_spending', 'date_joined']
        read_only_fields = ['id', 'date_joined']
    
    def get_total_expenses(self, obj):
        return obj.expenses.count()
    
    def get_monthly_spending(self, obj):
        from django.db.models import Sum
        from datetime import datetime, date
        
        today = date.today()
        current_month_expenses = obj.expenses.filter(
            date__year=today.year,
            date__month=today.month
        ).aggregate(total=Sum('amount'))['total']
        
        return current_month_expenses or 0
```

### Step 3: Create the Views (The Kitchen Operations)

```python
# expenses/views.py
from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Sum, Count, Q
from django.utils import timezone
from datetime import datetime, timedelta
from .models import Category, Expense, Budget
from .serializers import CategorySerializer, ExpenseSerializer, BudgetSerializer
from .filters import ExpenseFilter

class CategoryViewSet(viewsets.ModelViewSet):
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['name', 'description']
    ordering_fields = ['name', 'created_at']
    ordering = ['name']

class ExpenseViewSet(viewsets.ModelViewSet):
    serializer_class = ExpenseSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_class = ExpenseFilter
    search_fields = ['title', 'description', 'tags']
    ordering_fields = ['date', 'amount', 'created_at']
    ordering = ['-date', '-created_at']
    
    def get_queryset(self):
        return Expense.objects.filter(user=self.request.user).select_related('category')
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        """Get expense summary statistics"""
        queryset = self.get_queryset()
        
        # Total expenses
        total_count = queryset.count()
        total_amount = queryset.aggregate(total=Sum('amount'))['total'] or 0
        
        # Current month expenses
        today = timezone.now().date()
        current_month = queryset.filter(
            date__year=today.year,
            date__month=today.month
        )
        monthly_count = current_month.count()
        monthly_amount = current_month.aggregate(total=Sum('amount'))['total'] or 0
        
        # Category breakdown
        category_breakdown = queryset.values('category__name', 'category__color').annotate(
            total_amount=Sum('amount'),
            count=Count('id')
        ).order_by('-total_amount')
        
        # Recent expenses
        recent_expenses = ExpenseSerializer(
            queryset[:5], 
            many=True, 
            context={'request': request}
        ).data
        
        return Response({
            'total_expenses': {
                'count': total_count,
                'amount': total_amount
            },
            'monthly_expenses': {
                'count': monthly_count,
                'amount': monthly_amount
            },
            'category_breakdown': category_breakdown,
            'recent_expenses': recent_expenses
        })
    
    @action(detail=False, methods=['get'])
    def monthly_trend(self, request):
        """Get monthly spending trend for the last 12 months"""
        from django.db.models import DateTrunc
        
        twelve_months_ago = timezone.now().date() - timedelta(days=365)
        
        trend_data = self.get_queryset().filter(
            date__gte=twelve_months_ago
        ).extra(
            select={'month': "DATE_FORMAT(date, '%%Y-%%m')"}
        ).values('month').annotate(
            total_amount=Sum('amount'),
            count=Count('id')
        ).order_by('month')
        
        return Response(trend_data)
    
    @action(detail=False, methods=['get'])
    def by_category(self, request):
        """Get expenses grouped by category"""
        category_id = request.query_params.get('category')
        
        queryset = self.get_queryset()
        if category_id:
            queryset = queryset.filter(category_id=category_id)
        
        expenses = ExpenseSerializer(
            queryset, 
            many=True, 
            context={'request': request}
        ).data
        
        return Response(expenses)

class BudgetViewSet(viewsets.ModelViewSet):
    serializer_class = BudgetSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['category', 'month']
    ordering_fields = ['month', 'amount']
    ordering = ['-month']
    
    def get_queryset(self):
        return Budget.objects.filter(user=self.request.user).select_related('category')
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    
    @action(detail=False, methods=['get'])
    def current_month(self, request):
        """Get current month's budget status"""
        today = timezone.now().date()
        current_month = today.replace(day=1)
        
        budgets = self.get_queryset().filter(month=current_month)
        serializer = self.get_serializer(budgets, many=True)
        
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def alerts(self, request):
        """Get budget alerts (overbudget or close to limit)"""
        today = timezone.now().date()
        current_month = today.replace(day=1)
        
        budgets = self.get_queryset().filter(month=current_month)
        alerts = []
        
        for budget in budgets:
            serializer = self.get_serializer(budget)
            data = serializer.data
            
            if data['percentage_used'] >= 100:
                alerts.append({
                    'type': 'overbudget',
                    'message': f"You've exceeded your {data['category_name']} budget by ${data['spent_amount'] - data['amount']:.2f}",
                    'budget': data
                })
            elif data['percentage_used'] >= 80:
                alerts.append({
                    'type': 'warning',
                    'message': f"You've used {data['percentage_used']:.1f}% of your {data['category_name']} budget",
                    'budget': data
                })
        
        return Response(alerts)
```

### Step 4: Create Filters (The Kitchen Strainer)

```python
# expenses/filters.py
import django_filters
from django import forms
from .models import Expense, Category

class ExpenseFilter(django_filters.FilterSet):
    # Date range filtering
    date_from = django_filters.DateFilter(field_name='date', lookup_expr='gte')
    date_to = django_filters.DateFilter(field_name='date', lookup_expr='lte')
    
    # Amount range filtering
    amount_min = django_filters.NumberFilter(field_name='amount', lookup_expr='gte')
    amount_max = django_filters.NumberFilter(field_name='amount', lookup_expr='lte')
    
    # Category filtering
    category = django_filters.ModelChoiceFilter(queryset=Category.objects.all())
    category_name = django_filters.CharFilter(field_name='category__name', lookup_expr='icontains')
    
    # Payment method filtering
    payment_method = django_filters.ChoiceFilter(choices=Expense.PAYMENT_METHODS)
    
    # Tag filtering
    tags = django_filters.CharFilter(method='filter_tags')
    
    # Recurring expenses
    is_recurring = django_filters.BooleanFilter()
    
    class Meta:
        model = Expense
        fields = ['category', 'payment_method', 'is_recurring']
    
    def filter_tags(self, queryset, name, value):
        return queryset.filter(tags__icontains=value)
```

### Step 5: URL Configuration (The Kitchen Menu)

```python
# expenses/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CategoryViewSet, ExpenseViewSet, BudgetViewSet

router = DefaultRouter()
router.register(r'categories', CategoryViewSet)
router.register(r'expenses', ExpenseViewSet, basename='expense')
router.register(r'budgets', BudgetViewSet, basename='budget')

urlpatterns = [
    path('api/', include(router.urls)),
]
```

### Step 6: Main URL Configuration

```python
# expense_tracker/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.authtoken.views import obtain_auth_token

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('expenses.urls')),
    path('api-auth/', include('rest_framework.urls')),
    path('api/token/', obtain_auth_token, name='api_token_auth'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

### Step 7: Testing Your API (Quality Check)

```python
# Test your API endpoints using curl or a tool like Postman

# Get authentication token
curl -X POST http://localhost:8000/api/token/ \
     -H "Content-Type: application/json" \
     -d '{"username": "your_username", "password": "your_password"}'

# Create a category
curl -X POST http://localhost:8000/api/categories/ \
     -H "Authorization: Token YOUR_TOKEN_HERE" \
     -H "Content-Type: application/json" \
     -d '{"name": "Food", "description": "Restaurant and grocery expenses", "color": "#ff6b6b"}'

# Create an expense
curl -X POST http://localhost:8000/api/expenses/ \
     -H "Authorization: Token YOUR_TOKEN_HERE" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Lunch at Restaurant",
       "description": "Team lunch meeting",
       "amount": "45.50",
       "category": 1,
       "payment_method": "card",
       "date": "2024-07-14",
       "tags": "business, lunch, team"
     }'

# Get expense summary
curl -X GET http://localhost:8000/api/expenses/summary/ \
     -H "Authorization: Token YOUR_TOKEN_HERE"

# Filter expenses by date range
curl -X GET "http://localhost:8000/api/expenses/?date_from=2024-07-01&date_to=2024-07-31" \
     -H "Authorization: Token YOUR_TOKEN_HERE"

# Get monthly trend
curl -X GET http://localhost:8000/api/expenses/monthly_trend/ \
     -H "Authorization: Token YOUR_TOKEN_HERE"
```

## Project Features Implemented

Your expense tracker API now includes:

1. **Complete CRUD Operations** - Create, read, update, and delete expenses, categories, and budgets
2. **User Authentication** - Token-based authentication to secure user data
3. **Advanced Filtering** - Filter expenses by date range, amount, category, payment method, and tags
4. **Search Functionality** - Search expenses by title, description, and tags
5. **Pagination** - Automatic pagination for large datasets
6. **Summary Statistics** - Get spending summaries, monthly trends, and category breakdowns
7. **Budget Tracking** - Set budgets and get alerts when approaching limits
8. **File Upload** - Support for receipt images
9. **Data Validation** - Comprehensive validation for all input data
10. **Performance Optimization** - Efficient database queries with select_related

## API Endpoints Available

- `GET/POST /api/categories/` - List/create categories
- `GET/PUT/DELETE /api/categories/{id}/` - Retrieve/update/delete category
- `GET/POST /api/expenses/` - List/create expenses (with filtering)
- `GET/PUT/DELETE /api/expenses/{id}/` - Retrieve/update/delete expense
- `GET /api/expenses/summary/` - Get expense statistics
- `GET /api/expenses/monthly_trend/` - Get 12-month spending trend
- `GET /api/budgets/` - List/create budgets
- `GET /api/budgets/current_month/` - Get current month budgets
- `GET /api/budgets/alerts/` - Get budget alerts

Just like a well-organized kitchen where every chef knows exactly where to find ingredients and how to prepare each dish, your expense tracker API provides a clean, organized way for client applications to manage financial data with all the essential features a modern expense tracking system needs!

## Assignment: Recipe Rating System

### The Challenge

Your restaurant's API is working great, but customers want to rate and review recipes. You need to extend your existing recipe API to include a rating system.

**Requirements:**

1. Create a `Rating` model that allows users to rate recipes (1-5 stars) and leave optional comments
2. Each user can only rate a recipe once, but they can update their rating
3. Create a `RatingSerializer` that includes the user's name and rating details
4. Add a `RatingViewSet` with proper authentication (only authenticated users can rate)
5. Add a custom action to the `RecipeViewSet` called `ratings` that returns all ratings for a specific recipe
6. Add another custom action called `average_rating` that returns the average rating and total count for a recipe
7. Implement filtering on ratings (filter by star rating, date range)
8. Add pagination to the ratings

**Bonus Challenges:**
- Add validation to prevent users from rating their own recipes
- Create a "top_rated" action that returns recipes sorted by average rating
- Add a search filter to find ratings by comment content

**Expected API Endpoints:**
- `GET/POST /api/ratings/` - List all ratings or create a new rating
- `GET/PUT/DELETE /api/ratings/{id}/` - Get, update, or delete a specific rating
- `GET /api/recipes/{id}/ratings/` - Get all ratings for a specific recipe
- `GET /api/recipes/{id}/average_rating/` - Get average rating for a recipe
- `GET /api/recipes/top_rated/` - Get recipes ordered by rating

**Tips:**
- Think about the relationship between Recipe, Rating, and User models
- Consider using Django's `unique_together` constraint to prevent duplicate ratings
- Use Django's aggregation functions for calculating averages
- Remember to apply proper permissions (users can only edit their own ratings)

This assignment will test your understanding of model relationships, serializers, ViewSets, custom actions, filtering, and authentication - all the core concepts from today's lesson!

---

## Syntax Summary

Throughout this lesson, we've used several important Django REST Framework syntax patterns:

**Model Relationships:**
- `ForeignKey`: Creates a many-to-one relationship
- `on_delete=models.CASCADE`: Deletes related objects when parent is deleted

**Serializer Syntax:**
- `ModelSerializer`: Automatically generates fields from model
- `SerializerMethodField`: Creates computed fields
- `read_only_fields`: Fields that can't be modified via API
- `source`: Specifies where to get field data from

**ViewSet Syntax:**
- `ModelViewSet`: Provides CRUD operations automatically
- `@action`: Creates custom endpoints
- `detail=True/False`: Determines if action works on individual objects or collections
- `permission_classes`: Sets access control rules

**Authentication & Permissions:**
- `IsAuthenticated`: Requires user login
- `IsAuthenticatedOrReadOnly`: Allows read access to anyone, write access to authenticated users
- `BasePermission`: Base class for custom permissions

**Filtering & Pagination:**
- `FilterSet`: Defines filtering rules
- `lookup_expr`: Specifies filter type (gte, lte, icontains, etc.)
- `Q objects`: Builds complex queries with AND/OR logic
- `PageNumberPagination`: Provides page-based pagination

This foundation will serve you well as you build more complex APIs with Django REST Framework!