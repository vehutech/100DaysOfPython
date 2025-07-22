# Day 63: Django Extensions & Third-Party Packages

## Learning Objective
By the end of this lesson, you will be able to enhance your Django development workflow by integrating essential third-party packages, utilizing django-extensions for advanced development features, implementing django-debug-toolbar for performance optimization, and creating your own custom Django package for reusable functionality.

---

## Introduction: The Master Chef's Toolkit

Imagine that you're a master chef who has learned all the basic cooking techniques, but now you want to take your culinary skills to the next level. Just as a professional kitchen relies on specialized tools, high-quality ingredients from trusted suppliers, and custom spice blends to create exceptional dishes, Django developers use third-party packages to enhance their applications beyond the framework's built-in capabilities.

Today, we'll explore how to stock your Django kitchen with the finest tools and ingredients that will transform you from a home cook into a professional chef.

---

## Lesson 1: Essential Django Packages - Building Your Pantry

Just as every professional kitchen needs a well-stocked pantry with essential ingredients, every Django developer should know about the core packages that solve common problems.

### The Essential Pantry Items

Let's start by adding these must-have packages to our Django project:

```bash
# requirements.txt
Django==4.2.0
django-extensions==3.2.3
django-debug-toolbar==4.2.0
Pillow==10.0.0
python-dotenv==1.0.0
django-crispy-forms==2.0
```

**Syntax Explanation:**
- `requirements.txt` is like a shopping list for your project's dependencies
- Each line specifies a package name and version using `==` for exact version pinning
- This ensures all developers working on your project use the same "ingredients"

### Installing Your Pantry

```bash
pip install -r requirements.txt
```

### Configuring Your Kitchen (settings.py)

```python
# settings.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third-party packages - our premium ingredients
    'django_extensions',  # Swiss army knife for Django
    'debug_toolbar',      # Kitchen inspector
    'crispy_forms',       # Elegant form presentation
    
    # Your apps
    'blog',
]

MIDDLEWARE = [
    'debug_toolbar.middleware.DebugToolbarMiddleware',  # Must be near the top
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
```

**Syntax Explanation:**
- `INSTALLED_APPS` is a list that tells Django which "ingredients" are available in your kitchen
- Order matters in `MIDDLEWARE` - think of it as the sequence of preparation steps
- The debug toolbar middleware should be near the top to catch all requests

---

## Lesson 2: Django-Extensions - The Master Chef's Swiss Army Knife

Django-extensions is like having a Swiss army knife in your kitchen - it provides dozens of useful management commands and utilities that make development more efficient.

### Key Management Commands

```bash
# Generate a detailed model graph (like a recipe diagram)
python manage.py graph_models -a -g -o models.png

# Create a superuser with random password
python manage.py createsuperuser --username chef --email chef@kitchen.com

# Show all URLs in your project (your menu items)
python manage.py show_urls

# Shell with auto-imports (pre-heated oven)
python manage.py shell_plus

# Run development server with enhanced features
python manage.py runserver_plus
```

**Syntax Explanation:**
- `manage.py` is Django's command-line utility, like your main cooking utensil
- Commands follow the pattern: `python manage.py [command] [options]`
- The `-a` flag means "all apps", `-g` means "group by app", `-o` specifies output file

### Advanced Model Utilities

```python
# models.py
from django.db import models
from django_extensions.db.models import TimeStampedModel, TitleSlugDescriptionModel

class Recipe(TimeStampedModel, TitleSlugDescriptionModel):
    """
    Like inheriting cooking techniques from master chefs
    TimeStampedModel adds: created, modified fields
    TitleSlugDescriptionModel adds: title, slug, description fields
    """
    chef = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    ingredients = models.TextField()
    instructions = models.TextField()
    difficulty = models.CharField(max_length=20, choices=[
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    ])
    
    class Meta:
        ordering = ['-created']  # Newest recipes first
        
    def __str__(self):
        return f"{self.title} by {self.chef.username}"
```

**Syntax Explanation:**
- Multiple inheritance in Python uses comma-separated class names: `class Recipe(Parent1, Parent2)`
- `TimeStampedModel` automatically adds `created` and `modified` datetime fields
- `TitleSlugDescriptionModel` provides common fields used in many models
- `choices` parameter creates a dropdown menu in forms and admin

### Shell Plus Magic

```python
# When you run: python manage.py shell_plus
# All your models are automatically imported - no manual imports needed!

# Instead of:
# from blog.models import Recipe
# from django.contrib.auth.models import User

# You can directly use:
recipes = Recipe.objects.all()
users = User.objects.filter(is_staff=True)

# Show SQL queries generated
from django.db import connection
print(connection.queries)
```

---

## Lesson 3: Django-Debug-Toolbar - Your Kitchen Inspector

The debug toolbar is like having a kitchen inspector who watches everything you do and provides detailed reports on performance, ingredients used, and potential improvements.

### Configuration

```python
# settings.py
INTERNAL_IPS = [
    '127.0.0.1',  # Your local development IP
]

DEBUG_TOOLBAR_CONFIG = {
    'SHOW_TOOLBAR_CALLBACK': lambda request: True,  # Always show in development
}

# Debug toolbar panels - your inspection tools
DEBUG_TOOLBAR_PANELS = [
    'debug_toolbar.panels.history.HistoryPanel',      # Request history
    'debug_toolbar.panels.versions.VersionsPanel',     # Package versions
    'debug_toolbar.panels.timer.TimerPanel',          # Timing information
    'debug_toolbar.panels.settings.SettingsPanel',    # Settings inspection
    'debug_toolbar.panels.headers.HeadersPanel',      # HTTP headers
    'debug_toolbar.panels.request.RequestPanel',      # Request details
    'debug_toolbar.panels.sql.SQLPanel',              # Database queries
    'debug_toolbar.panels.staticfiles.StaticFilesPanel', # Static files
    'debug_toolbar.panels.templates.TemplatesPanel',   # Template usage
    'debug_toolbar.panels.cache.CachePanel',          # Cache usage
    'debug_toolbar.panels.signals.SignalsPanel',      # Django signals
]
```

### URL Configuration

```python
# urls.py
from django.conf import settings
from django.conf.urls import include
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('blog.urls')),
]

# Only include debug toolbar in development
if settings.DEBUG:
    import debug_toolbar
    urlpatterns = [
        path('__debug__/', include(debug_toolbar.urls)),
    ] + urlpatterns
```

**Syntax Explanation:**
- `INTERNAL_IPS` defines which IP addresses can see the debug toolbar
- `lambda request: True` is a function that always returns `True` - shows toolbar for all requests
- The `if settings.DEBUG:` block only adds debug URLs in development mode
- List concatenation using `+` adds debug URLs to the beginning of the URL list

### Reading the Inspector's Report

```python
# views.py - Example view with performance monitoring
from django.shortcuts import render
from django.db import connection
from .models import Recipe

def recipe_list(request):
    """
    The debug toolbar will show:
    - How many database queries this view makes
    - How long each query takes
    - Which templates are rendered
    - Memory usage
    """
    # Bad: This creates N+1 query problem
    recipes = Recipe.objects.all()  # 1 query
    # In template: recipe.chef.username creates 1 query per recipe
    
    # Good: Use select_related to optimize
    # recipes = Recipe.objects.select_related('chef')  # 1 query total
    
    context = {'recipes': recipes}
    return render(request, 'blog/recipe_list.html', context)
```

---

## Lesson 4: Custom Package Creation - Creating Your Own Spice Blend

Sometimes you need to create your own custom spice blend (package) that can be reused across multiple kitchens (projects).

### Package Structure

```
django-recipe-utils/
├── setup.py
├── README.md
├── recipe_utils/
│   ├── __init__.py
│   ├── models.py
│   ├── validators.py
│   └── templatetags/
│       ├── __init__.py
│       └── recipe_tags.py
```

### Creating the Package Files

```python
# setup.py - The recipe for your spice blend
from setuptools import setup, find_packages

setup(
    name='django-recipe-utils',
    version='1.0.0',
    description='Reusable utilities for Django recipe applications',
    author='Master Chef',
    author_email='chef@kitchen.com',
    packages=find_packages(),
    install_requires=[
        'Django>=4.0',
    ],
    classifiers=[
        'Environment :: Web Environment',
        'Framework :: Django',
        'Programming Language :: Python :: 3',
    ],
)
```

```python
# recipe_utils/validators.py
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

def validate_cooking_time(value):
    """
    Validate that cooking time is reasonable (between 1 minute and 24 hours)
    Like a cooking instructor checking if your timing makes sense
    """
    if value < 1:
        raise ValidationError(_('Cooking time must be at least 1 minute'))
    if value > 1440:  # 24 hours in minutes
        raise ValidationError(_('Cooking time cannot exceed 24 hours'))

def validate_difficulty_level(value):
    """Ensure difficulty is one of the accepted levels"""
    valid_levels = ['easy', 'medium', 'hard', 'expert']
    if value.lower() not in valid_levels:
        raise ValidationError(
            _('Difficulty must be one of: {}').format(', '.join(valid_levels))
        )
```

```python
# recipe_utils/templatetags/recipe_tags.py
from django import template
from django.utils.html import format_html

register = template.Library()

@register.simple_tag
def difficulty_badge(difficulty):
    """
    Create a styled badge for recipe difficulty
    Like adding garnish to make dishes look professional
    """
    colors = {
        'easy': 'success',
        'medium': 'warning', 
        'hard': 'danger',
        'expert': 'dark'
    }
    color = colors.get(difficulty, 'secondary')
    
    return format_html(
        '<span class="badge bg-{}">{}</span>',
        color,
        difficulty.title()
    )

@register.filter
def cooking_time_format(minutes):
    """Convert minutes to human-readable format"""
    if minutes < 60:
        return f"{minutes} min"
    else:
        hours = minutes // 60
        remaining_minutes = minutes % 60
        if remaining_minutes == 0:
            return f"{hours} hr"
        return f"{hours} hr {remaining_minutes} min"
```

**Syntax Explanation:**
- `find_packages()` automatically discovers Python packages in your directory
- `@register.simple_tag` creates a template tag that can be used like `{% difficulty_badge recipe.difficulty %}`
- `@register.filter` creates a template filter used like `{{ recipe.cooking_time|cooking_time_format }}`
- `format_html()` safely formats HTML strings, preventing XSS attacks
- `//` is integer division (floors the result), `%` is the modulo operator

### Using Your Custom Package

```python
# In any Django project, after installing your package:
# pip install -e /path/to/django-recipe-utils

# settings.py
INSTALLED_APPS = [
    # ...
    'recipe_utils',  # Your custom package
]

# models.py
from django.db import models
from recipe_utils.validators import validate_cooking_time, validate_difficulty_level

class Recipe(models.Model):
    title = models.CharField(max_length=200)
    cooking_time = models.IntegerField(
        validators=[validate_cooking_time],
        help_text="Cooking time in minutes"
    )
    difficulty = models.CharField(
        max_length=20,
        validators=[validate_difficulty_level]
    )
```

```html
<!-- In templates -->
{% load recipe_tags %}

<div class="recipe-card">
    <h3>{{ recipe.title }}</h3>
    {% difficulty_badge recipe.difficulty %}
    <p>Cooking time: {{ recipe.cooking_time|cooking_time_format }}</p>
</div>
```

---

## Final Project: Restaurant Management Enhancement

Now let's put all our new tools together to enhance a restaurant management system:

```python
# restaurant/models.py
from django.db import models
from django.contrib.auth.models import User
from django_extensions.db.models import TimeStampedModel, TitleSlugDescriptionModel
from recipe_utils.validators import validate_cooking_time, validate_difficulty_level

class Restaurant(TimeStampedModel):
    name = models.CharField(max_length=100)
    chef = models.ForeignKey(User, on_delete=models.CASCADE)
    cuisine_type = models.CharField(max_length=50)
    
    def __str__(self):
        return self.name

class MenuItem(TimeStampedModel, TitleSlugDescriptionModel):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=8, decimal_places=2)
    cooking_time = models.IntegerField(
        validators=[validate_cooking_time],
        help_text="Time in minutes"
    )
    difficulty = models.CharField(
        max_length=20,
        validators=[validate_difficulty_level]
    )
    is_available = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['restaurant', 'title']
```

```python
# restaurant/views.py
from django.shortcuts import render, get_object_or_404
from django.db.models import Avg, Count
from .models import Restaurant, MenuItem

def restaurant_dashboard(request, restaurant_id):
    """
    Dashboard view optimized for performance
    The debug toolbar will help us monitor query efficiency
    """
    restaurant = get_object_or_404(Restaurant, id=restaurant_id)
    
    # Optimized query using select_related and prefetch_related
    menu_items = MenuItem.objects.filter(
        restaurant=restaurant
    ).select_related('restaurant').prefetch_related('chef')
    
    # Aggregate data for dashboard metrics
    stats = menu_items.aggregate(
        avg_cooking_time=Avg('cooking_time'),
        total_items=Count('id'),
        available_items=Count('id', filter=models.Q(is_available=True))
    )
    
    context = {
        'restaurant': restaurant,
        'menu_items': menu_items,
        'stats': stats,
    }
    
    return render(request, 'restaurant/dashboard.html', context)
```

```html
<!-- restaurant/templates/restaurant/dashboard.html -->
{% extends 'base.html' %}
{% load recipe_tags %}

{% block content %}
<div class="container">
    <h1>{{ restaurant.name }} Dashboard</h1>
    
    <div class="row">
        <div class="col-md-4">
            <div class="card">
                <div class="card-body">
                    <h5>Total Menu Items</h5>
                    <h2>{{ stats.total_items }}</h2>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card">
                <div class="card-body">
                    <h5>Available Items</h5>
                    <h2>{{ stats.available_items }}</h2>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card">
                <div class="card-body">
                    <h5>Avg Cooking Time</h5>
                    <h2>{{ stats.avg_cooking_time|cooking_time_format }}</h2>
                </div>
            </div>
        </div>
    </div>
    
    <h3>Menu Items</h3>
    <div class="row">
        {% for item in menu_items %}
        <div class="col-md-6 mb-3">
            <div class="card">
                <div class="card-body">
                    <h5>{{ item.title }}</h5>
                    {% difficulty_badge item.difficulty %}
                    <p>{{ item.description|truncatewords:20 }}</p>
                    <p><strong>${{ item.price }}</strong></p>
                    <p>Cooking time: {{ item.cooking_time|cooking_time_format }}</p>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</div>
{% endblock %}
```

---

## Assignment: Package Integration Challenge

**Scenario:** You work for a food delivery startup and need to integrate multiple third-party packages to create a comprehensive order tracking system.

**Task:** Create a Django application that demonstrates the integration of django-extensions and django-debug-toolbar while building a custom package for order utilities.

**Requirements:**

1. **Create an Order Tracking App:**
   - Models: `Order`, `OrderItem`, `DeliveryAgent`
   - Use `TimeStampedModel` from django-extensions
   - Include custom validators for delivery time and order status

2. **Custom Package Creation:**
   - Create `django-order-utils` package
   - Include validators for phone numbers and delivery zones
   - Create template tags for order status badges and time formatting
   - Add a template filter for calculating delivery fees

3. **Debug Optimization:**
   - Create a view that initially has N+1 query problems
   - Use django-debug-toolbar to identify the issues
   - Optimize using `select_related` and `prefetch_related`
   - Document the before/after query counts

4. **Management Command:**
   - Use django-extensions to create a custom management command
   - Command should generate sample order data for testing
   - Use `shell_plus` to demonstrate the command working

**Deliverables:**
- Complete Django project with the order tracking app
- Separate custom package with proper setup.py
- Screenshots of debug toolbar showing optimization improvements
- README file documenting the packages used and their benefits

**Evaluation Criteria:**
- Proper integration of third-party packages (30%)
- Custom package structure and functionality (25%)
- Query optimization demonstrated through debug toolbar (25%)
- Code quality and documentation (20%)

**Bonus Points:**
- Add additional useful django-extensions commands to your workflow
- Create a comprehensive template tag library in your custom package
- Implement caching and show cache performance in debug toolbar

This assignment reinforces all the concepts learned while challenging you to apply them in a real-world scenario that's different from the restaurant management example in the lesson.

# Build: Enhanced Development Environment

## Project Overview
Create a comprehensive Django development environment that combines multiple third-party packages to streamline your workflow - like setting up a professional kitchen with all the essential tools a master chef needs.

## Project Structure
```
enhanced_django_project/
├── manage.py
├── requirements.txt
├── .env
├── config/
│   ├── __init__.py
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── urls.py
│   └── wsgi.py
├── apps/
│   ├── __init__.py
│   ├── users/
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   └── admin.py
│   └── blog/
│       ├── models.py
│       ├── views.py
│       ├── urls.py
│       └── admin.py
├── static/
├── media/
├── templates/
└── logs/
```

## Step 1: Project Setup and Requirements

Create `requirements.txt`:
```txt
Django>=4.2.0,<5.0.0
python-decouple==3.8
django-extensions==3.2.3
django-debug-toolbar==4.2.0
Pillow==10.0.0
django-crispy-forms==2.0
crispy-bootstrap5==0.7
django-environ==0.11.2
Werkzeug==2.3.7
ipython==8.15.0
```

Create `.env` file:
```env
DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///db.sqlite3
ALLOWED_HOSTS=localhost,127.0.0.1
```

## Step 2: Settings Configuration

Create `config/settings/base.py`:
```python
import os
from pathlib import Path
from decouple import config
import environ

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Environment variables
env = environ.Env()

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = config('SECRET_KEY')

# Application definition
DJANGO_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

THIRD_PARTY_APPS = [
    'django_extensions',
    'debug_toolbar',
    'crispy_forms',
    'crispy_bootstrap5',
]

LOCAL_APPS = [
    'apps.users',
    'apps.blog',
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
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

WSGI_APPLICATION = 'config.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Crispy Forms
CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'django.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
```

Create `config/settings/development.py`:
```python
from .base import *

DEBUG = config('DEBUG', default=True, cast=bool)

ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='localhost,127.0.0.1').split(',')

# Debug Toolbar
if DEBUG:
    MIDDLEWARE += ['debug_toolbar.middleware.DebugToolbarMiddleware']
    
    INTERNAL_IPS = [
        '127.0.0.1',
        'localhost',
    ]
    
    DEBUG_TOOLBAR_CONFIG = {
        'SHOW_TOOLBAR_CALLBACK': lambda request: True,
        'SHOW_COLLAPSED': True,
        'SHOW_TEMPLATE_CONTEXT': True,
    }
    
    DEBUG_TOOLBAR_PANELS = [
        'debug_toolbar.panels.versions.VersionsPanel',
        'debug_toolbar.panels.timer.TimerPanel',
        'debug_toolbar.panels.settings.SettingsPanel',
        'debug_toolbar.panels.headers.HeadersPanel',
        'debug_toolbar.panels.request.RequestPanel',
        'debug_toolbar.panels.sql.SQLPanel',
        'debug_toolbar.panels.staticfiles.StaticFilesPanel',
        'debug_toolbar.panels.templates.TemplatesPanel',
        'debug_toolbar.panels.cache.CachePanel',
        'debug_toolbar.panels.signals.SignalsPanel',
        'debug_toolbar.panels.logging.LoggingPanel',
        'debug_toolbar.panels.redirects.RedirectsPanel',
        'debug_toolbar.panels.profiling.ProfilingPanel',
    ]

# Django Extensions
SHELL_PLUS_PRINT_SQL = True
SHELL_PLUS_PRINT_SQL_TRUNCATE = 1000

# Email backend for development
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Cache
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
    }
}
```

## Step 3: URL Configuration

Create `config/urls.py`:
```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('apps.blog.urls')),
    path('users/', include('apps.users.urls')),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    
    # Debug toolbar
    if 'debug_toolbar' in settings.INSTALLED_APPS:
        import debug_toolbar
        urlpatterns = [
            path('__debug__/', include(debug_toolbar.urls)),
        ] + urlpatterns
```

## Step 4: User App with Custom Extensions

Create `apps/users/models.py`:
```python
from django.contrib.auth.models import AbstractUser
from django.db import models
from django_extensions.db.models import TimeStampedModel

class CustomUser(AbstractUser, TimeStampedModel):
    """
    Custom user model with additional fields
    Like adding special ingredients to enhance the basic recipe
    """
    email = models.EmailField(unique=True)
    bio = models.TextField(max_length=500, blank=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)
    is_verified = models.BooleanField(default=False)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
    
    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"
    
    def __str__(self):
        return self.email
```

Create `apps/users/admin.py`:
```python
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    """
    Enhanced admin interface - like a chef's organized workstation
    """
    list_display = ['email', 'username', 'is_verified', 'is_staff', 'created']
    list_filter = ['is_verified', 'is_staff', 'created']
    search_fields = ['email', 'username']
    ordering = ['-created']
    
    fieldsets = UserAdmin.fieldsets + (
        ('Additional Info', {
            'fields': ('bio', 'avatar', 'is_verified')
        }),
        ('Timestamps', {
            'fields': ('created', 'modified'),
            'classes': ('collapse',)
        }),
    )
    
    readonly_fields = ('created', 'modified')
```

## Step 5: Blog App with Advanced Features

Create `apps/blog/models.py`:
```python
from django.db import models
from django.conf import settings
from django.urls import reverse
from django.utils.text import slugify
from django_extensions.db.models import TimeStampedModel, TitleSlugDescriptionModel

class Category(TimeStampedModel):
    """Category model for organizing blog posts"""
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    
    class Meta:
        verbose_name_plural = "Categories"
        ordering = ['name']
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

class Post(TitleSlugDescriptionModel, TimeStampedModel):
    """
    Blog post model with rich features
    Like a signature dish with all the garnishes
    """
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE,
        related_name='posts'
    )
    category = models.ForeignKey(
        Category,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='posts'
    )
    content = models.TextField()
    featured_image = models.ImageField(
        upload_to='posts/', 
        blank=True, 
        null=True
    )
    is_published = models.BooleanField(default=False)
    published_at = models.DateTimeField(blank=True, null=True)
    views_count = models.PositiveIntegerField(default=0)
    tags = models.CharField(max_length=200, blank=True, help_text="Comma-separated tags")
    
    class Meta:
        ordering = ['-created']
        verbose_name = "Blog Post"
        verbose_name_plural = "Blog Posts"
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse('blog:post_detail', kwargs={'slug': self.slug})
    
    @property
    def get_tags_list(self):
        """Convert comma-separated tags to list"""
        return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
    
    def increment_views(self):
        """Increment view count"""
        self.views_count += 1
        self.save(update_fields=['views_count'])

class Comment(TimeStampedModel):
    """Comment model for blog posts"""
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    content = models.TextField(max_length=1000)
    is_approved = models.BooleanField(default=False)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True)
    
    class Meta:
        ordering = ['created']
    
    def __str__(self):
        return f"Comment by {self.author.username} on {self.post.title}"
```

Create `apps/blog/views.py`:
```python
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.contrib import messages
from django.db.models import Q, F
from .models import Post, Category, Comment

def post_list(request):
    """
    Display paginated list of published posts
    Like presenting a menu of today's specials
    """
    posts = Post.objects.filter(is_published=True).select_related('author', 'category')
    
    # Search functionality
    search_query = request.GET.get('search', '')
    if search_query:
        posts = posts.filter(
            Q(title__icontains=search_query) |
            Q(content__icontains=search_query) |
            Q(tags__icontains=search_query)
        )
    
    # Category filter
    category_slug = request.GET.get('category', '')
    if category_slug:
        posts = posts.filter(category__slug=category_slug)
    
    # Pagination
    paginator = Paginator(posts, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get all categories for sidebar
    categories = Category.objects.all()
    
    context = {
        'page_obj': page_obj,
        'categories': categories,
        'search_query': search_query,
        'selected_category': category_slug,
    }
    return render(request, 'blog/post_list.html', context)

def post_detail(request, slug):
    """
    Display individual post with comments
    Like serving a complete dish with all accompaniments
    """
    post = get_object_or_404(
        Post.objects.select_related('author', 'category'), 
        slug=slug, 
        is_published=True
    )
    
    # Increment view count
    Post.objects.filter(id=post.id).update(views_count=F('views_count') + 1)
    
    # Get approved comments
    comments = post.comments.filter(is_approved=True).select_related('author')
    
    context = {
        'post': post,
        'comments': comments,
        'related_posts': Post.objects.filter(
            category=post.category,
            is_published=True
        ).exclude(id=post.id)[:3]
    }
    return render(request, 'blog/post_detail.html', context)

@login_required
def add_comment(request, post_slug):
    """Add comment to a post"""
    if request.method == 'POST':
        post = get_object_or_404(Post, slug=post_slug, is_published=True)
        content = request.POST.get('content', '').strip()
        
        if content:
            Comment.objects.create(
                post=post,
                author=request.user,
                content=content
            )
            messages.success(request, 'Your comment has been submitted for review.')
        else:
            messages.error(request, 'Comment content cannot be empty.')
    
    return redirect('blog:post_detail', slug=post_slug)
```

## Step 6: Templates with Debug Information

Create `templates/base.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Enhanced Django Project{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    {% load static %}
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{% url 'blog:post_list' %}">My Enhanced Blog</a>
            <div class="navbar-nav ms-auto">
                {% if user.is_authenticated %}
                    <span class="navbar-text me-3">Hello, {{ user.username }}!</span>
                    <a class="nav-link" href="{% url 'admin:index' %}">Admin</a>
                {% else %}
                    <a class="nav-link" href="{% url 'admin:index' %}">Login</a>
                {% endif %}
            </div>
        </div>
    </nav>

    <main class="container my-4">
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}

        {% block content %}{% endblock %}
    </main>

    <!-- Development info panel (only in DEBUG mode) -->
    {% if debug %}
    <div class="bg-light border-top p-3 mt-5">
        <div class="container">
            <small class="text-muted">
                <strong>Development Mode:</strong> 
                Django {{ django_version }} | 
                Database queries: <span id="sql-queries">Check debug toolbar</span> |
                Page load time: <span id="page-time">Check debug toolbar</span>
            </small>
        </div>
    </div>
    {% endif %}

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

## Step 7: Management Commands

Create `apps/blog/management/commands/populate_blog.py`:
```python
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from apps.blog.models import Category, Post
import random

User = get_user_model()

class Command(BaseCommand):
    help = 'Populate blog with sample data'
    
    def add_arguments(self, parser):
        parser.add_argument('--posts', type=int, default=20, help='Number of posts to create')
    
    def handle(self, *args, **options):
        """
        Custom management command to populate sample data
        Like preparing ingredients for the kitchen
        """
        self.stdout.write('Creating sample blog data...')
        
        # Create superuser if doesn't exist
        if not User.objects.filter(is_superuser=True).exists():
            User.objects.create_superuser(
                username='admin',
                email='admin@example.com',
                password='admin123'
            )
            self.stdout.write(self.style.SUCCESS('Created superuser: admin/admin123'))
        
        # Create categories
        categories = [
            {'name': 'Technology', 'description': 'Tech-related posts'},
            {'name': 'Lifestyle', 'description': 'Lifestyle and personal posts'},
            {'name': 'Tutorial', 'description': 'How-to guides and tutorials'},
        ]
        
        for cat_data in categories:
            category, created = Category.objects.get_or_create(
                name=cat_data['name'],
                defaults={'description': cat_data['description']}
            )
            if created:
                self.stdout.write(f'Created category: {category.name}')
        
        # Create sample posts
        author = User.objects.filter(is_superuser=True).first()
        sample_posts = [
            {
                'title': 'Getting Started with Django Extensions',
                'content': 'Django extensions provide powerful tools for development...',
                'tags': 'django, python, web development'
            },
            {
                'title': 'Debugging Django Applications',
                'content': 'The debug toolbar is an essential tool for Django developers...',
                'tags': 'django, debugging, development'
            },
            {
                'title': 'Building Better Django Models',
                'content': 'Learn how to create efficient and maintainable Django models...',
                'tags': 'django, models, database'
            },
        ]
        
        for post_data in sample_posts:
            post, created = Post.objects.get_or_create(
                title=post_data['title'],
                defaults={
                    'author': author,
                    'content': post_data['content'],
                    'tags': post_data['tags'],
                    'category': random.choice(Category.objects.all()),
                    'is_published': True,
                }
            )
            if created:
                self.stdout.write(f'Created post: {post.title}')
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully populated blog with sample data!')
        )
```

## Step 8: Custom Settings Management

Create `manage.py` modification for environment-specific settings:
```python
#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    # Set default settings module based on environment
    environment = os.environ.get('DJANGO_ENVIRONMENT', 'development')
    
    if environment == 'production':
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.production')
    else:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
```

## Step 9: Advanced Development Commands

Create `apps/core/management/commands/dev_setup.py`:
```python
from django.core.management.base import BaseCommand
from django.core.management import call_command
import os

class Command(BaseCommand):
    help = 'Set up the complete development environment'
    
    def handle(self, *args, **options):
        """
        One-command setup for the entire development environment
        Like a master chef preparing the entire kitchen in one go
        """
        self.stdout.write('Setting up development environment...')
        
        # Create directories
        directories = ['logs', 'media', 'media/avatars', 'media/posts', 'static/css', 'static/js']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.stdout.write(f'Created directory: {directory}')
        
        # Run migrations
        self.stdout.write('Running migrations...')
        call_command('makemigrations')
        call_command('migrate')
        
        # Collect static files
        self.stdout.write('Collecting static files...')
        call_command('collectstatic', '--noinput')
        
        # Create sample data
        self.stdout.write('Populating with sample data...')
        call_command('populate_blog')
        
        # Show useful commands
        self.stdout.write(
            self.style.SUCCESS('\n✅ Development environment ready!')
        )
        self.stdout.write('\nUseful development commands:')
        self.stdout.write('• python manage.py runserver - Start development server')
        self.stdout.write('• python manage.py shell_plus - Enhanced Django shell')
        self.stdout.write('• python manage.py show_urls - Display all URL patterns')
        self.stdout.write('• python manage.py graph_models -a -o models.png - Generate model diagram')
```

## Step 10: Running and Testing the Enhanced Environment

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up the development environment:**
```bash
python manage.py dev_setup
```

3. **Run the development server:**
```bash
python manage.py runserver
```

4. **Test django-extensions commands:**
```bash
# Enhanced shell with auto-imports
python manage.py shell_plus

# Show all URL patterns
python manage.py show_urls

# Generate model graph (requires graphviz)
python manage.py graph_models -a -o models.png
```

5. **Access the application:**
   - Main site: http://127.0.0.1:8000/
   - Admin panel: http://127.0.0.1:8000/admin/ (admin/admin123)
   - Debug toolbar: Available on all pages in DEBUG mode

## Key Features Implemented

1. **Environment-based settings** - Different configurations for development/production
2. **Debug toolbar integration** - Complete debugging information
3. **Enhanced models** - Using TimeStampedModel and TitleSlugDescriptionModel
4. **Custom management commands** - Automated setup and data population
5. **Advanced admin interface** - Rich admin panels with custom fields
6. **Logging system** - File and console logging
7. **Media handling** - Proper static and media file configuration
8. **Search and filtering** - Blog with search and category filtering
9. **Performance optimization** - Query optimization with select_related
10. **Development tools** - Shell_plus, show_urls, and model graphing

This enhanced development environment provides a solid foundation for Django projects, combining multiple third-party packages into a cohesive, production-ready setup that streamlines the development workflow.