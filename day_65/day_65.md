# Django Deployment Preparation Course - Day 65

## Learning Objective
By the end of this lesson, you will be able to configure a Django application for production deployment by setting up proper production settings, managing environment variables, collecting static files, and configuring databases for a live environment.

---

## Introduction

Imagine that you've been perfecting your signature dish in your home kitchen for months. The recipe is fantastic, your family loves it, and now you're ready to serve it in a professional restaurant. However, cooking for 5 people at home is vastly different from cooking for 200 customers in a bustling restaurant kitchen. You need industrial-grade equipment, proper food storage systems, efficient workflows, and safety protocols.

Similarly, your Django application that works perfectly on your development machine needs significant preparation before it can serve real users in production. Just as a chef must adapt their cooking process for a commercial kitchen, we must configure our Django app for the demands of a production environment.

---

## Lesson 1: Production Settings Configuration

### The Chef's Professional Kitchen Setup

Think of your Django settings like the difference between a home kitchen and a professional restaurant kitchen. At home, you might leave ingredients on the counter, cook with the lights on full brightness, and not worry about someone stealing your secret recipes. But in a professional kitchen, you need:
- Secure storage for valuable ingredients (sensitive data)
- Efficient lighting that doesn't waste energy (optimized performance)
- Restricted access to recipe books (security settings)

### Understanding Settings Configuration

In Django, we separate development and production settings because they serve different purposes:

```python
# settings/base.py - Common settings for all environments
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Common settings that work everywhere
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'myproject.urls'
WSGI_APPLICATION = 'myproject.wsgi.application'
```

```python
# settings/development.py - Home kitchen settings
from .base import *

# Debug mode is like having all the lights on and detailed recipe notes visible
DEBUG = True

# Allow connections from anywhere (like letting neighbors drop by)
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# Simple database for development (like a small home refrigerator)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Simple email backend for testing
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
```

```python
# settings/production.py - Professional kitchen settings
from .base import *
import os

# Security first - no debug info for customers
DEBUG = False

# Only allow specific domains (like having a guest list)
ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']

# Security settings - like having security cameras and locked storage
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# Production database configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}

# Real email service
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('EMAIL_HOST')
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
```

**Syntax Explanation:**
- `from .base import *`: Imports all settings from the base configuration file
- `os.environ.get('VARIABLE_NAME')`: Retrieves environment variables safely
- `SECURE_*` settings: Django's built-in security configurations
- `DEBUG = False`: Disables detailed error pages that could expose sensitive information

---

## Lesson 2: Environment Variables

### The Chef's Secret Recipe Vault

Imagine a chef who writes their secret ingredients on a sticky note and leaves it on the kitchen counter where anyone can see it. That's essentially what happens when you hardcode sensitive information like database passwords directly in your code. Instead, professional chefs keep their secret recipes in a secure vault, accessible only when needed.

Environment variables are like that secure vault - they store sensitive configuration data outside your code.

### Setting Up Environment Variables

```python
# .env file (never commit this to version control!)
SECRET_KEY=your-super-secret-key-here
DEBUG=False
DB_NAME=myapp_production
DB_USER=myapp_user
DB_PASSWORD=super-secure-password
DB_HOST=localhost
DB_PORT=5432
EMAIL_HOST=smtp.gmail.com
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
```

```python
# settings/production.py - Updated to use environment variables
import os
from dotenv import load_dotenv

load_dotenv()

# Now your secrets are safely stored outside the code
SECRET_KEY = os.environ.get('SECRET_KEY')
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# Convert comma-separated string to list
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '').split(',')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}
```

```python
# utils/env_checker.py - A helper to validate required environment variables
import os
import sys

REQUIRED_ENV_VARS = [
    'SECRET_KEY',
    'DB_NAME',
    'DB_USER',
    'DB_PASSWORD',
    'EMAIL_HOST_USER',
    'EMAIL_HOST_PASSWORD',
]

def check_environment():
    """
    Like a chef checking that all essential ingredients are available
    before starting to cook for a big dinner service.
    """
    missing_vars = []
    
    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running in production.")
        sys.exit(1)
    else:
        print("✅ All required environment variables are set!")

if __name__ == "__main__":
    check_environment()
```

**Syntax Explanation:**
- `load_dotenv()`: Loads variables from a .env file into the environment
- `os.environ.get('VAR', 'default')`: Gets environment variable with optional default value
- `.split(',')`: Converts comma-separated string into a Python list
- `sys.exit(1)`: Exits the program with an error code if environment check fails

---

## Lesson 3: Static File Collection

### The Chef's Ingredient Prep Station

In a home kitchen, you might grab ingredients from different cabinets as you cook. But in a professional kitchen, chefs have a prep station where all ingredients for a dish are organized and ready to go. This is called "mise en place" - everything in its place.

Django's static file collection is like setting up your mise en place. It gathers all your CSS, JavaScript, images, and other static files from various locations and puts them in one organized place where your web server can efficiently serve them to users.

### Configuring Static Files

```python
# settings/production.py - Static files configuration
import os
from .base import *

# Static files settings - like organizing your prep station
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Additional locations to collect static files from
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# Static files storage - like having an efficient storage system
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'

# Media files (user uploads) - separate from static files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# For cloud storage (AWS S3, Google Cloud, etc.)
# DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
# AWS_STORAGE_BUCKET_NAME = os.environ.get('AWS_STORAGE_BUCKET_NAME')
```

```python
# management/commands/check_static.py - Custom management command
from django.core.management.base import BaseCommand
from django.conf import settings
import os

class Command(BaseCommand):
    help = 'Check static files configuration like a chef checking prep station'

    def handle(self, *args, **options):
        self.stdout.write("🔍 Checking static files configuration...")
        
        # Check if STATIC_ROOT directory exists
        if os.path.exists(settings.STATIC_ROOT):
            file_count = sum([len(files) for r, d, files in os.walk(settings.STATIC_ROOT)])
            self.stdout.write(
                self.style.SUCCESS(f"✅ STATIC_ROOT exists with {file_count} files")
            )
        else:
            self.stdout.write(
                self.style.WARNING("⚠️  STATIC_ROOT directory doesn't exist yet")
            )
            
        # Check STATICFILES_DIRS
        for static_dir in settings.STATICFILES_DIRS:
            if os.path.exists(static_dir):
                self.stdout.write(
                    self.style.SUCCESS(f"✅ Static directory found: {static_dir}")
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f"❌ Static directory missing: {static_dir}")
                )
```

```bash
# collectstatic.sh - Shell script to collect static files
#!/bin/bash
echo "🍽️  Preparing static files for production service..."

# Like a chef doing final prep before dinner service
python manage.py collectstatic --noinput --clear

echo "✅ Static files are ready to serve!"

# Check the results
python manage.py check_static
```

**Syntax Explanation:**
- `STATIC_ROOT`: The absolute path where collected static files will be stored
- `STATICFILES_DIRS`: List of additional directories to search for static files
- `--noinput --clear`: Command flags that run collectstatic without prompting and clear existing files
- `os.path.join()`: Safely joins file paths regardless of operating system

---

## Lesson 4: Database Configuration

### From Home Kitchen to Restaurant Database

Think of your development database like a small notebook where you jot down recipes and grocery lists. It works fine for personal use, but when you're running a restaurant, you need a professional inventory management system that can handle hundreds of orders simultaneously, track ingredients precisely, and never lose data even if the power goes out.

### Production Database Setup

```python
# settings/production.py - Production database configuration
import os
import dj_database_url
from .base import *

# Primary database configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
        'OPTIONS': {
            'sslmode': 'require',  # Secure connection like encrypted communication
        },
        'CONN_MAX_AGE': 600,  # Connection pooling - like keeping prep stations ready
    }
}

# Alternative: Using DATABASE_URL (common in cloud deployments)
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    DATABASES['default'] = dj_database_url.parse(DATABASE_URL)

# Database backup configuration
BACKUP_DATABASE = {
    'ENGINE': 'django.db.backends.postgresql',
    'NAME': os.environ.get('BACKUP_DB_NAME'),
    'USER': os.environ.get('BACKUP_DB_USER'),
    'PASSWORD': os.environ.get('BACKUP_DB_PASSWORD'),
    'HOST': os.environ.get('BACKUP_DB_HOST'),
    'PORT': os.environ.get('BACKUP_DB_PORT', '5432'),
}
```

```python
# utils/db_health.py - Database health checker
from django.core.management.base import BaseCommand
from django.db import connections
from django.db.utils import OperationalError
import time

class Command(BaseCommand):
    help = 'Check database connectivity like a chef testing all equipment'

    def add_arguments(self, parser):
        parser.add_argument(
            '--timeout',
            type=int,
            default=30,
            help='Maximum seconds to wait for database connection',
        )

    def handle(self, *args, **options):
        timeout = options['timeout']
        start_time = time.time()
        
        self.stdout.write("🔍 Testing database connection...")
        
        while True:
            try:
                # Test database connection
                db_conn = connections['default']
                db_conn.cursor()
                
                self.stdout.write(
                    self.style.SUCCESS("✅ Database connection successful!")
                )
                
                # Test a simple query
                from django.contrib.auth.models import User
                user_count = User.objects.count()
                self.stdout.write(f"📊 Database has {user_count} users")
                
                break
                
            except OperationalError as e:
                current_time = time.time()
                if current_time - start_time > timeout:
                    self.stdout.write(
                        self.style.ERROR(f"❌ Database connection failed after {timeout}s")
                    )
                    self.stdout.write(f"Error: {e}")
                    return
                
                self.stdout.write("⏳ Waiting for database...")
                time.sleep(2)
```

```python
# models/monitoring.py - Database performance monitoring
from django.db import models
from django.utils import timezone

class DatabaseHealthCheck(models.Model):
    """
    Like a chef's daily equipment check log
    """
    timestamp = models.DateTimeField(default=timezone.now)
    connection_time = models.FloatField()  # Time to connect in seconds
    query_time = models.FloatField()       # Time for test query in seconds
    status = models.CharField(max_length=20, choices=[
        ('healthy', 'Healthy'),
        ('slow', 'Slow'),
        ('error', 'Error'),
    ])
    error_message = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"DB Health Check - {self.timestamp.strftime('%Y-%m-%d %H:%M')} - {self.status}"

    @classmethod
    def perform_health_check(cls):
        """
        Perform a health check like a chef testing all burners before service
        """
        import time
        from django.db import connection
        
        start_time = time.time()
        
        try:
            # Test connection
            connection.cursor()
            connection_time = time.time() - start_time
            
            # Test query
            query_start = time.time()
            cls.objects.count()
            query_time = time.time() - query_start
            
            # Determine status
            if connection_time > 2 or query_time > 1:
                status = 'slow'
            else:
                status = 'healthy'
            
            # Save health check result
            return cls.objects.create(
                connection_time=connection_time,
                query_time=query_time,
                status=status
            )
            
        except Exception as e:
            return cls.objects.create(
                connection_time=0,
                query_time=0,
                status='error',
                error_message=str(e)
            )
```

**Syntax Explanation:**
- `dj_database_url.parse()`: Parses a database URL string into Django database configuration
- `CONN_MAX_AGE`: Keeps database connections alive for specified seconds to improve performance
- `sslmode='require'`: Forces encrypted connections to the database
- `@classmethod`: Creates a method that can be called on the class itself, not just instances
- `timezone.now()`: Django's timezone-aware current datetime function

---

# **Build**: Production-Ready Configuration Project

## Project Overview
Create a complete production-ready Django application configuration that demonstrates all the concepts from your deployment preparation lessons. Think of this as preparing your kitchen for a grand opening - everything must be perfectly organized, secure, and ready to serve hundreds of customers.

## Project Requirements

### 1. Create the Production Configuration Structure

```python
# settings/production.py
import os
from .base import *

# Debug must be False in production
DEBUG = False

# Allowed hosts for your production domain
ALLOWED_HOSTS = [
    'yourdomain.com',
    'www.yourdomain.com',
    os.environ.get('ALLOWED_HOST', 'localhost')
]

# Database configuration for production
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME', 'production_db'),
        'USER': os.environ.get('DB_USER', 'postgres'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}

# Security settings
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = 'DENY'

# Static files configuration
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Email configuration
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('EMAIL_HOST')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')

# Logging configuration
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
            'filename': 'django.log',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['file'],
    },
}
```

### 2. Environment Variables Configuration

```python
# .env.example (template for production environment)
SECRET_KEY=your-super-secret-key-here
DEBUG=False
DB_NAME=your_production_database
DB_USER=your_db_user
DB_PASSWORD=your_secure_password
DB_HOST=your_db_host
DB_PORT=5432
ALLOWED_HOST=yourdomain.com
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
```

### 3. Production-Ready Django Application

```python
# restaurant_app/models.py
from django.db import models
from django.contrib.auth.models import User

class Restaurant(models.Model):
    name = models.CharField(max_length=200)
    address = models.TextField()
    phone = models.CharField(max_length=20)
    email = models.EmailField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class MenuItem(models.Model):
    CATEGORY_CHOICES = [
        ('appetizer', 'Appetizer'),
        ('main', 'Main Course'),
        ('dessert', 'Dessert'),
        ('beverage', 'Beverage'),
    ]
    
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=8, decimal_places=2)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    is_available = models.BooleanField(default=True)
    image = models.ImageField(upload_to='menu_items/', blank=True, null=True)
    
    def __str__(self):
        return f"{self.name} - ${self.price}"

class Order(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('preparing', 'Preparing'),
        ('ready', 'Ready'),
        ('delivered', 'Delivered'),
    ]
    
    customer_name = models.CharField(max_length=200)
    customer_email = models.EmailField()
    items = models.ManyToManyField(MenuItem, through='OrderItem')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Order #{self.id} - {self.customer_name}"

class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    
    def get_total_price(self):
        return self.quantity * self.menu_item.price
```

### 4. Production Views with Error Handling

```python
# restaurant_app/views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import ListView
import logging
import json

logger = logging.getLogger(__name__)

class MenuListView(ListView):
    model = MenuItem
    template_name = 'restaurant/menu.html'
    context_object_name = 'menu_items'
    
    def get_queryset(self):
        return MenuItem.objects.filter(is_available=True).select_related('restaurant')

def place_order(request):
    if request.method == 'POST':
        try:
            customer_name = request.POST.get('customer_name')
            customer_email = request.POST.get('customer_email')
            selected_items = request.POST.getlist('menu_items')
            
            # Create order
            order = Order.objects.create(
                customer_name=customer_name,
                customer_email=customer_email,
                total_amount=0  # Will calculate below
            )
            
            total = 0
            for item_id in selected_items:
                menu_item = get_object_or_404(MenuItem, id=item_id)
                quantity = int(request.POST.get(f'quantity_{item_id}', 1))
                
                OrderItem.objects.create(
                    order=order,
                    menu_item=menu_item,
                    quantity=quantity
                )
                total += menu_item.price * quantity
            
            order.total_amount = total
            order.save()
            
            # Send confirmation email
            try:
                send_mail(
                    'Order Confirmation',
                    f'Thank you {customer_name}! Your order #{order.id} has been placed.',
                    settings.DEFAULT_FROM_EMAIL,
                    [customer_email],
                    fail_silently=False,
                )
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
            
            messages.success(request, f'Order #{order.id} placed successfully!')
            return redirect('order_success', order_id=order.id)
            
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            messages.error(request, 'Order placement failed. Please try again.')
            return redirect('menu')
    
    return redirect('menu')

def order_success(request, order_id):
    order = get_object_or_404(Order, id=order_id)
    return render(request, 'restaurant/order_success.html', {'order': order})

@csrf_exempt
def api_menu(request):
    """API endpoint for menu items - production ready with error handling"""
    try:
        menu_items = MenuItem.objects.filter(is_available=True).values(
            'id', 'name', 'description', 'price', 'category'
        )
        return JsonResponse({
            'status': 'success',
            'data': list(menu_items)
        })
    except Exception as e:
        logger.error(f"API menu error: {e}")
        return JsonResponse({
            'status': 'error',
            'message': 'Failed to fetch menu items'
        }, status=500)
```

### 5. Production Templates

```html
<!-- templates/restaurant/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Restaurant Management{% endblock %}</title>
    {% load static %}
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{% url 'menu' %}">🍽️ Chef's Kitchen</a>
        </div>
    </nav>
    
    <main class="container mt-4">
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}
        
        {% block content %}
        {% endblock %}
    </main>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

```html
<!-- templates/restaurant/menu.html -->
{% extends 'restaurant/base.html' %}

{% block title %}Menu - Chef's Kitchen{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <h2>Our Menu</h2>
        <form method="post" action="{% url 'place_order' %}">
            {% csrf_token %}
            <div class="row">
                {% for item in menu_items %}
                <div class="col-md-6 mb-4">
                    <div class="card">
                        {% if item.image %}
                            <img src="{{ item.image.url }}" class="card-img-top" alt="{{ item.name }}">
                        {% endif %}
                        <div class="card-body">
                            <h5 class="card-title">{{ item.name }}</h5>
                            <p class="card-text">{{ item.description }}</p>
                            <p class="card-text"><strong>${{ item.price }}</strong></p>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" name="menu_items" value="{{ item.id }}" id="item{{ item.id }}">
                                <label class="form-check-label" for="item{{ item.id }}">
                                    Add to order
                                </label>
                            </div>
                            <input type="number" name="quantity_{{ item.id }}" value="1" min="1" class="form-control mt-2" style="width: 80px;">
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <div class="mt-4">
                <h4>Customer Information</h4>
                <div class="row">
                    <div class="col-md-6">
                        <input type="text" name="customer_name" class="form-control" placeholder="Your Name" required>
                    </div>
                    <div class="col-md-6">
                        <input type="email" name="customer_email" class="form-control" placeholder="Your Email" required>
                    </div>
                </div>
                <button type="submit" class="btn btn-primary btn-lg mt-3">Place Order</button>
            </div>
        </form>
    </div>
</div>
{% endblock %}
```

### 6. Production URL Configuration

```python
# restaurant_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.MenuListView.as_view(), name='menu'),
    path('order/', views.place_order, name='place_order'),
    path('order/success/<int:order_id>/', views.order_success, name='order_success'),
    path('api/menu/', views.api_menu, name='api_menu'),
]
```

### 7. Production Deployment Scripts

```python
# manage.py modifications for production
#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    # Set default settings module based on environment
    if 'runserver' in sys.argv:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')
    else:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.production')
    
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

```bash
# deploy.sh - Production deployment script
#!/bin/bash
set -e

echo "🚀 Starting production deployment..."

# Load environment variables
source .env

# Install dependencies
pip install -r requirements.txt

# Collect static files
echo "📦 Collecting static files..."
python manage.py collectstatic --noinput --settings=config.settings.production

# Run database migrations
echo "🗄️ Running database migrations..."
python manage.py migrate --settings=config.settings.production

# Create superuser if it doesn't exist
echo "👤 Creating superuser..."
python manage.py shell --settings=config.settings.production << EOF
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', '$ADMIN_PASSWORD')
    print('Superuser created')
else:
    print('Superuser already exists')
EOF

# Test the deployment
echo "🧪 Running production checks..."
python manage.py check --deploy --settings=config.settings.production

echo "✅ Deployment completed successfully!"
echo "🍽️ Your restaurant kitchen is ready to serve customers!"
```

### 8. Requirements File for Production

```txt
# requirements.txt
Django==4.2.7
psycopg2-binary==2.9.7
python-decouple==3.8
whitenoise==6.5.0
gunicorn==21.2.0
Pillow==10.0.1
django-cors-headers==4.3.1
```

## Project Deliverables

Your production-ready configuration should include:

1. **Complete settings structure** with separate production configuration
2. **Environment variables** properly configured and documented
3. **Static files** collection and serving setup
4. **Database** configuration for production use
5. **Security settings** implemented and tested
6. **Error handling** and logging configured
7. **Email functionality** for order confirmations
8. **API endpoints** with proper error responses
9. **Deployment scripts** for automated deployment
10. **Production testing** with Django's deployment checklist

## Testing Your Production Setup

Run these commands to verify your configuration:

```bash
# Test production settings
python manage.py check --deploy --settings=config.settings.production

# Test static file collection
python manage.py collectstatic --noinput --settings=config.settings.production

# Test database connection
python manage.py showmigrations --settings=config.settings.production

# Test the application
python manage.py runserver --settings=config.settings.production
```

Just like a master chef preparing for a restaurant's grand opening, your Django application is now configured with all the production essentials - security, performance, reliability, and scalability. Every setting has been carefully chosen to ensure your digital kitchen can handle the demands of real-world traffic while keeping your data and users safe.

## Assignment: Environment Configuration Audit

### The Restaurant Health Inspector Challenge

You are a "Django Health Inspector" tasked with auditing a restaurant's (Django application's) readiness for production. Create a comprehensive audit tool that checks all the deployment preparation elements we've covered.

**Your Mission:**
Create a Django management command called `deployment_audit` that performs the following checks:

1. **Settings Audit**: Verify that production settings are properly configured
2. **Environment Variables Check**: Ensure all required environment variables are set
3. **Static Files Verification**: Confirm static files are properly configured and collected
4. **Database Health Test**: Test database connectivity and performance

**Requirements:**

```python
# management/commands/deployment_audit.py
from django.core.management.base import BaseCommand
from django.conf import settings
import os
import time

class Command(BaseCommand):
    help = 'Audit Django application deployment readiness'

    def handle(self, *args, **options):
        self.stdout.write("🔍 Starting Django Deployment Audit...")
        
        # Your code here - implement all four audit categories
        # Each should return a score and detailed feedback
        
        # Calculate and display overall readiness score
        pass
```

**Deliverables:**
1. A complete `deployment_audit.py` management command
2. A configuration file that defines what should be checked in each category
3. A colored output system (green for pass, yellow for warning, red for fail)
4. An overall "deployment readiness score" from 0-100

**Bonus Points:**
- Add the ability to export audit results to a JSON file
- Include suggestions for fixing any issues found
- Add a `--fix` flag that automatically resolves simple issues

This assignment is different from a typical "build a production configuration" project because it focuses on creating a diagnostic tool rather than just configuring settings. You'll need to think like both a developer (understanding the technical requirements) and a quality assurance engineer (creating comprehensive tests).

The audit tool should be something you could run before every deployment to ensure your application is production-ready - like a pre-flight checklist for pilots or a mise en place verification for chefs.