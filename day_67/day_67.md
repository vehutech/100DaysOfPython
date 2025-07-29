# Django Day 67: Cloud Deployment - From Kitchen to Cloud

## Learning Objective
By the end of this lesson, you will be able to deploy a Django application to the cloud, configure database hosting, set up static file serving, and secure your application with a custom domain and SSL certificate - transforming your local kitchen into a world-class restaurant accessible to everyone.

---

## Imagine That...

Imagine you've been perfecting your culinary skills in your home kitchen for months. You've created the most amazing restaurant management system - your Django application. Your recipes are perfect, your ingredients are fresh, and your cooking process is flawless. But here's the thing: no matter how incredible your food is, if customers can't find your restaurant or if you're only cooking for yourself at home, your culinary genius remains hidden from the world.

Cloud deployment is like taking your amazing home kitchen and opening it as a real restaurant in the heart of the city. You need to find the perfect location (cloud provider), set up professional equipment (server configuration), establish supply chains for fresh ingredients (database hosting), create an attractive storefront (static files), and put up a proper sign with your restaurant's name (domain and SSL). 

Let's transform your home kitchen into a world-renowned restaurant that serves customers 24/7!

---

## 1. Deploying to Cloud Platforms (Setting Up Your Restaurant Location)

### The Chef's Perspective
Just like choosing between opening your restaurant in a busy downtown area (AWS), a trendy neighborhood (Digital Ocean), or a food court (Heroku), each cloud platform offers different advantages for your Django kitchen.

### Deploying to Heroku (The Food Court Approach)

Heroku is like renting space in a well-managed food court - they handle most of the infrastructure while you focus on cooking.

**Step 1: Prepare Your Kitchen for the Move**

First, let's create the essential files your restaurant needs:

```python
# requirements.txt - Your ingredient list
Django==4.2.7
gunicorn==21.2.0
psycopg2-binary==2.9.7
whitenoise==6.6.0
python-decouple==3.8
```

```python
# Procfile - Your restaurant's operating instructions
web: gunicorn restaurant_project.wsgi:application
```

```python
# runtime.txt - Specify your cooking equipment version
python-3.11.5
```

**Step 2: Update Your Kitchen Settings**

```python
# settings.py - Your restaurant's configuration
import os
from decouple import config
import dj_database_url

# Your restaurant can operate in different environments
DEBUG = config('DEBUG', default=False, cast=bool)
ALLOWED_HOSTS = ['your-restaurant.herokuapp.com', 'localhost', '127.0.0.1']

# Database configuration - like setting up your refrigeration system
DATABASES = {
    'default': dj_database_url.config(
        default=config('DATABASE_URL', default='sqlite:///db.sqlite3')
    )
}

# Static files configuration - your restaurant's presentation materials
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Middleware for serving static files - like having a dedicated server
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # Your file server
    'django.contrib.sessions.middleware.SessionMiddleware',
    # ... other middleware
]

# Security settings - your restaurant's safety protocols
SECURE_SSL_REDIRECT = config('SECURE_SSL_REDIRECT', default=False, cast=bool)
SECURE_HSTS_SECONDS = config('SECURE_HSTS_SECONDS', default=0, cast=int)
```

**Step 3: Deploy Your Restaurant**

```bash
# Install Heroku CLI (your restaurant management tools)
# Create a new restaurant location
heroku create your-restaurant-name

# Set up your restaurant's environment variables (secret recipes)
heroku config:set DEBUG=False
heroku config:set SECRET_KEY='your-super-secret-key'

# Move your kitchen to the cloud
git add .
git commit -m "Ready to open our restaurant to the world"
git push heroku main

# Set up your dining area (run migrations)
heroku run python manage.py migrate
heroku run python manage.py collectstatic --noinput
```

### Deploying to Digital Ocean (The Trendy Neighborhood)

Digital Ocean is like renting a space in a hip neighborhood where you have more control but need to set up more infrastructure yourself.

```bash
# Create a droplet (rent your restaurant space)
# Connect to your server (enter your restaurant)
ssh root@your-server-ip

# Set up your kitchen environment
apt update && apt upgrade -y
apt install python3 python3-pip nginx postgresql postgresql-contrib

# Create your head chef user account
adduser restaurant_chef
usermod -aG sudo restaurant_chef
su - restaurant_chef

# Clone your restaurant recipes
git clone https://github.com/yourusername/your-restaurant.git
cd your-restaurant

# Set up your virtual kitchen environment
python3 -m venv restaurant_env
source restaurant_env/bin/activate
pip install -r requirements.txt
```

**Syntax Explanation:**
- `ssh`: Secure Shell - like having a secure key to enter your restaurant
- `apt update`: Updates the list of available software packages
- `usermod -aG sudo`: Adds user to admin group (makes them head chef)
- `source restaurant_env/bin/activate`: Activates virtual environment (enters your specialized kitchen)

---

## 2. Database Hosting (Setting Up Your Professional Refrigeration System)

### The Chef's Perspective
Moving from SQLite to MySQL/PostgreSQL is like upgrading from a home refrigerator to a professional-grade refrigeration system that can handle multiple chefs working simultaneously and never runs out of space.

### Setting Up MySQL on Cloud

```python
# Database configuration for production kitchen
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': config('DB_NAME'),
        'USER': config('DB_USER'),
        'PASSWORD': config('DB_PASSWORD'),
        'HOST': config('DB_HOST'),  # Your cloud database address
        'PORT': config('DB_PORT', default='3306'),
        'OPTIONS': {
            'sql_mode': 'traditional',  # Strict quality control
        }
    }
}

# Connection pooling - like having multiple refrigerator doors
DATABASES['default']['CONN_MAX_AGE'] = 60
```

### Environment Variables (.env file - Your Secret Recipe Book)

```bash
# .env - Keep your secret ingredients safe
DB_NAME=restaurant_production
DB_USER=head_chef
DB_PASSWORD=super_secret_sauce
DB_HOST=mysql-server.amazonaws.com
DB_PORT=3306
SECRET_KEY=your-django-secret-key
DEBUG=False
```

**Syntax Explanation:**
- `CONN_MAX_AGE`: Keeps database connections alive for 60 seconds to improve performance
- `sql_mode='traditional'`: Makes MySQL stricter about data validation
- Environment variables: Like having a locked recipe book that only authorized chefs can read

---

## 3. Static File Serving (Creating Your Restaurant's Visual Identity)

### The Chef's Perspective
Static files are like your restaurant's visual presentation - the beautiful plating, elegant menus, stylish uniforms, and attractive storefront that make customers want to dine with you.

### Setting Up AWS S3 for Static Files

```python
# Install the storage backend
# pip install django-storages boto3

# settings.py - Configure your presentation materials storage
import boto3
from botocore.exceptions import ClientError

# AWS S3 Configuration - Your professional photography studio
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
AWS_STORAGE_BUCKET_NAME = config('AWS_STORAGE_BUCKET_NAME')
AWS_S3_REGION_NAME = config('AWS_S3_REGION_NAME', default='us-east-1')

# Custom storage classes - Different areas of your restaurant
DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
STATICFILES_STORAGE = 'storages.backends.s3boto3.StaticS3Boto3Storage'

# File serving configuration
AWS_S3_CUSTOM_DOMAIN = f'{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com'
AWS_S3_OBJECT_PARAMETERS = {
    'CacheControl': 'max-age=86400',  # Cache for 24 hours (like daily specials)
}

# URLs for your presentation materials
STATIC_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/static/'
MEDIA_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/media/'
```

### CloudFront CDN Setup (Your Restaurant Chain)

```python
# CloudFront configuration - Opening branches worldwide
AWS_CLOUDFRONT_DOMAIN = config('AWS_CLOUDFRONT_DOMAIN', default=None)

if AWS_CLOUDFRONT_DOMAIN:
    # Use CloudFront URLs instead of direct S3
    STATIC_URL = f'https://{AWS_CLOUDFRONT_DOMAIN}/static/'
    MEDIA_URL = f'https://{AWS_CLOUDFRONT_DOMAIN}/media/'
```

**Syntax Explanation:**
- `CacheControl`: Tells browsers how long to keep files cached (like telling customers they can keep the menu for a day)
- `DEFAULT_FILE_STORAGE`: Where user uploads go (customer photos)
- `STATICFILES_STORAGE`: Where your CSS, JS, images go (restaurant's official materials)
- CloudFront: A Content Delivery Network that serves your files from locations closer to users

---

## 4. Domain and SSL Setup (Your Restaurant's Official Address and Security)

### The Chef's Perspective
Getting a custom domain and SSL is like getting your official business license, putting up a professional sign, and installing security systems. It makes customers trust your establishment and ensures their information is safe.

### Domain Configuration

```python
# settings.py - Your restaurant's official address
ALLOWED_HOSTS = [
    'www.your-restaurant.com',
    'your-restaurant.com',
    'your-restaurant.herokuapp.com',  # Keep backup address
]

# Security middleware - Your restaurant's security protocols
SECURE_SSL_REDIRECT = True  # Always use HTTPS (secure entrance)
SECURE_HSTS_SECONDS = 31536000  # Remember to use HTTPS for a year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# Cookie security - Protect customer information
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
```

### SSL Certificate Setup with Let's Encrypt

```bash
# Install Certbot (your security certificate manager)
sudo apt install certbot python3-certbot-nginx

# Get your security certificate
sudo certbot --nginx -d your-restaurant.com -d www.your-restaurant.com

# Nginx configuration for your restaurant
# /etc/nginx/sites-available/your-restaurant
server {
    listen 80;
    server_name your-restaurant.com www.your-restaurant.com;
    return 301 https://$server_name$request_uri;  # Redirect to secure entrance
}

server {
    listen 443 ssl;
    server_name your-restaurant.com www.your-restaurant.com;
    
    # SSL configuration (security certificates)
    ssl_certificate /etc/letsencrypt/live/your-restaurant.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-restaurant.com/privkey.pem;
    
    # Restaurant application
    location / {
        proxy_pass http://127.0.0.1:8000;  # Your kitchen
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Static files (presentation materials)
    location /static/ {
        alias /home/restaurant_chef/your-restaurant/staticfiles/;
    }
}
```

**Syntax Explanation:**
- `SECURE_SSL_REDIRECT`: Automatically redirects HTTP to HTTPS
- `SECURE_HSTS_SECONDS`: Tells browsers to only use HTTPS for this duration
- `proxy_pass`: Forwards requests to your Django application
- `ssl_certificate`: Points to your security certificate files
- `return 301`: Permanent redirect status code

### Environment-Specific Settings

```python
# Create different settings for different environments
# settings/production.py
from .base import *

DEBUG = False
ALLOWED_HOSTS = ['your-restaurant.com', 'www.your-restaurant.com']

# Production database (professional refrigeration)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('PROD_DB_NAME'),
        'USER': config('PROD_DB_USER'),
        'PASSWORD': config('PROD_DB_PASSWORD'),
        'HOST': config('PROD_DB_HOST'),
        'PORT': config('PROD_DB_PORT', default='5432'),
    }
}

# Production logging (restaurant's record keeping)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '/var/log/restaurant/django.log',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
```

---
# Django Cloud Deployment - Final Project
## **Build**: Live Deployed Application

### Project Overview
You'll deploy a complete Django restaurant management system to the cloud, making it accessible to real users worldwide. This project combines all your Django skills into a production-ready application.

### Project: "CloudChef Restaurant Manager"

A comprehensive restaurant management system with the following features:

#### Core Features to Implement:

1. **Menu Management System**
```python
# models.py
from django.db import models
from django.contrib.auth.models import User

class Category(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Categories"
    
    def __str__(self):
        return self.name

class MenuItem(models.Model):
    AVAILABILITY_CHOICES = [
        ('available', 'Available'),
        ('unavailable', 'Unavailable'),
        ('seasonal', 'Seasonal'),
    ]
    
    name = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=8, decimal_places=2)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='menu_items/', blank=True, null=True)
    availability = models.CharField(max_length=20, choices=AVAILABILITY_CHOICES, default='available')
    preparation_time = models.PositiveIntegerField(help_text="Time in minutes")
    ingredients = models.TextField(help_text="Comma-separated list")
    is_featured = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} - ${self.price}"
```

2. **Order Management System**
```python
# models.py (continued)
class Order(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('preparing', 'Preparing'),
        ('ready', 'Ready'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
    ]
    
    customer = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    table_number = models.PositiveIntegerField(blank=True, null=True)
    special_instructions = models.TextField(blank=True)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def calculate_total(self):
        total = sum(item.subtotal for item in self.orderitem_set.all())
        self.total_amount = total
        self.save()
        return total
    
    def __str__(self):
        return f"Order #{self.id} - {self.customer.username}"

class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    unit_price = models.DecimalField(max_digits=8, decimal_places=2)
    subtotal = models.DecimalField(max_digits=8, decimal_places=2)
    
    def save(self, *args, **kwargs):
        self.unit_price = self.menu_item.price
        self.subtotal = self.quantity * self.unit_price
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.quantity}x {self.menu_item.name}"
```

3. **Customer Management**
```python
# models.py (continued)
class CustomerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.CharField(max_length=15, blank=True)
    address = models.TextField(blank=True)
    loyalty_points = models.PositiveIntegerField(default=0)
    preferred_table = models.PositiveIntegerField(blank=True, null=True)
    dietary_restrictions = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def add_loyalty_points(self, amount):
        # Add 1 point for every $10 spent
        points = int(amount // 10)
        self.loyalty_points += points
        self.save()
        return points
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
```

4. **Views Implementation**
```python
# views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.core.paginator import Paginator
from .models import MenuItem, Category, Order, OrderItem, CustomerProfile
from .forms import OrderForm, MenuItemForm

def menu_view(request):
    categories = Category.objects.filter(is_active=True)
    category_filter = request.GET.get('category')
    
    if category_filter:
        menu_items = MenuItem.objects.filter(
            category_id=category_filter, 
            availability='available'
        )
    else:
        menu_items = MenuItem.objects.filter(availability='available')
    
    # Pagination
    paginator = Paginator(menu_items, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'categories': categories,
        'page_obj': page_obj,
        'current_category': category_filter
    }
    return render(request, 'restaurant/menu.html', context)

@login_required
def place_order(request):
    if request.method == 'POST':
        # Create new order
        order = Order.objects.create(
            customer=request.user,
            table_number=request.POST.get('table_number'),
            special_instructions=request.POST.get('special_instructions')
        )
        
        # Process cart items from session
        cart = request.session.get('cart', {})
        
        for item_id, quantity in cart.items():
            menu_item = get_object_or_404(MenuItem, id=item_id)
            OrderItem.objects.create(
                order=order,
                menu_item=menu_item,
                quantity=quantity
            )
        
        # Calculate total and clear cart
        order.calculate_total()
        request.session['cart'] = {}
        
        # Add loyalty points
        profile, created = CustomerProfile.objects.get_or_create(user=request.user)
        points_earned = profile.add_loyalty_points(order.total_amount)
        
        messages.success(request, f'Order placed successfully! You earned {points_earned} loyalty points.')
        return redirect('order_confirmation', order_id=order.id)
    
    return render(request, 'restaurant/place_order.html')

def add_to_cart(request, item_id):
    if request.method == 'POST':
        cart = request.session.get('cart', {})
        quantity = int(request.POST.get('quantity', 1))
        
        if str(item_id) in cart:
            cart[str(item_id)] += quantity
        else:
            cart[str(item_id)] = quantity
        
        request.session['cart'] = cart
        request.session.modified = True
        
        return JsonResponse({'success': True, 'cart_count': sum(cart.values())})
    
    return JsonResponse({'success': False})

@login_required
def order_history(request):
    orders = Order.objects.filter(customer=request.user).order_by('-created_at')
    paginator = Paginator(orders, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'restaurant/order_history.html', {'page_obj': page_obj})

def dashboard_view(request):
    if not request.user.is_staff:
        return redirect('menu')
    
    # Dashboard statistics
    from django.db.models import Count, Sum
    from datetime import datetime, timedelta
    
    today = datetime.now().date()
    last_30_days = today - timedelta(days=30)
    
    stats = {
        'total_orders_today': Order.objects.filter(created_at__date=today).count(),
        'pending_orders': Order.objects.filter(status='pending').count(),
        'revenue_today': Order.objects.filter(
            created_at__date=today
        ).aggregate(Sum('total_amount'))['total_amount__sum'] or 0,
        'popular_items': MenuItem.objects.annotate(
            order_count=Count('orderitem')
        ).order_by('-order_count')[:5]
    }
    
    return render(request, 'restaurant/dashboard.html', {'stats': stats})
```

5. **Forms Implementation**
```python
# forms.py
from django import forms
from .models import Order, MenuItem, CustomerProfile

class OrderForm(forms.ModelForm):
    class Meta:
        model = Order
        fields = ['table_number', 'special_instructions']
        widgets = {
            'table_number': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Table Number'
            }),
            'special_instructions': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Any special requests...'
            })
        }

class MenuItemForm(forms.ModelForm):
    class Meta:
        model = MenuItem
        fields = ['name', 'description', 'price', 'category', 'image', 
                 'availability', 'preparation_time', 'ingredients']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
            'ingredients': forms.Textarea(attrs={'rows': 3}),
            'price': forms.NumberInput(attrs={'step': '0.01'})
        }

class CustomerProfileForm(forms.ModelForm):
    class Meta:
        model = CustomerProfile
        fields = ['phone', 'address', 'preferred_table', 'dietary_restrictions']
        widgets = {
            'address': forms.Textarea(attrs={'rows': 3}),
            'dietary_restrictions': forms.Textarea(attrs={'rows': 2})
        }
```

6. **URL Configuration**
```python
# urls.py
from django.urls import path
from . import views

app_name = 'restaurant'

urlpatterns = [
    path('', views.menu_view, name='menu'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('order/', views.place_order, name='place_order'),
    path('add-to-cart/<int:item_id>/', views.add_to_cart, name='add_to_cart'),
    path('cart/', views.cart_view, name='cart'),
    path('orders/', views.order_history, name='order_history'),
    path('orders/<int:order_id>/', views.order_detail, name='order_detail'),
    path('profile/', views.profile_view, name='profile'),
    path('api/update-order-status/', views.update_order_status, name='update_order_status'),
]
```

7. **Template Structure**
```html
<!-- templates/restaurant/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}CloudChef Restaurant{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    {% load static %}
    <link href="{% static 'css/style.css' %}" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{% url 'restaurant:menu' %}">
                <i class="fas fa-utensils"></i> CloudChef
            </a>
            <div class="navbar-nav ms-auto">
                {% if user.is_authenticated %}
                    <a class="nav-link" href="{% url 'restaurant:cart' %}">
                        <i class="fas fa-shopping-cart"></i> Cart
                        <span class="badge bg-primary" id="cart-count">0</span>
                    </a>
                    <a class="nav-link" href="{% url 'restaurant:order_history' %}">Orders</a>
                    {% if user.is_staff %}
                        <a class="nav-link" href="{% url 'restaurant:dashboard' %}">Dashboard</a>
                    {% endif %}
                    <a class="nav-link" href="{% url 'logout' %}">Logout</a>
                {% else %}
                    <a class="nav-link" href="{% url 'login' %}">Login</a>
                {% endif %}
            </div>
        </div>
    </nav>

    <main class="container my-4">
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}

        {% block content %}
        {% endblock %}
    </main>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{% static 'js/app.js' %}"></script>
</body>
</html>
```

8. **Settings Configuration for Production**
```python
# settings.py (production additions)
import os
from pathlib import Path

# Security settings for production
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost').split(',')

# Database configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': os.environ.get('DB_NAME', 'cloudchef_db'),
        'USER': os.environ.get('DB_USER', 'root'),
        'PASSWORD': os.environ.get('DB_PASSWORD', ''),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '3306'),
        'OPTIONS': {
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
        },
    }
}

# Static files configuration
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# Media files configuration
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# AWS S3 Configuration (if using S3)
if os.environ.get('USE_S3') == 'TRUE':
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = os.environ.get('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_REGION_NAME = os.environ.get('AWS_S3_REGION_NAME', 'us-east-1')
    
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    STATICFILES_STORAGE = 'storages.backends.s3boto3.StaticS3Boto3Storage'

# Email configuration
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')

# Security settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
SECURE_HSTS_SECONDS = 86400 if not DEBUG else 0
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
```

9. **Requirements File**
```txt
# requirements.txt
Django==4.2.7
mysqlclient==2.2.0
Pillow==10.0.1
django-storages==1.14.2
boto3==1.29.7
gunicorn==21.2.0
whitenoise==6.6.0
python-decouple==3.8
django-crispy-forms==2.1
crispy-bootstrap5==0.7
```

10. **Environment Variables (.env file)**
```bash
# .env
DEBUG=False
SECRET_KEY=your-super-secret-key-here
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Database
DB_NAME=cloudchef_production
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your-db-host.amazonaws.com
DB_PORT=3306

# AWS S3 (Optional)
USE_S3=TRUE
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_STORAGE_BUCKET_NAME=cloudchef-static-files
AWS_S3_REGION_NAME=us-east-1

# Email
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
```

### Project Features Implemented:

1. **Complete Menu Management**: Dynamic menu with categories, pricing, and availability
2. **Order Processing**: Full cart and order system with status tracking
3. **User Authentication**: Customer registration, login, and profiles
4. **Loyalty System**: Points-based rewards for repeat customers
5. **Admin Dashboard**: Staff interface for order management and analytics
6. **Responsive Design**: Mobile-friendly interface using Bootstrap
7. **Real-time Updates**: AJAX-powered cart and order status updates
8. **Image Handling**: Menu item photos with proper optimization
9. **Search and Filtering**: Easy navigation through menu items
10. **Production Security**: Proper environment variable handling and security settings

### Deployment Checklist:

- ✅ Database models and migrations
- ✅ User authentication system
- ✅ Admin interface setup
- ✅ Static file collection
- ✅ Media file handling
- ✅ Environment variables configured
- ✅ Production settings optimized
- ✅ Error handling implemented
- ✅ Security measures in place
- ✅ Performance optimizations

This project demonstrates a complete Django application ready for cloud deployment, showcasing all the skills learned throughout your Django journey!

## Assignment: Cloud Migration Health Check

**Scenario:** You've just opened your restaurant in three different locations (local development, staging, and production). As the head chef, you need to ensure all locations are operating correctly and customers can access your restaurant safely.

**Task:** Create a Django management command called `deployment_health_check.py` that verifies your deployment is working correctly.

**Requirements:**

1. **Database Connectivity Test**: Verify the database connection works and can perform basic operations
2. **Static Files Test**: Check that static files are being served correctly
3. **Security Headers Test**: Verify that security headers are properly configured
4. **External Services Test**: Test any external APIs or services your restaurant uses

**File to Create:** `management/commands/deployment_health_check.py`

```python
# Your command should check:
# - Database connection and basic query
# - Static files serving (check if STATIC_URL is accessible)
# - Security settings (SSL redirect, HSTS headers)
# - Any external API connections (payment gateways, email services)
# - Log the results with timestamps

# Example usage: python manage.py deployment_health_check
# Should output a detailed report of all systems
```

**Deliverables:**
1. The complete management command file
2. A sample output showing what a successful health check looks like
3. Documentation explaining what each check does and why it's important for a production restaurant (deployment)

**Evaluation Criteria:**
- Command runs without errors
- Comprehensive testing of deployment components
- Clear, informative output messages
- Proper error handling for failed checks
- Good use of Django's management command structure

This assignment reinforces deployment concepts without building another full application, focusing instead on the critical skill of monitoring and verifying production deployments - something every chef (developer) needs to master when running a real restaurant (production system).

---

## Key Takeaways

You've successfully transformed your home kitchen into a world-class restaurant! You now understand how to:

- Choose the right cloud platform for your restaurant's needs
- Set up professional-grade database systems for reliable ingredient storage
- Serve your visual materials efficiently using cloud storage and CDNs
- Secure your restaurant with proper domain and SSL configuration
- Monitor your deployment's health to ensure smooth operations

Remember, deployment is not a one-time event - it's an ongoing process of maintaining and improving your restaurant's operations. Keep monitoring, updating, and optimizing to ensure your customers always have the best dining experience!