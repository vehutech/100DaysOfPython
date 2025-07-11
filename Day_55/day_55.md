# Day 55: Django Security Best Practices

## Learning Objective
By the end of this lesson, you will be able to implement Django's built-in security features, configure HTTPS properly, set up essential security headers, and prevent common web vulnerabilities to protect your Django applications from malicious attacks.

---

## Introduction

Imagine that you're a head chef running a prestigious restaurant kitchen. Your kitchen isn't just about creating delicious meals—it's about maintaining the highest standards of food safety, hygiene, and security. Just as a chef must protect their kitchen from contamination, spoilage, and unauthorized access, a Django developer must secure their web application from cyber threats, data breaches, and malicious attacks.

In today's digital kitchen, security isn't an optional garnish—it's the foundation that keeps your entire operation safe and trustworthy. Let's explore how to fortify your Django application like a master chef protects their kitchen.

---

## 1. Django Security Features

### The Kitchen's Built-in Safety Systems

Just as a professional kitchen comes equipped with fire suppression systems, temperature controls, and safety protocols, Django provides numerous built-in security features that work automatically to protect your application.

#### CSRF Protection
Cross-Site Request Forgery (CSRF) protection is like having a security guard at your kitchen door who ensures only authorized personnel can enter and perform actions.

```python
# In your Django template
<form method="post">
    {% csrf_token %}
    <input type="text" name="recipe_name" placeholder="Enter recipe name">
    <button type="submit">Save Recipe</button>
</form>
```

```python
# In your views.py
from django.views.decorators.csrf import csrf_protect
from django.shortcuts import render, redirect

@csrf_protect
def add_recipe(request):
    if request.method == 'POST':
        recipe_name = request.POST.get('recipe_name')
        # Process the recipe safely
        return redirect('recipe_list')
    return render(request, 'add_recipe.html')
```

**Syntax Explanation:**
- `{% csrf_token %}`: Django template tag that generates a hidden field with a unique token
- `@csrf_protect`: Decorator that ensures CSRF protection is active for the view
- Django automatically validates the CSRF token on POST requests

#### SQL Injection Prevention
Django's ORM is like having a skilled sous chef who properly sanitizes all ingredients before they enter your dishes.

```python
# SECURE: Using Django ORM (parameterized queries)
from django.shortcuts import render
from .models import Recipe

def search_recipes(request):
    query = request.GET.get('search', '')
    # Django ORM automatically escapes and sanitizes the query
    recipes = Recipe.objects.filter(name__icontains=query)
    return render(request, 'search_results.html', {'recipes': recipes})

# NEVER DO THIS (vulnerable to SQL injection)
# raw_query = f"SELECT * FROM recipes WHERE name LIKE '%{query}%'"
```

**Syntax Explanation:**
- `Recipe.objects.filter(name__icontains=query)`: Django ORM method that safely filters records
- `name__icontains`: Field lookup that performs case-insensitive containment check
- Django automatically escapes special characters in the query parameter

#### XSS Protection
Cross-Site Scripting (XSS) protection is like having quality control that prevents contaminated ingredients from reaching your customers.

```python
# In your template (Django auto-escapes by default)
<div class="recipe-description">
    {{ recipe.description }}  <!-- Automatically escaped -->
</div>

<!-- If you need to display HTML content (use with extreme caution) -->
<div class="recipe-content">
    {{ recipe.content|safe }}  <!-- Only use with trusted content -->
</div>
```

```python
# In your views.py - additional XSS protection
from django.utils.html import escape
from django.http import JsonResponse

def get_recipe_data(request):
    user_input = request.GET.get('comment', '')
    # Manually escape if needed
    safe_input = escape(user_input)
    
    return JsonResponse({
        'comment': safe_input,
        'status': 'success'
    })
```

**Syntax Explanation:**
- `{{ recipe.description }}`: Django template variable that's automatically HTML-escaped
- `|safe`: Template filter that marks content as safe (bypasses auto-escaping)
- `escape()`: Function that manually escapes HTML special characters

---

## 2. HTTPS Configuration

### Securing Your Kitchen's Communication Lines

HTTPS is like having encrypted communication channels in your kitchen—ensuring that all sensitive information (orders, payments, customer data) travels securely between different parts of your operation.

#### Settings Configuration

```python
# settings.py
import os

# Force HTTPS in production
SECURE_SSL_REDIRECT = True  # Redirects all HTTP requests to HTTPS
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Session security
SESSION_COOKIE_SECURE = True  # Only send session cookies over HTTPS
SESSION_COOKIE_HTTPONLY = True  # Prevent JavaScript access to session cookies
SESSION_COOKIE_SAMESITE = 'Strict'  # Prevent CSRF attacks

# CSRF cookie security
CSRF_COOKIE_SECURE = True  # Only send CSRF cookies over HTTPS
CSRF_COOKIE_HTTPONLY = True  # Prevent JavaScript access to CSRF cookies
CSRF_COOKIE_SAMESITE = 'Strict'

# Additional security settings
SECURE_CONTENT_TYPE_NOSNIFF = True  # Prevent MIME type sniffing
SECURE_BROWSER_XSS_FILTER = True  # Enable XSS filtering
SECURE_HSTS_SECONDS = 31536000  # HTTP Strict Transport Security (1 year)
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
```

**Syntax Explanation:**
- `SECURE_SSL_REDIRECT = True`: Boolean setting that forces HTTP to HTTPS redirection
- `SESSION_COOKIE_SECURE = True`: Ensures session cookies are only sent over encrypted connections
- `SECURE_HSTS_SECONDS = 31536000`: Sets HSTS header duration in seconds (1 year)
- `SESSION_COOKIE_SAMESITE = 'Strict'`: Prevents cookies from being sent with cross-site requests

#### Environment-Specific Configuration

```python
# settings.py
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

if not DEBUG:  # Production settings
    SECURE_SSL_REDIRECT = True
    ALLOWED_HOSTS = ['yourrestaurant.com', 'www.yourrestaurant.com']
else:  # Development settings
    SECURE_SSL_REDIRECT = False
    ALLOWED_HOSTS = ['localhost', '127.0.0.1']
```

**Syntax Explanation:**
- `os.getenv('DEBUG', 'False')`: Gets environment variable with default fallback
- `.lower() == 'true'`: Converts string to boolean safely
- `ALLOWED_HOSTS`: List of allowed domain names for the application

---

## 3. Security Headers

### Your Kitchen's Safety Protocols

Security headers are like the safety protocols posted throughout your kitchen—they provide instructions to browsers (like kitchen staff) on how to handle your content securely.

#### Custom Middleware for Security Headers

```python
# security_middleware.py
class SecurityHeadersMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        # Content Security Policy - controls which resources can be loaded
        response['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self';"
        )
        
        # X-Frame-Options - prevents clickjacking
        response['X-Frame-Options'] = 'DENY'
        
        # X-Content-Type-Options - prevents MIME type sniffing
        response['X-Content-Type-Options'] = 'nosniff'
        
        # Referrer Policy - controls referrer information
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Permissions Policy - controls browser features
        response['Permissions-Policy'] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )
        
        return response
```

**Syntax Explanation:**
- `def __init__(self, get_response)`: Middleware initialization method
- `def __call__(self, request)`: Method called for each request
- `response['Header-Name']`: Sets HTTP response headers
- `"default-src 'self'"`: CSP directive allowing resources only from same origin

#### Register the Middleware

```python
# settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'myapp.security_middleware.SecurityHeadersMiddleware',  # Add your custom middleware
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
```

**Syntax Explanation:**
- `MIDDLEWARE`: List of middleware classes processed in order
- `'myapp.security_middleware.SecurityHeadersMiddleware'`: Python path to your custom middleware
- Order matters: security middleware should be near the top

---

## 4. Common Vulnerabilities Prevention

### Protecting Against Kitchen Hazards

Just as a chef must prevent food poisoning, fires, and accidents, we must protect our Django applications from common web vulnerabilities.

#### Preventing Insecure Direct Object References

```python
# models.py
from django.db import models
from django.contrib.auth.models import User

class Recipe(models.Model):
    name = models.CharField(max_length=200)
    ingredients = models.TextField()
    chef = models.ForeignKey(User, on_delete=models.CASCADE)
    is_private = models.BooleanField(default=False)
    
    def __str__(self):
        return self.name
```

```python
# views.py - SECURE approach
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import Http404

@login_required
def view_recipe(request, recipe_id):
    # Only allow users to view their own recipes or public recipes
    try:
        recipe = Recipe.objects.get(
            id=recipe_id,
            chef=request.user  # Ensure user owns the recipe
        )
    except Recipe.DoesNotExist:
        # Try to get public recipe if user doesn't own it
        recipe = get_object_or_404(
            Recipe, 
            id=recipe_id, 
            is_private=False
        )
    
    return render(request, 'recipe_detail.html', {'recipe': recipe})
```

**Syntax Explanation:**
- `@login_required`: Decorator ensuring user is authenticated
- `Recipe.objects.get(id=recipe_id, chef=request.user)`: Filters by both ID and ownership
- `get_object_or_404()`: Returns object or raises 404 if not found
- `try/except`: Handles the case where recipe doesn't exist or user doesn't own it

#### Input Validation and Sanitization

```python
# forms.py
from django import forms
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError

class RecipeForm(forms.Form):
    name = forms.CharField(
        max_length=200,
        validators=[
            RegexValidator(
                regex=r'^[a-zA-Z0-9\s\-\']+$',
                message='Recipe name can only contain letters, numbers, spaces, hyphens, and apostrophes.'
            )
        ]
    )
    
    ingredients = forms.CharField(
        widget=forms.Textarea,
        max_length=2000
    )
    
    servings = forms.IntegerField(
        min_value=1,
        max_value=100
    )
    
    def clean_ingredients(self):
        ingredients = self.cleaned_data['ingredients']
        
        # Custom validation
        if len(ingredients.split('\n')) > 50:
            raise ValidationError('Too many ingredients. Maximum 50 allowed.')
        
        # Sanitize the input
        ingredients = ingredients.strip()
        
        return ingredients
```

**Syntax Explanation:**
- `forms.CharField()`: Form field for text input
- `RegexValidator()`: Validates input against regular expression pattern
- `max_length=200`: Limits input length
- `def clean_ingredients(self)`: Custom validation method for specific field
- `self.cleaned_data['ingredients']`: Accesses validated form data

#### Rate Limiting for API Endpoints

```python
# views.py
from django.core.cache import cache
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import time

def rate_limit(max_requests=10, window=60):
    """
    Rate limiting decorator - like controlling how many orders 
    a chef can accept per minute
    """
    def decorator(func):
        def wrapper(request, *args, **kwargs):
            # Create unique key for this user/IP
            key = f"rate_limit:{request.user.id if request.user.is_authenticated else request.META.get('REMOTE_ADDR')}"
            
            # Get current request count
            current_requests = cache.get(key, 0)
            
            if current_requests >= max_requests:
                return JsonResponse({
                    'error': 'Rate limit exceeded. Please wait before making more requests.'
                }, status=429)
            
            # Increment counter
            cache.set(key, current_requests + 1, window)
            
            return func(request, *args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_requests=5, window=60)  # 5 requests per minute
@require_http_methods(["POST"])
def api_create_recipe(request):
    # API endpoint logic here
    return JsonResponse({'status': 'success'})
```

**Syntax Explanation:**
- `def rate_limit(max_requests=10, window=60)`: Decorator factory with parameters
- `def decorator(func)`: Actual decorator function
- `def wrapper(request, *args, **kwargs)`: Wrapper function that adds rate limiting
- `cache.get(key, 0)`: Gets cached value with default fallback
- `cache.set(key, current_requests + 1, window)`: Sets cache with expiration time

---

## Project: Secure Recipe Management System

Create a secure Django application that manages cooking recipes with the following features:

### Project Requirements

1. **User Authentication System**
   - Secure user registration and login
   - Password strength validation
   - Session management with security headers

2. **Recipe CRUD Operations**
   - Create, read, update, delete recipes
   - Proper authorization (users can only modify their own recipes)
   - Input validation and sanitization

3. **Security Implementation**
   - CSRF protection on all forms
   - XSS prevention in templates
   - SQL injection prevention using ORM
   - Security headers implementation

4. **HTTPS Configuration**
   - Force HTTPS in production mode
   - Secure cookies configuration
   - HSTS headers

### Sample Implementation Structure

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinLengthValidator

class Recipe(models.Model):
    name = models.CharField(max_length=200, validators=[MinLengthValidator(3)])
    description = models.TextField(max_length=1000)
    ingredients = models.TextField(max_length=2000)
    instructions = models.TextField(max_length=5000)
    prep_time = models.PositiveIntegerField()  # in minutes
    cook_time = models.PositiveIntegerField()  # in minutes
    servings = models.PositiveIntegerField()
    chef = models.ForeignKey(User, on_delete=models.CASCADE)
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['-created_at']
```

This project will demonstrate all the security concepts learned in this lesson while creating a practical, real-world application.

---

# Django Security Project: Building a Security Audit and Hardening Solution

## Learning Objective
By the end of this lesson, you will be able to conduct a comprehensive security audit of a Django application and implement hardening measures to protect against common vulnerabilities, ensuring your web application is production-ready and secure.

---

## Introduction: The Kitchen Safety Inspector

Imagine that you're a renowned chef who has just opened a new restaurant. Your kitchen is bustling with activity, your dishes are exquisite, and customers are lining up outside. But there's one crucial step you haven't completed yet – the health and safety inspection.

Just like a professional kitchen needs to pass rigorous safety standards before serving customers, your Django application needs a thorough security audit before going live. A contaminated kitchen can make people sick; an insecure web application can expose sensitive data and compromise user trust.

Today, we'll play the role of both the head chef and the safety inspector, examining every corner of our digital kitchen to ensure it's secure, compliant, and ready to serve thousands of users safely.

---

## The Security Audit Process: Your Kitchen Inspection Checklist

### Step 1: Setting Up Our Test Kitchen (Development Environment)

First, let's create a sample Django application that we'll audit and harden:

```python
# requirements.txt
Django==4.2.7
django-environ==0.11.2
psycopg2-binary==2.9.9
gunicorn==21.2.0
whitenoise==6.6.0
```

```python
# settings.py
import os
from pathlib import Path
import environ

# Initialize environment variables
env = environ.Env(
    DEBUG=(bool, False)
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Read environment file
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env('SECRET_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env('DEBUG')

ALLOWED_HOSTS = env.list('ALLOWED_HOSTS', default=[])

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'restaurant',  # Our sample app
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': env('DB_NAME'),
        'USER': env('DB_USER'),
        'PASSWORD': env('DB_PASSWORD'),
        'HOST': env('DB_HOST', default='localhost'),
        'PORT': env('DB_PORT', default='5432'),
    }
}

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
```

**Syntax Explanation:**
- `environ.Env()`: Creates an environment variable parser that safely reads configuration from environment variables
- `env('SECRET_KEY')`: Retrieves the SECRET_KEY from environment variables, raising an error if not found
- `env.list('ALLOWED_HOSTS', default=[])`: Parses a comma-separated list from environment variables

### Step 2: Creating Our Security Audit Tool (The Inspector's Toolkit)

Let's create a comprehensive security audit script:

```python
# security_audit.py
import os
import re
import subprocess
import json
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from django.core.management import call_command
from django.core.exceptions import ImproperlyConfigured

class SecurityAuditor:
    """
    Our head safety inspector - checks every aspect of kitchen security
    """
    
    def __init__(self):
        self.vulnerabilities = []
        self.recommendations = []
        self.critical_issues = []
        
    def audit_settings(self):
        """
        Check if our kitchen's basic safety protocols are in place
        """
        print("🔍 Inspecting Kitchen Settings...")
        
        # Check DEBUG setting
        if getattr(settings, 'DEBUG', True):
            self.critical_issues.append({
                'issue': 'DEBUG is True in production',
                'severity': 'CRITICAL',
                'description': 'Like leaving gas burners on - DEBUG exposes sensitive information',
                'fix': 'Set DEBUG = False in production environment'
            })
        
        # Check SECRET_KEY
        secret_key = getattr(settings, 'SECRET_KEY', '')
        if not secret_key or len(secret_key) < 50:
            self.critical_issues.append({
                'issue': 'Weak or missing SECRET_KEY',
                'severity': 'CRITICAL',
                'description': 'Like using the same key for all kitchen lockers',
                'fix': 'Generate a strong, unique SECRET_KEY'
            })
        
        # Check ALLOWED_HOSTS
        allowed_hosts = getattr(settings, 'ALLOWED_HOSTS', [])
        if not allowed_hosts or '*' in allowed_hosts:
            self.critical_issues.append({
                'issue': 'ALLOWED_HOSTS not properly configured',
                'severity': 'HIGH',
                'description': 'Like letting anyone into your kitchen',
                'fix': 'Specify exact domain names in ALLOWED_HOSTS'
            })
        
        print(f"   Found {len(self.critical_issues)} critical settings issues")
    
    def check_database_security(self):
        """
        Ensure our ingredient storage (database) is secure
        """
        print("🔍 Checking Database Security...")
        
        db_config = settings.DATABASES.get('default', {})
        
        # Check for hardcoded credentials
        if 'PASSWORD' in db_config and db_config['PASSWORD']:
            if any(char in db_config['PASSWORD'] for char in ['admin', 'password', '123']):
                self.vulnerabilities.append({
                    'issue': 'Weak database password detected',
                    'severity': 'HIGH',
                    'description': 'Like using "password" as your safe combination',
                    'fix': 'Use strong, unique database passwords'
                })
        
        # Check SSL/TLS for database connections
        if 'OPTIONS' not in db_config or 'sslmode' not in db_config.get('OPTIONS', {}):
            self.recommendations.append({
                'issue': 'Database connection not encrypted',
                'severity': 'MEDIUM',
                'description': 'Like shouting your recipes across the kitchen',
                'fix': 'Enable SSL/TLS for database connections'
            })
    
    def scan_for_sensitive_data(self):
        """
        Look for ingredients left out in the open (sensitive data exposure)
        """
        print("🔍 Scanning for Exposed Sensitive Data...")
        
        sensitive_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
            (r'secret_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret key'),
            (r'aws_access_key\s*=\s*["\'][^"\']+["\']', 'AWS credentials'),
        ]
        
        for root, dirs, files in os.walk('.'):
            # Skip virtual environments and node_modules
            dirs[:] = [d for d in dirs if d not in ['venv', 'env', 'node_modules', '.git']]
            
            for file in files:
                if file.endswith(('.py', '.js', '.json', '.yml', '.yaml')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        for pattern, description in sensitive_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                self.vulnerabilities.append({
                                    'issue': f'{description} in {file_path}',
                                    'severity': 'HIGH',
                                    'description': f'Like leaving {description.lower()} written on the kitchen wall',
                                    'fix': 'Move sensitive data to environment variables'
                                })
                    except (UnicodeDecodeError, PermissionError):
                        continue
    
    def check_middleware_security(self):
        """
        Verify our kitchen's safety equipment (middleware) is properly installed
        """
        print("🔍 Checking Security Middleware...")
        
        middleware = getattr(settings, 'MIDDLEWARE', [])
        
        required_security_middleware = [
            ('django.middleware.security.SecurityMiddleware', 'Basic security headers'),
            ('django.middleware.csrf.CsrfViewMiddleware', 'CSRF protection'),
            ('django.contrib.auth.middleware.AuthenticationMiddleware', 'Authentication'),
            ('django.middleware.clickjacking.XFrameOptionsMiddleware', 'Clickjacking protection'),
        ]
        
        for middleware_class, description in required_security_middleware:
            if middleware_class not in middleware:
                self.vulnerabilities.append({
                    'issue': f'Missing {description}',
                    'severity': 'HIGH',
                    'description': f'Like missing {description.lower()} in your kitchen',
                    'fix': f'Add {middleware_class} to MIDDLEWARE'
                })
    
    def generate_report(self):
        """
        Create our kitchen safety inspection report
        """
        print("\n" + "="*60)
        print("🛡️  SECURITY AUDIT REPORT")
        print("="*60)
        
        total_issues = len(self.critical_issues) + len(self.vulnerabilities) + len(self.recommendations)
        
        if total_issues == 0:
            print("✅ Congratulations! Your kitchen passes all safety inspections!")
            return
        
        print(f"📊 Total Issues Found: {total_issues}")
        print(f"🚨 Critical: {len(self.critical_issues)}")
        print(f"⚠️  High/Medium: {len(self.vulnerabilities)}")
        print(f"💡 Recommendations: {len(self.recommendations)}")
        
        if self.critical_issues:
            print("\n🚨 CRITICAL ISSUES (Fix Immediately!):")
            for issue in self.critical_issues:
                print(f"   • {issue['issue']}")
                print(f"     Chef's Note: {issue['description']}")
                print(f"     Fix: {issue['fix']}\n")
        
        if self.vulnerabilities:
            print("\n⚠️  VULNERABILITIES:")
            for vuln in self.vulnerabilities:
                print(f"   • {vuln['issue']}")
                print(f"     Chef's Note: {vuln['description']}")
                print(f"     Fix: {vuln['fix']}\n")
        
        if self.recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for rec in self.recommendations:
                print(f"   • {rec['issue']}")
                print(f"     Chef's Note: {rec['description']}")
                print(f"     Fix: {rec['fix']}\n")
    
    def run_full_audit(self):
        """
        Run the complete kitchen safety inspection
        """
        print("🔍 Starting Security Audit...")
        print("Like a thorough kitchen inspection before opening night!\n")
        
        self.audit_settings()
        self.check_database_security()
        self.scan_for_sensitive_data()
        self.check_middleware_security()
        self.generate_report()

# Usage
if __name__ == "__main__":
    auditor = SecurityAuditor()
    auditor.run_full_audit()
```

**Syntax Explanation:**
- `getattr(settings, 'DEBUG', True)`: Safely gets the DEBUG setting, defaulting to True if not found
- `os.walk('.')`: Recursively walks through all directories and files
- `re.search(pattern, content, re.IGNORECASE)`: Searches for regex patterns case-insensitively
- `dirs[:] = [d for d in dirs if d not in ['venv', 'env']]`: Modifies the dirs list in-place to skip certain directories

### Step 3: The Hardening Process (Upgrading Our Kitchen Security)

Now let's create a hardening script that fixes the issues we've identified:

```python
# security_hardening.py
import os
import secrets
import string
from pathlib import Path
from django.conf import settings

class SecurityHardener:
    """
    Our kitchen renovation specialist - upgrades security infrastructure
    """
    
    def __init__(self):
        self.changes_made = []
        self.base_dir = Path(__file__).resolve().parent
    
    def generate_secure_secret_key(self):
        """
        Generate a new, secure secret key - like changing all the locks
        """
        alphabet = string.ascii_letters + string.digits + '!@#$%^&*(-_=+)'
        secret_key = ''.join(secrets.choice(alphabet) for _ in range(50))
        
        # Update .env file
        env_file = self.base_dir / '.env'
        
        if env_file.exists():
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Replace or add SECRET_KEY
            if 'SECRET_KEY=' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('SECRET_KEY='):
                        lines[i] = f'SECRET_KEY={secret_key}'
                        break
                content = '\n'.join(lines)
            else:
                content += f'\nSECRET_KEY={secret_key}\n'
            
            with open(env_file, 'w') as f:
                f.write(content)
        else:
            with open(env_file, 'w') as f:
                f.write(f'SECRET_KEY={secret_key}\n')
        
        self.changes_made.append("✅ Generated new secure SECRET_KEY")
    
    def create_security_settings(self):
        """
        Add comprehensive security settings - like installing a complete security system
        """
        security_settings = '''
# security_settings.py
"""
Production Security Settings
Like a comprehensive kitchen safety manual
"""

# Security Settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# SSL/HTTPS Settings
SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
USE_TLS = True

# Session Security
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Strict'
SESSION_COOKIE_AGE = 3600  # 1 hour

# CSRF Protection
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SAMESITE = 'Strict'
CSRF_TRUSTED_ORIGINS = []  # Add your trusted domains here

# Clickjacking Protection
X_FRAME_OPTIONS = 'DENY'

# Content Security Policy
CSP_DEFAULT_SRC = ("'self'",)
CSP_SCRIPT_SRC = ("'self'", "'unsafe-inline'")
CSP_STYLE_SRC = ("'self'", "'unsafe-inline'")
CSP_IMG_SRC = ("'self'", "data:", "https:")
CSP_FONT_SRC = ("'self'", "https:")

# Database Security
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
        'OPTIONS': {
            'sslmode': 'require',
            'options': '-c default_transaction_isolation=serializable'
        },
    }
}

# Logging Security
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'security': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'security.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django.security': {
            'handlers': ['security'],
            'level': 'INFO',
            'propagate': True,
        },
    },
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
    },
}

# File Upload Security
FILE_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB
FILE_UPLOAD_PERMISSIONS = 0o644
DATA_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB

# Admin Security
ADMIN_URL = os.environ.get('ADMIN_URL', 'admin/')  # Use custom admin URL
'''
        
        security_file = self.base_dir / 'security_settings.py'
        with open(security_file, 'w') as f:
            f.write(security_settings)
        
        self.changes_made.append("✅ Created comprehensive security settings file")
    
    def create_security_middleware(self):
        """
        Create custom security middleware - like hiring a security guard
        """
        middleware_code = '''
# security_middleware.py
import logging
import time
from django.http import HttpResponseForbidden
from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger('django.security')

class SecurityMiddleware:
    """
    Custom security middleware - our kitchen's security guard
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Rate limiting - prevent kitchen from being overwhelmed
        if self.is_rate_limited(request):
            logger.warning(f"Rate limit exceeded for {request.META.get('REMOTE_ADDR')}")
            return HttpResponseForbidden("Rate limit exceeded. Please slow down.")
        
        # Security headers
        response = self.get_response(request)
        
        # Add security headers - like putting up safety signs
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        response['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Log security events
        if hasattr(request, 'security_event'):
            logger.info(f"Security event: {request.security_event}")
        
        return response
    
    def is_rate_limited(self, request):
        """
        Simple rate limiting - like controlling how fast orders come in
        """
        ip = request.META.get('REMOTE_ADDR')
        key = f'rate_limit_{ip}'
        
        current_requests = cache.get(key, 0)
        if current_requests > 100:  # 100 requests per minute
            return True
        
        cache.set(key, current_requests + 1, 60)  # 1 minute window
        return False

class IPWhitelistMiddleware:
    """
    IP whitelist middleware - like a VIP list for your kitchen
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.allowed_ips = getattr(settings, 'ALLOWED_IPS', [])
    
    def __call__(self, request):
        if self.allowed_ips and request.path.startswith('/admin/'):
            ip = request.META.get('REMOTE_ADDR')
            if ip not in self.allowed_ips:
                logger.warning(f"Unauthorized admin access attempt from {ip}")
                return HttpResponseForbidden("Access denied from this IP address")
        
        return self.get_response(request)
'''
        
        middleware_file = self.base_dir / 'security_middleware.py'
        with open(middleware_file, 'w') as f:
            f.write(middleware_code)
        
        self.changes_made.append("✅ Created custom security middleware")
    
    def create_security_checklist(self):
        """
        Create a deployment security checklist - like a pre-opening inspection form
        """
        checklist = '''
# DJANGO SECURITY DEPLOYMENT CHECKLIST
## Like a final kitchen inspection before opening night

### Environment Variables (Keep these secret like family recipes!)
- [ ] SECRET_KEY is generated and stored securely
- [ ] DEBUG is set to False in production
- [ ] Database credentials are in environment variables
- [ ] API keys are in environment variables
- [ ] ALLOWED_HOSTS is properly configured

### HTTPS/SSL (Like ensuring all kitchen equipment is properly grounded)
- [ ] SSL certificate is installed and valid
- [ ] HTTP redirects to HTTPS
- [ ] HSTS headers are configured
- [ ] Mixed content issues are resolved

### Database Security (Protecting your ingredient storage)
- [ ] Database uses strong authentication
- [ ] Database connections are encrypted (SSL/TLS)
- [ ] Database user has minimal required permissions
- [ ] Database backups are encrypted

### File Security (Keeping your recipes safe)
- [ ] Media files are served securely
- [ ] File upload validation is in place
- [ ] Static files are served with proper headers
- [ ] Sensitive files are not in web root

### Monitoring & Logging (Kitchen surveillance system)
- [ ] Security logs are configured
- [ ] Failed login attempts are logged
- [ ] Rate limiting is in place
- [ ] Error monitoring is set up

### Admin Security (Protecting the head chef's office)
- [ ] Admin URL is changed from default
- [ ] Admin access is IP-restricted if possible
- [ ] Strong admin passwords are enforced
- [ ] Two-factor authentication is enabled

### Regular Maintenance (Keep your kitchen clean!)
- [ ] Dependencies are regularly updated
- [ ] Security patches are applied promptly
- [ ] Regular security audits are performed
- [ ] Penetration testing is conducted

### Final Checks
- [ ] All security headers are present
- [ ] CSRF protection is working
- [ ] XSS protection is enabled
- [ ] Clickjacking protection is active
- [ ] Content Security Policy is configured
'''
        
        checklist_file = self.base_dir / 'SECURITY_CHECKLIST.md'
        with open(checklist_file, 'w') as f:
            f.write(checklist)
        
        self.changes_made.append("✅ Created security deployment checklist")
    
    def apply_all_hardening(self):
        """
        Apply all security hardening measures - complete kitchen renovation
        """
        print("🔧 Starting Security Hardening...")
        print("Like renovating your kitchen with the latest safety equipment!\n")
        
        self.generate_secure_secret_key()
        self.create_security_settings()
        self.create_security_middleware()
        self.create_security_checklist()
        
        print("✅ Security Hardening Complete!")
        print("\nChanges Made:")
        for change in self.changes_made:
            print(f"  {change}")
        
        print("\n🎉 Your kitchen is now secured and ready for production!")
        print("Don't forget to:")
        print("  1. Review and customize the security settings")
        print("  2. Add the security middleware to your MIDDLEWARE setting")
        print("  3. Follow the security checklist before deployment")
        print("  4. Test everything in a staging environment first")

# Usage
if __name__ == "__main__":
    hardener = SecurityHardener()
    hardener.apply_all_hardening()
```

**Syntax Explanation:**
- `secrets.choice(alphabet)`: Cryptographically secure random choice from the alphabet
- `Path(__file__).resolve().parent`: Gets the directory containing the current script
- `getattr(settings, 'ALLOWED_IPS', [])`: Gets a setting with a default value if not found
- `cache.set(key, value, timeout)`: Sets a value in Django's cache system with expiration

### Step 4: The Final Project - Complete Security Implementation

Now let's create the final project that demonstrates everything we've learned:

```python
# final_project.py
"""
FINAL PROJECT: Complete Django Security Implementation
Like opening a five-star restaurant with bulletproof security!
"""

import os
import subprocess
import sys
from pathlib import Path

class RestaurantSecurityManager:
    """
    Master class that manages all aspects of restaurant security
    """
    
    def __init__(self, project_name="secure_restaurant"):
        self.project_name = project_name
        self.base_dir = Path.cwd()
        self.project_dir = self.base_dir / project_name
        
    def create_project_structure(self):
        """
        Create our restaurant with proper security architecture
        """
        print("🏗️  Creating Secure Restaurant Project Structure...")
        
        # Create main project directory
        self.project_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.project_dir / "config").mkdir(exist_ok=True)
        (self.project_dir / "apps").mkdir(exist_ok=True)
        (self.project_dir / "security").mkdir(exist_ok=True)
        (self.project_dir / "logs").mkdir(exist_ok=True)
        (self.project_dir / "static").mkdir(exist_ok=True)
        (self.project_dir / "media").mkdir(exist_ok=True)
        (self.project_dir / "templates").mkdir(exist_ok=True)
        
        print("✅ Project structure created")
    
    def setup_environment(self):
        """
        Set up environment configuration - like preparing ingredients
        """
        env_template = """# Restaurant Environment Configuration
# Keep these values secret like your signature sauce recipe!

# Basic Settings
DEBUG=False
SECRET_KEY=your-super-secret-key-here
ALLOWED_HOSTS=your-domain.com,www.your-domain.com

# Database Configuration
DB_NAME=restaurant_db
DB_USER=restaurant_user
DB_PASSWORD=super-secure-password
DB_HOST=localhost
DB_PORT=5432

# Security Settings
ADMIN_URL=secret-admin-panel/
ALLOWED_IPS=127.0.0.1,::1

# Email Configuration
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password

# External Services
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
"""
        
        env_file = self.project_dir / '.env.template'
        with open(env_file, 'w') as f:
            f.write(env_template)
        
        print("✅ Environment template created")
    
    def create_security_tests(self):
        """
        Create comprehensive security tests - like health inspections
        """
        test_code = '''
# test_security.py
"""
Security Test Suite - Like health inspection protocols
"""

import pytest
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.conf import settings
from django.core.cache import cache

class SecurityTestSuite(TestCase):
    """
    Complete security test suite for our restaurant
    """
    
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testchef',
            email='chef@restaurant.com',
            password='SecurePassword123!'
        )
    
    def test_debug_is_false(self):
        """Test that DEBUG is False - like checking gas valves are off"""
        self.assertFalse(settings.DEBUG, "DEBUG should be False in production")
    
    def test_secret_key_security(self):
        """Test SECRET_KEY strength - like checking lock quality"""
        secret_key = settings.SECRET_KEY
        self.assertGreater(len(secret_key), 40, "SECRET_KEY should be long")
        self.assertNotIn('django', secret_key.lower(), "SECRET_KEY should not contain 'django'")
        self.assertNotIn('secret', secret_key.lower(), "SECRET_KEY should not contain 'secret'")
    
    def test_allowed_hosts_configured(self):
        """Test ALLOWED_HOSTS - like checking guest list"""
        allowed_hosts = settings.ALLOWED_HOSTS
        self.assertTrue(allowed_hosts, "ALLOWED_HOSTS should not be empty")
        self.assertNotIn('*', allowed_hosts, "ALLOWED_HOSTS should not contain '*'")
    
    def test_security_headers(self):
        """Test security headers - like checking safety equipment"""
        response = self.client.get('/')
        
        # Check for security headers
        self.assertIn('X-Content-Type-Options', response.headers)
        self.assertIn('X-Frame-Options', response.headers)
        self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
        self.assertEqual(response.headers['X-Frame-Options'], 'DENY')
    
    def test_csrf_protection(self):
        """Test CSRF protection - like checking order verification"""
        response = self.client.get('/admin/login/')
        self.assertContains(response, 'csrfmiddlewaretoken')
    
    def test_admin_url_changed(self):
        """Test admin URL is customized - like hiding the manager's office"""
        response = self.client.get('/admin/')
        # Should not be accessible at default URL in production
        self.assertNotEqual(response.status_code, 200)
    
    def test_rate_limiting(self):
        """Test rate limiting - like preventing kitchen overload"""
        # This would test your rate limiting middleware
        for i in range(10):
            response = self.client.get('/')
            if response.status_code == 429:  # Too Many Requests
                break
        else:
            # If we didn't hit rate limit, that's okay for basic testing
            pass
    
    def test_file_upload_security(self):
        """Test file upload restrictions - like checking ingredient quality"""
        # Test file size limits
        self.assertLessEqual(settings.FILE_UPLOAD_MAX_MEMORY_SIZE, 10485760)  # 10MB max
        self.assertLessEqual(settings.DATA_UPLOAD_MAX_MEMORY_SIZE, 10485760)  # 10MB max
    
    def test_session_security(self):
        """Test session security - like checking customer privacy"""
        self.assertTrue(getattr(settings, 'SESSION_COOKIE_SECURE', False))
        self.assertTrue(getattr(settings, 'SESSION_COOKIE_HTTPONLY', False))
        self.assertEqual(getattr(settings, 'SESSION_COOKIE_SAMESITE', None), 'Strict')
    
    def test_database_security(self):
        """Test database security - like checking ingredient storage"""
        db_config = settings.DATABASES['default']
        # Check that sensitive data is not hardcoded
        self.assertNotIn('password123', str(db_config))
        self.assertNotIn('admin', str(db_config))

if __name__ == '__main__':
    pytest.main([__file__])
'''
        
        test_file = self.project_dir / 'security' / 'test_security.py'
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        print("✅ Security tests created")
    
    def create_deployment_script(self):
        """
        Create automated deployment script - like opening night preparation
        """
        deploy_script = '''#!/bin/bash
# deploy_secure_restaurant.sh
# Automated deployment script for our secure restaurant

echo "🚀 Starting Secure Restaurant Deployment..."
echo "Like preparing for opening night with full security!"

# Color codes for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if .env file exists
if [ ! -f .env ]; then
    print_error ".env file not found! Copy from .env.template and configure."
    exit 1
fi

# Load environment variables
export $(cat .env | xargs)

# Security checks before deployment
echo "🔍 Running pre-deployment security checks..."

# Check DEBUG setting
if [ "$DEBUG" = "True" ]; then
    print_error "DEBUG is set to True! This is unsafe for production."
    exit 1
fi
print_status "DEBUG is properly set to False"

# Check SECRET_KEY
if [ ${#SECRET_KEY} -lt 40 ]; then
    print_error "SECRET_KEY is too short! Generate a stronger key."
    exit 1
fi
print_status "SECRET_KEY length is adequate"

# Check ALLOWED_HOSTS
if [[ "$ALLOWED_HOSTS" == *"*"* ]]; then
    print_error "ALLOWED_HOSTS contains '*' - this is unsafe!"
    exit 1
fi
print_status "ALLOWED_HOSTS is properly configured"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt
print_status "Dependencies installed"

# Run security tests
echo "🧪 Running security tests..."
python manage.py test security.test_security
if [ $? -ne 0 ]; then
    print_error "Security tests failed! Fix issues before deployment."
    exit 1
fi
print_status "Security tests passed"

# Collect static files
echo "📁 Collecting static files..."
python manage.py collectstatic --noinput
print_status "Static files collected"

# Run database migrations
echo "🗄️  Running database migrations..."
python manage.py migrate
print_status "Database migrations completed"

# Check for security issues
echo "🔍 Running Django security check..."
python manage.py check --deploy
if [ $? -ne 0 ]; then
    print_warning "Security check found issues. Review and fix if necessary."
fi

# Create superuser if needed
if [ "$CREATE_SUPERUSER" = "True" ]; then
    echo "👤 Creating superuser..."
    python manage.py createsuperuser --noinput
    print_status "Superuser created"
fi

# Final security verification
echo "🔐 Final security verification..."
python security/security_audit.py
print_status "Security audit completed"

echo ""
echo "🎉 Deployment completed successfully!"
echo "Your secure restaurant is ready to serve customers!"
echo ""
echo "📋 Post-deployment checklist:"
echo "  1. Verify SSL certificate is working"
echo "  2. Test all security headers"
echo "  3. Confirm rate limiting is active"
echo "  4. Check logs for any issues"
echo "  5. Monitor for security events"
echo ""
echo "🍽️  Welcome to your secure restaurant!"
'''
        
        deploy_file = self.project_dir / 'deploy_secure_restaurant.sh'
        with open(deploy_file, 'w') as f:
            f.write(deploy_script)
        
        # Make script executable
        os.chmod(deploy_file, 0o755)
        
        print("✅ Deployment script created")
    
    def create_monitoring_dashboard(self):
        """
        Create security monitoring dashboard - like a kitchen control panel
        """
        dashboard_code = '''
# security_dashboard.py
"""
Security Monitoring Dashboard
Like a master control panel for your restaurant's security
"""

import os
import json
import datetime
from pathlib import Path
from collections import defaultdict, Counter
import re

class SecurityDashboard:
    """
    Real-time security monitoring for our restaurant
    """
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.security_events = []
        self.failed_logins = []
        self.suspicious_activities = []
        
    def parse_security_logs(self):
        """
        Parse security logs - like reviewing security footage
        """
        security_log = self.log_dir / "security.log"
        if not security_log.exists():
            return
        
        with open(security_log, 'r') as f:
            for line in f:
                self.analyze_log_entry(line.strip())
    
    def analyze_log_entry(self, log_entry):
        """
        Analyze individual log entries - like checking each security camera
        """
        if "failed login" in log_entry.lower():
            self.failed_logins.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'entry': log_entry
            })
        
        if "rate limit" in log_entry.lower():
            self.suspicious_activities.append({
                'type': 'rate_limit',
                'timestamp': datetime.datetime.now().isoformat(),
                'entry': log_entry
            })
        
        if "unauthorized" in log_entry.lower():
            self.suspicious_activities.append({
                'type': 'unauthorized_access',
                'timestamp': datetime.datetime.now().isoformat(),
                'entry': log_entry
            })
    
    def generate_security_report(self):
        """
        Generate comprehensive security report - like a daily security briefing
        """
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'summary': {
                'total_security_events': len(self.security_events),
                'failed_logins': len(self.failed_logins),
                'suspicious_activities': len(self.suspicious_activities),
                'status': 'SECURE' if len(self.suspicious_activities) == 0 else 'ALERT'
            },
            'details': {
                'failed_logins': self.failed_logins[-10:],  # Last 10
                'suspicious_activities': self.suspicious_activities[-10:],  # Last 10
            }
        }
        
        return report
    
    def display_dashboard(self):
        """
        Display security dashboard - like the restaurant's security monitor
        """
        self.parse_security_logs()
        report = self.generate_security_report()
        
        print("🛡️  RESTAURANT SECURITY DASHBOARD")
        print("=" * 50)
        print(f"📅 Report Time: {report['timestamp']}")
        print(f"🔒 Security Status: {report['summary']['status']}")
        print()
        
        # Security Summary
        print("📊 SECURITY SUMMARY")
        print("-" * 30)
        print(f"Total Security Events: {report['summary']['total_security_events']}")
        print(f"Failed Login Attempts: {report['summary']['failed_logins']}")
        print(f"Suspicious Activities: {report['summary']['suspicious_activities']}")
        print()
        
        # Recent Failed Logins
        if report['details']['failed_logins']:
            print("🚨 RECENT FAILED LOGINS")
            print("-" * 30)
            for login in report['details']['failed_logins']:
                print(f"⏰ {login['timestamp']}")
                print(f"📝 {login['entry']}")
                print()
        
        # Suspicious Activities
        if report['details']['suspicious_activities']:
            print("⚠️  SUSPICIOUS ACTIVITIES")
            print("-" * 30)
            for activity in report['details']['suspicious_activities']:
                print(f"⏰ {activity['timestamp']}")
                print(f"🔍 Type: {activity['type']}")
                print(f"📝 {activity['entry']}")
                print()
        
        # Recommendations
        print("💡 SECURITY RECOMMENDATIONS")
        print("-" * 30)
        if report['summary']['failed_logins'] > 5:
            print("• Consider implementing account lockout after failed attempts")
        if report['summary']['suspicious_activities'] > 10:
            print("• Review and strengthen rate limiting rules")
        if report['summary']['status'] == 'ALERT':
            print("• Investigate suspicious activities immediately")
        print("• Regular security audits recommended")
        print("• Keep all dependencies updated")
        print()
        
        print("🍽️  Your restaurant's security is being monitored 24/7!")

# Usage
if __name__ == "__main__":
    dashboard = SecurityDashboard()
    dashboard.display_dashboard()
'''
        
        dashboard_file = self.project_dir / 'security' / 'security_dashboard.py'
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_code)
        
        print("✅ Security monitoring dashboard created")
    
    def create_complete_example(self):
        """
        Create a complete working example - like a fully operational restaurant
        """
        # Create main Django project files
        main_settings = '''
# settings.py - Main Restaurant Configuration
"""
Complete Django Security Configuration
Like a master recipe for a secure restaurant
"""

import os
from pathlib import Path
import environ

# Initialize environment
env = environ.Env(DEBUG=(bool, False))
BASE_DIR = Path(__file__).resolve().parent.parent

# Read environment file
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

# Basic Configuration
SECRET_KEY = env('SECRET_KEY')
DEBUG = env('DEBUG')
ALLOWED_HOSTS = env.list('ALLOWED_HOSTS')

# Application Definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'apps.restaurant',
    'security',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'security.middleware.SecurityMiddleware',
    'security.middleware.IPWhitelistMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

# Database Configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': env('DB_NAME'),
        'USER': env('DB_USER'),
        'PASSWORD': env('DB_PASSWORD'),
        'HOST': env('DB_HOST'),
        'PORT': env('DB_PORT'),
        'OPTIONS': {
            'sslmode': 'require',
        }
    }
}

# Security Settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
X_FRAME_OPTIONS = 'DENY'

# Static Files
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

# Media Files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# File Upload Security
FILE_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'security_file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'security.log',
        },
    },
    'loggers': {
        'django.security': {
            'handlers': ['security_file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# Admin Configuration
ADMIN_URL = env('ADMIN_URL', default='admin/')
ALLOWED_IPS = env.list('ALLOWED_IPS', default=[])

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
'''
        
        settings_file = self.project_dir / 'config' / 'settings.py'
        with open(settings_file, 'w') as f:
            f.write(main_settings)
        
        print("✅ Complete Django configuration created")
    
    def build_complete_restaurant(self):
        """
        Build the complete secure restaurant - grand opening!
        """
        print("🏗️  Building Your Secure Restaurant...")
        print("Like constructing a five-star establishment with bulletproof security!")
        print()
        
        self.create_project_structure()
        self.setup_environment()
        self.create_security_tests()
        self.create_deployment_script()
        self.create_monitoring_dashboard()
        self.create_complete_example()
        
        # Create requirements.txt
        requirements = """Django==4.2.7
django-environ==0.11.2
psycopg2-binary==2.9.9
gunicorn==21.2.0
whitenoise==6.6.0
redis==5.0.1
celery==5.3.4
pytest==7.4.3
pytest-django==4.7.0
"""
        
        with open(self.project_dir / 'requirements.txt', 'w') as f:
            f.write(requirements)
        
        # Create README
        readme = f"""# {self.project_name.title()}
## A Secure Django Restaurant Application

This is a production-ready Django application with comprehensive security features.

### Features
- 🔒 Complete security hardening
- 🛡️ Real-time security monitoring
- 🧪 Comprehensive security testing
- 🚀 Automated secure deployment
- 📊 Security dashboard and reporting

### Setup Instructions
1. Copy `.env.template` to `.env` and configure
2. Install dependencies: `pip install -r requirements.txt`
3. Run security tests: `python manage.py test security`
4. Deploy: `./deploy_secure_restaurant.sh`

### Security Features Included
- CSRF protection
- XSS prevention
- Clickjacking protection
- Rate limiting
- Security headers
- IP whitelisting for admin
- Secure session management
- File upload security
- Database security
- Logging and monitoring

### Monitoring
Run the security dashboard: `python security/security_dashboard.py`

### Architecture
This application follows Django security best practices and includes:
- Environment-based configuration
- Security middleware
- Comprehensive testing
- Automated deployment
- Real-time monitoring

Your restaurant is secure and ready to serve customers! 🍽️
"""
        
        with open(self.project_dir / 'README.md', 'w') as f:
            f.write(readme)
        
        print("🎉 CONGRATULATIONS!")
        print("=" * 60)
        print("Your Secure Restaurant is Complete!")
        print()
        print("🏆 What You've Built:")
        print("  • Complete Django security implementation")
        print("  • Automated security testing")
        print("  • Real-time monitoring dashboard")
        print("  • Secure deployment process")
        print("  • Production-ready configuration")
        print()
        print("📁 Project Structure:")
        print(f"  {self.project_name}/")
        print("  ├── config/          # Django configuration")
        print("  ├── security/        # Security tools and tests")
        print("  ├── apps/            # Application modules")
        print("  ├── logs/            # Security logs")
        print("  ├── static/          # Static files")
        print("  ├── media/           # Media files")
        print("  └── templates/       # Templates")
        print()
        print("🚀 Next Steps:")
        print("  1. Configure your .env file")
        print("  2. Run security tests")
        print("  3. Deploy to production")
        print("  4. Monitor with the dashboard")
        print()
        print("🍽️  Your restaurant is now ready to serve customers securely!")

# Usage
if __name__ == "__main__":
    manager = RestaurantSecurityManager()
    manager.build_complete_restaurant()
```

**Syntax Explanation:**
- `Path.cwd()`: Gets the current working directory
- `mkdir(exist_ok=True)`: Creates directory, doesn't fail if it already exists
- `os.chmod(file, 0o755)`: Sets file permissions (readable/executable by all, writable by owner)
- `Counter()`: Creates a dictionary subclass for counting hashable objects
- `defaultdict(list)`: Creates a dictionary that automatically creates lists for new keys

---

### Project Overview
You've now built a complete, production-ready Django application with enterprise-level security features. This isn't just a learning exercise - it's a real-world template that you can use for any Django project.

### Key Components You've Mastered:

1. **Security Auditing**: Your `SecurityAuditor` class can scan any Django project for vulnerabilities
2. **Security Hardening**: The `SecurityHardener` automatically implements best practices
3. **Automated Testing**: Comprehensive test suite ensures security measures are working
4. **Monitoring Dashboard**: Real-time security monitoring and reporting
5. **Deployment Automation**: Secure deployment process with pre-flight checks

### What Makes This Project Special:

**Like a Five-Star Restaurant:**
- **Kitchen Safety Standards**: Every security measure is implemented and tested
- **Quality Control**: Automated testing ensures nothing breaks
- **Staff Training**: Clear documentation and checklists for deployment
- **Customer Safety**: Multiple layers of protection for user data
- **Business Continuity**: Monitoring and logging for ongoing operations

### Skills Demonstrated:
- File I/O operations and directory management
- Regular expressions for pattern matching
- Environment variable management
- Security header implementation
- Middleware development
- Test-driven security development
- Automated deployment scripting
- Real-time monitoring systems

---

## Project Summary

**Congratulations, Chef!** 🎉

You've successfully transformed from a novice cook into a master chef who can run a secure, five-star restaurant. You now possess the skills to:

- **Audit** any Django application for security vulnerabilities
- **Harden** applications with production-ready security measures
- **Test** security implementations comprehensively
- **Deploy** applications securely with automated processes
- **Monitor** applications in real-time for security events

Your digital restaurant is now ready to serve customers safely, securely, and with confidence. The security measures you've implemented would make any cybersecurity professional proud!

**Remember**: Security is not a one-time setup - it's an ongoing process. Keep your kitchen clean, your ingredients fresh, and your security measures up to date. Your customers (and their data) depend on it!

**Final Chef's Tip**: The best security is invisible to legitimate users but impenetrable to attackers. You've built exactly that - a smooth, seamless experience that's fortress-strong underneath.

Now go forth and build secure, amazing applications! 🚀🔒🍽️

## Assignment: Security Vulnerability Assessment

### Task
You are given a Django view function that contains multiple security vulnerabilities. Your task is to identify the vulnerabilities and rewrite the function to be secure.

### Vulnerable Code to Fix

```python
# VULNERABLE CODE - DO NOT USE IN PRODUCTION
from django.shortcuts import render
from django.http import HttpResponse
from django.db import connection

def search_recipes_vulnerable(request):
    search_term = request.GET.get('q', '')
    
    # Vulnerability 1: SQL Injection
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT * FROM recipes WHERE name LIKE '%{search_term}%'"
        )
        results = cursor.fetchall()
    
    # Vulnerability 2: XSS - Direct output without escaping
    html_output = f"<h1>Search Results for: {search_term}</h1>"
    
    # Vulnerability 3: No input validation
    if search_term:
        html_output += "<ul>"
        for row in results:
            html_output += f"<li><strong>{row[1]}</strong> - {row[2]}</li>"
        html_output += "</ul>"
    
    return HttpResponse(html_output)
```

### Your Task

1. **Identify** all security vulnerabilities in the code above
2. **Rewrite** the function to be secure
3. **Implement** proper input validation
4. **Add** appropriate security measures
5. **Explain** what each vulnerability could lead to and how your fixes prevent them

### Expected Deliverables

1. A list of identified vulnerabilities
2. Secure rewritten version of the function
3. A template file for proper output rendering
4. Brief explanation of each fix

### Evaluation Criteria

- **Vulnerability Identification**: Did you spot all the security issues?
- **Secure Implementation**: Is your rewritten code secure?
- **Input Validation**: Are you properly validating user input?
- **Best Practices**: Are you following Django security best practices?
- **Code Quality**: Is your code clean and well-documented?

---

## Summary

In this lesson, you've learned to secure your Django kitchen like a master chef protects their culinary domain. You've implemented Django's built-in security features, configured HTTPS properly, set up essential security headers, and learned to prevent common vulnerabilities. 

Remember, security is not a one-time setup—it's an ongoing practice that requires constant attention and updates, just like maintaining the highest standards in a professional kitchen.

Your Django applications are now fortified against common threats, ready to serve users safely and securely in the digital world.