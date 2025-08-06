# Day 46: Django User Authentication System - Complete Course

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand and implement Django's built-in authentication system
- Create secure login and logout functionality
- Build password reset flows with email verification
- Implement user registration with proper validation
- Apply authentication decorators to protect views
- Customize Django's authentication templates

---

## Course Introduction

**Imagine that** you're running a high-end restaurant kitchen where only authorized chefs can access certain cooking stations, recipes, and equipment. You wouldn't want just anyone walking in off the street to start cooking in your kitchen, right? 

In the same way, web applications need a security system to control who can access what. Django's authentication system is like having a master chef who manages all the kitchen passes, checks credentials, and ensures only the right people can access the right areas of your digital kitchen.

Today, we'll learn how to be that master chef, implementing a complete authentication system that keeps your web application secure while providing a smooth experience for legitimate users.

---

## Lesson 1: Django's Built-in Authentication System

### The Foundation - Understanding Django's Auth Framework

**Think of Django's authentication system as your restaurant's comprehensive security protocol.** It includes:
- **User accounts** (like chef profiles)
- **Permissions** (what each chef can do)
- **Groups** (kitchen teams)
- **Sessions** (tracking who's currently working)

### Setting Up Authentication

First, let's ensure authentication is properly configured in your Django project:

```python
# settings.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',          # Authentication framework
    'django.contrib.contenttypes',  # Content type system
    'django.contrib.sessions',      # Session framework
    'django.contrib.messages',      # Messaging framework
    'django.contrib.staticfiles',
    # Your apps here
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',  # Session management
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',  # Authentication
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Authentication URLs
LOGIN_URL = '/accounts/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'
```

**Syntax Explanation:**
- `INSTALLED_APPS`: Lists all Django applications to include in your project
- `MIDDLEWARE`: Ordered list of middleware components that process requests/responses
- `LOGIN_URL`: Where to redirect users who need to log in
- `LOGIN_REDIRECT_URL`: Where to send users after successful login

### The User Model

Django provides a built-in User model that's like a standard chef profile template:

```python
# Understanding the built-in User model
from django.contrib.auth.models import User

# The User model includes these fields by default:
# - username (unique identifier)
# - email (email address)
# - first_name (first name)
# - last_name (last name)
# - password (encrypted password)
# - is_active (account status)
# - is_staff (can access admin)
# - is_superuser (has all permissions)
# - date_joined (when account was created)
# - last_login (when user last logged in)
```

### URL Configuration

Set up your project's URL routing:

```python
# main_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),  # Built-in auth URLs
    path('', include('your_app.urls')),  # Your app URLs
]
```

**Syntax Explanation:**
- `include()`: Includes URL patterns from another URL configuration
- `django.contrib.auth.urls`: Provides pre-built authentication URLs like login, logout, password reset

---

## Lesson 2: Login/Logout Functionality

### Creating the Login System

**Imagine your kitchen has a time clock system** where chefs must punch in and out. That's exactly what our login/logout system does digitally.

### Login View and Template

```python
# views.py
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages

def custom_login(request):
    """
    Custom login view - like a chef checking in at the kitchen
    """
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        # Authenticate user (check credentials)
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Valid credentials - log them in
            login(request, user)
            messages.success(request, f'Welcome back, {user.first_name}!')
            return redirect('dashboard')  # Redirect to main kitchen area
        else:
            # Invalid credentials
            messages.error(request, 'Invalid username or password. Please try again.')
    
    return render(request, 'registration/login.html')

def custom_logout(request):
    """
    Custom logout view - like a chef clocking out
    """
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')

@login_required
def dashboard(request):
    """
    Protected view - only authenticated users can access
    Like a restricted kitchen area only for authorized chefs
    """
    return render(request, 'dashboard.html', {
        'user': request.user
    })
```

**Syntax Explanation:**
- `authenticate()`: Verifies username and password
- `login()`: Logs in a user and creates a session
- `logout()`: Logs out a user and destroys the session
- `@login_required`: Decorator that requires authentication to access the view
- `request.user`: The currently logged-in user object

### Login Template

```html
<!-- templates/registration/login.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kitchen Access - Login</title>
    <style>
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"], input[type="password"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        .btn {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        }
        .btn:hover {
            background-color: #0056b3;
        }
        .messages {
            margin-bottom: 15px;
        }
        .alert {
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .alert-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .alert-error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h2>🍳 Kitchen Access Portal</h2>
        <p>Welcome, Chef! Please enter your credentials to access the kitchen.</p>
        
        <!-- Display messages -->
        {% if messages %}
            <div class="messages">
                {% for message in messages %}
                    <div class="alert alert-{{ message.tags }}">
                        {{ message }}
                    </div>
                {% endfor %}
            </div>
        {% endif %}
        
        <form method="post">
            {% csrf_token %}
            <div class="form-group">
                <label for="username">Chef Username:</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-group">
                <label for="password">Kitchen Password:</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit" class="btn">Enter Kitchen</button>
        </form>
        
        <p style="text-align: center; margin-top: 20px;">
            <a href="{% url 'password_reset' %}">Forgot your kitchen password?</a> |
            <a href="{% url 'register' %}">New chef? Register here</a>
        </p>
    </div>
</body>
</html>
```

**Syntax Explanation:**
- `{% csrf_token %}`: Django's Cross-Site Request Forgery protection token
- `{% if messages %}`: Template tag to check if there are any messages
- `{% for message in messages %}`: Loop through all messages
- `{% url 'name' %}`: Generate URL from URL name

### Dashboard Template

```html
<!-- templates/dashboard.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kitchen Dashboard</title>
    <style>
        .dashboard-container {
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }
        .welcome-banner {
            background-color: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #007bff;
        }
        .logout-btn {
            background-color: #dc3545;
            color: white;
            padding: 8px 16px;
            text-decoration: none;
            border-radius: 4px;
            float: right;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="welcome-banner">
            <a href="{% url 'logout' %}" class="logout-btn">🚪 Leave Kitchen</a>
            <h1>🍳 Welcome to the Kitchen, {{ user.first_name|default:user.username }}!</h1>
            <p>You've successfully entered the kitchen area. Time to start cooking!</p>
            <p><strong>Chef Status:</strong> 
                {% if user.is_staff %}
                    Head Chef 👨‍🍳
                {% else %}
                    Line Cook 🧑‍🍳
                {% endif %}
            </p>
        </div>
        
        <div class="kitchen-stats">
            <h3>Kitchen Information</h3>
            <ul>
                <li><strong>Username:</strong> {{ user.username }}</li>
                <li><strong>Email:</strong> {{ user.email }}</li>
                <li><strong>Joined Kitchen:</strong> {{ user.date_joined|date:"F d, Y" }}</li>
                <li><strong>Last Visit:</strong> {{ user.last_login|date:"F d, Y g:i A" }}</li>
            </ul>
        </div>
    </div>
</body>
</html>
```

---

## Lesson 3: Password Reset Flows

### Setting Up Email Backend

**Think of password reset as having a master key system** where if a chef forgets their key, they can request a new one through a secure process.

```python
# settings.py - Email configuration
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'your-email@gmail.com'
EMAIL_HOST_PASSWORD = 'your-app-password'
DEFAULT_FROM_EMAIL = 'Kitchen Management <your-email@gmail.com>'

# For development, you can use console backend
# EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
```

### Password Reset Templates

```html
<!-- templates/registration/password_reset_form.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Kitchen Password</title>
    <style>
        .reset-container {
            max-width: 500px;
            margin: 100px auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="email"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        .btn {
            background-color: #28a745;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="reset-container">
        <h2>🔑 Reset Kitchen Password</h2>
        <p>Forgot your kitchen password? No worries! Enter your email address and we'll send you a link to reset it.</p>
        
        <form method="post">
            {% csrf_token %}
            <div class="form-group">
                <label for="email">Chef Email Address:</label>
                <input type="email" id="email" name="email" required>
            </div>
            <button type="submit" class="btn">Send Reset Link</button>
        </form>
        
        <p style="text-align: center; margin-top: 20px;">
            <a href="{% url 'login' %}">← Back to Kitchen Login</a>
        </p>
    </div>
</body>
</html>
```

```html
<!-- templates/registration/password_reset_done.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Password Reset Sent</title>
    <style>
        .success-container {
            max-width: 500px;
            margin: 100px auto;
            padding: 20px;
            border: 1px solid #28a745;
            border-radius: 8px;
            background-color: #d4edda;
            color: #155724;
        }
    </style>
</head>
<body>
    <div class="success-container">
        <h2>📧 Reset Link Sent!</h2>
        <p>We've emailed you instructions for setting your password. If an account exists with the email you entered, you should receive them shortly.</p>
        <p>If you don't receive an email, please make sure you've entered the address you registered with, and check your spam folder.</p>
        <p><a href="{% url 'login' %}">← Return to Kitchen Login</a></p>
    </div>
</body>
</html>
```

### Custom Password Reset View

```python
# views.py - Custom password reset handling
from django.contrib.auth.views import PasswordResetView
from django.urls import reverse_lazy

class CustomPasswordResetView(PasswordResetView):
    """
    Custom password reset view - like a kitchen manager handling lost keys
    """
    template_name = 'registration/password_reset_form.html'
    email_template_name = 'registration/password_reset_email.html'
    subject_template_name = 'registration/password_reset_subject.txt'
    success_url = reverse_lazy('password_reset_done')
    
    def form_valid(self, form):
        """
        Add custom logic when form is valid
        """
        response = super().form_valid(form)
        # You can add custom logic here, like logging the reset attempt
        return response
```

---

## Lesson 4: User Registration

### Registration View

**Think of user registration as the hiring process** where new chefs apply to join your kitchen team.

```python
# views.py
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib import messages
from django.contrib.auth.models import User

def register(request):
    """
    User registration view - like hiring a new chef
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            # Create the new user
            user = form.save()
            
            # Add additional user information
            user.email = request.POST.get('email', '')
            user.first_name = request.POST.get('first_name', '')
            user.last_name = request.POST.get('last_name', '')
            user.save()
            
            # Log the user in immediately after registration
            login(request, user)
            messages.success(request, f'Welcome to the kitchen, {user.first_name}! Your account has been created.')
            return redirect('dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserCreationForm()
    
    return render(request, 'registration/register.html', {'form': form})

# Custom registration form for more fields
from django import forms
from django.contrib.auth.forms import UserCreationForm

class CustomUserCreationForm(UserCreationForm):
    """
    Extended registration form - like a detailed chef application
    """
    email = forms.EmailField(required=True, help_text='Required. Enter a valid email address.')
    first_name = forms.CharField(max_length=30, required=True, help_text='Required. Enter your first name.')
    last_name = forms.CharField(max_length=30, required=True, help_text='Required. Enter your last name.')
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
    
    def save(self, commit=True):
        """
        Save the user with additional fields
        """
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
        return user

def custom_register(request):
    """
    Registration view using custom form
    """
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome to the kitchen team, {user.first_name}!')
            return redirect('dashboard')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'registration/register.html', {'form': form})
```

**Syntax Explanation:**
- `UserCreationForm`: Django's built-in form for user registration
- `forms.EmailField()`: Form field for email input with validation
- `class Meta`: Defines metadata for the form, including which model and fields to use
- `cleaned_data`: Dictionary containing validated form data
- `commit=False`: Prevents saving to database immediately, allowing modifications

### Registration Template

```html
<!-- templates/registration/register.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Join Our Kitchen Team</title>
    <style>
        .register-container {
            max-width: 500px;
            margin: 50px auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"], input[type="email"], input[type="password"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        .btn {
            background-color: #28a745;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        }
        .btn:hover {
            background-color: #218838;
        }
        .help-text {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        .errorlist {
            color: #dc3545;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .messages {
            margin-bottom: 15px;
        }
        .alert {
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .alert-error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="register-container">
        <h2>👨‍🍳 Join Our Kitchen Team</h2>
        <p>Ready to start your culinary journey? Fill out this application to join our kitchen!</p>
        
        <!-- Display messages -->
        {% if messages %}
            <div class="messages">
                {% for message in messages %}
                    <div class="alert alert-{{ message.tags }}">
                        {{ message }}
                    </div>
                {% endfor %}
            </div>
        {% endif %}
        
        <form method="post">
            {% csrf_token %}
            
            <!-- Username field -->
            <div class="form-group">
                <label for="{{ form.username.id_for_label }}">Chef Username:</label>
                {{ form.username }}
                {% if form.username.help_text %}
                    <div class="help-text">{{ form.username.help_text }}</div>
                {% endif %}
                {% if form.username.errors %}
                    <div class="errorlist">
                        {% for error in form.username.errors %}
                            <div>{{ error }}</div>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
            
            <!-- First Name field -->
            {% if form.first_name %}
            <div class="form-group">
                <label for="{{ form.first_name.id_for_label }}">First Name:</label>
                {{ form.first_name }}
                {% if form.first_name.help_text %}
                    <div class="help-text">{{ form.first_name.help_text }}</div>
                {% endif %}
                {% if form.first_name.errors %}
                    <div class="errorlist">
                        {% for error in form.first_name.errors %}
                            <div>{{ error }}</div>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
            {% endif %}
            
            <!-- Last Name field -->
            {% if form.last_name %}
            <div class="form-group">
                <label for="{{ form.last_name.id_for_label }}">Last Name:</label>
                {{ form.last_name }}
                {% if form.last_name.help_text %}
                    <div class="help-text">{{ form.last_name.help_text }}</div>
                {% endif %}
                {% if form.last_name.errors %}
                    <div class="errorlist">
                        {% for error in form.last_name.errors %}
                            <div>{{ error }}</div>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
            {% endif %}
            
            <!-- Email field -->
            {% if form.email %}
            <div class="form-group">
                <label for="{{ form.email.id_for_label }}">Email Address:</label>
                {{ form.email }}
                {% if form.email.help_text %}
                    <div class="help-text">{{ form.email.help_text }}</div>
                {% endif %}
                {% if form.email.errors %}
                    <div class="errorlist">
                        {% for error in form.email.errors %}
                            <div>{{ error }}</div>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
            {% endif %}
            
            <!-- Password1 field -->
            <div class="form-group">
                <label for="{{ form.password1.id_for_label }}">Kitchen Password:</label>
                {{ form.password1 }}
                {% if form.password1.help_text %}
                    <div class="help-text">{{ form.password1.help_text }}</div>
                {% endif %}
                {% if form.password1.errors %}
                    <div class="errorlist">
                        {% for error in form.password1.errors %}
                            <div>{{ error }}</div>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
            
            <!-- Password2 field -->
            <div class="form-group">
                <label for="{{ form.password2.id_for_label }}">Confirm Password:</label>
                {{ form.password2 }}
                {% if form.password2.help_text %}
                    <div class="help-text">{{ form.password2.help_text }}</div>
                {% endif %}
                {% if form.password2.errors %}
                    <div class="errorlist">
                        {% for error in form.password2.errors %}
                            <div>{{ error }}</div>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
            
            <button type="submit" class="btn">🍳 Join the Kitchen Team</button>
        </form>
        
        <p style="text-align: center; margin-top: 20px;">
            Already part of our kitchen? <a href="{% url 'login' %}">Sign in here</a>
        </p>
    </div>
</body>
</html>
```

### URL Configuration

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('login/', views.custom_login, name='login'),
    path('logout/', views.custom_logout, name='logout'),
    path('register/', views.custom_register, name='register'),
    path('password-reset/', views.CustomPasswordResetView.as_view(), name='password_reset'),
    # Django's built-in auth URLs will handle the rest
]
```

---

# Project: Complete User Authentication System for Expense Tracker

## Project Objective
By the end of this project, you will be able to implement a complete user authentication system for your expense tracker application, including user registration, login/logout functionality, password reset flows, and secure user session management - transforming your single-user expense tracker into a multi-user platform where each chef can manage their own financial kitchen.

---

## Introduction: The Restaurant Empire

Imagine that you've been running a successful neighborhood restaurant with just one chef (yourself) keeping track of all the expenses in a single ledger. Your restaurant has grown so popular that you now want to expand into a restaurant empire with multiple locations, each with their own head chef. 

Each chef needs their own private kitchen (user account) where they can track their location's expenses without seeing or interfering with other chefs' financial records. Just like how a restaurant chain has a secure system for each manager to access only their location's data, we're going to build a robust authentication system that gives each user their own secure "kitchen" in our expense tracker.

Think of authentication as the restaurant's security system - it checks who's trying to enter (login), gives them the right keys to their kitchen (session management), helps them if they forget their keys (password reset), and allows new chefs to join the team (registration).

---

## Phase 1: Setting Up the Authentication Foundation

### Understanding Django's Authentication Architecture

Django's authentication system is like having a master key system for your restaurant empire. Let's start by understanding the components:

```python
# settings.py - The master security blueprint for your restaurant chain
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',           # The security guard
    'django.contrib.contenttypes',   # The inventory system
    'django.contrib.sessions',       # The key tracking system
    'django.contrib.messages',       # The communication system
    'django.contrib.staticfiles',
    'expenses',                      # Your expense tracking kitchen
]

# Security settings - like the restaurant's safety protocols
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 8,
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Login/logout redirect settings - where chefs go after entering/leaving
LOGIN_REDIRECT_URL = 'expense_list'
LOGOUT_REDIRECT_URL = 'home'
LOGIN_URL = 'login'
```

**Syntax Explanation:**
- `AUTH_PASSWORD_VALIDATORS`: A list of dictionaries defining password strength rules
- `LOGIN_REDIRECT_URL`: Where users go after successful login (like directing a chef to their kitchen)
- `LOGOUT_REDIRECT_URL`: Where users go after logging out (like the restaurant's front entrance)
- `LOGIN_URL`: The default login page URL for protected views

---

## Phase 2: Creating the Authentication Views

### The Security Checkpoint System

Let's create our authentication views - think of these as the different security checkpoints in your restaurant:

```python
# views.py - The security checkpoint controllers
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.views import PasswordResetView
from django.urls import reverse_lazy
from django.core.mail import send_mail
from django.contrib.auth.models import User

def home(request):
    """
    The restaurant's front entrance - what everyone sees first
    """
    return render(request, 'registration/home.html')

def register_view(request):
    """
    The hiring office - where new chefs join the team
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            # Create the new chef's account
            user = form.save()
            username = form.cleaned_data.get('username')
            
            # Give them their kitchen keys immediately
            login(request, user)
            messages.success(request, f'Welcome to the team, Chef {username}! Your kitchen is ready.')
            return redirect('expense_list')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserCreationForm()
    
    return render(request, 'registration/register.html', {'form': form})

def login_view(request):
    """
    The main entrance security checkpoint
    """
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        # Check if this chef has valid credentials
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Give them access to their kitchen
            login(request, user)
            messages.success(request, f'Welcome back, Chef {username}!')
            
            # Take them to their intended destination or their kitchen
            next_url = request.GET.get('next', 'expense_list')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid credentials. Please try again.')
    
    return render(request, 'registration/login.html')

def logout_view(request):
    """
    The exit checkpoint - secure logout
    """
    logout(request)
    messages.success(request, 'You have been logged out successfully. Come back soon!')
    return redirect('home')

class CustomPasswordResetView(PasswordResetView):
    """
    The lost keys service - helping chefs who forgot their passwords
    """
    template_name = 'registration/password_reset_form.html'
    email_template_name = 'registration/password_reset_email.html'
    success_url = reverse_lazy('password_reset_done')
    
    def form_valid(self, form):
        messages.success(self.request, 'Password reset instructions have been sent to your email.')
        return super().form_valid(form)
```

**Syntax Explanation:**
- `authenticate(request, username=username, password=password)`: Checks if credentials are valid
- `login(request, user)`: Creates a session for the user (gives them their kitchen keys)
- `logout(request)`: Destroys the session (takes away the keys)
- `@login_required` decorator: Ensures only authenticated users can access certain views
- `request.GET.get('next', 'expense_list')`: Gets the 'next' parameter from URL or defaults to 'expense_list'

---

## Phase 3: URL Configuration - The Restaurant's Directory

```python
# urls.py - The restaurant's directory system
from django.urls import path, include
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Main entrance and reception
    path('', views.home, name='home'),
    
    # Registration desk - where new chefs sign up
    path('register/', views.register_view, name='register'),
    
    # Security checkpoints
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Lost keys service (password reset)
    path('password_reset/', views.CustomPasswordResetView.as_view(), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),
    
    # The kitchen area - where the real work happens
    path('expenses/', include('expenses.urls')),
]
```

**Syntax Explanation:**
- `<uidb64>/<token>/`: URL parameters for password reset security
- `auth_views.PasswordResetDoneView.as_view()`: Django's built-in class-based view
- `include('expenses.urls')`: Includes URLs from the expenses app

---

## Phase 4: Securing the Kitchen - Protecting Expense Views

Now let's modify your expense tracker to ensure each chef only sees their own expenses:

```python
# expenses/models.py - The recipe book with ownership
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class Expense(models.Model):
    """
    Each expense belongs to a specific chef (user)
    """
    CATEGORY_CHOICES = [
        ('food', 'Food & Ingredients'),
        ('utilities', 'Utilities'),
        ('equipment', 'Kitchen Equipment'),
        ('supplies', 'Supplies'),
        ('maintenance', 'Maintenance'),
        ('other', 'Other'),
    ]
    
    # The chef who owns this expense record
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='expenses')
    
    title = models.CharField(max_length=200)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    date = models.DateField()
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date', '-created_at']
    
    def __str__(self):
        return f"{self.title} - ${self.amount} ({self.user.username})"
    
    def get_absolute_url(self):
        return reverse('expense_detail', kwargs={'pk': self.pk})

# expenses/views.py - The secure kitchen operations
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Sum
from django.utils import timezone
from .models import Expense
from .forms import ExpenseForm

@login_required
def expense_list(request):
    """
    Chef's personal expense dashboard - their own kitchen's financial records
    """
    # Only show expenses belonging to this chef
    expenses = Expense.objects.filter(user=request.user)
    
    # Calculate this chef's total spending
    total_spending = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
    
    # Get spending by category for this chef
    category_spending = {}
    for category, category_name in Expense.CATEGORY_CHOICES:
        amount = expenses.filter(category=category).aggregate(Sum('amount'))['amount__sum'] or 0
        if amount > 0:
            category_spending[category_name] = amount
    
    context = {
        'expenses': expenses,
        'total_spending': total_spending,
        'category_spending': category_spending,
        'chef_name': request.user.username,
    }
    
    return render(request, 'expenses/expense_list.html', context)

@login_required
def expense_create(request):
    """
    Adding a new expense to the chef's records
    """
    if request.method == 'POST':
        form = ExpenseForm(request.POST)
        if form.is_valid():
            # Create the expense but don't save to database yet
            expense = form.save(commit=False)
            # Assign it to the current chef
            expense.user = request.user
            expense.save()
            
            messages.success(request, 'Expense added successfully!')
            return redirect('expense_list')
    else:
        form = ExpenseForm()
    
    return render(request, 'expenses/expense_form.html', {'form': form, 'title': 'Add New Expense'})

@login_required
def expense_detail(request, pk):
    """
    View a specific expense - but only if it belongs to this chef
    """
    expense = get_object_or_404(Expense, pk=pk, user=request.user)
    return render(request, 'expenses/expense_detail.html', {'expense': expense})

@login_required
def expense_update(request, pk):
    """
    Update an expense - but only if it belongs to this chef
    """
    expense = get_object_or_404(Expense, pk=pk, user=request.user)
    
    if request.method == 'POST':
        form = ExpenseForm(request.POST, instance=expense)
        if form.is_valid():
            form.save()
            messages.success(request, 'Expense updated successfully!')
            return redirect('expense_detail', pk=expense.pk)
    else:
        form = ExpenseForm(instance=expense)
    
    return render(request, 'expenses/expense_form.html', {
        'form': form, 
        'expense': expense,
        'title': 'Update Expense'
    })

@login_required
def expense_delete(request, pk):
    """
    Delete an expense - but only if it belongs to this chef
    """
    expense = get_object_or_404(Expense, pk=pk, user=request.user)
    
    if request.method == 'POST':
        expense.delete()
        messages.success(request, 'Expense deleted successfully!')
        return redirect('expense_list')
    
    return render(request, 'expenses/expense_confirm_delete.html', {'expense': expense})
```

**Syntax Explanation:**
- `models.ForeignKey(User, on_delete=models.CASCADE)`: Links each expense to a user
- `@login_required`: Decorator that ensures only logged-in users can access the view
- `Expense.objects.filter(user=request.user)`: Filters expenses to only show current user's records
- `get_object_or_404(Expense, pk=pk, user=request.user)`: Gets object or returns 404 if not found or not owned by user
- `form.save(commit=False)`: Creates model instance without saving to database
- `expense.user = request.user`: Assigns the expense to the current user

---

## Phase 5: The Templates - The Restaurant's Visual Identity

### Base Template - The Restaurant's Foundation

```html
<!-- templates/base.html - The restaurant's main layout -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Chef's Expense Tracker{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .navbar-brand {
            font-weight: bold;
            color: #dc3545 !important;
        }
        .chef-welcome {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
    </style>
</head>
<body>
    <!-- Navigation - The restaurant's main menu -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{% url 'home' %}">
                <i class="fas fa-utensils"></i> Chef's Expense Tracker
            </a>
            
            <div class="navbar-nav ms-auto">
                {% if user.is_authenticated %}
                    <a class="nav-link" href="{% url 'expense_list' %}">
                        <i class="fas fa-clipboard-list"></i> My Kitchen
                    </a>
                    <a class="nav-link" href="{% url 'expense_create' %}">
                        <i class="fas fa-plus"></i> Add Expense
                    </a>
                    <a class="nav-link" href="{% url 'logout' %}">
                        <i class="fas fa-sign-out-alt"></i> Logout
                    </a>
                {% else %}
                    <a class="nav-link" href="{% url 'login' %}">
                        <i class="fas fa-sign-in-alt"></i> Login
                    </a>
                    <a class="nav-link" href="{% url 'register' %}">
                        <i class="fas fa-user-plus"></i> Join the Team
                    </a>
                {% endif %}
            </div>
        </div>
    </nav>

    <!-- Main content area -->
    <div class="container mt-4">
        <!-- Welcome message for authenticated chefs -->
        {% if user.is_authenticated %}
            <div class="chef-welcome">
                <h4><i class="fas fa-chef-hat"></i> Welcome back, Chef {{ user.username }}!</h4>
                <p class="mb-0">Your kitchen is ready. Let's manage those expenses!</p>
            </div>
        {% endif %}
        
        <!-- Messages - Kitchen announcements -->
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}
        
        <!-- Page content -->
        {% block content %}
        {% endblock %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

### Login Template - The Security Checkpoint

```html
<!-- templates/registration/login.html -->
{% extends 'base.html' %}

{% block title %}Login - Chef's Expense Tracker{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header text-center">
                <h3><i class="fas fa-key"></i> Kitchen Access</h3>
                <p class="text-muted">Enter your credentials to access your kitchen</p>
            </div>
            <div class="card-body">
                <form method="post">
                    {% csrf_token %}
                    <div class="mb-3">
                        <label for="username" class="form-label">Chef Username</label>
                        <input type="text" class="form-control" id="username" name="username" required>
                    </div>
                    <div class="mb-3">
                        <label for="password" class="form-label">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">
                        <i class="fas fa-sign-in-alt"></i> Enter Kitchen
                    </button>
                </form>
                
                <div class="text-center mt-3">
                    <a href="{% url 'password_reset' %}">Forgot your keys?</a>
                </div>
                
                <hr>
                <div class="text-center">
                    <p>New chef? <a href="{% url 'register' %}">Join our team</a></p>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

### Registration Template - The Hiring Office

```html
<!-- templates/registration/register.html -->
{% extends 'base.html' %}

{% block title %}Join the Team - Chef's Expense Tracker{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header text-center">
                <h3><i class="fas fa-user-plus"></i> Join Our Kitchen Team</h3>
                <p class="text-muted">Create your chef account and get your own kitchen</p>
            </div>
            <div class="card-body">
                <form method="post">
                    {% csrf_token %}
                    {{ form.as_p }}
                    <button type="submit" class="btn btn-success w-100">
                        <i class="fas fa-chef-hat"></i> Get My Kitchen
                    </button>
                </form>
                
                <hr>
                <div class="text-center">
                    <p>Already have an account? <a href="{% url 'login' %}">Login here</a></p>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

---

## Phase 6: The Final Quality Project - Complete Integration

Let's create a comprehensive dashboard that showcases all authentication features:

```python
# expenses/views.py - Enhanced dashboard view
@login_required
def dashboard(request):
    """
    Chef's complete kitchen dashboard with analytics
    """
    # Get all expenses for this chef
    expenses = Expense.objects.filter(user=request.user)
    
    # Calculate statistics
    total_expenses = expenses.count()
    total_amount = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
    
    # Get recent expenses (last 5)
    recent_expenses = expenses[:5]
    
    # Monthly spending
    from django.db.models import Extract
    current_month = timezone.now().month
    monthly_spending = expenses.filter(
        date__month=current_month
    ).aggregate(Sum('amount'))['amount__sum'] or 0
    
    # Category breakdown
    category_data = []
    for category, category_name in Expense.CATEGORY_CHOICES:
        amount = expenses.filter(category=category).aggregate(Sum('amount'))['amount__sum'] or 0
        if amount > 0:
            category_data.append({
                'category': category_name,
                'amount': amount,
                'percentage': (amount / total_amount * 100) if total_amount > 0 else 0
            })
    
    context = {
        'total_expenses': total_expenses,
        'total_amount': total_amount,
        'monthly_spending': monthly_spending,
        'recent_expenses': recent_expenses,
        'category_data': category_data,
        'chef_name': request.user.username,
    }
    
    return render(request, 'expenses/dashboard.html', context)
```

### Dashboard Template - The Master Kitchen View

```html
<!-- templates/expenses/dashboard.html -->
{% extends 'base.html' %}

{% block title %}Dashboard - Chef {{ chef_name }}'s Kitchen{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h2><i class="fas fa-tachometer-alt"></i> Kitchen Dashboard</h2>
        <p class="text-muted">Your complete expense management overview</p>
    </div>
</div>

<!-- Statistics Cards -->
<div class="row mb-4">
    <div class="col-md-3">
        <div class="card text-white bg-primary">
            <div class="card-body">
                <h5 class="card-title">Total Expenses</h5>
                <h3>{{ total_expenses }}</h3>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-white bg-success">
            <div class="card-body">
                <h5 class="card-title">Total Amount</h5>
                <h3>${{ total_amount }}</h3>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-white bg-info">
            <div class="card-body">
                <h5 class="card-title">This Month</h5>
                <h3>${{ monthly_spending }}</h3>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-white bg-warning">
            <div class="card-body">
                <h5 class="card-title">Average</h5>
                <h3>${{ total_amount|floatformat:2 }}</h3>
            </div>
        </div>
    </div>
</div>

<!-- Recent Expenses and Categories -->
<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5>Recent Expenses</h5>
            </div>
            <div class="card-body">
                {% for expense in recent_expenses %}
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <div>
                            <strong>{{ expense.title }}</strong><br>
                            <small class="text-muted">{{ expense.date }}</small>
                        </div>
                        <span class="badge bg-primary">${{ expense.amount }}</span>
                    </div>
                {% empty %}
                    <p class="text-muted">No expenses yet. <a href="{% url 'expense_create' %}">Add your first expense</a></p>
                {% endfor %}
            </div>
        </div>
    </div>
    
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5>Spending by Category</h5>
            </div>
            <div class="card-body">
                {% for item in category_data %}
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <span>{{ item.category }}</span>
                            <span>${{ item.amount }}</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar" style="width: {{ item.percentage }}%"></div>
                        </div>
                    </div>
                {% empty %}
                    <p class="text-muted">No expenses to categorize yet.</p>
                {% endfor %}
            </div>
        </div>
    </div>
</div>

<!-- Quick Actions -->
<div class="row mt-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5>Quick Actions</h5>
            </div>
            <div class="card-body">
                <a href="{% url 'expense_create' %}" class="btn btn-primary me-2">
                    <i class="fas fa-plus"></i> Add New Expense
                </a>
                <a href="{% url 'expense_list' %}" class="btn btn-outline-primary">
                    <i class="fas fa-list"></i> View All Expenses
                </a>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

---

## Putting It All Together - The Complete Authentication System

Your expense tracker now has a complete multi-user authentication system where:

1. **New chefs can join** through the registration system
2. **Existing chefs can access their kitchen** through secure login
3. **Each chef sees only their own expenses** through user-specific filtering
4. **Forgotten passwords can be reset** through the email system
5. **Sessions are managed securely** with proper logout functionality

The analogy helps us understand that each user has their own private space (kitchen) where they can manage their expenses without interference from others, just like how restaurant managers each have their own office and records.

## Key Security Features Implemented:

- **User isolation**: Each user only sees their own data
- **Session management**: Secure login/logout with proper redirects
- **Password validation**: Strong password requirements
- **CSRF protection**: All forms include CSRF tokens
- **Access control**: Login required for all expense operations
- **Object-level permissions**: Users can only modify their own expenses

This authentication system transforms your single-user expense tracker into a robust multi-user platform suitable for teams, families, or any group where multiple people need to track expenses separately but within the same application framework.

**Syntax Summary:**
- `@login_required`: Protects views from unauthorized access
- `request.user`: The currently logged-in user object
- `user.is_authenticated`: Boolean check for authentication status
- `{% csrf_token %}`: Template tag for CSRF protection
- `get_object_or_404(Model, pk=pk, user=request.user)`: Secure object retrieval with ownership check

## Assignment: Kitchen Staff Management System

### Project Description
Create a complete authentication system for a restaurant's staff management portal. This system should allow kitchen staff to register, login, logout, and reset their passwords.

### Requirements

1. **User Registration**
   - Custom registration form with fields: username, first name, last name, email, password
   - Form validation and error handling
   - Automatic login after successful registration
   - Welcome message after registration

2. **Login System**
   - Custom login form with username and password
   - Remember user sessions
   - Redirect to dashboard after successful login
   - Error messages for invalid credentials

3. **Password Reset**
   - Email-based password reset system
   - Custom email templates
   - Success and confirmation pages

4. **Protected Dashboard**
   - Only accessible to authenticated users
   - Display user information and kitchen stats
   - Logout functionality

5. **Design Requirements**
   - Use the kitchen/chef theme throughout
   - Responsive design that works on mobile and desktop
   - Clean, professional styling
   - Proper error handling and user feedback

### Deliverables

1. **Django Views** (`views.py`)
   - Custom registration view using `CustomUserCreationForm`
   - Custom login/logout views
   - Password reset view
   - Protected dashboard view

2. **Templates**
   - `registration/register.html`
   - `registration/login.html`
   - `registration/password_reset_form.html`
   - `registration/password_reset_done.html`
   - `dashboard.html`

3. **URL Configuration**
   - Properly configured URL patterns
   - Named URLs for easy navigation

4. **Settings Configuration**
   - Email backend setup
   - Authentication settings
   - Static files configuration

### Bonus Features (Optional)
- Add user profile pictures
- Implement user role-based access (Head Chef, Sous Chef, Line Cook)
- Add password strength indicator
- Email verification for new registrations

### Evaluation Criteria
- **Functionality** (40%): All authentication features work correctly
- **Code Quality** (30%): Clean, well-commented, and organized code
- **User Experience** (20%): Intuitive interface and proper error handling
- **Design** (10%): Professional appearance and responsive layout

### Submission Guidelines
1. Create a new Django project called `kitchen_auth`
2. Implement all required features
3. Test all functionality thoroughly
4. Document any setup instructions
5. Submit the complete project folder

**Good luck, chef! 👨‍🍳**

---

## Course Summary

**Congratulations!** You've successfully learned how to implement a complete authentication system in Django. Just like a well-organized kitchen needs proper access control, your web applications now have the security infrastructure they need.

### Key Concepts Covered
- Django's built-in authentication framework
- User model and session management
- Login/logout functionality with custom views
- Password reset flows with email integration
- User registration with form validation
- Template integration and URL routing

Remember: Authentication is like the foundation of a kitchen - it needs to be solid and secure before you can build amazing applications on top of it!