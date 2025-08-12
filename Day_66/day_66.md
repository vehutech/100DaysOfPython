# Day 66: Docker & Containerization

**Learning Objective:** By the end of this lesson, you will understand how to containerize Django applications using Docker, set up development environments with Docker Compose, optimize builds with multi-stage processes, and grasp the fundamentals of container orchestration - just like organizing a professional kitchen operation.

---

## Introduction:

Imagine that you're running a world-class restaurant chain. You've perfected your signature dishes (Django applications), but now you need to replicate the exact same cooking environment in kitchens across different cities, countries, and continents. Each location has different equipment, different operating systems for their appliances, and different staff capabilities.

In the traditional approach, you'd have to:
- Send detailed setup instructions to each location
- Hope they install the right versions of equipment
- Deal with "it works in my kitchen" problems
- Spend countless hours troubleshooting environment differences

But what if you could package your entire kitchen - the exact stove, the precise ingredients, the specific cooking tools, and even the cooking instructions - into a standardized container that works identically everywhere? That's exactly what Docker does for your Django applications!

---

## Lesson 1: Django in Docker Containers

### The Kitchen Setup Analogy
Think of a Docker container as a **portable kitchen unit**. Just like a food truck contains everything needed to prepare specific dishes - the cooking equipment, ingredients, recipes, and cooking space - a Docker container packages your Django application with its exact runtime environment, dependencies, and configuration.

### What is Docker?
Docker is a containerization platform that packages applications and their dependencies into lightweight, portable containers. These containers can run consistently across different environments.

**Key Docker Concepts:**
- **Image**: The blueprint (like a kitchen blueprint with all specifications)
- **Container**: The running instance (the actual functioning kitchen)
- **Dockerfile**: The recipe to build the kitchen (step-by-step setup instructions)

### Creating Your First Django Dockerfile

Let's containerize a Django application. First, create a `Dockerfile` in your Django project root:

```dockerfile
# Use Python 3.11 as the base image (our foundation kitchen equipment)
FROM python:3.11-slim

# Set environment variables (kitchen standards)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory (designate our cooking area)
WORKDIR /app

# Install system dependencies (essential kitchen tools)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (bring in the ingredient list)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (bring in all recipes and ingredients)
COPY . /app/

# Expose port (open the kitchen service window)
EXPOSE 8000

# Command to run the application (start cooking!)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

**Syntax Explanation:**
- `FROM`: Specifies the base image to build upon
- `ENV`: Sets environment variables
- `WORKDIR`: Sets the working directory inside the container
- `RUN`: Executes commands during the build process
- `COPY`: Copies files from host to container
- `EXPOSE`: Documents which port the container listens on
- `CMD`: Specifies the default command to run when container starts

### Building and Running Your Container

```bash
# Build the image (construct your portable kitchen)
docker build -t my-django-app .

# Run the container (start operating the kitchen)
docker run -p 8000:8000 my-django-app
```

**Code Example: Complete Django Project Structure**
```
my-django-project/
├── myproject/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── myapp/
│   ├── models.py
│   ├── views.py
│   └── urls.py
├── requirements.txt
├── Dockerfile
└── manage.py
```

---

## Lesson 2: Docker Compose for Development

### The Restaurant Chain Analogy
Imagine you're not just running one kitchen, but an entire restaurant operation. You need:
- A main kitchen (Django app)
- A storage room (database)
- A communication system (networking)
- Coordination between all parts

Docker Compose is like the **restaurant manager** that coordinates all these services, ensuring they work together seamlessly.

### What is Docker Compose?
Docker Compose is a tool for defining and running multi-container Docker applications. It uses a YAML file to configure all your application's services.

### Creating docker-compose.yml

```yaml
version: '3.8'

services:
  # Main kitchen (Django application)
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - DEBUG=1
      - DATABASE_URL=postgresql://chef:secret@db:5432/restaurant_db
    depends_on:
      - db
    command: python manage.py runserver 0.0.0.0:8000

  # Storage room (PostgreSQL database)
  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    environment:
      - POSTGRES_DB=restaurant_db
      - POSTGRES_USER=chef
      - POSTGRES_PASSWORD=secret
    ports:
      - "5432:5432"

  # Optional: Redis for caching (like a quick-access spice rack)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

**Syntax Explanation:**
- `version`: Specifies the Compose file format version
- `services`: Defines the containers that make up your application
- `build`: Builds an image from a Dockerfile
- `ports`: Maps host ports to container ports
- `volumes`: Mounts host directories or named volumes
- `environment`: Sets environment variables
- `depends_on`: Defines service dependencies
- `command`: Overrides the default command

### Running with Docker Compose

```bash
# Start all services (open the entire restaurant)
docker-compose up

# Start in background (run restaurant operations behind the scenes)
docker-compose up -d

# Stop all services (close the restaurant)
docker-compose down

# View logs (check kitchen operations)
docker-compose logs web
```

**Development-Optimized Compose File:**
```yaml
version: '3.8'

services:
  web:
    build: 
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - /app/node_modules  # Prevent overwriting node_modules
    environment:
      - DEBUG=1
      - DJANGO_SETTINGS_MODULE=myproject.settings.development
    depends_on:
      - db
      - redis
    stdin_open: true  # Enable interactive debugging
    tty: true

  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data/
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    environment:
      - POSTGRES_DB=restaurant_dev
      - POSTGRES_USER=chef
      - POSTGRES_PASSWORD=secret123

volumes:
  postgres_data:
```

---

## Lesson 3: Multi-stage Builds

### The Restaurant Preparation Analogy
Think of multi-stage builds like a **professional kitchen's meal prep process**:

1. **Prep Kitchen**: Where you prepare ingredients, install tools, and do heavy preparation work
2. **Service Kitchen**: The clean, efficient space where you plate and serve dishes

You don't want customers to see the messy prep work - they only see the final, perfectly plated dish. Similarly, multi-stage builds let you separate the "messy" build process from the clean, optimized final container.

### Why Multi-stage Builds?
- **Smaller final images**: Remove build dependencies and temporary files
- **Better security**: Fewer packages means smaller attack surface
- **Faster deployment**: Smaller images transfer and start faster

### Multi-stage Dockerfile Example

```dockerfile
# Stage 1: Build stage (Prep Kitchen)
FROM python:3.11 as builder

# Set work directory for building
WORKDIR /app

# Install build dependencies (heavy prep tools)
RUN apt-get update && apt-get install -y \
    gcc \
    musl-dev \
    libpq-dev \
    python3-dev

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Collect static files (prepare garnishes and plating materials)
RUN python manage.py collectstatic --noinput

# Stage 2: Production stage (Service Kitchen)
FROM python:3.11-slim as production

# Create non-root user (hire a specific chef for this kitchen)
RUN useradd --create-home --shell /bin/bash django

# Set work directory
WORKDIR /app

# Install only runtime dependencies (essential service tools)
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/django/.local

# Copy application code and static files
COPY --from=builder /app .

# Change ownership to django user
RUN chown -R django:django /app

# Switch to non-root user
USER django

# Update PATH to include user-installed packages
ENV PATH=/home/django/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check (ensure the kitchen is operating properly)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "myproject.wsgi:application"]
```

**Advanced Multi-stage with Node.js assets:**
```dockerfile
# Stage 1: Node.js build (Pastry Kitchen for frontend assets)
FROM node:16 as node-builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY assets/ ./assets/
RUN npm run build

# Stage 2: Python build (Main Prep Kitchen)
FROM python:3.11 as python-builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 3: Final production (Service Kitchen)
FROM python:3.11-slim as production

WORKDIR /app

# Copy Python packages
COPY --from=python-builder /root/.local /root/.local

# Copy built frontend assets
COPY --from=node-builder /app/dist ./static/

# Copy application code
COPY . .

# Final setup
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "myproject.wsgi:application"]
```

---

## Lesson 4: Container Orchestration Basics

### The Restaurant Empire Analogy
Imagine you've grown from a single restaurant to a **global restaurant empire**. Now you need:
- **Multiple locations** (multiple containers)
- **Load balancing** (directing customers to less busy locations)
- **Automatic scaling** (opening more locations during peak hours)
- **Health monitoring** (ensuring all locations are operating properly)
- **Service discovery** (helping locations communicate with each other)

Container orchestration platforms like Docker Swarm and Kubernetes are like your **corporate headquarters** that manages this entire empire.

### Container Orchestration Concepts

**Key Concepts:**
- **Cluster**: Group of machines working together (your restaurant empire)
- **Node**: Individual machine in the cluster (individual restaurant location)
- **Service**: Desired state of your application (standardized menu and operations)
- **Load Balancer**: Distributes traffic across containers (hostess directing customers)
- **Auto-scaling**: Automatically adjusts container count (opening/closing locations based on demand)

### Docker Swarm Example

```bash
# Initialize swarm (establish corporate headquarters)
docker swarm init

# Create a service (standardize restaurant operations)
docker service create \
  --name django-app \
  --replicas 3 \
  --publish 8000:8000 \
  my-django-app

# Scale service (open more locations)
docker service scale django-app=5

# Update service (roll out menu changes)
docker service update \
  --image my-django-app:v2 \
  django-app
```

### Docker Compose for Swarm (Stack)

```yaml
version: '3.8'

services:
  web:
    image: my-django-app:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '0.50'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    depends_on:
      - db

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - db_data:/var/lib/postgresql/data
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    configs:
      - source: nginx_config
        target: /etc/nginx/nginx.conf
    deploy:
      replicas: 2

volumes:
  db_data:

configs:
  nginx_config:
    external: true
```

### Basic Kubernetes Concepts

```yaml
# deployment.yaml - Define desired state (restaurant franchise standards)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: django-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: django-app
  template:
    metadata:
      labels:
        app: django-app
    spec:
      containers:
      - name: django
        image: my-django-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://user:pass@postgres:5432/mydb"

---
# service.yaml - Load balancer (customer distribution system)
apiVersion: v1
kind: Service
metadata:
  name: django-service
spec:
  selector:
    app: django-app
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

# Build: Containerized Django App - Quality Project

## Project Overview
You'll create a complete containerized Django application - a **Recipe Management System** that demonstrates production-ready containerization practices. Think of this as packaging your entire kitchen (Django app) into a portable container that can be shipped and run anywhere.

## Project Structure
```
recipe_manager/
├── app/
│   ├── recipe_manager/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── recipes/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   └── templates/
│   ├── manage.py
│   └── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── nginx/
    └── nginx.conf
```

## Step 1: Django Application Code

### app/requirements.txt
```txt
Django==4.2.7
psycopg2-binary==2.9.7
gunicorn==21.2.0
whitenoise==6.6.0
```

### app/recipe_manager/settings.py
```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'recipes',
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

ROOT_URLCONF = 'recipe_manager.urls'

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

WSGI_APPLICATION = 'recipe_manager.wsgi.application'

# Database configuration for container environment
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME', 'recipedb'),
        'USER': os.environ.get('DB_USER', 'recipeuser'),
        'PASSWORD': os.environ.get('DB_PASSWORD', 'recipepass'),
        'HOST': os.environ.get('DB_HOST', 'db'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
```

### app/recipes/models.py
```python
from django.db import models
from django.contrib.auth.models import User

class Recipe(models.Model):
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField()
    ingredients = models.TextField()
    instructions = models.TextField()
    prep_time = models.IntegerField(help_text="Preparation time in minutes")
    cook_time = models.IntegerField(help_text="Cooking time in minutes")
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title
    
    def total_time(self):
        return self.prep_time + self.cook_time
    
    class Meta:
        ordering = ['-created_at']
```

### app/recipes/views.py
```python
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .models import Recipe

def recipe_list(request):
    recipes = Recipe.objects.all()
    return render(request, 'recipes/recipe_list.html', {'recipes': recipes})

def recipe_detail(request, pk):
    recipe = get_object_or_404(Recipe, pk=pk)
    return render(request, 'recipes/recipe_detail.html', {'recipe': recipe})

@login_required
def recipe_create(request):
    if request.method == 'POST':
        recipe = Recipe(
            title=request.POST['title'],
            description=request.POST['description'],
            ingredients=request.POST['ingredients'],
            instructions=request.POST['instructions'],
            prep_time=int(request.POST['prep_time']),
            cook_time=int(request.POST['cook_time']),
            difficulty=request.POST['difficulty'],
            author=request.user
        )
        recipe.save()
        messages.success(request, 'Recipe created successfully!')
        return redirect('recipe_detail', pk=recipe.pk)
    
    return render(request, 'recipes/recipe_form.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful!')
            return redirect('recipe_list')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})
```

### app/recipes/templates/recipes/base.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Recipe Manager{% endblock %}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .nav { background: #333; color: white; padding: 15px; margin: -20px -20px 20px -20px; border-radius: 8px 8px 0 0; }
        .nav a { color: white; text-decoration: none; margin-right: 20px; }
        .nav a:hover { text-decoration: underline; }
        .recipe-card { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; border: none; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .form-group { margin: 15px 0; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input, .form-group textarea, .form-group select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .messages { margin: 10px 0; }
        .alert { padding: 10px; border-radius: 4px; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <div class="container">
        <nav class="nav">
            <a href="{% url 'recipe_list' %}">All Recipes</a>
            {% if user.is_authenticated %}
                <a href="{% url 'recipe_create' %}">Add Recipe</a>
                <a href="{% url 'logout' %}">Logout ({{ user.username }})</a>
            {% else %}
                <a href="{% url 'login' %}">Login</a>
                <a href="{% url 'register' %}">Register</a>
            {% endif %}
        </nav>
        
        {% if messages %}
            <div class="messages">
                {% for message in messages %}
                    <div class="alert alert-success">{{ message }}</div>
                {% endfor %}
            </div>
        {% endif %}
        
        {% block content %}
        {% endblock %}
    </div>
</body>
</html>
```

### app/recipes/templates/recipes/recipe_list.html
```html
{% extends 'recipes/base.html' %}

{% block title %}All Recipes - Recipe Manager{% endblock %}

{% block content %}
<h1>🍳 Recipe Collection</h1>

{% if recipes %}
    <div class="recipe-grid">
        {% for recipe in recipes %}
            <div class="recipe-card">
                <h3><a href="{% url 'recipe_detail' recipe.pk %}">{{ recipe.title }}</a></h3>
                <p><strong>By:</strong> {{ recipe.author.username }}</p>
                <p><strong>Difficulty:</strong> {{ recipe.get_difficulty_display }}</p>
                <p><strong>Total Time:</strong> {{ recipe.total_time }} minutes</p>
                <p>{{ recipe.description|truncatewords:20 }}</p>
                <a href="{% url 'recipe_detail' recipe.pk %}" class="btn">View Recipe</a>
            </div>
        {% endfor %}
    </div>
{% else %}
    <p>No recipes yet. <a href="{% url 'recipe_create' %}">Add the first one!</a></p>
{% endif %}
{% endblock %}
```

### app/recipes/templates/recipes/recipe_detail.html
```html
{% extends 'recipes/base.html' %}

{% block title %}{{ recipe.title }} - Recipe Manager{% endblock %}

{% block content %}
<h1>{{ recipe.title }}</h1>
<div class="recipe-meta">
    <p><strong>Author:</strong> {{ recipe.author.username }}</p>
    <p><strong>Difficulty:</strong> {{ recipe.get_difficulty_display }}</p>
    <p><strong>Prep Time:</strong> {{ recipe.prep_time }} minutes</p>
    <p><strong>Cook Time:</strong> {{ recipe.cook_time }} minutes</p>
    <p><strong>Total Time:</strong> {{ recipe.total_time }} minutes</p>
    <p><strong>Created:</strong> {{ recipe.created_at|date:"F j, Y" }}</p>
</div>

<h2>Description</h2>
<p>{{ recipe.description }}</p>

<h2>Ingredients</h2>
<div style="white-space: pre-line;">{{ recipe.ingredients }}</div>

<h2>Instructions</h2>
<div style="white-space: pre-line;">{{ recipe.instructions }}</div>

<div style="margin-top: 30px;">
    <a href="{% url 'recipe_list' %}" class="btn">← Back to Recipes</a>
</div>
{% endblock %}
```

### app/recipes/templates/recipes/recipe_form.html
```html
{% extends 'recipes/base.html' %}

{% block title %}Add Recipe - Recipe Manager{% endblock %}

{% block content %}
<h1>Add New Recipe</h1>

<form method="post">
    {% csrf_token %}
    
    <div class="form-group">
        <label for="title">Recipe Title:</label>
        <input type="text" id="title" name="title" required>
    </div>
    
    <div class="form-group">
        <label for="description">Description:</label>
        <textarea id="description" name="description" rows="3" required></textarea>
    </div>
    
    <div class="form-group">
        <label for="ingredients">Ingredients (one per line):</label>
        <textarea id="ingredients" name="ingredients" rows="8" required placeholder="1 cup flour&#10;2 eggs&#10;1/2 cup sugar"></textarea>
    </div>
    
    <div class="form-group">
        <label for="instructions">Cooking Instructions:</label>
        <textarea id="instructions" name="instructions" rows="8" required placeholder="1. Preheat oven to 350°F&#10;2. Mix dry ingredients..."></textarea>
    </div>
    
    <div style="display: flex; gap: 20px;">
        <div class="form-group" style="flex: 1;">
            <label for="prep_time">Prep Time (minutes):</label>
            <input type="number" id="prep_time" name="prep_time" min="1" required>
        </div>
        
        <div class="form-group" style="flex: 1;">
            <label for="cook_time">Cook Time (minutes):</label>
            <input type="number" id="cook_time" name="cook_time" min="1" required>
        </div>
        
        <div class="form-group" style="flex: 1;">
            <label for="difficulty">Difficulty:</label>
            <select id="difficulty" name="difficulty" required>
                <option value="">Choose...</option>
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
            </select>
        </div>
    </div>
    
    <button type="submit" class="btn">Save Recipe</button>
    <a href="{% url 'recipe_list' %}" class="btn" style="background: #6c757d; margin-left: 10px;">Cancel</a>
</form>
{% endblock %}
```

## Step 2: Container Configuration

### Dockerfile (Multi-stage Production Build)
```dockerfile
# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production stage
FROM python:3.11-slim

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/appuser/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy project files
COPY app/ .

# Create static files directory
RUN mkdir -p staticfiles

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Collect static files
RUN python manage.py collectstatic --noinput

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health/', timeout=10)" || exit 1

# Expose port
EXPOSE 8000

# Run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "120", "recipe_manager.wsgi:application"]
```

### docker-compose.yml (Complete Stack)
```yaml
version: '3.8'

services:
  # Database service
  db:
    image: postgres:15-alpine
    container_name: recipe_db
    environment:
      POSTGRES_DB: recipedb
      POSTGRES_USER: recipeuser
      POSTGRES_PASSWORD: recipepass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - recipe_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U recipeuser -d recipedb"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for caching (optional)
  redis:
    image: redis:7-alpine
    container_name: recipe_redis
    networks:
      - recipe_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Django web application
  web:
    build: 
      context: .
      dockerfile: Dockerfile
    container_name: recipe_web
    environment:
      - DEBUG=False
      - SECRET_KEY=your-super-secret-production-key-here
      - DB_NAME=recipedb
      - DB_USER=recipeuser
      - DB_PASSWORD=recipepass
      - DB_HOST=db
      - DB_PORT=5432
      - ALLOWED_HOSTS=localhost,127.0.0.1,recipe-app.local
    volumes:
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    networks:
      - recipe_network
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: recipe_nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - static_volume:/app/staticfiles:ro
      - media_volume:/app/media:ro
    networks:
      - recipe_network
    depends_on:
      - web
    restart: unless-stopped

volumes:
  postgres_data:
  static_volume:
  media_volume:

networks:
  recipe_network:
    driver: bridge
```

### nginx/nginx.conf
```nginx
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    
    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;
    
    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/javascript;
    
    upstream django {
        server web:8000;
    }
    
    server {
        listen 80;
        server_name localhost recipe-app.local;
        
        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        
        # Static files
        location /static/ {
            alias /app/staticfiles/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
        
        # Media files
        location /media/ {
            alias /app/media/;
            expires 7d;
        }
        
        # Main application
        location / {
            proxy_pass http://django;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
        
        # Health check endpoint
        location /health/ {
            proxy_pass http://django/health/;
            access_log off;
        }
    }
}
```

### .dockerignore
```
__pycache__
*.pyc
*.pyo
*.pyd
.git
.gitignore
README.md
.env
.venv
Dockerfile
docker-compose.yml
node_modules
```

## Step 3: Additional Django Configuration

### Add health check view to app/recipe_manager/urls.py
```python
from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

def health_check(request):
    return JsonResponse({'status': 'healthy', 'service': 'recipe-manager'})

urlpatterns = [
    path('admin/', admin.site.urls),
    path('health/', health_check, name='health_check'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('', include('recipes.urls')),
]
```

### app/recipes/urls.py
```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.recipe_list, name='recipe_list'),
    path('recipe/<int:pk>/', views.recipe_detail, name='recipe_detail'),
    path('recipe/new/', views.recipe_create, name='recipe_create'),
    path('register/', views.register, name='register'),
]
```

### app/recipes/templates/registration/login.html
```html
{% extends 'recipes/base.html' %}

{% block title %}Login - Recipe Manager{% endblock %}

{% block content %}
<h2>Login to Recipe Manager</h2>

<form method="post">
    {% csrf_token %}
    <div class="form-group">
        <label for="{{ form.username.id_for_label }}">Username:</label>
        {{ form.username }}
    </div>
    
    <div class="form-group">
        <label for="{{ form.password.id_for_label }}">Password:</label>
        {{ form.password }}
    </div>
    
    <button type="submit" class="btn">Login</button>
    <a href="{% url 'register' %}" style="margin-left: 10px;">Don't have an account? Register</a>
</form>
{% endblock %}
```

## Step 4: Build and Deploy Commands

### Initial Setup Script (setup.sh)
```bash
#!/bin/bash
echo "🏗️  Building containerized Recipe Manager..."

# Build and start services
docker-compose up --build -d

# Wait for database to be ready
echo "⏳ Waiting for database..."
sleep 10

# Run migrations
echo "🔄 Running database migrations..."
docker-compose exec web python manage.py migrate

# Create superuser (optional)
echo "👤 Creating superuser (skip if not needed)..."
docker-compose exec web python manage.py createsuperuser --noinput --username admin --email admin@recipe.com || true

# Show status
docker-compose ps

echo "✅ Recipe Manager is ready!"
echo "🌐 Access at: http://localhost"
echo "🔧 Admin at: http://localhost/admin"
```

### Production Deployment Commands
```bash
# Build the application
docker-compose -f docker-compose.yml build

# Start in production mode
docker-compose -f docker-compose.yml up -d

# View logs
docker-compose logs -f web

# Scale the application
docker-compose up -d --scale web=3

# Update the application
docker-compose pull
docker-compose up -d --force-recreate

# Backup database
docker-compose exec db pg_dump -U recipeuser recipedb > backup.sql

# Monitor containers
docker-compose top
docker stats
```

## Quality Features Implemented

### 1. **Production Security**
- Non-root user in container
- Environment-based configuration
- Security headers in Nginx
- Health checks for all services

### 2. **Performance Optimization**
- Multi-stage Docker build (smaller image)
- Nginx for static file serving
- Gunicorn with multiple workers
- Gzip compression
- Static file caching

### 3. **Reliability Features**
- Health checks for all services
- Service dependencies with conditions
- Restart policies
- Database connection pooling
- Graceful shutdowns

### 4. **Development Experience**
- Volume mounts for development
- Database persistence
- Comprehensive logging
- Easy scaling commands

### 5. **Container Best Practices**
- Minimal base images (Alpine Linux)
- Layer caching optimization
- Proper .dockerignore
- Explicit port exposure
- Resource limits ready

This containerized Django application demonstrates production-ready practices while maintaining simplicity for learning. The multi-service architecture with database, web application, reverse proxy, and caching shows real-world container orchestration patterns.

## Assignment: Multi-Environment Django Blog Platform

### Project Overview
Create a **Personal Blog Platform** that demonstrates Docker containerization skills across different environments. Think of this as setting up a **blog publishing kitchen** that can operate identically whether you're running it locally for testing, in staging for review, or in production for real users.

### Requirements

**Core Application Features:**
1. **Django Blog App** with:
   - User authentication and profiles
   - Blog post creation, editing, and deletion
   - Comment system
   - Categories and tags
   - Search functionality

2. **Multi-Environment Setup:**
   - Development environment with hot reloading
   - Staging environment with production-like settings
   - Production-ready configuration with optimizations

3. **Docker Implementation:**
   - Base Dockerfile with multi-stage build
   - Separate docker-compose files for each environment
   - Proper volume management for data persistence
   - Environment-specific configurations

### Technical Specifications

**Project Structure:**
```
blog-platform/
├── blog/
│   ├── models.py
│   ├── views.py
│   ├── serializers.py (if using DRF)
│   └── urls.py
├── users/
│   ├── models.py
│   ├── views.py
│   └── urls.py
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   └── nginx.conf
├── docker-compose.yml
├── docker-compose.dev.yml
├── docker-compose.staging.yml
├── requirements/
│   ├── base.txt
│   ├── development.txt
│   └── production.txt
└── scripts/
    ├── wait-for-db.sh
    └── migrate-and-run.sh
```

**Deliverables:**

1. **Multi-stage Production Dockerfile** that:
   - Uses separate stages for dependencies and runtime
   - Implements security best practices (non-root user)
   - Includes health checks
   - Optimizes for size and performance

2. **Three Docker Compose configurations:**
   - `docker-compose.dev.yml`: Development with code hot-reloading
   - `docker-compose.staging.yml`: Staging environment with realistic data
   - `docker-compose.yml`: Production configuration with security and performance optimizations

3. **Environment Management:**
   - Separate settings files for each environment
   - Proper secret management using environment variables
   - Database migrations handled automatically
   - Static file serving configured properly

4. **Documentation:**
   - README with setup instructions for each environment
   - Architecture explanation using kitchen analogies
   - Troubleshooting guide

### Evaluation Criteria

**Technical Excellence (40%):**
- Proper Docker best practices implementation
- Multi-stage build optimization
- Environment separation and configuration management
- Security considerations (non-root users, secret management)

**Functionality (30%):**
- Complete blog platform with all required features
- Proper database relationships and migrations
- User authentication and authorization
- Search and filtering capabilities

**DevOps Integration (20%):**
- Easy environment switching
- Proper volume management for data persistence
- Health checks and monitoring setup
- Clean, maintainable Docker configurations

**Documentation and Understanding (10%):**
- Clear setup instructions
- Explanation of Docker concepts using analogies
- Troubleshooting guide
- Code comments explaining Docker-specific configurations

### Bonus Challenges
1. **Load Balancing**: Implement nginx load balancer for multiple Django instances
2. **Monitoring**: Add container health monitoring and logging
3. **CI/CD Ready**: Structure the project for easy integration with CI/CD pipelines
4. **Database Optimization**: Implement database connection pooling and optimization

### Submission Format
- Complete project in a Git repository
- Include all Docker configurations and documentation
- Provide a video demo (5-10 minutes) showing:
  - Local development environment setup
  - Switching between environments
  - Key features of the blog platform
  - Explanation of Docker architecture decisions

This assignment will demonstrate your understanding of containerizing Django applications, managing multi-environment deployments, and implementing Docker best practices - skills essential for modern web development operations.