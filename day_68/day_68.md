# Django CI/CD Pipeline Course - Day 68

## Learning Objective
By the end of this lesson, you will understand how to implement a complete CI/CD (Continuous Integration/Continuous Deployment) pipeline for Django applications, including automated testing, deployment automation, environment management, and rollback strategies. You'll be able to set up a production-ready deployment workflow that ensures code quality and reliable releases.

---

## Introduction: Imagine That...

Imagine that you're running a bustling restaurant kitchen where multiple chefs are constantly creating new dishes and improving existing recipes. Without proper coordination, one chef might accidentally serve raw chicken while another burns the pasta because they're not following the same quality checks. Now imagine if you could create a magical kitchen system where:

- Every dish is automatically tested for quality before it reaches customers
- New recipes are seamlessly added to the menu without disrupting service
- You can instantly switch between different kitchen setups (lunch rush vs. dinner service)
- If a dish goes wrong, you can immediately return to the previous working recipe

This is exactly what a CI/CD pipeline does for your Django applications - it's your automated kitchen management system that ensures every code change is tested, deployed safely, and can be rolled back if needed.

---

## Lesson 1: Automated Testing Pipeline

### The Quality Control Station

In our kitchen analogy, this is like having a dedicated quality control station where every dish is automatically tested for taste, temperature, and presentation before it leaves the kitchen.

### Setting Up GitHub Actions for Django

First, let's create our automated testing pipeline using GitHub Actions:

```yaml
# .github/workflows/django.yml
name: Django CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run migrations
      run: |
        python manage.py migrate
      env:
        DATABASE_URL: postgres://postgres:postgres@localhost:5432/test_db
        
    - name: Run tests
      run: |
        python manage.py test
        coverage run --source='.' manage.py test
        coverage report
      env:
        DATABASE_URL: postgres://postgres:localhost:5432/test_db
        
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
```

### Django Test Configuration

Create a comprehensive test setup in your Django project:

```python
# tests/test_models.py
from django.test import TestCase
from django.contrib.auth.models import User
from myapp.models import Recipe

class RecipeModelTest(TestCase):
    def setUp(self):
        """Set up test dependencies - like prepping ingredients"""
        self.user = User.objects.create_user(
            username='chef_gordon',
            email='gordon@kitchen.com',
            password='secret_sauce'
        )
        
    def test_recipe_creation(self):
        """Test that we can create a recipe - like testing a new dish"""
        recipe = Recipe.objects.create(
            title='Perfect Pasta',
            chef=self.user,
            ingredients='Pasta, Tomatoes, Garlic',
            instructions='Cook pasta, make sauce, combine',
            cooking_time=30
        )
        
        self.assertEqual(recipe.title, 'Perfect Pasta')
        self.assertEqual(recipe.chef, self.user)
        self.assertTrue(recipe.created_at)
        
    def test_recipe_str_representation(self):
        """Test string representation - like checking the menu display"""
        recipe = Recipe.objects.create(
            title='Chocolate Cake',
            chef=self.user,
            ingredients='Flour, Chocolate, Eggs',
            instructions='Mix and bake',
            cooking_time=45
        )
        
        self.assertEqual(str(recipe), 'Chocolate Cake')
```

### Syntax Explanation:
- `on: push/pull_request`: Triggers the pipeline when code is pushed or PR is created (like starting quality control when a new dish is ready)
- `services: postgres`: Sets up a database service for testing (like having a test kitchen with all equipment)
- `steps`: Sequential actions performed (like quality control checklist)
- `env`: Environment variables (like kitchen settings - temperature, timing, etc.)

---

## Lesson 2: Deployment Automation

### The Automatic Serving System

This is like having an automated system that takes approved dishes from quality control and serves them to customers in the dining room, ensuring consistent presentation and timing.

### Docker Configuration for Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set environment variables - like setting kitchen temperature
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory - like designating prep area
WORKDIR /app

# Install system dependencies - like having proper kitchen tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies - like stocking ingredients
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project - like bringing recipe to kitchen
COPY . /app/

# Collect static files - like plating and garnishing
RUN python manage.py collectstatic --noinput

# Run the application - like opening the restaurant
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "myproject.wsgi:application"]
```

### GitHub Actions Deployment Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    needs: test  # Only deploy if tests pass - like quality control approval
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t myapp:${{ github.sha }} .
        docker tag myapp:${{ github.sha }} myapp:latest
        
    - name: Deploy to staging
      run: |
        # Deploy to staging first - like testing in small dining room
        echo "Deploying to staging environment..."
        # Your staging deployment commands here
        
    - name: Run smoke tests
      run: |
        # Basic health checks - like final taste test
        curl -f http://staging.myapp.com/health/ || exit 1
        
    - name: Deploy to production
      if: success()  # Only if staging succeeds
      run: |
        # Deploy to production - opening main dining room
        echo "Deploying to production environment..."
        # Your production deployment commands here
```

### Django Production Settings

```python
# settings/production.py
from .base import *
import os

# Security settings - like restaurant security measures
DEBUG = False
ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']

# Database configuration - like main kitchen database
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

# Static files - like plated presentations
STATIC_URL = '/static/'
STATIC_ROOT = '/app/staticfiles/'

# Media files - like photo gallery of dishes
MEDIA_URL = '/media/'
MEDIA_ROOT = '/app/media/'

# Logging - like kitchen activity log
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '/app/logs/django.log',
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

### Syntax Explanation:
- `FROM python:3.9-slim`: Base image (like starting with a clean kitchen)
- `ENV`: Environment variables (kitchen settings that persist)
- `WORKDIR`: Working directory (designated prep area)
- `COPY`: Copy files into container (bringing ingredients and recipes)
- `CMD`: Default command when container starts (opening restaurant)

---

## Lesson 3: Environment Management

### Multiple Kitchen Configurations

Think of this as having different kitchen setups for different occasions - a breakfast kitchen, lunch rush configuration, and elegant dinner service setup, each optimized for specific needs.

### Environment Configuration with Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    command: gunicorn myproject.wsgi:application --bind 0.0.0.0:8000
    volumes:
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    ports:
      - "8000:8000"
    env_file:
      - .env.production
    depends_on:
      - db
      - redis
    
  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    env_file:
      - .env.production
      
  redis:
    image: redis:6-alpine
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - static_volume:/app/staticfiles
      - media_volume:/app/media
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web

volumes:
  postgres_data:
  static_volume:
  media_volume:
```

### Environment-Specific Settings

```python
# settings/environments.py
import os

class Environment:
    """Environment management - like different kitchen configurations"""
    
    @staticmethod
    def get_env():
        return os.environ.get('DJANGO_ENV', 'development')
    
    @staticmethod
    def is_production():
        return Environment.get_env() == 'production'
    
    @staticmethod
    def is_staging():
        return Environment.get_env() == 'staging'
    
    @staticmethod
    def is_development():
        return Environment.get_env() == 'development'

# Environment-specific configurations
ENVIRONMENT_CONFIGS = {
    'development': {
        'DEBUG': True,
        'ALLOWED_HOSTS': ['localhost', '127.0.0.1'],
        'DATABASE_URL': 'sqlite:///db.sqlite3',
        'CACHE_BACKEND': 'dummy',  # No caching in dev
    },
    'staging': {
        'DEBUG': False,
        'ALLOWED_HOSTS': ['staging.myapp.com'],
        'DATABASE_URL': os.environ.get('STAGING_DATABASE_URL'),
        'CACHE_BACKEND': 'redis',
    },
    'production': {
        'DEBUG': False,
        'ALLOWED_HOSTS': ['myapp.com', 'www.myapp.com'],
        'DATABASE_URL': os.environ.get('DATABASE_URL'),
        'CACHE_BACKEND': 'redis',
        'SECURE_SSL_REDIRECT': True,  # Extra security in production
    }
}

# Apply environment configuration
current_env = Environment.get_env()
config = ENVIRONMENT_CONFIGS.get(current_env, ENVIRONMENT_CONFIGS['development'])

DEBUG = config['DEBUG']
ALLOWED_HOSTS = config['ALLOWED_HOSTS']
```

### Environment Variables Management

```bash
# .env.production
DJANGO_ENV=production
SECRET_KEY=your-super-secret-production-key
DATABASE_URL=postgresql://user:password@db:5432/myapp_production
REDIS_URL=redis://redis:6379/0

# Email settings - like restaurant communication system
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=noreply@myapp.com
EMAIL_HOST_PASSWORD=your-email-password

# Third-party integrations - like supplier connections
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_STORAGE_BUCKET_NAME=myapp-static-files
```

### Syntax Explanation:
- `version: '3.8'`: Docker Compose file version (like recipe format version)
- `services`: Different components (like different stations in kitchen)
- `volumes`: Persistent storage (like pantry storage that survives kitchen renovations)
- `env_file`: Environment variables file (like kitchen settings manual)
- `depends_on`: Service dependencies (like ensuring ingredients arrive before cooking)

---

## Lesson 4: Rollback Strategies

### The Emergency Recipe Restoration

This is like having a backup of every successful recipe and the ability to instantly return to a previous menu if a new dish causes problems in the dining room.

### Database Migration Rollback Strategy

```python
# management/commands/rollback_migration.py
from django.core.management.base import BaseCommand
from django.db import connection
from django.db.migrations.executor import MigrationExecutor

class Command(BaseCommand):
    """Emergency rollback command - like reverting to previous menu"""
    help = 'Rollback database migrations safely'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--target',
            type=str,
            help='Target migration to rollback to',
            required=True
        )
        parser.add_argument(
            '--app',
            type=str,
            help='App name',
            required=True
        )
        
    def handle(self, *args, **options):
        target = options['target']
        app = options['app']
        
        self.stdout.write(
            f"Rolling back {app} to migration {target}..."
        )
        
        # Create backup before rollback - like saving current recipe
        self.create_backup()
        
        # Perform rollback
        executor = MigrationExecutor(connection)
        try:
            executor.migrate([(app, target)])
            self.stdout.write(
                self.style.SUCCESS(f'Successfully rolled back to {target}')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Rollback failed: {str(e)}')
            )
            # Restore from backup if rollback fails
            self.restore_backup()
    
    def create_backup(self):
        """Create database backup - like saving current recipes"""
        import subprocess
        import datetime
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f'backup_{timestamp}.sql'
        
        subprocess.run([
            'pg_dump',
            '--host=localhost',
            '--username=postgres',
            '--dbname=myapp',
            '--file=' + backup_file
        ])
        
        self.stdout.write(f'Backup created: {backup_file}')
        
    def restore_backup(self):
        """Restore from backup if needed"""
        self.stdout.write('Restoring from backup...')
        # Implementation for backup restoration
```

### Application Version Rollback

```python
# utils/deployment.py
import subprocess
import json
from datetime import datetime

class DeploymentManager:
    """Manages deployments and rollbacks - like restaurant service manager"""
    
    def __init__(self):
        self.deployment_history = []
        
    def deploy_version(self, version, environment='production'):
        """Deploy specific version - like introducing new menu"""
        deployment_record = {
            'version': version,
            'environment': environment,
            'deployed_at': datetime.now().isoformat(),
            'status': 'deploying'
        }
        
        try:
            # Stop current services - like pausing kitchen
            self.stop_services()
            
            # Deploy new version - like switching to new recipes
            self.deploy_application(version)
            
            # Health check - like tasting new dishes
            if self.health_check():
                deployment_record['status'] = 'success'
                self.deployment_history.append(deployment_record)
                return True
            else:
                # Auto-rollback if health check fails
                self.rollback_to_previous()
                deployment_record['status'] = 'failed_auto_rollback'
                return False
                
        except Exception as e:
            deployment_record['status'] = f'failed: {str(e)}'
            self.rollback_to_previous()
            return False
            
    def rollback_to_previous(self):
        """Rollback to previous successful deployment"""
        successful_deployments = [
            d for d in self.deployment_history 
            if d['status'] == 'success'
        ]
        
        if successful_deployments:
            previous = successful_deployments[-1]
            self.deploy_application(previous['version'])
            print(f"Rolled back to version {previous['version']}")
        else:
            print("No previous successful deployment found!")
            
    def deploy_application(self, version):
        """Deploy specific application version"""
        commands = [
            f'docker pull myapp:{version}',
            f'docker stop myapp_container || true',
            f'docker rm myapp_container || true',
            f'docker run -d --name myapp_container myapp:{version}'
        ]
        
        for cmd in commands:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Command failed: {cmd}")
                
    def health_check(self):
        """Perform health check - like final quality inspection"""
        import requests
        
        try:
            response = requests.get('http://localhost:8000/health/', timeout=30)
            return response.status_code == 200
        except:
            return False
            
    def stop_services(self):
        """Stop current services gracefully"""
        subprocess.run(['docker', 'stop', 'myapp_container'], 
                      capture_output=True)
```

### Automated Rollback in CI/CD

```yaml
# .github/workflows/deploy-with-rollback.yml
name: Deploy with Rollback Strategy

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Get previous successful deployment
      id: previous
      run: |
        # Get last successful deployment tag - like previous working menu
        PREVIOUS_TAG=$(git describe --tags --abbrev=0 HEAD~1)
        echo "previous_tag=$PREVIOUS_TAG" >> $GITHUB_OUTPUT
        
    - name: Deploy new version
      id: deploy
      run: |
        # Deploy current version
        NEW_TAG=v$(date +%Y%m%d%H%M%S)
        echo "Deploying version $NEW_TAG"
        
        # Your deployment commands here
        ./deploy.sh $NEW_TAG
        
        echo "new_tag=$NEW_TAG" >> $GITHUB_OUTPUT
        
    - name: Health check
      id: health
      run: |
        # Wait for deployment to be ready
        sleep 30
        
        # Perform health checks - like quality control
        if curl -f http://your-app.com/health/; then
          echo "health_status=healthy" >> $GITHUB_OUTPUT
        else
          echo "health_status=unhealthy" >> $GITHUB_OUTPUT
        fi
        
    - name: Rollback on failure
      if: steps.health.outputs.health_status == 'unhealthy'
      run: |
        echo "Health check failed, rolling back to ${{ steps.previous.outputs.previous_tag }}"
        ./deploy.sh ${{ steps.previous.outputs.previous_tag }}
        
        # Notify team of rollback - like alerting restaurant manager
        curl -X POST -H 'Content-type: application/json' \
          --data '{"text":"🚨 Deployment failed, rolled back to previous version"}' \
          $SLACK_WEBHOOK_URL
```

### Syntax Explanation:
- `git describe --tags`: Gets the most recent tag (like finding the last successful recipe version)
- `capture_output=True`: Captures command output (like recording kitchen activities)
- `subprocess.run()`: Executes system commands (like following kitchen procedures)
- `$GITHUB_OUTPUT`: GitHub Actions output variable (like kitchen communication board)
- `if: steps.health.outputs.health_status == 'unhealthy'`: Conditional execution (like emergency protocols)

---
# Complete CI/CD Setup Project

## Project Overview
Build a complete CI/CD pipeline for a Django restaurant management application that automatically tests, builds, and deploys your code when changes are pushed to your repository.

## Project Structure
```
restaurant_manager/
├── .github/
│   └── workflows/
│       └── django.yml
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
├── scripts/
│   ├── deploy.sh
│   └── health_check.sh
├── restaurant/
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   └── ...
├── requirements/
│   ├── base.txt
│   ├── development.txt
│   └── production.txt
└── manage.py
```

## 1. Environment Configuration Setup

### settings/base.py
```python
import os
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = config('SECRET_KEY', default='your-secret-key-here')
DEBUG = config('DEBUG', default=False, cast=bool)

ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'menu',
    'orders',
    'staff',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'restaurant.urls'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME', default='restaurant_db'),
        'USER': config('DB_USER', default='postgres'),
        'PASSWORD': config('DB_PASSWORD', default='password'),
        'HOST': config('DB_HOST', default='localhost'),
        'PORT': config('DB_PORT', default='5432'),
    }
}

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

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
            'filename': BASE_DIR / 'logs' / 'django.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}
```

### settings/production.py
```python
from .base import *

# Security settings for production
SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# Database connection pooling
DATABASES['default'].update({
    'CONN_MAX_AGE': 600,
    'OPTIONS': {
        'MAX_CONNS': 20,
        'CONN_HEALTH_CHECKS': True,
    }
})

# Cache configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': config('REDIS_URL', default='redis://localhost:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Email configuration
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = config('EMAIL_HOST')
EMAIL_PORT = config('EMAIL_PORT', cast=int)
EMAIL_USE_TLS = True
EMAIL_HOST_USER = config('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD')
```

## 2. Docker Configuration

### Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=restaurant.settings.production

# Install Python dependencies
COPY requirements/production.txt .
RUN pip install --no-cache-dir -r production.txt

# Copy project
COPY . .

# Create logs directory
RUN mkdir -p logs

# Collect static files
RUN python manage.py collectstatic --noinput

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python scripts/health_check.py

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "--worker-class", "gevent", "restaurant.wsgi:application"]
```

### docker-compose.prod.yml
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DJANGO_SETTINGS_MODULE=restaurant.settings.production
      - SECRET_KEY=${SECRET_KEY}
      - DB_HOST=db
      - DB_NAME=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
      - REDIS_URL=redis://redis:6379/1
    depends_on:
      - db
      - redis
    volumes:
      - static_volume:/app/staticfiles
      - media_volume:/app/media
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - static_volume:/var/www/static
      - media_volume:/var/www/media
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped

volumes:
  postgres_data:
  static_volume:
  media_volume:
```

## 3. GitHub Actions CI/CD Pipeline

### .github/workflows/django.yml
```yaml
name: Django CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_restaurant
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ env.PYTHON_VERSION }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/development.txt
    
    - name: Set up environment variables
      run: |
        echo "SECRET_KEY=test-secret-key-for-ci" >> $GITHUB_ENV
        echo "DEBUG=True" >> $GITHUB_ENV
        echo "DB_HOST=localhost" >> $GITHUB_ENV
        echo "DB_NAME=test_restaurant" >> $GITHUB_ENV
        echo "DB_USER=postgres" >> $GITHUB_ENV
        echo "DB_PASSWORD=postgres" >> $GITHUB_ENV
        echo "REDIS_URL=redis://localhost:6379/0" >> $GITHUB_ENV
    
    - name: Run Django system checks
      run: python manage.py check --settings=restaurant.settings.development
    
    - name: Run migrations
      run: python manage.py migrate --settings=restaurant.settings.development
    
    - name: Run tests with coverage
      run: |
        coverage run --source='.' manage.py test --settings=restaurant.settings.development
        coverage report --show-missing
        coverage xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml
    
    - name: Run linting
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

  security:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install security tools
      run: |
        pip install bandit safety
    
    - name: Run security checks
      run: |
        bandit -r . -x tests/
        safety check -r requirements/production.txt

  build:
    runs-on: ubuntu-latest
    needs: [test, security]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ secrets.DOCKER_USERNAME }}/restaurant-manager
        tags: |
          type=ref,event=branch
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to production server
      uses: appleboy/ssh-action@v1.0.0
      with:
        host: ${{ secrets.PRODUCTION_HOST }}
        username: ${{ secrets.PRODUCTION_USER }}
        key: ${{ secrets.PRODUCTION_SSH_KEY }}
        port: ${{ secrets.PRODUCTION_PORT }}
        script: |
          cd /opt/restaurant-manager
          
          # Backup current deployment
          ./scripts/backup.sh
          
          # Pull latest changes
          git pull origin main
          
          # Update Docker images
          docker compose -f docker-compose.prod.yml pull
          
          # Deploy with zero-downtime
          docker compose -f docker-compose.prod.yml up -d --no-deps web
          
          # Run health checks
          ./scripts/health_check.sh
          
          # Run database migrations if needed
          docker compose -f docker-compose.prod.yml exec -T web python manage.py migrate --noinput
          
          # Collect static files
          docker compose -f docker-compose.prod.yml exec -T web python manage.py collectstatic --noinput
          
          # Clear cache
          docker compose -f docker-compose.prod.yml exec -T web python manage.py clear_cache
          
          # Final health check
          sleep 30
          ./scripts/health_check.sh
    
    - name: Notify deployment status
      if: always()
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 4. Deployment Scripts

### scripts/deploy.sh
```bash
#!/bin/bash

set -e

echo "🚀 Starting deployment process..."

# Configuration
PROJECT_DIR="/opt/restaurant-manager"
BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root"
fi

# Navigate to project directory
cd $PROJECT_DIR || error "Project directory not found"

# Check Docker and Docker Compose
command -v docker >/dev/null 2>&1 || error "Docker is not installed"
command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is not installed"

# Create backup
log "Creating backup..."
mkdir -p $BACKUP_DIR
docker-compose -f docker-compose.prod.yml exec -T db pg_dump -U postgres restaurant_db > $BACKUP_DIR/db_backup_$DATE.sql
tar -czf $BACKUP_DIR/media_backup_$DATE.tar.gz media/
log "Backup created successfully"

# Pull latest code
log "Pulling latest code..."
git fetch origin
git reset --hard origin/main

# Check for environment file
if [ ! -f .env ]; then
    error ".env file not found"
fi

# Build and deploy
log "Building and deploying containers..."
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml build --no-cache web

# Zero-downtime deployment
log "Performing zero-downtime deployment..."
docker-compose -f docker-compose.prod.yml up -d --no-deps --scale web=2 web
sleep 10

# Health check
log "Running health checks..."
./scripts/health_check.sh || error "Health check failed"

# Scale down old container
docker-compose -f docker-compose.prod.yml up -d --no-deps --scale web=1 web

# Run migrations
log "Running database migrations..."
docker-compose -f docker-compose.prod.yml exec -T web python manage.py migrate --noinput

# Collect static files
log "Collecting static files..."
docker-compose -f docker-compose.prod.yml exec -T web python manage.py collectstatic --noinput

# Clear cache
log "Clearing cache..."
docker-compose -f docker-compose.prod.yml exec -T web python manage.py clear_cache

# Final health check
log "Running final health check..."
sleep 30
./scripts/health_check.sh || error "Final health check failed"

# Cleanup old backups (keep last 7 days)
log "Cleaning up old backups..."
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

log "🎉 Deployment completed successfully!"
```

### scripts/health_check.sh
```bash
#!/bin/bash

# Health check script for the restaurant management application

set -e

# Configuration
APP_URL="http://localhost:8000"
MAX_RETRIES=10
RETRY_INTERVAL=5

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Check if application is responding
check_app_health() {
    local url=$1
    local expected_status=$2
    
    log "Checking $url..."
    
    for i in $(seq 1 $MAX_RETRIES); do
        if response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null); then
            if [ "$response" = "$expected_status" ]; then
                log "✅ $url is healthy (HTTP $response)"
                return 0
            else
                warn "❌ $url returned HTTP $response, expected $expected_status"
            fi
        else
            warn "❌ Failed to connect to $url"
        fi
        
        if [ $i -lt $MAX_RETRIES ]; then
            log "Retrying in $RETRY_INTERVAL seconds... (attempt $i/$MAX_RETRIES)"
            sleep $RETRY_INTERVAL
        fi
    done
    
    error "Health check failed for $url after $MAX_RETRIES attempts"
}

# Check database connectivity
check_database() {
    log "Checking database connectivity..."
    
    if docker-compose -f docker-compose.prod.yml exec -T web python manage.py check --database default; then
        log "✅ Database is healthy"
    else
        error "❌ Database check failed"
    fi
}

# Check Redis connectivity
check_redis() {
    log "Checking Redis connectivity..."
    
    if docker-compose -f docker-compose.prod.yml exec -T redis redis-cli ping | grep -q "PONG"; then
        log "✅ Redis is healthy"
    else
        error "❌ Redis check failed"
    fi
}

# Check disk space
check_disk_space() {
    log "Checking disk space..."
    
    used_space=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$used_space" -gt 80 ]; then
        warn "⚠️ Disk usage is ${used_space}% (threshold: 80%)"
    else
        log "✅ Disk usage is ${used_space}%"
    fi
}

# Check memory usage
check_memory() {
    log "Checking memory usage..."
    
    memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    if (( $(echo "$memory_usage > 80" | bc -l) )); then
        warn "⚠️ Memory usage is ${memory_usage}% (threshold: 80%)"
    else
        log "✅ Memory usage is ${memory_usage}%"
    fi
}

# Main health check routine
main() {
    log "🏥 Starting comprehensive health check..."
    
    # Check application endpoints
    check_app_health "$APP_URL/health/" "200"
    check_app_health "$APP_URL/api/v1/menu/" "200"
    check_app_health "$APP_URL/admin/" "302"
    
    # Check services
    check_database
    check_redis
    
    # Check system resources
    check_disk_space
    check_memory
    
    log "🎉 All health checks passed!"
}

# Run health checks
main "$@"
```

## 5. Management Commands

### restaurant/management/commands/clear_cache.py
```python
from django.core.management.base import BaseCommand
from django.core.cache import cache
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Clear all caches'

    def add_arguments(self, parser):
        parser.add_argument(
            '--pattern',
            type=str,
            help='Clear cache keys matching pattern',
        )

    def handle(self, *args, **options):
        """Clear cache based on pattern or clear all."""
        try:
            if options['pattern']:
                # Clear specific pattern (Redis implementation)
                if hasattr(cache, 'delete_pattern'):
                    deleted = cache.delete_pattern(options['pattern'])
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'Successfully cleared {deleted} cache keys matching "{options["pattern"]}"'
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING('Pattern deletion not supported by current cache backend')
                    )
            else:
                # Clear all cache
                cache.clear()
                self.stdout.write(
                    self.style.SUCCESS('Successfully cleared all cache')
                )
                
            logger.info(f"Cache cleared: pattern={options.get('pattern', 'all')}")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            self.stdout.write(
                self.style.ERROR(f'Failed to clear cache: {e}')
            )
            raise
```

## 6. Health Check Endpoint

### restaurant/views.py
```python
from django.http import JsonResponse
from django.db import connection
from django.core.cache import cache
from django.conf import settings
import redis
import time

def health_check(request):
    """Comprehensive health check endpoint."""
    health_status = {
        'status': 'healthy',
        'timestamp': time.time(),
        'checks': {}
    }
    
    try:
        # Database check
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            health_status['checks']['database'] = {
                'status': 'healthy',
                'message': 'Database connection successful'
            }
    except Exception as e:
        health_status['checks']['database'] = {
            'status': 'unhealthy',
            'message': f'Database error: {str(e)}'
        }
        health_status['status'] = 'unhealthy'
    
    try:
        # Cache check
        cache_key = 'health_check_test'
        cache.set(cache_key, 'test_value', 10)
        if cache.get(cache_key) == 'test_value':
            health_status['checks']['cache'] = {
                'status': 'healthy',
                'message': 'Cache working correctly'
            }
        else:
            raise Exception("Cache test failed")
    except Exception as e:
        health_status['checks']['cache'] = {
            'status': 'unhealthy',
            'message': f'Cache error: {str(e)}'
        }
        health_status['status'] = 'unhealthy'
    
    # Return appropriate status code
    status_code = 200 if health_status['status'] == 'healthy' else 503
    
    return JsonResponse(health_status, status=status_code)
```

## 7. Environment Files

### .env.example
```bash
# Django Configuration
SECRET_KEY=your-super-secret-key-here
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Database Configuration
DB_NAME=restaurant_db
DB_USER=restaurant_user
DB_PASSWORD=your-db-password
DB_HOST=db
DB_PORT=5432

# Redis Configuration
REDIS_URL=redis://redis:6379/1

# Email Configuration
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password

# AWS S3 Configuration (optional)
USE_S3=False
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_STORAGE_BUCKET_NAME=your-bucket-name
AWS_S3_REGION_NAME=us-east-1

# Monitoring (optional)
SENTRY_DSN=your-sentry-dsn
```

## Project Features Implemented

### 🔧 Infrastructure as Code
- **Docker multi-stage builds** for optimized production images
- **Docker Compose** orchestration for development and production
- **Environment-specific configurations** with Django settings modules

### 🚀 Automated CI/CD Pipeline
- **GitHub Actions workflow** with parallel job execution
- **Automated testing** with coverage reporting
- **Security scanning** with Bandit and Safety
- **Multi-platform Docker builds** for AMD64 and ARM64
- **Zero-downtime deployments** with health checks

### 📊 Monitoring & Health Checks
- **Comprehensive health endpoints** checking database, cache, and system resources
- **Application monitoring** with structured logging
- **Resource monitoring** for disk space and memory usage
- **Automated backup creation** before deployments

### 🔒 Security & Production Readiness
- **Security headers** and SSL configuration
- **Connection pooling** for database optimization
- **Static file compression** with WhiteNoise
- **Redis caching** for improved performance
- **Non-root container execution** for security

### 🎯 Operational Excellence
- **Rollback capabilities** with automated backups
- **Environment management** with separate configurations
- **Notification integration** with Slack for deployment status
- **Cleanup automation** for old backups and logs

This complete CI/CD setup transforms your Django restaurant management application into a production-ready system with automated testing, building, and deployment. The pipeline ensures code quality, security, and reliable deployments while maintaining zero downtime for your users.


## Assignment: Restaurant Monitoring Dashboard

**Objective**: Create a Django monitoring dashboard that tracks deployment health and provides rollback capabilities for a fictional restaurant chain's online ordering system.

### Requirements:

1. **Create a Django app called `monitoring`** that includes:
   - A model to track deployment history (version, timestamp, status, environment)
   - A model to track application health metrics (response time, error rate, active users)

2. **Build a dashboard view** that displays:
   - Current deployment status across different environments (development, staging, production)
   - Health metrics graphs for the last 24 hours
   - List of recent deployments with rollback buttons
   - System alerts if any metrics exceed thresholds

3. **Implement a health check endpoint** (`/health/`) that:
   - Checks database connectivity
   - Verifies critical services are running
   - Returns JSON response with system status
   - Logs health check results to your monitoring models

4. **Create a management command** (`python manage.py check_deployment_health`) that:
   - Runs automated health checks
   - Compares current metrics with previous deployments
   - Sends alerts if performance degrades significantly
   - Can be scheduled to run every 5 minutes

### Deliverables:
- Django models for deployment tracking and health metrics
- Dashboard templates with basic CSS styling
- Health check endpoint with comprehensive system verification
- Management command for automated monitoring
- Brief documentation explaining how a restaurant manager would use this system

### Success Criteria:
- Dashboard loads without errors and displays meaningful data
- Health check endpoint responds correctly and logs data
- Management command runs successfully and provides useful output
- Code includes proper error handling and logging
- Models include appropriate relationships and constraints

This assignment reinforces CI/CD concepts while being practical - every restaurant chain needs to monitor their online systems to ensure customers can place orders smoothly, just like our kitchen analogy where we need to monitor that all stations are working properly!