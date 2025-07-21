# Day 59: Asynchronous Django - Complete Course

## Learning Objective
By the end of this course, you will understand how to implement asynchronous functionality in Django applications, enabling real-time communication, non-blocking database operations, and background task processing - transforming your Django skills from a short-order cook to a master chef orchestrating multiple kitchen stations simultaneously.

---

## Lesson 1: Django Channels Introduction

**Imagine that...** you're running a busy restaurant kitchen. In traditional Django (synchronous cooking), you can only prepare one dish at a time - you start the soup, wait for it to finish, then move to the salad, then the main course. But with Django Channels, you become like a head chef managing multiple cooks simultaneously - one chef handles appetizers, another works on mains, and a third prepares desserts, all cooking at the same time and communicating seamlessly.

### What are Django Channels?

Django Channels extends Django beyond HTTP to handle WebSockets, chat protocols, IoT protocols, and more. It's like upgrading your kitchen from a single burner stove to a professional kitchen with multiple stations, each specialized for different tasks.

### Installation and Setup

```python
# requirements.txt
django==4.2.7
channels==4.0.0
channels-redis==4.1.0
redis==5.0.1
```

```bash
pip install channels channels-redis redis
```

### Basic Configuration

```python
# settings.py
import os
from django.core.asgi import get_asgi_application

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'channels',  # Add channels
    'your_app',  # Your app name
]

# Channels configuration
ASGI_APPLICATION = 'your_project.asgi.application'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}
```

### ASGI Application Setup

```python
# asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import your_app.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            your_app.routing.websocket_urlpatterns
        )
    ),
})
```

**Kitchen Analogy**: Think of ASGI as your kitchen's central command center - it decides whether incoming orders (requests) go to the regular cooking station (HTTP) or the live cooking show station (WebSocket).

---

## Lesson 2: WebSockets Implementation

**Imagine that...** you're running a cooking show where viewers can ask questions in real-time. Instead of waiting for commercial breaks to read letters (like traditional HTTP requests), you have a live microphone system where viewers can speak directly to you while you cook, and you can respond immediately without stopping your demonstration.

### Creating a Consumer

A consumer in Channels is like a specialized chef who handles specific types of communication:

```python
# consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """When a diner sits at your chef's counter"""
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'chat_{self.room_name}'

        # Join room group (like seating diners at the same counter section)
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        """When a diner leaves the counter"""
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        """Receiving orders from diners"""
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        username = text_data_json['username']

        # Send message to room group (announce to all diners at the counter)
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message,
                'username': username
            }
        )

    async def chat_message(self, event):
        """Receive message from room group and send to WebSocket"""
        message = event['message']
        username = event['username']

        # Send message to WebSocket (deliver the prepared dish)
        await self.send(text_data=json.dumps({
            'message': message,
            'username': username
        }))
```

### Routing Configuration

```python
# routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/chat/(?P<room_name>\w+)/$', consumers.ChatConsumer.as_asgi()),
]
```

### Frontend WebSocket Connection

```html
<!-- chat.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Chef's Live Kitchen</title>
</head>
<body>
    <div id="chat-log"></div>
    <input id="chat-message-input" type="text" placeholder="Ask the chef...">
    <button id="chat-message-submit">Send</button>

    <script>
        const roomName = 'general';
        const chatSocket = new WebSocket(
            'ws://' + window.location.host + '/ws/chat/' + roomName + '/'
        );

        // Like having a direct line to the chef
        chatSocket.onmessage = function(e) {
            const data = JSON.parse(e.data);
            const chatLog = document.querySelector('#chat-log');
            chatLog.innerHTML += '<div><b>' + data.username + ':</b> ' + data.message + '</div>';
        };

        chatSocket.onclose = function(e) {
            console.error('Chat socket closed unexpectedly');
        };

        // Send message when button clicked or Enter pressed
        document.querySelector('#chat-message-submit').onclick = function(e) {
            const messageInputDom = document.querySelector('#chat-message-input');
            const message = messageInputDom.value;
            chatSocket.send(JSON.stringify({
                'message': message,
                'username': 'Chef Student'  // In real app, get from authentication
            }));
            messageInputDom.value = '';
        };

        document.querySelector('#chat-message-input').onkeyup = function(e) {
            if (e.keyCode === 13) {  // Enter key
                document.querySelector('#chat-message-submit').click();
            }
        };
    </script>
</body>
</html>
```

**Kitchen Analogy**: The WebSocket connection is like having a speaking tube between the dining room and kitchen - diners can call out questions and the chef can respond instantly without leaving their cooking station.

---

## Lesson 3: Async Views and ORM

**Imagine that...** you're a chef who can now chop vegetables while simultaneously checking if the oven is ready, tasting the soup, and answering customer questions - all without having to stop one task completely before starting another. This is async Django in action!

### Async Views

```python
# views.py
import asyncio
import aiohttp
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import sync_to_async
from .models import Recipe, Ingredient

async def async_recipe_view(request):
    """Like a chef who can prep multiple dishes simultaneously"""
    
    # Simulate fetching data from external API (like checking supplier inventory)
    async def fetch_ingredient_prices():
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.ingredient-prices.com/today') as response:
                return await response.json()
    
    # Async database operations
    @sync_to_async
    def get_recipes():
        return list(Recipe.objects.select_related('chef').all()[:10])
    
    @sync_to_async
    def get_ingredients():
        return list(Ingredient.objects.all()[:20])
    
    # Run multiple operations concurrently (like having multiple cooks)
    recipes, ingredients, prices = await asyncio.gather(
        get_recipes(),
        get_ingredients(),
        fetch_ingredient_prices(),
        return_exceptions=True
    )
    
    return JsonResponse({
        'recipes': [{'name': r.name, 'chef': r.chef.name} for r in recipes],
        'ingredients': [{'name': i.name, 'stock': i.stock} for i in ingredients],
        'market_prices': prices if not isinstance(prices, Exception) else None
    })

# Traditional sync view for comparison
def sync_recipe_view(request):
    """Traditional chef - one task at a time"""
    recipes = list(Recipe.objects.select_related('chef').all()[:10])
    ingredients = list(Ingredient.objects.all()[:20])
    # Would need separate request for prices - no concurrent execution
    
    return JsonResponse({
        'recipes': [{'name': r.name, 'chef': r.chef.name} for r in recipes],
        'ingredients': [{'name': i.name, 'stock': i.stock} for i in ingredients],
    })
```

### Async Model Operations

```python
# models.py
from django.db import models
from asgiref.sync import sync_to_async

class Recipe(models.Model):
    name = models.CharField(max_length=200)
    chef = models.ForeignKey('Chef', on_delete=models.CASCADE)
    prep_time = models.IntegerField()  # minutes
    difficulty = models.IntegerField(choices=[(1, 'Easy'), (2, 'Medium'), (3, 'Hard')])
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    # Async class methods
    @classmethod
    async def aget_popular_recipes(cls):
        """Get popular recipes asynchronously - like checking what's trending"""
        return await sync_to_async(list)(
            cls.objects.filter(difficulty__lte=2).order_by('-created_at')[:5]
        )
    
    @classmethod
    async def aget_by_chef(cls, chef_name):
        """Find recipes by chef name asynchronously"""
        return await sync_to_async(list)(
            cls.objects.filter(chef__name__icontains=chef_name)
        )

class Chef(models.Model):
    name = models.CharField(max_length=100)
    speciality = models.CharField(max_length=100)
    experience_years = models.IntegerField()

class Ingredient(models.Model):
    name = models.CharField(max_length=100)
    stock = models.IntegerField()
    unit = models.CharField(max_length=20)
```

### URL Configuration

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('recipes/async/', views.async_recipe_view, name='async_recipes'),
    path('recipes/sync/', views.sync_recipe_view, name='sync_recipes'),
]
```

**Kitchen Analogy**: Async views are like having a sous chef who can coordinate multiple cooking processes - while the soup simmers, they can prep vegetables, check inventory, and even call suppliers, all without waiting for each task to completely finish before starting the next.

---

## Lesson 4: Background Tasks with Celery

**Imagine that...** your restaurant is so busy that some tasks need to be handled by a separate prep kitchen. When customers order a birthday cake, you don't make them wait 2 hours while you bake it - instead, you send the order to your pastry chef in the back (Celery worker) who handles it separately while you continue serving other customers.

### Celery Setup and Configuration

```python
# celery.py (in your project root)
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')

app = Celery('your_project')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
```

```python
# settings.py additions
import os

# Celery Configuration (Redis as message broker - like the order ticket system)
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'

# Email configuration for notifications
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
```

### Creating Celery Tasks

```python
# tasks.py
from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from .models import Recipe, Chef
import time
import requests

@shared_task
def send_recipe_notification(chef_email, recipe_name):
    """
    Like sending a message to the pastry chef about a special order
    This runs in the background without blocking the main kitchen
    """
    subject = f'New Recipe Request: {recipe_name}'
    message = f'''
    Hello Chef!
    
    A customer has requested the recipe for {recipe_name}.
    Please prepare the detailed instructions.
    
    Best regards,
    Kitchen Management System
    '''
    
    try:
        send_mail(
            subject,
            message,
            settings.DEFAULT_FROM_EMAIL,
            [chef_email],
            fail_silently=False,
        )
        return f'Recipe notification sent to {chef_email}'
    except Exception as e:
        return f'Failed to send notification: {str(e)}'

@shared_task
def process_bulk_recipe_import(recipe_data_list):
    """
    Like having your prep cook process a large shipment of ingredients
    Heavy lifting done in background while main kitchen keeps serving
    """
    results = []
    
    for recipe_data in recipe_data_list:
        try:
            # Simulate time-consuming processing
            time.sleep(1)  # Imagine complex recipe validation
            
            recipe = Recipe.objects.create(
                name=recipe_data['name'],
                chef_id=recipe_data['chef_id'],
                prep_time=recipe_data['prep_time'],
                difficulty=recipe_data['difficulty']
            )
            results.append(f'Created recipe: {recipe.name}')
            
        except Exception as e:
            results.append(f'Failed to create {recipe_data["name"]}: {str(e)}')
    
    return results

@shared_task
def update_ingredient_prices():
    """
    Like having someone check market prices every morning
    Scheduled task that runs automatically
    """
    try:
        # Simulate external API call
        response = requests.get('https://api.ingredient-market.com/prices')
        if response.status_code == 200:
            price_data = response.json()
            # Update your ingredient prices
            # (In real app, you'd update your database)
            return f'Updated {len(price_data)} ingredient prices'
        else:
            return 'Failed to fetch price data'
    except Exception as e:
        return f'Error updating prices: {str(e)}'

@shared_task(bind=True, max_retries=3)
def send_chef_report(self, chef_id):
    """
    Task with retry logic - like making sure important messages get delivered
    """
    try:
        chef = Chef.objects.get(id=chef_id)
        recipes_count = Recipe.objects.filter(chef=chef).count()
        
        # Simulate potential failure (network issue, etc.)
        if recipes_count == 0:
            raise Exception("No recipes found for chef")
        
        message = f'''
        Daily Report for Chef {chef.name}
        
        Total Recipes: {recipes_count}
        Speciality: {chef.speciality}
        Experience: {chef.experience_years} years
        '''
        
        send_mail(
            f'Daily Report - {chef.name}',
            message,
            settings.DEFAULT_FROM_EMAIL,
            [f'{chef.name.lower().replace(" ", ".")}@restaurant.com'],
            fail_silently=False,
        )
        
        return f'Report sent to {chef.name}'
        
    except Exception as exc:
        # Retry with exponential backoff (like trying to reach chef again later)
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
```

### Using Celery Tasks in Views

```python
# views.py (addition to previous views)
from django.shortcuts import render, redirect
from django.contrib import messages
from .tasks import send_recipe_notification, process_bulk_recipe_import
from .models import Chef, Recipe

def request_recipe(request):
    """View where customers can request recipes"""
    if request.method == 'POST':
        recipe_name = request.POST['recipe_name']
        chef_id = request.POST['chef_id']
        
        try:
            chef = Chef.objects.get(id=chef_id)
            
            # Send notification asynchronously (like sending order to back kitchen)
            task = send_recipe_notification.delay(
                f'{chef.name.lower().replace(" ", ".")}@restaurant.com',
                recipe_name
            )
            
            messages.success(request, 
                f'Recipe request sent! Your task ID is: {task.id}')
            
        except Chef.DoesNotExist:
            messages.error(request, 'Chef not found!')
        
        return redirect('request_recipe')
    
    chefs = Chef.objects.all()
    return render(request, 'request_recipe.html', {'chefs': chefs})

def bulk_import_recipes(request):
    """Import many recipes at once - background processing"""
    if request.method == 'POST':
        # In real app, you'd parse uploaded file
        sample_recipes = [
            {'name': 'Async Soup', 'chef_id': 1, 'prep_time': 30, 'difficulty': 2},
            {'name': 'Concurrent Curry', 'chef_id': 1, 'prep_time': 45, 'difficulty': 3},
            {'name': 'Background Bread', 'chef_id': 2, 'prep_time': 120, 'difficulty': 2},
        ]
        
        # Process in background (like having prep kitchen handle bulk orders)
        task = process_bulk_recipe_import.delay(sample_recipes)
        
        messages.success(request, 
            f'Bulk import started! Task ID: {task.id}')
        
        return redirect('bulk_import_recipes')
    
    return render(request, 'bulk_import.html')
```

### Periodic Tasks with Celery Beat

```python
# celery.py (addition)
from celery.schedules import crontab

app.conf.beat_schedule = {
    # Update ingredient prices every morning at 6 AM (like daily market check)
    'update-ingredient-prices': {
        'task': 'your_app.tasks.update_ingredient_prices',
        'schedule': crontab(hour=6, minute=0),
    },
    # Send daily reports every day at 5 PM
    'send-daily-reports': {
        'task': 'your_app.tasks.send_chef_report',
        'schedule': crontab(hour=17, minute=0),
        'args': (1,)  # Chef ID 1
    },
}

app.conf.timezone = 'UTC'
```

### Running Celery

```bash
# Terminal 1 - Start Redis (the message broker/order ticket system)
redis-server

# Terminal 2 - Start Celery Worker (the background chefs)
celery -A your_project worker --loglevel=info

# Terminal 3 - Start Celery Beat (the scheduler/kitchen timer)
celery -A your_project beat --loglevel=info

# Terminal 4 - Run Django development server
python manage.py runserver
```

**Kitchen Analogy**: Celery is like having a separate prep kitchen with specialized chefs. Your main kitchen (Django views) can send orders (tasks) to the prep kitchen via a ticket system (Redis broker). The prep chefs (Celery workers) handle time-consuming tasks like making stocks, prep work, or special orders while your main kitchen continues serving customers without delays.

---

## Assignment: Real-time Recipe Collaboration System

### Project Description
Create a real-time recipe sharing platform where multiple chefs can collaborate on creating recipes simultaneously, with background processing for notifications and ingredient price updates.

### Requirements:

1. **WebSocket Implementation**: Create a real-time recipe editing room where multiple users can see changes as they happen
2. **Async Views**: Implement async views that fetch recipe data and external ingredient prices concurrently  
3. **Background Tasks**: Use Celery to send notifications when recipes are completed and to update ingredient prices periodically
4. **Models**: Create Recipe, Chef, and Ingredient models with proper relationships

### Expected Features:

```python
# Key components you should implement:

# 1. Real-time recipe collaboration WebSocket consumer
class RecipeCollaborationConsumer(AsyncWebsocketConsumer):
    # Handle multiple chefs editing same recipe
    
# 2. Async view for recipe dashboard
async def recipe_dashboard(request):
    # Fetch recipes, ingredients, and prices concurrently
    
# 3. Celery tasks for notifications
@shared_task
def notify_recipe_completion(recipe_id):
    # Send emails to all collaborating chefs
    
# 4. Celery beat schedule for price updates
app.conf.beat_schedule = {
    'update-prices-hourly': {
        'task': 'your_app.tasks.update_ingredient_prices',
        'schedule': crontab(minute=0),  # Every hour
    },
}
```

### Deliverables:
- Working Django Channels WebSocket for real-time collaboration
- At least 2 async views using concurrent operations  
- 3 different Celery tasks including one scheduled task
- Frontend with WebSocket integration showing real-time updates
- README with setup instructions and explanation of async concepts used

### Success Criteria:
- Multiple browser tabs can edit same recipe and see real-time changes
- Async views respond faster than synchronous equivalents
- Background tasks process without blocking main application
- All components work together seamlessly

**Kitchen Analogy**: You're building a digital version of a professional kitchen where multiple chefs can work on the same recipe simultaneously (WebSocket), the head chef can check multiple stations at once (async views), and there's a dedicated staff handling notifications and supply management (Celery) - all working together to create an efficient, modern cooking operation.

---

## Code Syntax Explanations

### async/await Keywords
```python
async def my_function():  # Declares an asynchronous function
    result = await some_async_operation()  # Pauses until operation completes
    return result
```
- `async`: Declares a function as asynchronous (can be paused and resumed)
- `await`: Pauses function execution until the awaited operation completes

### AsyncWebsocketConsumer Methods
```python
async def connect(self):     # When WebSocket connection is established
async def disconnect(self):  # When WebSocket connection is closed  
async def receive(self):     # When message received from frontend
```

### Celery Decorators
```python
@shared_task                 # Makes function a Celery task
@shared_task(bind=True)      # Gives access to task instance (self)
@shared_task(max_retries=3)  # Sets maximum retry attempts
```

### Django Channel Layers
```python
self.channel_layer.group_add()     # Add connection to group
self.channel_layer.group_discard() # Remove connection from group  
self.channel_layer.group_send()    # Send message to all in group
```

### Async Database Operations
```python
@sync_to_async               # Converts sync function to async
await sync_to_async(list)(QuerySet)  # Convert QuerySet to list asynchronously
```

This course transforms you from a synchronous cook into an asynchronous chef master, capable of handling multiple real-time operations while maintaining the smooth flow of your Django kitchen!

# Real-time Expense Notifications Project

## Project Overview
Build a real-time expense tracking system where users receive instant notifications when expenses are added, updated, or when spending limits are exceeded.

## Project Structure
```
expense_tracker/
├── manage.py
├── requirements.txt
├── expense_tracker/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── asgi.py
├── expenses/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── consumers.py
│   ├── routing.py
│   ├── tasks.py
│   └── templates/
│       └── expenses/
│           ├── dashboard.html
│           └── expense_form.html
└── static/
    ├── css/
    └── js/
```

## Models (expenses/models.py)
```python
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Category(models.Model):
    name = models.CharField(max_length=100)
    spending_limit = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    def __str__(self):
        return self.name

class Expense(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.CharField(max_length=200)
    date_created = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.description} - ${self.amount}"

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    monthly_limit = models.DecimalField(max_digits=10, decimal_places=2, default=1000)
    notifications_enabled = models.BooleanField(default=True)
```

## WebSocket Consumer (expenses/consumers.py)
```python
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import User
from .models import Expense, Category

class ExpenseConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        if self.user.is_anonymous:
            await self.close()
            return
            
        self.room_group_name = f"expense_{self.user.id}"
        
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Send welcome message
        await self.send(text_data=json.dumps({
            'type': 'connection',
            'message': 'Connected to expense notifications'
        }))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        data = json.loads(text_data)
        
        if data['type'] == 'expense_added':
            await self.handle_expense_added(data)

    async def handle_expense_added(self, data):
        # Save expense to database
        expense = await self.save_expense(data)
        
        # Check spending limits
        limit_exceeded = await self.check_spending_limits(expense)
        
        # Broadcast to user's group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'expense_notification',
                'expense': {
                    'id': expense.id,
                    'amount': str(expense.amount),
                    'description': expense.description,
                    'category': expense.category.name,
                    'date': expense.date_created.isoformat()
                },
                'limit_exceeded': limit_exceeded
            }
        )

    @database_sync_to_async
    def save_expense(self, data):
        category = Category.objects.get(id=data['category_id'])
        expense = Expense.objects.create(
            user=self.user,
            category=category,
            amount=data['amount'],
            description=data['description']
        )
        return expense

    @database_sync_to_async
    def check_spending_limits(self, expense):
        from django.db.models import Sum
        from datetime import datetime, timedelta
        
        # Check monthly spending
        current_month = datetime.now().replace(day=1)
        monthly_total = Expense.objects.filter(
            user=expense.user,
            date_created__gte=current_month
        ).aggregate(Sum('amount'))['amount__sum'] or 0
        
        user_profile = expense.user.userprofile
        
        return monthly_total > user_profile.monthly_limit

    async def expense_notification(self, event):
        await self.send(text_data=json.dumps({
            'type': 'expense_added',
            'expense': event['expense'],
            'limit_exceeded': event['limit_exceeded']
        }))
```

## Async Views (expenses/views.py)
```python
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from asgiref.sync import sync_to_async
import json
from .models import Expense, Category, UserProfile
from .tasks import send_expense_email_notification

@login_required
def dashboard(request):
    expenses = Expense.objects.filter(user=request.user).order_by('-date_created')[:10]
    categories = Category.objects.all()
    
    context = {
        'expenses': expenses,
        'categories': categories,
        'user_id': request.user.id
    }
    return render(request, 'expenses/dashboard.html', context)

@method_decorator(csrf_exempt, name='dispatch')
class AsyncExpenseView(View):
    async def post(self, request):
        try:
            data = json.loads(request.body)
            
            # Create expense asynchronously
            expense = await sync_to_async(Expense.objects.create)(
                user=request.user,
                category_id=data['category_id'],
                amount=data['amount'],
                description=data['description']
            )
            
            # Trigger background email notification
            send_expense_email_notification.delay(expense.id)
            
            return JsonResponse({
                'status': 'success',
                'expense_id': expense.id
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
```

## Background Tasks (expenses/tasks.py)
```python
from celery import shared_task
from django.core.mail import send_mail
from django.template.loader import render_to_string
from .models import Expense

@shared_task
def send_expense_email_notification(expense_id):
    try:
        expense = Expense.objects.get(id=expense_id)
        
        # Check if user has notifications enabled
        if not expense.user.userprofile.notifications_enabled:
            return
        
        subject = f"New Expense Added: ${expense.amount}"
        message = render_to_string('expenses/email_notification.html', {
            'expense': expense,
            'user': expense.user
        })
        
        send_mail(
            subject,
            message,
            'noreply@expensetracker.com',
            [expense.user.email],
            fail_silently=False,
        )
        
        return f"Email sent for expense {expense_id}"
        
    except Expense.DoesNotExist:
        return f"Expense {expense_id} not found"

@shared_task
def check_monthly_limits():
    """Background task to check all users' monthly spending limits"""
    from django.contrib.auth.models import User
    from django.db.models import Sum
    from datetime import datetime
    
    current_month = datetime.now().replace(day=1)
    
    for user in User.objects.all():
        monthly_total = Expense.objects.filter(
            user=user,
            date_created__gte=current_month
        ).aggregate(Sum('amount'))['amount__sum'] or 0
        
        if monthly_total > user.userprofile.monthly_limit:
            send_limit_exceeded_notification.delay(user.id, monthly_total)

@shared_task
def send_limit_exceeded_notification(user_id, amount):
    from django.contrib.auth.models import User
    
    user = User.objects.get(id=user_id)
    subject = "Monthly Spending Limit Exceeded!"
    message = f"You have exceeded your monthly limit. Current spending: ${amount}"
    
    send_mail(
        subject,
        message,
        'noreply@expensetracker.com',
        [user.email],
        fail_silently=False,
    )
```

## WebSocket Routing (expenses/routing.py)
```python
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/expenses/$', consumers.ExpenseConsumer.as_asgi()),
]
```

## Dashboard Template (expenses/templates/expenses/dashboard.html)
```html
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Expense Tracker</title>
    <style>
        .notification {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            animation: slideIn 0.3s ease-in-out;
        }
        .success { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .expense-item {
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <h1>Real-time Expense Tracker</h1>
    
    <div id="notifications"></div>
    
    <div class="expense-form">
        <h3>Add New Expense</h3>
        <form id="expenseForm">
            <select id="category" required>
                <option value="">Select Category</option>
                {% for category in categories %}
                    <option value="{{ category.id }}">{{ category.name }}</option>
                {% endfor %}
            </select>
            
            <input type="number" id="amount" placeholder="Amount" step="0.01" required>
            <input type="text" id="description" placeholder="Description" required>
            
            <button type="submit">Add Expense</button>
        </form>
    </div>
    
    <div class="expenses-list">
        <h3>Recent Expenses</h3>
        <div id="expensesList">
            {% for expense in expenses %}
                <div class="expense-item">
                    <strong>${{ expense.amount }}</strong> - {{ expense.description }}
                    <br><small>{{ expense.category.name }} | {{ expense.date_created }}</small>
                </div>
            {% endfor %}
        </div>
    </div>

    <script>
        // WebSocket connection
        const socket = new WebSocket(`ws://${window.location.host}/ws/expenses/`);
        
        socket.onopen = function(e) {
            console.log('WebSocket connected');
        };
        
        socket.onmessage = function(e) {
            const data = JSON.parse(e.data);
            handleWebSocketMessage(data);
        };
        
        socket.onclose = function(e) {
            console.log('WebSocket disconnected');
        };
        
        function handleWebSocketMessage(data) {
            const notificationsDiv = document.getElementById('notifications');
            
            if (data.type === 'expense_added') {
                // Show notification
                const notification = document.createElement('div');
                notification.className = data.limit_exceeded ? 'notification warning' : 'notification success';
                notification.innerHTML = `
                    <strong>Expense Added!</strong><br>
                    $${data.expense.amount} - ${data.expense.description}
                    ${data.limit_exceeded ? '<br><em>⚠️ Monthly limit exceeded!</em>' : ''}
                `;
                notificationsDiv.appendChild(notification);
                
                // Add to expenses list
                addExpenseToList(data.expense);
                
                // Remove notification after 5 seconds
                setTimeout(() => {
                    notification.remove();
                }, 5000);
            }
        }
        
        function addExpenseToList(expense) {
            const expensesList = document.getElementById('expensesList');
            const expenseItem = document.createElement('div');
            expenseItem.className = 'expense-item';
            expenseItem.innerHTML = `
                <strong>$${expense.amount}</strong> - ${expense.description}
                <br><small>${expense.category} | ${new Date(expense.date).toLocaleString()}</small>
            `;
            expensesList.insertBefore(expenseItem, expensesList.firstChild);
        }
        
        // Form submission
        document.getElementById('expenseForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = {
                category_id: document.getElementById('category').value,
                amount: document.getElementById('amount').value,
                description: document.getElementById('description').value
            };
            
            // Send via WebSocket
            socket.send(JSON.stringify({
                type: 'expense_added',
                ...formData
            }));
            
            // Reset form
            this.reset();
        });
    </script>
</body>
</html>
```

## Settings Configuration (expense_tracker/settings.py)
```python
# Add to INSTALLED_APPS
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'channels',
    'expenses',
]

# Channels configuration
ASGI_APPLICATION = 'expense_tracker.asgi.application'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}

# Celery configuration
CELERY_BROKER_URL = 'redis://localhost:6379'
CELERY_RESULT_BACKEND = 'redis://localhost:6379'
CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TASK_SERIALIZER = 'json'
```

## ASGI Configuration (expense_tracker/asgi.py)
```python
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import expenses.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'expense_tracker.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            expenses.routing.websocket_urlpatterns
        )
    ),
})
```

## Requirements (requirements.txt)
```
Django==4.2.7
channels==4.0.0
channels-redis==4.1.0
celery==5.3.4
redis==5.0.1
```
