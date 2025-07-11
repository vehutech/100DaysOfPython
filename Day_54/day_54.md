# Day 54: Django Caching Strategies

## Learning Objective
By the end of this lesson, you will understand Django's caching framework and be able to implement different caching strategies to optimize your web application's performance, just like a chef who knows when to prep ingredients ahead of time versus cooking them fresh to order.

---

## 1. Django Caching Framework

### Imagine that...
You're running a busy restaurant kitchen. Every time a customer orders your famous tomato soup, you could start from scratch - washing tomatoes, chopping them, simmering for hours. But what if you prepared a large batch in the morning and kept it warm? That's exactly what caching does for your Django application - it stores frequently requested data so you don't have to "cook" it from scratch every time.

### The Chef's Caching Philosophy
Just as a smart chef prepares mise en place (everything in its place), Django's caching framework helps you organize and store your "digital ingredients" for quick access.

### Basic Caching Setup

```python
# settings.py - Setting up your kitchen's storage system
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
        'TIMEOUT': 300,  # 5 minutes - like keeping soup warm
        'OPTIONS': {
            'MAX_ENTRIES': 1000,
        }
    }
}
```

**Syntax Explanation:**
- `CACHES`: Dictionary defining cache configurations
- `'default'`: The primary cache backend (like your main prep station)
- `'BACKEND'`: Specifies the cache storage method
- `'TIMEOUT'`: How long to keep cached data (in seconds)
- `'MAX_ENTRIES'`: Maximum number of cached items

### Using Cache in Views

```python
# views.py - Your kitchen operations
from django.core.cache import cache
from django.shortcuts import render
from django.http import JsonResponse
from .models import MenuItem

def expensive_menu_calculation(request):
    # Check if we already have this "dish" prepared
    cached_result = cache.get('daily_menu_stats')
    
    if cached_result is None:
        # Time to cook from scratch - expensive operation
        menu_items = MenuItem.objects.all()
        total_revenue = sum(item.price * item.orders_count for item in menu_items)
        avg_rating = sum(item.rating for item in menu_items) / len(menu_items)
        
        # Prepare our "dish" for storage
        result = {
            'total_revenue': total_revenue,
            'avg_rating': avg_rating,
            'items_count': len(menu_items)
        }
        
        # Store in our prep station for 1 hour
        cache.set('daily_menu_stats', result, 3600)
        cached_result = result
    
    return JsonResponse(cached_result)
```

**Syntax Explanation:**
- `cache.get('key')`: Retrieves cached data (like checking if soup is ready)
- `cache.set('key', value, timeout)`: Stores data in cache (like putting soup in the warmer)
- Cache keys are strings that uniquely identify cached data

---

## 2. Cache Backends (Redis, Memcached)

### Imagine that...
Your local kitchen storage (LocMemCache) is like your countertop - great for small items, but what if you're running a chain of restaurants? You need a central refrigerated warehouse (Redis) or a distributed cold storage system (Memcached) that all your kitchen locations can access.

### Redis Backend - The Premium Cold Storage

```python
# settings.py - Setting up Redis (premium storage)
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'KEY_PREFIX': 'restaurant_app',
        'TIMEOUT': 300,
    }
}

# Alternative configuration for production
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://redis-server:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            }
        }
    }
}
```

**Syntax Explanation:**
- `django_redis.cache.RedisCache`: Redis backend implementation
- `LOCATION`: Redis server address and database number
- `KEY_PREFIX`: Namespace for cache keys (prevents conflicts)
- `CONNECTION_POOL_KWARGS`: Performance optimization settings

### Memcached Backend - The Distributed Network

```python
# settings.py - Setting up Memcached (distributed storage)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.memcached.PyMemcacheCache',
        'LOCATION': [
            '127.0.0.1:11211',
            '127.0.0.1:11212',  # Multiple servers for redundancy
        ],
        'TIMEOUT': 300,
        'OPTIONS': {
            'MAX_ENTRIES': 1000,
        }
    }
}
```

### Advanced Caching Patterns

```python
# utils/cache_helpers.py - Your caching recipe book
from django.core.cache import cache
from django.core.cache.utils import make_template_fragment_key
import hashlib

class ChefCacheManager:
    """A master chef's approach to caching"""
    
    @staticmethod
    def get_or_set_recipe(key, callable_func, timeout=300):
        """Get cached result or cook fresh if needed"""
        result = cache.get(key)
        if result is None:
            result = callable_func()
            cache.set(key, result, timeout)
        return result
    
    @staticmethod
    def invalidate_menu_cache(menu_id):
        """Remove old menu from cache when updated"""
        cache.delete(f'menu_{menu_id}')
        cache.delete('all_menus')
    
    @staticmethod
    def generate_cache_key(*args):
        """Create unique cache keys like recipe names"""
        key_string = ':'.join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()

# views.py - Using your caching recipes
from .utils.cache_helpers import ChefCacheManager

def get_restaurant_menu(request, restaurant_id):
    cache_key = ChefCacheManager.generate_cache_key('menu', restaurant_id)
    
    def fetch_menu():
        return MenuItem.objects.filter(restaurant_id=restaurant_id).select_related('category')
    
    menu_items = ChefCacheManager.get_or_set_recipe(cache_key, fetch_menu, 1800)
    
    return render(request, 'menu.html', {'menu_items': menu_items})
```

**Syntax Explanation:**
- `@staticmethod`: Methods that don't need class instance (utility functions)
- `hashlib.md5()`: Creates unique hash for cache keys
- `select_related()`: Django ORM optimization (reduces database queries)

---

## 3. Template Fragment Caching

### Imagine that...
You're preparing a complex plated dessert. The chocolate sauce and fruit garnish change daily, but the cake base is always the same. Instead of rebuilding the entire dessert each time, you can cache the cake base and only prepare the toppings fresh. Template fragment caching works the same way - cache the parts that don't change often.

### Basic Fragment Caching

```html
<!-- templates/restaurant/menu.html -->
{% load cache %}

<div class="menu-container">
    <h1>Today's Menu</h1>
    
    <!-- Cache the expensive header calculation for 1 hour -->
    {% cache 3600 menu_header restaurant.id %}
        <div class="menu-stats">
            <h2>{{ restaurant.name }}</h2>
            <p>Average Rating: {{ restaurant.avg_rating|floatformat:1 }}</p>
            <p>Total Items: {{ restaurant.menu_items.count }}</p>
            <p>Chef's Special: {{ restaurant.get_daily_special }}</p>
        </div>
    {% endcache %}
    
    <!-- Dynamic content that changes frequently -->
    <div class="current-orders">
        <h3>Live Orders: {{ current_orders.count }}</h3>
        {% for order in current_orders %}
            <div class="order-item">{{ order.item_name }} - {{ order.status }}</div>
        {% endfor %}
    </div>
    
    <!-- Cache menu items by category -->
    {% for category in categories %}
        {% cache 1800 menu_category category.id %}
            <div class="category-section">
                <h3>{{ category.name }}</h3>
                {% for item in category.menu_items.all %}
                    <div class="menu-item">
                        <span class="item-name">{{ item.name }}</span>
                        <span class="item-price">${{ item.price }}</span>
                    </div>
                {% endfor %}
            </div>
        {% endcache %}
    {% endfor %}
</div>
```

**Syntax Explanation:**
- `{% load cache %}`: Loads template caching tags
- `{% cache timeout key_var1 key_var2 %}`: Caches template fragment
- `timeout`: Cache duration in seconds
- `key_var1, key_var2`: Variables that make cache key unique
- `{% endcache %}`: Closes the cached section

### Advanced Fragment Caching with Vary-On

```html
<!-- templates/restaurant/user_dashboard.html -->
{% load cache %}

<div class="dashboard">
    <!-- Cache varies by user and their role -->
    {% cache 1800 user_sidebar user.id user.role %}
        <div class="sidebar">
            <h3>Welcome, {{ user.first_name }}!</h3>
            {% if user.role == 'chef' %}
                <ul>
                    <li><a href="{% url 'kitchen_orders' %}">Kitchen Orders</a></li>
                    <li><a href="{% url 'inventory' %}">Inventory</a></li>
                </ul>
            {% elif user.role == 'waiter' %}
                <ul>
                    <li><a href="{% url 'table_assignments' %}">Tables</a></li>
                    <li><a href="{% url 'order_history' %}">Order History</a></li>
                </ul>
            {% endif %}
        </div>
    {% endcache %}
    
    <!-- Never cache real-time data -->
    <div class="live-updates">
        <h3>Current Status</h3>
        <p>Orders in queue: {{ live_orders.count }}</p>
        <p>Last update: {{ now|date:"H:i:s" }}</p>
    </div>
</div>
```

### Cache Invalidation in Views

```python
# views.py - Managing your cached fragments
from django.core.cache import cache
from django.core.cache.utils import make_template_fragment_key

def update_menu_item(request, item_id):
    if request.method == 'POST':
        menu_item = MenuItem.objects.get(id=item_id)
        # Update menu item
        menu_item.name = request.POST.get('name')
        menu_item.price = request.POST.get('price')
        menu_item.save()
        
        # Clear related caches - like removing old prep from storage
        category_key = make_template_fragment_key('menu_category', [menu_item.category.id])
        cache.delete(category_key)
        
        # Also clear any user-specific caches if needed
        for user_id in [1, 2, 3]:  # In real app, get affected users
            user_key = make_template_fragment_key('user_sidebar', [user_id, 'chef'])
            cache.delete(user_key)
        
        return JsonResponse({'status': 'success'})
```

**Syntax Explanation:**
- `make_template_fragment_key()`: Creates the same key format as template cache
- `cache.delete()`: Removes specific cached item
- Cache invalidation ensures users see fresh data after updates

---

## 4. Database Query Caching

### Imagine that...
Every time someone asks "What's the most popular dish?", you could run to the dining room, count every plate, and calculate. But a smart chef keeps a running tally throughout the day. Database query caching does the same - it stores the results of expensive database operations so you don't have to "count the plates" every time.

### Query-Level Caching

```python
# models.py - Your recipe database
from django.db import models
from django.core.cache import cache

class Restaurant(models.Model):
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=200)
    
    def get_popular_dishes(self, limit=5):
        """Get popular dishes with caching - like keeping a best-sellers list"""
        cache_key = f'popular_dishes_{self.id}_{limit}'
        popular_dishes = cache.get(cache_key)
        
        if popular_dishes is None:
            # Expensive query - like counting all orders
            popular_dishes = list(
                self.menu_items.annotate(
                    total_orders=models.Count('orders')
                ).order_by('-total_orders')[:limit]
            )
            # Cache for 2 hours
            cache.set(cache_key, popular_dishes, 7200)
        
        return popular_dishes
    
    def get_daily_revenue(self, date):
        """Calculate daily revenue with caching"""
        cache_key = f'daily_revenue_{self.id}_{date}'
        revenue = cache.get(cache_key)
        
        if revenue is None:
            revenue = self.orders.filter(
                created_at__date=date,
                status='completed'
            ).aggregate(
                total=models.Sum('total_amount')
            )['total'] or 0
            
            # Cache completed day's revenue permanently
            timeout = None if date < timezone.now().date() else 3600
            cache.set(cache_key, revenue, timeout)
        
        return revenue

class MenuItem(models.Model):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE, related_name='menu_items')
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.ForeignKey('Category', on_delete=models.CASCADE)
    
    @property
    def avg_rating(self):
        """Cached average rating calculation"""
        cache_key = f'avg_rating_{self.id}'
        rating = cache.get(cache_key)
        
        if rating is None:
            rating = self.reviews.aggregate(
                avg=models.Avg('rating')
            )['avg'] or 0
            cache.set(cache_key, rating, 1800)  # 30 minutes
        
        return rating
```

**Syntax Explanation:**
- `annotate()`: Adds calculated fields to query results
- `Count()`: Counts related objects
- `aggregate()`: Performs calculations across multiple rows
- `Sum()`, `Avg()`: Aggregate functions for calculations
- `related_name`: How to access related objects in reverse

### Advanced Query Caching Patterns

```python
# utils/query_cache.py - Your advanced caching cookbook
from django.core.cache import cache
from django.db import models
from functools import wraps

def cache_query_result(timeout=300, key_prefix='query'):
    """Decorator for caching query results - like labeling your prep containers"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create unique cache key from function and parameters
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            result = cache.get(cache_key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(cache_key, result, timeout)
            return result
        return wrapper
    return decorator

class CachedQueryManager:
    """Master chef's query caching system"""
    
    @staticmethod
    @cache_query_result(timeout=1800, key_prefix='restaurant_stats')
    def get_restaurant_statistics(restaurant_id):
        """Get comprehensive restaurant stats"""
        from .models import Restaurant
        
        restaurant = Restaurant.objects.get(id=restaurant_id)
        
        stats = {
            'total_menu_items': restaurant.menu_items.count(),
            'avg_item_price': restaurant.menu_items.aggregate(
                avg_price=models.Avg('price')
            )['avg_price'],
            'total_orders_today': restaurant.orders.filter(
                created_at__date=timezone.now().date()
            ).count(),
            'popular_categories': list(
                restaurant.menu_items.values('category__name')
                .annotate(count=models.Count('id'))
                .order_by('-count')[:3]
            )
        }
        
        return stats
    
    @staticmethod
    def invalidate_restaurant_cache(restaurant_id):
        """Clear all caches for a restaurant - like cleaning the prep station"""
        cache_patterns = [
            f'popular_dishes_{restaurant_id}*',
            f'daily_revenue_{restaurant_id}*',
            f'restaurant_stats:*{restaurant_id}*'
        ]
        
        # Note: This is a simplified example
        # In production, use Redis patterns or keep track of cache keys
        for pattern in cache_patterns:
            cache.delete(pattern)

# views.py - Using your cached queries
from .utils.query_cache import CachedQueryManager

def restaurant_dashboard(request, restaurant_id):
    # Get cached statistics
    stats = CachedQueryManager.get_restaurant_statistics(restaurant_id)
    
    # Get real-time data (never cached)
    live_orders = Order.objects.filter(
        restaurant_id=restaurant_id,
        status__in=['pending', 'preparing']
    )
    
    context = {
        'stats': stats,
        'live_orders': live_orders,
        'last_updated': timezone.now()
    }
    
    return render(request, 'dashboard.html', context)
```

**Syntax Explanation:**
- `@wraps(func)`: Preserves original function metadata in decorators
- `hash()`: Creates unique identifier from function parameters
- `f-strings`: Modern Python string formatting
- `timezone.now()`: Django's timezone-aware current time
- `values()`: Returns dictionary-like objects instead of model instances

### Cache Warming Strategy

```python
# management/commands/warm_cache.py - Prepping your kitchen
from django.core.management.base import BaseCommand
from django.core.cache import cache
from myapp.models import Restaurant

class Command(BaseCommand):
    help = 'Warm up the cache with frequently accessed data'
    
    def handle(self, *args, **options):
        """Pre-cook popular dishes for the day"""
        self.stdout.write('Starting cache warming...')
        
        restaurants = Restaurant.objects.all()
        
        for restaurant in restaurants:
            # Pre-cache popular dishes
            restaurant.get_popular_dishes()
            
            # Pre-cache today's revenue
            restaurant.get_daily_revenue(timezone.now().date())
            
            # Pre-cache restaurant statistics
            CachedQueryManager.get_restaurant_statistics(restaurant.id)
            
            self.stdout.write(f'Warmed cache for {restaurant.name}')
        
        self.stdout.write(self.style.SUCCESS('Cache warming completed!'))
```

**Syntax Explanation:**
- `BaseCommand`: Django's management command base class
- `handle()`: Main method that runs when command is executed
- `self.stdout.write()`: Outputs messages to terminal
- `self.style.SUCCESS()`: Colors terminal output green

---
# Building an Optimized Expense Dashboard

## Project Objective
By the end of this project, you will build a high-performance expense dashboard that combines multiple caching strategies to deliver lightning-fast data visualization and user experience.

---

## The Master Chef's Kitchen: Performance Optimization

Imagine that you're the head chef of a prestigious restaurant that serves hundreds of customers daily. Your kitchen operates like a well-oiled machine, but during peak hours, orders pile up and customers wait too long for their meals. What do you do?

A master chef doesn't just cook faster – they optimize their entire kitchen workflow. They pre-prepare ingredients (cache frequently used data), set up mise en place stations (template fragment caching), and create signature dishes that can be partially prepared in advance (database query optimization).

Today, we're going to transform your Django application into that master chef's kitchen, building an expense dashboard that serves data as efficiently as a five-star restaurant serves its signature dishes.

---

## Building the Optimized Expense Dashboard

### Step 1: Setting Up the Foundation

First, let's create our expense dashboard models and views. Think of this as setting up your kitchen's basic equipment:

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import datetime, timedelta

class ExpenseCategory(models.Model):
    name = models.CharField(max_length=100)
    color = models.CharField(max_length=7, default='#3498db')  # Hex color
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name_plural = "Expense Categories"

class Expense(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(ExpenseCategory, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField(blank=True)
    date = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.amount} - {self.category.name}"
    
    class Meta:
        ordering = ['-date']
```

### Step 2: Creating the Optimized Dashboard View

Now, let's create our main dashboard view. This is like being the executive chef who orchestrates the entire kitchen:

```python
# views.py
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.db.models import Sum, Count, Avg
from django.db.models.functions import TruncMonth, TruncWeek
from django.utils import timezone
from datetime import datetime, timedelta
import json

@login_required
def expense_dashboard(request):
    """
    Main dashboard view with comprehensive caching strategy
    Like a master chef's command center - everything optimized for speed
    """
    user = request.user
    cache_key_prefix = f"expense_dashboard_{user.id}"
    
    # Try to get cached data first (like checking if ingredients are pre-prepped)
    cached_data = cache.get(f"{cache_key_prefix}_main")
    
    if cached_data:
        context = cached_data
    else:
        # If not cached, prepare fresh data (like cooking from scratch)
        context = _prepare_dashboard_data(user)
        # Cache for 15 minutes (like keeping prepared ingredients fresh)
        cache.set(f"{cache_key_prefix}_main", context, 900)
    
    return render(request, 'expenses/dashboard.html', context)

def _prepare_dashboard_data(user):
    """
    Prepare all dashboard data - like a sous chef prepping all ingredients
    """
    # Current month expenses
    current_month = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Monthly summary (using select_related for efficient queries)
    monthly_expenses = Expense.objects.filter(
        user=user,
        date__gte=current_month
    ).select_related('category').aggregate(
        total=Sum('amount'),
        count=Count('id'),
        average=Avg('amount')
    )
    
    # Category breakdown
    category_data = Expense.objects.filter(
        user=user,
        date__gte=current_month
    ).values('category__name', 'category__color').annotate(
        total=Sum('amount'),
        count=Count('id')
    ).order_by('-total')
    
    # Monthly trends (last 6 months)
    six_months_ago = current_month - timedelta(days=180)
    monthly_trends = Expense.objects.filter(
        user=user,
        date__gte=six_months_ago
    ).annotate(
        month=TruncMonth('date')
    ).values('month').annotate(
        total=Sum('amount')
    ).order_by('month')
    
    # Weekly trends (last 8 weeks)
    eight_weeks_ago = timezone.now() - timedelta(weeks=8)
    weekly_trends = Expense.objects.filter(
        user=user,
        date__gte=eight_weeks_ago
    ).annotate(
        week=TruncWeek('date')
    ).values('week').annotate(
        total=Sum('amount')
    ).order_by('week')
    
    # Recent expenses (cached separately for frequent updates)
    recent_expenses = list(Expense.objects.filter(
        user=user
    ).select_related('category')[:10].values(
        'amount', 'description', 'date', 'category__name', 'category__color'
    ))
    
    # Top spending categories
    top_categories = Expense.objects.filter(
        user=user,
        date__gte=six_months_ago
    ).values('category__name', 'category__color').annotate(
        total=Sum('amount')
    ).order_by('-total')[:5]
    
    return {
        'monthly_summary': monthly_expenses,
        'category_data': list(category_data),
        'monthly_trends': list(monthly_trends),
        'weekly_trends': list(weekly_trends),
        'recent_expenses': recent_expenses,
        'top_categories': list(top_categories),
        'chart_data': _prepare_chart_data(category_data, monthly_trends, weekly_trends)
    }

def _prepare_chart_data(category_data, monthly_trends, weekly_trends):
    """
    Prepare data for JavaScript charts - like plating the final dish
    """
    return {
        'category_chart': {
            'labels': [item['category__name'] for item in category_data],
            'data': [float(item['total']) for item in category_data],
            'colors': [item['category__color'] for item in category_data]
        },
        'monthly_chart': {
            'labels': [item['month'].strftime('%b %Y') for item in monthly_trends],
            'data': [float(item['total']) for item in monthly_trends]
        },
        'weekly_chart': {
            'labels': [item['week'].strftime('Week %U') for item in weekly_trends],
            'data': [float(item['total']) for item in weekly_trends]
        }
    }
```

### Step 3: Creating the Dashboard Template

Now let's create our template with fragment caching - like having different stations in the kitchen that can work independently:

```html
<!-- templates/expenses/dashboard.html -->
{% extends 'base.html' %}
{% load cache %}
{% load humanize %}

{% block title %}Expense Dashboard{% endblock %}

{% block extra_css %}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
    .dashboard-card {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .chart-container {
        position: relative;
        height: 400px;
        margin-bottom: 20px;
    }
    
    .expense-list {
        max-height: 300px;
        overflow-y: auto;
    }
    
    .expense-item {
        display: flex;
        justify-content: space-between;
        padding: 10px;
        border-bottom: 1px solid #eee;
    }
    
    .category-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
</style>
{% endblock %}

{% block content %}
<div class="container-fluid">
    <div class="row">
        <div class="col-12">
            <h1>Expense Dashboard</h1>
            <p class="text-muted">Your financial overview at a glance</p>
        </div>
    </div>
    
    <!-- Monthly Summary Cards (cached for 30 minutes) -->
    {% cache 1800 monthly_summary user.id %}
    <div class="row">
        <div class="col-md-4">
            <div class="metric-card">
                <div class="metric-value">${{ monthly_summary.total|floatformat:2|default:"0.00" }}</div>
                <div>Total This Month</div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="metric-card">
                <div class="metric-value">{{ monthly_summary.count|default:"0" }}</div>
                <div>Transactions</div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="metric-card">
                <div class="metric-value">${{ monthly_summary.average|floatformat:2|default:"0.00" }}</div>
                <div>Average Amount</div>
            </div>
        </div>
    </div>
    {% endcache %}
    
    <!-- Charts Row -->
    <div class="row">
        <!-- Category Breakdown (cached for 1 hour) -->
        {% cache 3600 category_breakdown user.id %}
        <div class="col-lg-6">
            <div class="dashboard-card">
                <h4>Spending by Category</h4>
                <div class="chart-container">
                    <canvas id="categoryChart"></canvas>
                </div>
            </div>
        </div>
        {% endcache %}
        
        <!-- Monthly Trends (cached for 2 hours) -->
        {% cache 7200 monthly_trends user.id %}
        <div class="col-lg-6">
            <div class="dashboard-card">
                <h4>Monthly Trends</h4>
                <div class="chart-container">
                    <canvas id="monthlyChart"></canvas>
                </div>
            </div>
        </div>
        {% endcache %}
    </div>
    
    <!-- Recent Expenses and Top Categories -->
    <div class="row">
        <!-- Recent Expenses (cached for 5 minutes - frequent updates) -->
        {% cache 300 recent_expenses user.id %}
        <div class="col-lg-6">
            <div class="dashboard-card">
                <h4>Recent Expenses</h4>
                <div class="expense-list">
                    {% for expense in recent_expenses %}
                    <div class="expense-item">
                        <div>
                            <span class="category-dot" style="background-color: {{ expense.category__color }}"></span>
                            {{ expense.description|default:"No description" }}
                            <small class="text-muted d-block">{{ expense.date|date:"M d, Y" }}</small>
                        </div>
                        <div class="font-weight-bold">
                            ${{ expense.amount|floatformat:2 }}
                        </div>
                    </div>
                    {% empty %}
                    <p class="text-muted">No expenses yet. Start tracking your spending!</p>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endcache %}
        
        <!-- Top Categories (cached for 1 hour) -->
        {% cache 3600 top_categories user.id %}
        <div class="col-lg-6">
            <div class="dashboard-card">
                <h4>Top Spending Categories</h4>
                {% for category in top_categories %}
                <div class="expense-item">
                    <div>
                        <span class="category-dot" style="background-color: {{ category.category__color }}"></span>
                        {{ category.category__name }}
                    </div>
                    <div class="font-weight-bold">
                        ${{ category.total|floatformat:2 }}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endcache %}
    </div>
    
    <!-- Weekly Trends -->
    {% cache 1800 weekly_trends user.id %}
    <div class="row">
        <div class="col-12">
            <div class="dashboard-card">
                <h4>Weekly Spending Pattern</h4>
                <div class="chart-container">
                    <canvas id="weeklyChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    {% endcache %}
</div>

<script>
// Chart data from Django (like recipes passed from the chef to the plating station)
const chartData = {{ chart_data|safe }};

// Category Pie Chart
const categoryCtx = document.getElementById('categoryChart').getContext('2d');
new Chart(categoryCtx, {
    type: 'doughnut',
    data: {
        labels: chartData.category_chart.labels,
        datasets: [{
            data: chartData.category_chart.data,
            backgroundColor: chartData.category_chart.colors,
            borderWidth: 2,
            borderColor: '#fff'
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom'
            }
        }
    }
});

// Monthly Trends Line Chart
const monthlyCtx = document.getElementById('monthlyChart').getContext('2d');
new Chart(monthlyCtx, {
    type: 'line',
    data: {
        labels: chartData.monthly_chart.labels,
        datasets: [{
            label: 'Monthly Spending',
            data: chartData.monthly_chart.data,
            borderColor: '#667eea',
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    callback: function(value) {
                        return '$' + value.toFixed(2);
                    }
                }
            }
        }
    }
});

// Weekly Trends Bar Chart
const weeklyCtx = document.getElementById('weeklyChart').getContext('2d');
new Chart(weeklyCtx, {
    type: 'bar',
    data: {
        labels: chartData.weekly_chart.labels,
        datasets: [{
            label: 'Weekly Spending',
            data: chartData.weekly_chart.data,
            backgroundColor: 'rgba(118, 75, 162, 0.8)',
            borderColor: '#764ba2',
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    callback: function(value) {
                        return '$' + value.toFixed(2);
                    }
                }
            }
        }
    }
});
</script>
{% endblock %}
```

### Step 4: Adding Cache Invalidation

Let's add intelligent cache invalidation - like updating your prep stations when new ingredients arrive:

```python
# signals.py
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.core.cache import cache
from .models import Expense

@receiver([post_save, post_delete], sender=Expense)
def invalidate_expense_cache(sender, instance, **kwargs):
    """
    Clear cache when expenses are modified
    Like updating the prep station when ingredients change
    """
    user_id = instance.user.id
    cache_patterns = [
        f"expense_dashboard_{user_id}_main",
        f"monthly_summary_{user_id}",
        f"recent_expenses_{user_id}",
        f"category_breakdown_{user_id}",
        f"monthly_trends_{user_id}",
        f"weekly_trends_{user_id}",
        f"top_categories_{user_id}",
    ]
    
    for pattern in cache_patterns:
        cache.delete(pattern)
```

### Step 5: URL Configuration

```python
# urls.py
from django.urls import path
from . import views

app_name = 'expenses'

urlpatterns = [
    path('dashboard/', views.expense_dashboard, name='dashboard'),
    # Add other expense-related URLs here
]
```

---

## Coding Syntax Explanation

Let me break down the key coding concepts we used:

### 1. **Django ORM Optimization**
```python
# select_related() - Joins related tables in a single query
.select_related('category')
# Like getting all ingredients and their storage locations in one trip

# aggregate() - Performs database-level calculations
.aggregate(total=Sum('amount'), count=Count('id'))
# Like having the kitchen calculate totals instead of counting manually
```

### 2. **Template Caching**
```html
{% cache 1800 monthly_summary user.id %}
<!-- Content here is cached for 30 minutes (1800 seconds) -->
{% endcache %}
```
The syntax: `{% cache [timeout] [cache_key] [variables] %}`

### 3. **Database Annotations**
```python
# annotate() adds calculated fields to each record
.annotate(month=TruncMonth('date'))
# Like adding a "prepared_month" label to each expense
```

### 4. **JavaScript Template Integration**
```javascript
// Django passes data to JavaScript using the |safe filter
const chartData = {{ chart_data|safe }};
// The |safe filter tells Django not to escape the JSON
```

---

## The Final Dish: Your Optimized Dashboard

Congratulations! You've just built a restaurant-quality expense dashboard that:

- **Serves data lightning-fast** with multi-level caching
- **Minimizes database queries** with smart ORM optimization  
- **Updates intelligently** when data changes
- **Provides rich visualizations** with Chart.js integration
- **Scales efficiently** as your user base grows

Just like a master chef's kitchen, every component works together harmoniously. Your caching strategy ensures that frequently requested data is always ready to serve, while your database optimizations mean that even complex queries run smoothly during peak usage.

Your dashboard now performs like a five-star restaurant - fast, efficient, and always ready to impress your users with lightning-quick insights into their spending patterns.

## Assignment: Restaurant Performance Optimizer

### Your Mission
You're the head chef of a popular restaurant chain, and your digital ordering system is running slow during peak hours. Your task is to implement a comprehensive caching strategy that reduces database load and improves response times.

### Requirements

Create a Django application with the following models and implement caching at multiple levels:

1. **Models to Create:**
   - `Restaurant`: name, location, phone
   - `Category`: name, description
   - `MenuItem`: name, price, category, restaurant, description
   - `Order`: restaurant, total_amount, status, created_at
   - `OrderItem`: order, menu_item, quantity, price

2. **Caching Implementation:**
   - **View-level caching**: Cache expensive dashboard calculations
   - **Template fragment caching**: Cache menu sections by category
   - **Query caching**: Cache popular items and daily statistics
   - **Cache invalidation**: Properly clear caches when data changes

3. **Performance Features:**
   - Dashboard showing restaurant statistics (cached for 1 hour)
   - Menu display with category sections (cached for 30 minutes)
   - Popular items list (cached for 2 hours)
   - Real-time order status (never cached)

4. **Bonus Points:**
   - Implement cache warming management command
   - Add cache hit/miss monitoring
   - Create cache invalidation signals

### Deliverables
- Working Django application with all models
- At least 3 views demonstrating different caching strategies
- Templates showing fragment caching implementation
- Management command for cache warming
- Brief documentation explaining your caching decisions

### Success Criteria
Your application should demonstrate understanding of:
- When to cache vs. when to fetch fresh data
- Proper cache key generation and management
- Cache invalidation strategies
- Performance vs. data freshness trade-offs

Remember: A good chef knows that some ingredients are best fresh, while others improve with proper storage. Your caching strategy should reflect this wisdom!