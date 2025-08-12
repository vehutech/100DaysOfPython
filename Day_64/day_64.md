# Day 64: Django Performance Monitoring Course

## Learning Objective
By the end of this lesson, you will be able to identify performance bottlenecks in Django applications, monitor database queries effectively, track application metrics, and implement error tracking to ensure your web application runs as smoothly as a well-orchestrated kitchen during dinner rush.

---

## Introduction:

Imagine that you're the head chef of a bustling five-star restaurant. Your kitchen is your Django application, your sous chefs are your database queries, your wait staff represents your application metrics, and your restaurant manager is your error tracking system. 

Just like a successful restaurant needs constant monitoring to ensure orders go out quickly, ingredients don't spoil, and customers leave satisfied, your Django application needs performance monitoring to ensure users have a seamless experience. Today, we'll learn how to be the head chef who keeps everything running perfectly, even during the busiest times.

---

## Lesson 1: Django Performance Profiling

### The Kitchen Timer Analogy
Think of performance profiling as timing each station in your kitchen. You need to know if your appetizer station takes too long, if your grill is the bottleneck, or if plating is slowing down service.

### Code Example: Using Django Debug Toolbar

First, let's install and set up Django Debug Toolbar - your kitchen's timing system:

```bash
pip install django-debug-toolbar
```

```python
# settings.py
INSTALLED_APPS = [
    # ... your apps
    'debug_toolbar',
]

MIDDLEWARE = [
    'debug_toolbar.middleware.DebugToolbarMiddleware',
    # ... your other middleware
]

# Show toolbar only in development
INTERNAL_IPS = [
    '127.0.0.1',
]

# Debug toolbar configuration
DEBUG_TOOLBAR_CONFIG = {
    'SHOW_TOOLBAR_CALLBACK': lambda request: True,  # Always show in development
}
```

```python
# urls.py
from django.conf import settings
from django.urls import path, include

urlpatterns = [
    # ... your URLs
]

if settings.DEBUG:
    import debug_toolbar
    urlpatterns = [
        path('__debug__/', include(debug_toolbar.urls)),
    ] + urlpatterns
```

### Custom Profiling Decorator

Create a decorator to time your view functions like timing each dish preparation:

```python
# utils/profiling.py
import time
import functools
import logging

logger = logging.getLogger(__name__)

def profile_view(func):
    """
    Decorator to profile view execution time
    Like timing how long each dish takes to prepare
    """
    @functools.wraps(func)
    def wrapper(request, *args, **kwargs):
        start_time = time.time()
        result = func(request, *args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"View {func.__name__} took {execution_time:.2f} seconds")
        
        return result
    return wrapper

# Usage in views.py
from utils.profiling import profile_view

@profile_view
def restaurant_menu_view(request):
    """View that shows restaurant menu - needs to be fast!"""
    menu_items = MenuItem.objects.select_related('category').all()
    return render(request, 'menu.html', {'items': menu_items})
```

**Syntax Explanation:**
- `@functools.wraps(func)`: Preserves the original function's metadata
- `time.time()`: Gets current timestamp in seconds
- `logger.info()`: Logs information to Django's logging system
- `select_related()`: Optimizes database queries by fetching related objects in one query

---

## Lesson 2: Database Query Monitoring

### The Sous Chef Efficiency Analogy
Your database queries are like sous chefs preparing ingredients. You need to monitor if they're working efficiently or if someone is taking too long to chop vegetables, causing a backup in the entire kitchen.

### Code Example: Query Optimization and Monitoring

```python
# models.py
from django.db import models

class Restaurant(models.Model):
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=200)
    
class MenuItem(models.Model):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=6, decimal_places=2)
    category = models.CharField(max_length=50)
    
class Order(models.Model):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
```

```python
# views.py - Bad vs Good Query Examples
from django.shortcuts import render
from django.db import connection
from django.conf import settings

def inefficient_kitchen_view(request):
    """
    BAD: Like having each sous chef ask for ingredients one by one
    This creates N+1 query problems
    """
    restaurants = Restaurant.objects.all()
    data = []
    
    for restaurant in restaurants:
        # This creates one query per restaurant - inefficient!
        recent_orders = restaurant.order_set.all()[:5]
        data.append({
            'restaurant': restaurant,
            'recent_orders': recent_orders
        })
    
    return render(request, 'inefficient.html', {'data': data})

def efficient_kitchen_view(request):
    """
    GOOD: Like having all ingredients prepped and ready
    Uses select_related and prefetch_related for optimization
    """
    restaurants = Restaurant.objects.prefetch_related(
        'order_set__menu_item'  # Prefetch orders with their menu items
    ).select_related()
    
    data = []
    for restaurant in restaurants:
        # No additional queries needed - data is already fetched
        recent_orders = restaurant.order_set.all()[:5]
        data.append({
            'restaurant': restaurant,
            'recent_orders': recent_orders
        })
    
    return render(request, 'efficient.html', {'data': data})

def query_monitor_view(request):
    """Monitor queries like watching your sous chefs work"""
    initial_queries = len(connection.queries)
    
    # Your view logic here
    menu_items = MenuItem.objects.select_related('restaurant').all()
    
    final_queries = len(connection.queries)
    query_count = final_queries - initial_queries
    
    if settings.DEBUG:
        print(f"This view executed {query_count} database queries")
        for query in connection.queries[initial_queries:]:
            print(f"Query: {query['sql']}")
            print(f"Time: {query['time']}s")
    
    return render(request, 'menu.html', {
        'items': menu_items,
        'query_count': query_count
    })
```

**Syntax Explanation:**
- `select_related()`: Performs SQL JOIN to fetch related objects in one query
- `prefetch_related()`: Performs separate queries but reduces total query count for many-to-many relationships
- `connection.queries`: List of all database queries executed (only available in DEBUG mode)
- `len()`: Gets the length/count of a list or queryset

---

## Lesson 3: Application Performance Metrics

### The Restaurant Dashboard Analogy
Just like a restaurant manager needs a dashboard showing table turnover, average meal preparation time, and customer wait times, your Django app needs metrics to track response times, memory usage, and user activity patterns.

### Code Example: Custom Metrics Middleware

```python
# middleware/metrics.py
import time
import psutil
import logging
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger('performance')

class PerformanceMetricsMiddleware(MiddlewareMixin):
    """
    Middleware to collect performance metrics
    Like a restaurant manager timing everything
    """
    
    def process_request(self, request):
        # Start timing - like when customer places order
        request._start_time = time.time()
        request._start_memory = psutil.Process().memory_info().rss
    
    def process_response(self, request, response):
        # Calculate metrics - like when order is delivered
        if hasattr(request, '_start_time'):
            duration = time.time() - request._start_time
            memory_used = psutil.Process().memory_info().rss - request._start_memory
            
            # Log metrics like a restaurant's daily report
            logger.info(f"Path: {request.path}")
            logger.info(f"Method: {request.method}")
            logger.info(f"Duration: {duration:.2f}s")
            logger.info(f"Memory Delta: {memory_used / 1024 / 1024:.2f}MB")
            logger.info(f"Status Code: {response.status_code}")
            logger.info("---")
            
            # Add performance headers (like receipt timestamps)
            response['X-Response-Time'] = f"{duration:.2f}s"
            response['X-Memory-Usage'] = f"{memory_used / 1024 / 1024:.2f}MB"
        
        return response
```

```python
# utils/metrics_collector.py
from django.core.cache import cache
from django.utils import timezone
import json

class MetricsCollector:
    """
    Collect and store performance metrics
    Like keeping a restaurant's performance logbook
    """
    
    @staticmethod
    def record_page_view(path, duration, status_code):
        """Record a page view like logging a completed order"""
        today = timezone.now().strftime('%Y-%m-%d')
        cache_key = f"metrics:{today}"
        
        # Get existing metrics or create new
        metrics = cache.get(cache_key, {
            'page_views': 0,
            'total_response_time': 0,
            'error_count': 0,
            'popular_pages': {}
        })
        
        # Update metrics
        metrics['page_views'] += 1
        metrics['total_response_time'] += duration
        
        if status_code >= 400:
            metrics['error_count'] += 1
        
        # Track popular pages
        if path in metrics['popular_pages']:
            metrics['popular_pages'][path] += 1
        else:
            metrics['popular_pages'][path] = 1
        
        # Store back in cache
        cache.set(cache_key, metrics, 60 * 60 * 24)  # Store for 24 hours
    
    @staticmethod
    def get_daily_metrics():
        """Get today's metrics like daily restaurant report"""
        today = timezone.now().strftime('%Y-%m-%d')
        return cache.get(f"metrics:{today}", {})

# Usage in views
def dashboard_view(request):
    metrics = MetricsCollector.get_daily_metrics()
    avg_response_time = 0
    
    if metrics.get('page_views', 0) > 0:
        avg_response_time = metrics['total_response_time'] / metrics['page_views']
    
    return render(request, 'dashboard.html', {
        'metrics': metrics,
        'avg_response_time': avg_response_time
    })
```

**Syntax Explanation:**
- `MiddlewareMixin`: Base class for Django middleware
- `psutil.Process().memory_info().rss`: Gets current memory usage in bytes
- `cache.get(key, default)`: Retrieves value from cache or returns default
- `timezone.now().strftime()`: Formats current datetime as string
- `hasattr(obj, attr)`: Checks if object has specified attribute

---

## Lesson 4: Error Tracking with Sentry

### The Restaurant Manager Alert System Analogy
Sentry is like having a restaurant manager who immediately knows when something goes wrong in the kitchen - a burnt dish, a dropped plate, or an unhappy customer. Instead of finding out problems later, you get instant alerts with detailed information about what went wrong and where.

### Code Example: Sentry Integration

```bash
pip install sentry-sdk[django]
```

```python
# settings.py
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

# Sentry configuration - your restaurant's alert system
sentry_logging = LoggingIntegration(
    level=logging.INFO,        # Capture info and above as breadcrumbs
    event_level=logging.ERROR  # Send errors and above as events
)

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN_HERE",  # Replace with your actual DSN
    integrations=[
        DjangoIntegration(
            transaction_style='url',  # Track URLs as transactions
        ),
        sentry_logging,
    ],
    traces_sample_rate=0.1,  # Sample 10% of transactions for performance
    send_default_pii=False,  # Don't send personally identifiable information
    environment='production' if not DEBUG else 'development',
)
```

```python
# views.py - Error handling with Sentry
from sentry_sdk import capture_exception, capture_message, add_breadcrumb
from django.http import JsonResponse
from django.shortcuts import get_object_or_404

def order_processing_view(request):
    """
    Process restaurant orders with comprehensive error tracking
    Like having a manager monitor every step of order preparation
    """
    try:
        # Add breadcrumb - like leaving notes for the manager
        add_breadcrumb(
            message='Starting order processing',
            category='order',
            level='info'
        )
        
        order_id = request.POST.get('order_id')
        if not order_id:
            # Log warning - like noting a minor kitchen issue
            capture_message('Order ID missing from request', 'warning')
            return JsonResponse({'error': 'Order ID required'}, status=400)
        
        add_breadcrumb(
            message=f'Processing order {order_id}',
            category='order',
            level='info'
        )
        
        # Get order with error handling
        order = get_object_or_404(Order, id=order_id)
        
        # Simulate processing that might fail
        if order.menu_item.price > 100:
            # Custom exception handling
            raise ValueError(f"High-value order {order_id} needs manager approval")
        
        # Process the order
        order.status = 'processing'
        order.save()
        
        add_breadcrumb(
            message=f'Order {order_id} processed successfully',
            category='order',
            level='info'
        )
        
        return JsonResponse({'status': 'success', 'order_id': order_id})
        
    except ValueError as e:
        # Capture business logic errors
        capture_exception(e)
        return JsonResponse({
            'error': 'Order requires special handling',
            'message': str(e)
        }, status=422)
        
    except Exception as e:
        # Capture unexpected errors - like kitchen emergencies
        capture_exception(e)
        return JsonResponse({
            'error': 'Internal server error',
            'message': 'Something went wrong processing your order'
        }, status=500)

# Custom error handler
def custom_404_handler(request, exception):
    """Custom 404 handler with Sentry tracking"""
    capture_message(
        f'404 error: {request.path}',
        'warning',
        extras={'user_agent': request.META.get('HTTP_USER_AGENT')}
    )
    
    return render(request, '404.html', status=404)

# Performance monitoring example
def slow_kitchen_operation(request):
    """Monitor slow operations like tracking prep times"""
    with sentry_sdk.start_transaction(name="slow_kitchen_operation", op="view"):
        # Simulate slow database operation
        expensive_query_span = sentry_sdk.start_span(op="db", description="Complex menu query")
        
        try:
            # This represents a slow database query
            menu_items = MenuItem.objects.select_related('restaurant').filter(
                price__gte=50
            ).order_by('-price')[:100]
            
            expensive_query_span.set_data("query_count", len(menu_items))
            expensive_query_span.finish()
            
            return render(request, 'premium_menu.html', {'items': menu_items})
            
        except Exception as e:
            expensive_query_span.finish()
            raise e
```

**Syntax Explanation:**
- `capture_exception(e)`: Sends exception details to Sentry
- `capture_message(msg, level)`: Sends custom messages to Sentry
- `add_breadcrumb()`: Adds context information for debugging
- `sentry_sdk.start_transaction()`: Begins performance monitoring
- `sentry_sdk.start_span()`: Monitors specific operations within transactions
- `get_object_or_404()`: Django shortcut that raises 404 if object not found

---

## Assignment: Restaurant Performance Audit

**Scenario:** You've been hired as a technical consultant for "Django Delights," a restaurant chain that's experiencing performance issues with their online ordering system. Customer complaints include slow page loads, occasional errors during peak hours, and the system sometimes timing out.

**Your Task:** Create a comprehensive performance monitoring system for their Django application that includes:

1. **Performance Profiling Setup**
   - Implement the performance metrics middleware for all views
   - Create a custom decorator to monitor the top 3 slowest API endpoints: `/api/menu/`, `/api/orders/`, and `/api/restaurants/`
   - Set up Django Debug Toolbar for development environment

2. **Database Query Optimization**
   - Write a management command that analyzes and reports the 5 most expensive database queries
   - Create optimized views for the restaurant listing page that shows restaurants with their average rating and total orders
   - Implement query monitoring that alerts when any single request executes more than 10 database queries

3. **Metrics Dashboard**
   - Build a simple metrics dashboard view that displays:
     - Average response time for the current day
     - Total number of requests
     - Error rate (4xx and 5xx responses)
     - Most popular pages
   - Store metrics in Django cache with 24-hour expiration

4. **Error Tracking Implementation**
   - Integrate Sentry (you can use a dummy DSN for this exercise)
   - Add proper error handling to all views with meaningful breadcrumbs
   - Create custom error pages (404, 500) that log errors to Sentry
   - Implement performance transaction tracking for the checkout process

**Deliverables:**
- Complete Django project with all monitoring features implemented
- A README.md file explaining how to set up and use each monitoring feature
- Screenshots or logs showing the monitoring system in action
- A brief report (300-500 words) analyzing what performance bottlenecks you discovered and how your monitoring solution helps identify them

**Bonus Challenge:** Create a simple alert system that sends an email notification when average response time exceeds 2 seconds or error rate goes above 5% within any 10-minute period.

**Evaluation Criteria:**
- Correct implementation of all monitoring features
- Code quality and proper error handling
- Effectiveness of the monitoring solution
- Quality of documentation and analysis report

This assignment will test your ability to implement real-world performance monitoring solutions that every Django developer needs to master for production applications.

# Django Production Monitoring Setup - Final Project

## Project: Restaurant Chain Monitoring Dashboard

You'll build a comprehensive monitoring system for a multi-location restaurant chain's Django application, just like how a head chef monitors all kitchen operations across different restaurant branches.

### Project Structure

```
restaurant_monitor/
├── monitor_app/
│   ├── models.py
│   ├── views.py
│   ├── middleware.py
│   └── monitoring.py
├── templates/
│   └── dashboard.html
├── static/
│   └── dashboard.js
└── requirements.txt
```

### Step 1: Install Required Packages

```bash
pip install django-debug-toolbar
pip install psutil
pip install requests
pip install django-extensions
```

### Step 2: Custom Monitoring Middleware

```python
# monitor_app/middleware.py
import time
import psutil
import logging
from django.utils.deprecation import MiddlewareMixin
from django.db import connection
from django.core.cache import cache

logger = logging.getLogger('restaurant_monitor')

class RestaurantMonitoringMiddleware(MiddlewareMixin):
    """
    Like a sous chef keeping track of every order's timing and kitchen performance
    """
    
    def process_request(self, request):
        request.start_time = time.time()
        request.db_queries_before = len(connection.queries)
        
    def process_response(self, request, response):
        # Calculate response time (like timing how long each dish takes)
        if hasattr(request, 'start_time'):
            response_time = time.time() - request.start_time
            
            # Count database queries (like counting ingredients used)
            db_queries = len(connection.queries) - getattr(request, 'db_queries_before', 0)
            
            # Get system metrics (like checking kitchen equipment status)
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Store metrics in cache (like writing on the kitchen whiteboard)
            metrics_key = f"metrics_{int(time.time())}"
            cache.set(metrics_key, {
                'path': request.path,
                'method': request.method,
                'response_time': response_time,
                'db_queries': db_queries,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'status_code': response.status_code,
                'timestamp': time.time()
            }, timeout=3600)  # Keep for 1 hour
            
            # Log slow requests (like flagging orders taking too long)
            if response_time > 1.0:  # Over 1 second
                logger.warning(f"Slow request: {request.path} took {response_time:.2f}s")
                
        return response
```

### Step 3: Performance Monitoring Models

```python
# monitor_app/models.py
from django.db import models
from django.utils import timezone

class PerformanceLog(models.Model):
    """
    Like a kitchen log book tracking every service period
    """
    timestamp = models.DateTimeField(default=timezone.now)
    endpoint = models.CharField(max_length=200)
    response_time = models.FloatField()
    db_queries_count = models.IntegerField()
    cpu_usage = models.FloatField()
    memory_usage = models.FloatField()
    status_code = models.IntegerField()
    user_agent = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        
    def __str__(self):
        return f"{self.endpoint} - {self.response_time:.2f}s"

class ErrorLog(models.Model):
    """
    Like an incident report book for kitchen accidents
    """
    timestamp = models.DateTimeField(default=timezone.now)
    error_type = models.CharField(max_length=100)
    error_message = models.TextField()
    endpoint = models.CharField(max_length=200)
    stack_trace = models.TextField()
    user_id = models.IntegerField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']

class SystemMetrics(models.Model):
    """
    Like recording kitchen equipment readings every hour
    """
    timestamp = models.DateTimeField(default=timezone.now)
    cpu_usage = models.FloatField()
    memory_usage = models.FloatField()
    disk_usage = models.FloatField()
    active_connections = models.IntegerField()
    
    @classmethod
    def record_current_metrics(cls):
        """Record current system state like a chef checking all stations"""
        import psutil
        from django.db import connections
        
        # Get system metrics
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        # Count active database connections
        db_connections = 0
        for conn in connections.all():
            if conn.connection is not None:
                db_connections += 1
                
        cls.objects.create(
            cpu_usage=cpu,
            memory_usage=memory,
            disk_usage=disk,
            active_connections=db_connections
        )
```

### Step 4: Monitoring Dashboard Views

```python
# monitor_app/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.core.cache import cache
from django.db.models import Avg, Count
from django.utils import timezone
from datetime import timedelta
from .models import PerformanceLog, ErrorLog, SystemMetrics
import json

def monitoring_dashboard(request):
    """
    The head chef's command center - overview of all kitchen operations
    """
    # Get recent performance data (last 24 hours)
    yesterday = timezone.now() - timedelta(days=1)
    
    performance_stats = {
        'avg_response_time': PerformanceLog.objects.filter(
            timestamp__gte=yesterday
        ).aggregate(Avg('response_time'))['response_time__avg'] or 0,
        
        'total_requests': PerformanceLog.objects.filter(
            timestamp__gte=yesterday
        ).count(),
        
        'error_count': ErrorLog.objects.filter(
            timestamp__gte=yesterday
        ).count(),
        
        'slow_requests': PerformanceLog.objects.filter(
            timestamp__gte=yesterday,
            response_time__gt=1.0
        ).count()
    }
    
    # Get current system status
    latest_metrics = SystemMetrics.objects.first()
    
    context = {
        'performance_stats': performance_stats,
        'system_metrics': latest_metrics,
        'recent_errors': ErrorLog.objects.all()[:10]
    }
    
    return render(request, 'dashboard.html', context)

def api_metrics(request):
    """
    Live metrics API - like asking the kitchen for current order status
    """
    # Get cached metrics from last hour
    current_hour = int(timezone.now().timestamp() // 3600)
    metrics_keys = [f"metrics_{current_hour - i}" for i in range(24)]
    
    all_metrics = []
    for key in metrics_keys:
        hourly_metrics = cache.get(key, [])
        if hourly_metrics:
            all_metrics.extend(hourly_metrics if isinstance(hourly_metrics, list) else [hourly_metrics])
    
    # Calculate averages (like averaging cooking times per hour)
    if all_metrics:
        avg_response_time = sum(m.get('response_time', 0) for m in all_metrics) / len(all_metrics)
        avg_db_queries = sum(m.get('db_queries', 0) for m in all_metrics) / len(all_metrics)
        avg_cpu = sum(m.get('cpu_percent', 0) for m in all_metrics) / len(all_metrics)
        avg_memory = sum(m.get('memory_percent', 0) for m in all_metrics) / len(all_metrics)
    else:
        avg_response_time = avg_db_queries = avg_cpu = avg_memory = 0
    
    return JsonResponse({
        'response_time': round(avg_response_time, 3),
        'db_queries': round(avg_db_queries, 1),
        'cpu_usage': round(avg_cpu, 1),
        'memory_usage': round(avg_memory, 1),
        'timestamp': timezone.now().isoformat()
    })

def health_check(request):
    """
    Quick health check - like a chef's 30-second kitchen inspection
    """
    try:
        # Test database connection
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            
        # Test cache
        cache.set('health_check', 'ok', timeout=10)
        cache_status = cache.get('health_check') == 'ok'
        
        # Check system resources
        import psutil
        cpu_ok = psutil.cpu_percent() < 90
        memory_ok = psutil.virtual_memory().percent < 90
        disk_ok = psutil.disk_usage('/').percent < 90
        
        all_systems_good = all([cache_status, cpu_ok, memory_ok, disk_ok])
        
        return JsonResponse({
            'status': 'healthy' if all_systems_good else 'warning',
            'database': 'ok',
            'cache': 'ok' if cache_status else 'error',
            'cpu': 'ok' if cpu_ok else 'high',
            'memory': 'ok' if memory_ok else 'high',
            'disk': 'ok' if disk_ok else 'full'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
```

### Step 5: Real-time Dashboard Template

```html
<!-- templates/dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Restaurant Chain Monitoring - Head Chef Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #1a1a1a; color: white; }
        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; padding: 20px; }
        .metric-card { background: #2d2d2d; border-radius: 8px; padding: 20px; border-left: 4px solid #4CAF50; }
        .metric-card.warning { border-left-color: #FF9800; }
        .metric-card.error { border-left-color: #f44336; }
        .metric-value { font-size: 2.5em; font-weight: bold; margin: 10px 0; }
        .chart-container { height: 300px; background: #333; border-radius: 8px; padding: 10px; }
        .error-log { background: #2d2d2d; border-radius: 8px; padding: 15px; margin: 20px; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }
        .status-ok { background: #4CAF50; }
        .status-warning { background: #FF9800; }
        .status-error { background: #f44336; }
    </style>
</head>
<body>
    <h1 style="text-align: center; padding: 20px;">🍳 Restaurant Chain Monitoring Dashboard</h1>
    
    <div class="dashboard">
        <!-- Performance Metrics -->
        <div class="metric-card">
            <h3>⏱️ Average Response Time</h3>
            <div class="metric-value" id="response-time">{{ performance_stats.avg_response_time|floatformat:3 }}s</div>
            <small>Kitchen speed (last 24h)</small>
        </div>
        
        <div class="metric-card">
            <h3>📊 Total Requests</h3>
            <div class="metric-value" id="total-requests">{{ performance_stats.total_requests }}</div>
            <small>Orders processed (last 24h)</small>
        </div>
        
        <div class="metric-card {% if performance_stats.error_count > 0 %}warning{% endif %}">
            <h3>🚨 Error Count</h3>
            <div class="metric-value" id="error-count">{{ performance_stats.error_count }}</div>
            <small>Kitchen incidents (last 24h)</small>
        </div>
        
        <div class="metric-card {% if performance_stats.slow_requests > 5 %}warning{% endif %}">
            <h3>🐌 Slow Requests</h3>
            <div class="metric-value" id="slow-requests">{{ performance_stats.slow_requests }}</div>
            <small>Orders taking >1s (last 24h)</small>
        </div>
        
        <!-- System Metrics -->
        {% if system_metrics %}
        <div class="metric-card {% if system_metrics.cpu_usage > 80 %}warning{% endif %}">
            <h3>🖥️ CPU Usage</h3>
            <div class="metric-value">{{ system_metrics.cpu_usage|floatformat:1 }}%</div>
            <small>Kitchen equipment load</small>
        </div>
        
        <div class="metric-card {% if system_metrics.memory_usage > 80 %}warning{% endif %}">
            <h3>🧠 Memory Usage</h3>
            <div class="metric-value">{{ system_metrics.memory_usage|floatformat:1 }}%</div>
            <small>Storage space used</small>
        </div>
        {% endif %}
    </div>
    
    <!-- Real-time Chart -->
    <div class="chart-container" id="real-time-chart">
        <h3>📈 Live Performance Metrics</h3>
        <canvas id="metricsChart" width="800" height="200"></canvas>
    </div>
    
    <!-- Recent Errors -->
    <div class="error-log">
        <h3>🔥 Recent Kitchen Incidents</h3>
        {% for error in recent_errors %}
        <div style="border-bottom: 1px solid #555; padding: 10px;">
            <span class="status-indicator status-error"></span>
            <strong>{{ error.error_type }}</strong> at {{ error.endpoint }}
            <br><small>{{ error.timestamp }} - {{ error.error_message|truncatechars:100 }}</small>
        </div>
        {% empty %}
        <p style="color: #4CAF50;">✅ All systems running smoothly - no recent incidents!</p>
        {% endfor %}
    </div>

    <script>
        // Real-time updates - like a chef constantly checking order status
        let metricsData = [];
        const maxDataPoints = 50;
        
        async function updateMetrics() {
            try {
                const response = await fetch('/api/metrics/');
                const data = await response.json();
                
                // Update metric cards
                document.getElementById('response-time').textContent = data.response_time + 's';
                
                // Update chart data
                metricsData.push({
                    timestamp: new Date(data.timestamp),
                    response_time: data.response_time,
                    cpu_usage: data.cpu_usage,
                    memory_usage: data.memory_usage
                });
                
                // Keep only recent data
                if (metricsData.length > maxDataPoints) {
                    metricsData.shift();
                }
                
                drawChart();
                
            } catch (error) {
                console.error('Failed to update metrics:', error);
            }
        }
        
        function drawChart() {
            const canvas = document.getElementById('metricsChart');
            const ctx = canvas.getContext('2d');
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (metricsData.length === 0) return;
            
            // Draw response time line (like tracking order completion times)
            ctx.strokeStyle = '#4CAF50';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            metricsData.forEach((point, index) => {
                const x = (index / (maxDataPoints - 1)) * canvas.width;
                const y = canvas.height - (point.response_time / 2.0) * canvas.height; // Scale to 2s max
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
            
            // Draw CPU usage line
            ctx.strokeStyle = '#FF9800';
            ctx.lineWidth = 1;
            ctx.beginPath();
            
            metricsData.forEach((point, index) => {
                const x = (index / (maxDataPoints - 1)) * canvas.width;
                const y = canvas.height - (point.cpu_usage / 100) * canvas.height;
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
        }
        
        // Update every 5 seconds (like checking the kitchen every few minutes)
        updateMetrics();
        setInterval(updateMetrics, 5000);
        
        // Health check indicator
        async function checkHealth() {
            try {
                const response = await fetch('/health/');
                const data = await response.json();
                
                const statusColor = {
                    'healthy': '#4CAF50',
                    'warning': '#FF9800',
                    'error': '#f44336'
                }[data.status];
                
                // Update page title color to show status
                document.querySelector('h1').style.borderLeft = `5px solid ${statusColor}`;
                
            } catch (error) {
                document.querySelector('h1').style.borderLeft = '5px solid #f44336';
            }
        }
        
        checkHealth();
        setInterval(checkHealth, 30000); // Check every 30 seconds
    </script>
</body>
</html>
```

### Step 6: Management Commands for System Monitoring

```python
# monitor_app/management/commands/collect_metrics.py
from django.core.management.base import BaseCommand
from monitor_app.models import SystemMetrics
import time

class Command(BaseCommand):
    help = 'Collect system metrics like a chef doing hourly kitchen checks'
    
    def add_arguments(self, parser):
        parser.add_argument('--interval', type=int, default=300, 
                          help='Collection interval in seconds (default: 300)')
        parser.add_argument('--duration', type=int, default=0,
                          help='How long to run (0 = forever)')
    
    def handle(self, *args, **options):
        interval = options['interval']
        duration = options['duration']
        start_time = time.time()
        
        self.stdout.write(f"🍳 Starting metrics collection every {interval} seconds...")
        
        while True:
            try:
                # Record current metrics (like a chef checking all stations)
                SystemMetrics.record_current_metrics()
                self.stdout.write(f"✅ Metrics recorded at {time.strftime('%H:%M:%S')}")
                
                # Clean old data (keep only last 7 days)
                from django.utils import timezone
                from datetime import timedelta
                week_ago = timezone.now() - timedelta(days=7)
                
                deleted_count = SystemMetrics.objects.filter(
                    timestamp__lt=week_ago
                ).delete()[0]
                
                if deleted_count > 0:
                    self.stdout.write(f"🧹 Cleaned {deleted_count} old metric records")
                
                # Check if we should stop
                if duration > 0 and (time.time() - start_time) >= duration:
                    self.stdout.write("⏰ Collection duration reached, stopping...")
                    break
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                self.stdout.write("🛑 Stopping metrics collection...")
                break
            except Exception as e:
                self.stdout.write(f"❌ Error collecting metrics: {e}")
                time.sleep(interval)
```

### Step 7: Django Settings Configuration

```python
# settings.py additions
import logging

# Add monitoring middleware
MIDDLEWARE = [
    'monitor_app.middleware.RestaurantMonitoringMiddleware',
    # ... your existing middleware
]

# Cache configuration for storing metrics
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Logging configuration (like keeping kitchen logbooks)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'WARNING',
            'class': 'logging.FileHandler',
            'filename': 'restaurant_monitor.log',
            'formatter': 'detailed',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
        },
    },
    'loggers': {
        'restaurant_monitor': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# Performance monitoring settings
PERFORMANCE_MONITORING = {
    'SLOW_REQUEST_THRESHOLD': 1.0,  # Log requests over 1 second
    'MAX_DB_QUERIES': 50,  # Alert if more than 50 queries per request
    'METRICS_RETENTION_HOURS': 24,  # Keep metrics for 24 hours in cache
}
```

### Step 8: URL Configuration

```python
# urls.py
from django.urls import path
from monitor_app import views

urlpatterns = [
    path('dashboard/', views.monitoring_dashboard, name='monitoring_dashboard'),
    path('api/metrics/', views.api_metrics, name='api_metrics'),
    path('health/', views.health_check, name='health_check'),
    # ... your other URLs
]
```

### Step 9: Deployment Setup Script

```python
# deploy_monitoring.py
#!/usr/bin/env python3
"""
Production monitoring deployment script - like setting up a new kitchen's monitoring system
"""
import os
import sys
import subprocess
import django

def setup_monitoring():
    """Set up monitoring like preparing a restaurant for opening"""
    
    print("🍳 Setting up Restaurant Chain Monitoring System...")
    
    # Run migrations
    print("📋 Creating database tables (setting up logbooks)...")
    subprocess.run([sys.executable, 'manage.py', 'makemigrations', 'monitor_app'])
    subprocess.run([sys.executable, 'manage.py', 'migrate'])
    
    # Create superuser if needed
    print("👨‍🍳 Creating head chef account...")
    subprocess.run([sys.executable, 'manage.py', 'createsuperuser', '--noinput'], 
                   input=b'yes\n', capture_output=True)
    
    # Collect static files
    print("🎨 Gathering dashboard assets...")
    subprocess.run([sys.executable, 'manage.py', 'collectstatic', '--noinput'])
    
    # Start metrics collection
    print("📊 Starting background metrics collection...")
    subprocess.Popen([sys.executable, 'manage.py', 'collect_metrics'])
    
    print("✅ Monitoring system is ready!")
    print("🌐 Access dashboard at: http://your-domain.com/dashboard/")
    print("💓 Health check at: http://your-domain.com/health/")

if __name__ == '__main__':
    setup_monitoring()
```

### Running the Complete System

1. **Install and migrate:**
```bash
pip install -r requirements.txt
python manage.py makemigrations monitor_app
python manage.py migrate
```

2. **Start metrics collection:**
```bash
python manage.py collect_metrics --interval=60  # Every minute
```

3. **Run the development server:**
```bash
python manage.py runserver
```

4. **Access the dashboard:**
- Main dashboard: `http://localhost:8000/dashboard/`
- API metrics: `http://localhost:8000/api/metrics/`
- Health check: `http://localhost:8000/health/`

This complete production monitoring setup gives you real-time performance tracking, error logging, system metrics, and a beautiful dashboard - just like a head chef having complete visibility into all restaurant operations across the entire chain. The system automatically collects metrics, stores historical data, and provides alerts when things go wrong.