# Day 50: Django Middleware & Request Processing - Complete Course

## Learning Objective
By the end of this lesson, you will understand Django's middleware system and be able to create custom middleware components that process requests and responses, monitor performance, and add functionality to your Django applications.

---

## Introduction: The Kitchen Brigade System

Imagine that you're running a high-end restaurant, and every customer order must pass through a well-orchestrated kitchen brigade system. Before a dish reaches the customer, it goes through multiple stations: the prep cook cleans and prepares ingredients, the sauce chef adds the perfect seasoning, the grill master cooks it to perfection, and the expediter does the final plating and quality check.

Django's middleware system works exactly like this kitchen brigade. When a request comes into your Django application, it passes through multiple "stations" (middleware components) before reaching your view function, and then passes through them again in reverse order before the response is sent back to the user. Each middleware component can inspect, modify, or even reject the request/response, just like how each chef can enhance or quality-check the dish at their station.

---

## Lesson 1: Understanding Django's Middleware System

### The Kitchen Brigade Analogy
In our restaurant analogy, middleware components are like specialized chefs in the kitchen brigade:

- **Security Chef** (SecurityMiddleware): Checks if ingredients are safe and fresh
- **Session Chef** (SessionMiddleware): Remembers customer preferences and allergies
- **Authentication Chef** (AuthenticationMiddleware): Verifies customer identity and membership status
- **CSRF Chef** (CsrfViewMiddleware): Prevents food tampering and ensures order authenticity

### How Middleware Works
Just like orders flow through the kitchen brigade, HTTP requests flow through middleware:

1. **Request Phase**: Request enters through each middleware (like ingredients going through each chef)
2. **View Processing**: Your view function processes the request (like the main chef creating the dish)
3. **Response Phase**: Response travels back through middleware in reverse order (like final quality checks)

### Default Django Middleware
Here's Django's default middleware stack (like your standard kitchen brigade):

```python
# settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',           # Security Chef
    'django.contrib.sessions.middleware.SessionMiddleware',    # Session Chef
    'django.middleware.common.CommonMiddleware',               # Common Tasks Chef
    'django.middleware.csrf.CsrfViewMiddleware',              # CSRF Protection Chef
    'django.contrib.auth.middleware.AuthenticationMiddleware', # Authentication Chef
    'django.contrib.messages.middleware.MessageMiddleware',    # Messaging Chef
    'django.middleware.clickjacking.XFrameOptionsMiddleware',  # Frame Protection Chef
]
```

### Code Example: Understanding the Flow
```python
# Example showing how a request flows through middleware
class ExampleMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        print("🍳 Middleware initialized - Kitchen is ready!")
    
    def __call__(self, request):
        print("📥 Request entering kitchen - Order received!")
        
        # This runs before the view
        print("👨‍🍳 Preprocessing ingredients...")
        
        # Get response from the next middleware/view
        response = self.get_response(request)
        
        # This runs after the view
        print("🍽️ Final plating and quality check...")
        print("📤 Dish ready to serve!")
        
        return response
```

---

## Lesson 2: Creating Custom Middleware

### The Specialized Chef Concept
Creating custom middleware is like training a specialized chef for your kitchen. Let's create a "Time Tracking Chef" who measures how long each order takes to prepare.

### Basic Custom Middleware Structure
```python
# middleware/timing_middleware.py
import time
from django.utils.deprecation import MiddlewareMixin

class TimingMiddleware:
    """
    Like a kitchen timer chef who tracks cooking times
    """
    def __init__(self, get_response):
        self.get_response = get_response
        # One-time initialization when Django starts
        print("⏱️ Timing Chef reporting for duty!")
    
    def __call__(self, request):
        # Start timing - Chef starts the timer
        start_time = time.time()
        print(f"🕐 Order started at: {time.strftime('%H:%M:%S')}")
        
        # Process request through other middleware and view
        response = self.get_response(request)
        
        # Calculate processing time - Chef checks the timer
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Add timing info to response headers
        response['X-Processing-Time'] = f"{processing_time:.4f}s"
        
        print(f"✅ Order completed in {processing_time:.4f} seconds")
        return response
```

### Advanced Custom Middleware with Process Methods
```python
# middleware/advanced_middleware.py
import logging
from django.http import HttpResponse
from django.shortcuts import redirect

class KitchenQualityMiddleware:
    """
    Like a head chef who oversees the entire kitchen operation
    """
    def __init__(self, get_response):
        self.get_response = get_response
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, request):
        return self.get_response(request)
    
    def process_request(self, request):
        """
        Check ingredients before cooking starts
        """
        # Block suspicious requests (like rejecting bad ingredients)
        if request.META.get('HTTP_USER_AGENT', '').lower().startswith('badbot'):
            self.logger.warning(f"🚫 Rejected suspicious request from {request.META.get('REMOTE_ADDR')}")
            return HttpResponse("Access Denied", status=403)
        
        # Log all incoming orders
        self.logger.info(f"📋 New order: {request.method} {request.path}")
        return None  # Continue processing
    
    def process_response(self, request, response):
        """
        Final quality check before serving
        """
        # Add custom headers (like adding garnish)
        response['X-Kitchen-Quality'] = 'Premium'
        response['X-Chef-Approved'] = 'Yes'
        
        self.logger.info(f"🍽️ Order served: {response.status_code}")
        return response
    
    def process_exception(self, request, exception):
        """
        Handle kitchen disasters gracefully
        """
        self.logger.error(f"🔥 Kitchen fire! {exception}")
        return HttpResponse("Kitchen temporarily closed", status=503)
```

### Registering Custom Middleware
```python
# settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'myapp.middleware.timing_middleware.TimingMiddleware',      # Our timing chef
    'myapp.middleware.advanced_middleware.KitchenQualityMiddleware',  # Our quality chef
    'django.contrib.sessions.middleware.SessionMiddleware',
    # ... other middleware
]
```

---

## Lesson 3: Request/Response Cycle Deep Dive

### The Complete Kitchen Journey
Let's trace a request through the entire kitchen brigade system:

```python
# middleware/request_tracker.py
import uuid
from django.utils.deprecation import MiddlewareMixin

class RequestTrackerMiddleware:
    """
    Like a kitchen manager who tracks each order through the entire process
    """
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Generate unique order ID
        order_id = str(uuid.uuid4())[:8]
        request.order_id = order_id
        
        print(f"📋 ORDER #{order_id} - Kitchen brigade process starting")
        print(f"🚪 Customer entered: {request.method} {request.path}")
        
        # Station 1: Ingredient preparation
        self.prep_station(request)
        
        # Send to next middleware/view (like passing to next chef)
        response = self.get_response(request)
        
        # Station 2: Final quality control
        self.quality_control(request, response)
        
        print(f"✅ ORDER #{order_id} - Served successfully!")
        return response
    
    def prep_station(self, request):
        """Ingredient preparation station"""
        print(f"👨‍🍳 Prep Chef: Preparing ingredients for order #{request.order_id}")
        
        # Add request metadata (like preparing ingredient list)
        request.prep_time = time.time()
        request.customer_info = {
            'ip': request.META.get('REMOTE_ADDR'),
            'user_agent': request.META.get('HTTP_USER_AGENT', 'Unknown'),
            'method': request.method
        }
    
    def quality_control(self, request, response):
        """Final quality control station"""
        print(f"🔍 Quality Chef: Final inspection for order #{request.order_id}")
        
        # Add quality stamps to response
        response['X-Order-ID'] = request.order_id
        response['X-Kitchen-Station'] = 'Quality-Approved'
        
        # Calculate total kitchen time
        if hasattr(request, 'prep_time'):
            kitchen_time = time.time() - request.prep_time
            response['X-Kitchen-Time'] = f"{kitchen_time:.4f}s"
```

### Understanding Middleware Ordering
```python
# Example showing middleware execution order
class FirstMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        print("🥗 Salad Chef initialized")
    
    def __call__(self, request):
        print("1️⃣ Salad Chef: Processing request")
        response = self.get_response(request)
        print("6️⃣ Salad Chef: Processing response")
        return response

class SecondMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        print("🍲 Soup Chef initialized")
    
    def __call__(self, request):
        print("2️⃣ Soup Chef: Processing request")
        response = self.get_response(request)
        print("5️⃣ Soup Chef: Processing response")
        return response

# View function
def my_view(request):
    print("3️⃣ Main Chef: Cooking the main dish")
    return HttpResponse("🍽️ Delicious meal ready!")
    print("4️⃣ Main Chef: Dish completed")

# Execution order:
# 1️⃣ Salad Chef: Processing request
# 2️⃣ Soup Chef: Processing request  
# 3️⃣ Main Chef: Cooking the main dish
# 4️⃣ Main Chef: Dish completed
# 5️⃣ Soup Chef: Processing response
# 6️⃣ Salad Chef: Processing response
```

---

## Lesson 4: Performance Monitoring Middleware

### The Kitchen Performance Manager
Let's create a comprehensive performance monitoring system, like having a kitchen performance manager who tracks efficiency:

```python
# middleware/performance_monitor.py
import time
import psutil
import threading
from django.http import JsonResponse
from django.conf import settings
from collections import defaultdict, deque

class PerformanceMonitorMiddleware:
    """
    Like a kitchen performance manager who tracks all kitchen metrics
    """
    def __init__(self, get_response):
        self.get_response = get_response
        
        # Kitchen performance metrics storage
        self.metrics = {
            'response_times': deque(maxlen=100),  # Last 100 orders
            'request_count': defaultdict(int),     # Orders per endpoint
            'error_count': defaultdict(int),       # Kitchen mistakes
            'peak_memory': 0,                      # Peak kitchen resource usage
            'active_requests': 0,                  # Current orders being processed
        }
        
        # Thread lock for thread-safe operations
        self.lock = threading.Lock()
        
        print("📊 Performance Manager reporting for duty!")
    
    def __call__(self, request):
        # Start performance tracking
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        with self.lock:
            self.metrics['active_requests'] += 1
        
        print(f"⚡ Performance tracking started for: {request.path}")
        
        try:
            # Process request
            response = self.get_response(request)
            
            # Track successful completion
            self.record_success_metrics(request, response, start_time, start_memory)
            
        except Exception as e:
            # Track errors
            self.record_error_metrics(request, e, start_time)
            raise
        
        finally:
            with self.lock:
                self.metrics['active_requests'] -= 1
        
        return response
    
    def record_success_metrics(self, request, response, start_time, start_memory):
        """Record successful request metrics"""
        end_time = time.time()
        response_time = end_time - start_time
        memory_used = self.get_memory_usage() - start_memory
        
        with self.lock:
            # Update metrics
            self.metrics['response_times'].append(response_time)
            self.metrics['request_count'][request.path] += 1
            self.metrics['peak_memory'] = max(self.metrics['peak_memory'], memory_used)
        
        # Add performance headers
        response['X-Response-Time'] = f"{response_time:.4f}s"
        response['X-Memory-Used'] = f"{memory_used:.2f}MB"
        
        # Log performance data
        print(f"📈 Performance: {request.path} - {response_time:.4f}s - {memory_used:.2f}MB")
    
    def record_error_metrics(self, request, exception, start_time):
        """Record error metrics"""
        response_time = time.time() - start_time
        
        with self.lock:
            self.metrics['error_count'][request.path] += 1
            self.metrics['response_times'].append(response_time)
        
        print(f"❌ Error tracked: {request.path} - {exception} - {response_time:.4f}s")
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_performance_report(self):
        """Generate performance report"""
        with self.lock:
            avg_response_time = sum(self.metrics['response_times']) / len(self.metrics['response_times']) if self.metrics['response_times'] else 0
            
            return {
                'average_response_time': f"{avg_response_time:.4f}s",
                'total_requests': sum(self.metrics['request_count'].values()),
                'active_requests': self.metrics['active_requests'],
                'peak_memory_usage': f"{self.metrics['peak_memory']:.2f}MB",
                'error_rate': sum(self.metrics['error_count'].values()) / max(sum(self.metrics['request_count'].values()), 1) * 100,
                'top_endpoints': dict(sorted(self.metrics['request_count'].items(), key=lambda x: x[1], reverse=True)[:5])
            }

# Global instance for accessing metrics
performance_monitor = PerformanceMonitorMiddleware(None)

# View to display performance dashboard
def performance_dashboard(request):
    """Kitchen performance dashboard"""
    if not settings.DEBUG:
        return JsonResponse({'error': 'Dashboard only available in debug mode'})
    
    report = performance_monitor.get_performance_report()
    return JsonResponse(report, json_dumps_params={'indent': 2})
```

---

## Final Quality Project: Custom Analytics Middleware

Now let's create a comprehensive analytics middleware that combines all concepts learned - like building a complete kitchen management system:

```python
# middleware/analytics_middleware.py
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin
from django.conf import settings
import threading

class KitchenAnalyticsMiddleware:
    """
    Complete kitchen analytics system that tracks everything
    """
    def __init__(self, get_response):
        self.get_response = get_response
        self.lock = threading.Lock()
        
        # Analytics storage (like a kitchen logbook)
        self.analytics = {
            'daily_orders': defaultdict(int),           # Orders per day
            'hourly_patterns': defaultdict(int),        # Peak hours
            'popular_dishes': defaultdict(int),         # Most requested endpoints
            'customer_satisfaction': deque(maxlen=1000), # Response times
            'kitchen_errors': defaultdict(list),        # Error tracking
            'chef_performance': {},                     # Individual middleware performance
            'customer_demographics': defaultdict(int),  # User agents, IPs
            'order_history': deque(maxlen=500),        # Recent order details
        }
        
        print("📊 Kitchen Analytics System activated!")
    
    def __call__(self, request):
        # Start order tracking
        order_start = time.time()
        order_date = datetime.now()
        
        # Generate unique order tracking ID
        order_id = f"ORDER_{int(time.time() * 1000) % 100000}"
        request.analytics_id = order_id
        
        print(f"📋 {order_id}: New order received - {request.method} {request.path}")
        
        # Pre-processing analytics
        self.track_order_start(request, order_date)
        
        try:
            # Process through kitchen brigade
            response = self.get_response(request)
            
            # Success analytics
            self.track_order_completion(request, response, order_start, order_date)
            
            return response
            
        except Exception as e:
            # Error analytics
            self.track_order_error(request, e, order_start, order_date)
            raise
    
    def track_order_start(self, request, order_date):
        """Track when order starts processing"""
        with self.lock:
            # Daily order counting
            date_str = order_date.strftime('%Y-%m-%d')
            self.analytics['daily_orders'][date_str] += 1
            
            # Peak hour analysis
            hour = order_date.hour
            self.analytics['hourly_patterns'][hour] += 1
            
            # Customer demographics
            user_agent = request.META.get('HTTP_USER_AGENT', 'Unknown')
            browser = self.extract_browser(user_agent)
            self.analytics['customer_demographics'][browser] += 1
    
    def track_order_completion(self, request, response, order_start, order_date):
        """Track successful order completion"""
        completion_time = time.time() - order_start
        
        with self.lock:
            # Track popular dishes (endpoints)
            self.analytics['popular_dishes'][request.path] += 1
            
            # Customer satisfaction (response time)
            satisfaction_score = self.calculate_satisfaction(completion_time)
            self.analytics['customer_satisfaction'].append(satisfaction_score)
            
            # Order history
            order_record = {
                'id': request.analytics_id,
                'path': request.path,
                'method': request.method,
                'completion_time': completion_time,
                'status_code': response.status_code,
                'timestamp': order_date.isoformat(),
                'satisfaction': satisfaction_score
            }
            self.analytics['order_history'].append(order_record)
        
        # Add analytics headers to response
        response['X-Analytics-ID'] = request.analytics_id
        response['X-Completion-Time'] = f"{completion_time:.4f}s"
        response['X-Satisfaction-Score'] = f"{satisfaction_score}/5"
        
        print(f"✅ {request.analytics_id}: Order completed successfully - {completion_time:.4f}s")
    
    def track_order_error(self, request, exception, order_start, order_date):
        """Track order errors and failures"""
        error_time = time.time() - order_start
        
        with self.lock:
            error_record = {
                'id': request.analytics_id,
                'path': request.path,
                'error': str(exception),
                'error_time': error_time,
                'timestamp': order_date.isoformat()
            }
            self.analytics['kitchen_errors'][request.path].append(error_record)
        
        print(f"❌ {request.analytics_id}: Order failed - {exception}")
    
    def calculate_satisfaction(self, completion_time):
        """Calculate customer satisfaction based on response time"""
        if completion_time < 0.1:
            return 5  # Excellent
        elif completion_time < 0.5:
            return 4  # Good
        elif completion_time < 1.0:
            return 3  # Average
        elif completion_time < 2.0:
            return 2  # Below Average
        else:
            return 1  # Poor
    
    def extract_browser(self, user_agent):
        """Extract browser type from user agent"""
        user_agent_lower = user_agent.lower()
        if 'chrome' in user_agent_lower:
            return 'Chrome'
        elif 'firefox' in user_agent_lower:
            return 'Firefox'
        elif 'safari' in user_agent_lower:
            return 'Safari'
        elif 'edge' in user_agent_lower:
            return 'Edge'
        else:
            return 'Other'
    
    def generate_analytics_report(self):
        """Generate comprehensive analytics report"""
        with self.lock:
            # Calculate averages and insights
            avg_satisfaction = sum(self.analytics['customer_satisfaction']) / len(self.analytics['customer_satisfaction']) if self.analytics['customer_satisfaction'] else 0
            
            total_orders = sum(self.analytics['daily_orders'].values())
            total_errors = sum(len(errors) for errors in self.analytics['kitchen_errors'].values())
            error_rate = (total_errors / total_orders * 100) if total_orders > 0 else 0
            
            # Find peak hour
            peak_hour = max(self.analytics['hourly_patterns'].items(), key=lambda x: x[1]) if self.analytics['hourly_patterns'] else (0, 0)
            
            # Generate report
            report = {
                'kitchen_performance': {
                    'total_orders': total_orders,
                    'average_satisfaction': f"{avg_satisfaction:.2f}/5",
                    'error_rate': f"{error_rate:.2f}%",
                    'peak_hour': f"{peak_hour[0]}:00 ({peak_hour[1]} orders)"
                },
                'popular_dishes': dict(sorted(self.analytics['popular_dishes'].items(), key=lambda x: x[1], reverse=True)[:10]),
                'customer_demographics': dict(self.analytics['customer_demographics']),
                'daily_orders': dict(self.analytics['daily_orders']),
                'recent_orders': list(self.analytics['order_history'])[-10:],
                'kitchen_errors': {path: len(errors) for path, errors in self.analytics['kitchen_errors'].items()}
            }
            
            return report

# Global analytics instance
kitchen_analytics = KitchenAnalyticsMiddleware(None)

# Views for analytics dashboard
def analytics_dashboard(request):
    """Kitchen analytics dashboard"""
    if not settings.DEBUG:
        return JsonResponse({'error': 'Analytics dashboard only available in debug mode'})
    
    report = kitchen_analytics.generate_analytics_report()
    return JsonResponse(report, json_dumps_params={'indent': 2})

def analytics_summary(request):
    """Quick analytics summary"""
    with kitchen_analytics.lock:
        summary = {
            'total_orders_today': kitchen_analytics.analytics['daily_orders'][datetime.now().strftime('%Y-%m-%d')],
            'current_satisfaction': f"{sum(kitchen_analytics.analytics['customer_satisfaction'][-10:]) / 10:.1f}/5" if len(kitchen_analytics.analytics['customer_satisfaction']) >= 10 else "N/A",
            'most_popular_dish': max(kitchen_analytics.analytics['popular_dishes'].items(), key=lambda x: x[1])[0] if kitchen_analytics.analytics['popular_dishes'] else "None",
            'kitchen_status': "🟢 Operating smoothly" if sum(len(errors) for errors in kitchen_analytics.analytics['kitchen_errors'].values()) < 10 else "🟡 Minor issues detected"
        }
    
    return JsonResponse(summary, json_dumps_params={'indent': 2})
```

### URLs Configuration
```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('analytics/', views.analytics_dashboard, name='analytics_dashboard'),
    path('analytics/summary/', views.analytics_summary, name='analytics_summary'),
    path('performance/', views.performance_dashboard, name='performance_dashboard'),
]
```

### Settings Configuration
```python
# settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'myapp.middleware.analytics_middleware.KitchenAnalyticsMiddleware',  # Our analytics system
    'myapp.middleware.performance_monitor.PerformanceMonitorMiddleware',  # Performance monitoring
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Enable logging for better debugging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'myapp.middleware': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}
```

---

## Code Syntax Explanations

### 1. Middleware Class Structure
```python
class MyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        # Initialization code here
    
    def __call__(self, request):
        # Code before view execution
        response = self.get_response(request)
        # Code after view execution
        return response
```

**Explanation:**
- `__init__`: Called once when Django starts - like hiring a chef
- `get_response`: Function that calls the next middleware/view in the chain
- `__call__`: Makes the class callable - executed for each request
- `self.get_response(request)`: Passes control to the next middleware/view

### 2. Thread Safety with Locks
```python
import threading
self.lock = threading.Lock()

with self.lock:
    # Thread-safe operations here
    self.shared_data += 1
```

**Explanation:**
- `threading.Lock()`: Creates a lock object for thread synchronization
- `with self.lock:`: Context manager that automatically acquires and releases the lock
- Prevents race conditions when multiple requests access shared data simultaneously

### 3. Collections for Data Storage
```python
from collections import defaultdict, deque

# Auto-creates missing keys with default values
self.counters = defaultdict(int)  # Default to 0
self.lists = defaultdict(list)    # Default to empty list

# Fixed-size queue that automatically removes old items
self.recent_items = deque(maxlen=100)  # Keeps only last 100 items
```

**Explanation:**
- `defaultdict(int)`: Dictionary that creates missing keys with default value 0
- `deque(maxlen=100)`: Double-ended queue with maximum size, automatically removes old items
- These are memory-efficient for tracking metrics without manual cleanup

### 4. Exception Handling in Middleware
```python
def __call__(self, request):
    try:
        response = self.get_response(request)
        return response
    except Exception as e:
        # Handle errors gracefully
        self.log_error(request, e)
        raise  # Re-raise to let Django handle it
```

**Explanation:**
- `try/except`: Catches exceptions that occur in subsequent middleware/views
- `raise`: Re-raises the exception after logging, allowing Django's error handling to continue
- This pattern allows middleware to observe errors without interrupting the error handling flow

---

## Assignment: Kitchen Health Monitor

Create a middleware called `KitchenHealthMonitor` that acts like a health inspector for your Django kitchen. Your middleware should:

1. **Track kitchen health metrics:**
   - Response times (healthy: <0.5s, warning: 0.5-2s, critical: >2s)
   - Error rates (healthy: <1%, warning: 1-5%, critical: >5%)
   - Memory usage patterns
   - Request volume per minute

2. **Implement health status levels:**
   - 🟢 **Healthy**: All metrics normal
   - 🟡 **Warning**: Some metrics elevated
   - 🔴 **Critical**: Immediate attention needed

3. **Create health endpoints:**
   - `/health/status/` - Current health status
   - `/health/detailed/` - Detailed health metrics
   - `/health/alerts/` - Recent health alerts

4. **Add automatic alerts:**
   - Log warning messages when thresholds are exceeded
   - Add health status headers to all responses
   - Generate health reports every 100 requests

**Bonus challenges:**
- Implement health history tracking
- Add a simple HTML health dashboard
- Create health trend analysis (improving/declining)

Your middleware should be production-ready, thread-safe, and include proper error handling. Submit your complete middleware class with example usage and test cases.

---

## Summary

You've learned to build Django's middleware system like managing a professional kitchen brigade. Key concepts mastered:

1. **Middleware Architecture**: Understanding the request/response cycle flow
2. **Custom Middleware Creation**: Building specialized middleware components
3. **Request/Response Processing**: Implementing comprehensive request tracking
4. **Performance Monitoring**: Creating analytics and monitoring systems

Your final analytics middleware demonstrates real-world middleware capabilities, combining performance monitoring, error tracking, and comprehensive analytics - like having a complete kitchen management system that tracks every aspect of your restaurant operation.

Remember: middleware is powerful but should be used judiciously. Each middleware adds processing time to every request, so keep them efficient and focused on their specific purpose.