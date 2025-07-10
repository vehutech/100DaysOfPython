# Django Sessions & Cookies: The Kitchen Memory System

## Learning Objective
By the end of this lesson, you will understand how Django manages user sessions and cookies, implement custom session backends, handle cookie security, and build a secure session-based system that remembers user interactions across multiple requests.

---

## Imagine That...

Imagine that you're running a busy restaurant kitchen where dozens of customers come and go throughout the day. Each customer has specific preferences, dietary restrictions, and order history that you need to remember. But here's the challenge: you can't physically follow each customer around to remember their details.

Instead, you give each customer a special numbered ticket when they first arrive. This ticket corresponds to a detailed card in your kitchen filing system where you write down everything about that customer - their favorite dishes, allergies, previous orders, and current meal preferences. Every time they return to your restaurant, they show their ticket, and you instantly know everything about them by looking up their card.

This is exactly how Django sessions work! The "ticket" is a session cookie stored in the user's browser, and the "filing system" is Django's session backend that stores all the user's information on the server.

---

## Lesson 1: Django Sessions Framework

### The Kitchen Filing System

In our kitchen analogy, Django's session framework is like having a sophisticated filing system that automatically manages customer information cards.

```python
# views.py - The Head Chef's Orders
from django.shortcuts import render
from django.http import HttpResponse

def customer_preferences(request):
    """
    Like a chef remembering a customer's preferences
    """
    # Check if this customer has visited before
    if 'favorite_dish' in request.session:
        # We remember this customer!
        favorite = request.session['favorite_dish']
        visit_count = request.session.get('visits', 0) + 1
        request.session['visits'] = visit_count
        
        message = f"Welcome back! Your favorite dish is {favorite}. This is visit #{visit_count}"
    else:
        # New customer - let's start their preference card
        request.session['favorite_dish'] = 'Not set yet'
        request.session['visits'] = 1
        message = "Welcome to our restaurant! Let's start building your preference profile."
    
    return HttpResponse(message)

def set_favorite_dish(request):
    """
    Like a chef updating a customer's preference card
    """
    if request.method == 'POST':
        dish = request.POST.get('dish')
        # Update the customer's preference card
        request.session['favorite_dish'] = dish
        request.session['dietary_restrictions'] = request.POST.get('restrictions', '')
        
        # Mark the session as modified (like updating the filing system)
        request.session.modified = True
        
        return HttpResponse(f"Got it! Your favorite dish is now {dish}")
    
    return render(request, 'set_preferences.html')
```

```python
# settings.py - Kitchen Rules and Regulations
INSTALLED_APPS = [
    'django.contrib.sessions',  # The filing system manager
    # ... other apps
]

MIDDLEWARE = [
    'django.contrib.sessions.middleware.SessionMiddleware',  # The ticket checker
    # ... other middleware
]

# Session configuration - like setting up your filing system rules
SESSION_ENGINE = 'django.contrib.sessions.backends.db'  # Store cards in database
SESSION_COOKIE_AGE = 1209600  # Tickets expire after 2 weeks
SESSION_EXPIRE_AT_BROWSER_CLOSE = False  # Keep tickets even after customer leaves
SESSION_SAVE_EVERY_REQUEST = True  # Update timestamp on every visit
```

**Syntax Explanation:**
- `request.session` acts like a dictionary where you can store user-specific data
- `request.session.get('key', default)` safely retrieves values with a fallback
- `request.session.modified = True` tells Django to save changes to the session
- Session middleware automatically handles the cookie creation and retrieval

---

## Lesson 2: Custom Session Backends

### Specialized Filing Systems

Sometimes our standard kitchen filing system isn't enough. Maybe we need faster access (Redis), or we want to store customer cards in a special secure vault (custom database), or we need to handle massive volumes (distributed storage).

```python
# custom_session_backend.py - Your Special Filing System
import json
import redis
from django.contrib.sessions.backends.base import SessionBase

class RedisSessionStore(SessionBase):
    """
    Like having a super-fast electronic filing system
    """
    def __init__(self, session_key=None):
        super().__init__(session_key)
        # Connect to our speed-of-light filing system
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
    
    def load(self):
        """
        Like quickly finding a customer's card
        """
        try:
            # Look up the customer's card by ticket number
            session_data = self.redis_client.get(f"session:{self.session_key}")
            if session_data:
                return json.loads(session_data)
        except Exception as e:
            # If we can't find the card, start fresh
            print(f"Error loading session: {e}")
        return {}
    
    def save(self, must_create=False):
        """
        Like filing away a customer's updated card
        """
        try:
            # Convert the customer data to a format we can store
            session_data = json.dumps(self._get_session(no_load=must_create))
            
            # File it away with an expiration date
            self.redis_client.setex(
                f"session:{self.session_key}",
                self.get_expiry_age(),
                session_data
            )
        except Exception as e:
            print(f"Error saving session: {e}")
    
    def delete(self, session_key=None):
        """
        Like throwing away a customer's card when they ask us to forget them
        """
        if session_key is None:
            session_key = self.session_key
        
        try:
            self.redis_client.delete(f"session:{session_key}")
        except Exception as e:
            print(f"Error deleting session: {e}")
    
    def exists(self, session_key):
        """
        Like checking if we have a card for this customer
        """
        return self.redis_client.exists(f"session:{session_key}")
```

```python
# settings.py - Telling Django about our custom filing system
SESSION_ENGINE = 'myapp.custom_session_backend'

# Or for file-based sessions (like physical filing cabinets)
SESSION_ENGINE = 'django.contrib.sessions.backends.file'
SESSION_FILE_PATH = '/tmp/django_sessions'  # Where to store the files

# Or for cache-based sessions (like a quick-access memory system)
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'  # Which cache system to use
```

**Syntax Explanation:**
- Custom session backends inherit from `SessionBase` and implement key methods
- `load()` retrieves session data, `save()` stores it, `delete()` removes it
- `json.dumps()` and `json.loads()` convert Python objects to/from JSON strings
- Redis `setex()` sets a value with an expiration time

---

## Lesson 3: Cookie Handling

### The Customer Ticket System

Cookies are like the tickets we give customers. They're small pieces of information that the customer's browser holds onto and shows us every time they visit.

```python
# views.py - Managing Customer Tickets
from django.http import HttpResponse
from django.shortcuts import render
import datetime

def issue_customer_ticket(request):
    """
    Like giving a customer their first ticket with basic info
    """
    response = HttpResponse("Welcome! Here's your customer ticket.")
    
    # Issue a basic ticket (cookie)
    response.set_cookie(
        'customer_id',
        'CUST_12345',
        max_age=3600,  # Ticket expires in 1 hour
        secure=True,   # Only send over HTTPS (secure kitchen)
        httponly=True, # Can't be accessed by JavaScript (security measure)
        samesite='Strict'  # Only send to our restaurant
    )
    
    # Issue a preference ticket
    response.set_cookie(
        'preferred_cuisine',
        'Italian',
        max_age=7*24*3600,  # Valid for a week
        path='/menu/'  # Only valid in the menu section
    )
    
    return response

def read_customer_ticket(request):
    """
    Like reading what's written on a customer's ticket
    """
    # Check what tickets the customer is carrying
    customer_id = request.COOKIES.get('customer_id', 'Unknown Customer')
    preferred_cuisine = request.COOKIES.get('preferred_cuisine', 'No preference')
    
    # Special handling for signed cookies (tamper-proof tickets)
    loyalty_level = request.get_signed_cookie(
        'loyalty_level',
        default='Bronze',
        salt='restaurant-loyalty',  # Secret ingredient for security
        max_age=30*24*3600  # Must be used within 30 days
    )
    
    ticket_info = f"""
    Customer ID: {customer_id}
    Preferred Cuisine: {preferred_cuisine}
    Loyalty Level: {loyalty_level}
    """
    
    return HttpResponse(ticket_info)

def destroy_customer_ticket(request):
    """
    Like asking a customer to throw away their old ticket
    """
    response = HttpResponse("Your tickets have been destroyed. Come back anytime!")
    
    # Destroy specific tickets
    response.delete_cookie('customer_id')
    response.delete_cookie('preferred_cuisine')
    
    # For secure cookies, we need to match the original settings
    response.delete_cookie('loyalty_level', path='/menu/', domain='restaurant.com')
    
    return response
```

```python
# Advanced cookie handling - Like VIP ticket management
def vip_customer_system(request):
    """
    Managing premium customer tickets with extra security
    """
    if request.method == 'POST':
        # Create a VIP ticket with extra security
        response = HttpResponse("VIP status activated!")
        
        # Encrypted ticket that can't be forged
        response.set_signed_cookie(
            'vip_status',
            'PLATINUM',
            salt='super-secret-kitchen-salt',
            max_age=365*24*3600,  # Valid for a year
            secure=True,
            httponly=True,
            samesite='Strict'
        )
        
        # Store complex data in the ticket
        vip_data = {
            'member_since': '2024-01-15',
            'points': 15000,
            'special_requests': ['gluten-free', 'extra-spicy']
        }
        
        response.set_signed_cookie(
            'vip_data',
            json.dumps(vip_data),
            salt='vip-data-salt',
            max_age=365*24*3600
        )
        
        return response
    
    return render(request, 'vip_signup.html')
```

**Syntax Explanation:**
- `response.set_cookie()` creates cookies with various security options
- `max_age` sets expiration time in seconds
- `secure=True` ensures cookies only travel over HTTPS
- `httponly=True` prevents JavaScript access, improving security
- `samesite` controls when cookies are sent with cross-site requests
- `request.get_signed_cookie()` retrieves tamper-proof cookies

---

## Lesson 4: Session Security

### Protecting Your Kitchen's Filing System

Just like you wouldn't want unauthorized people accessing your customer files, session security is crucial for protecting user data.

```python
# security_views.py - The Kitchen Security System
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect
from django.http import HttpResponse
from django.shortcuts import redirect
import secrets
import hashlib

@csrf_protect
def secure_customer_area(request):
    """
    Like having a bouncer check IDs before entering the VIP area
    """
    # Check if the customer has a valid session
    if not request.session.get('authenticated', False):
        return redirect('login')
    
    # Regenerate session ID for extra security (like issuing a new ticket)
    if request.session.get('just_logged_in', False):
        request.session.cycle_key()  # New ticket number
        del request.session['just_logged_in']
    
    # Add security headers (like security cameras)
    response = HttpResponse("Welcome to the secure area!")
    response['X-Frame-Options'] = 'DENY'
    response['X-Content-Type-Options'] = 'nosniff'
    
    return response

def login_customer(request):
    """
    Like verifying a customer's identity before giving them access
    """
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # In a real system, you'd verify credentials against the database
        if verify_credentials(username, password):
            # Create a secure session
            request.session['authenticated'] = True
            request.session['username'] = username
            request.session['just_logged_in'] = True
            
            # Add security metadata
            request.session['login_time'] = datetime.datetime.now().isoformat()
            request.session['ip_address'] = get_client_ip(request)
            
            # Set session expiry based on user choice
            if request.POST.get('remember_me'):
                request.session.set_expiry(30*24*3600)  # 30 days
            else:
                request.session.set_expiry(0)  # Expire when browser closes
            
            return redirect('secure_area')
    
    return render(request, 'login.html')

def session_security_check(request):
    """
    Like having a security guard check if customer tickets are still valid
    """
    # Check for suspicious activity
    current_ip = get_client_ip(request)
    stored_ip = request.session.get('ip_address')
    
    if stored_ip and stored_ip != current_ip:
        # IP address changed - potential security risk
        request.session.flush()  # Destroy everything and start fresh
        return redirect('login')
    
    # Check session age
    login_time = request.session.get('login_time')
    if login_time:
        # Convert string back to datetime
        login_datetime = datetime.datetime.fromisoformat(login_time)
        time_since_login = datetime.datetime.now() - login_datetime
        
        if time_since_login.total_seconds() > 8*3600:  # 8 hours
            request.session.flush()
            return redirect('login')
    
    return HttpResponse("Security check passed!")

def get_client_ip(request):
    """
    Like checking which entrance a customer used
    """
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
```

```python
# settings.py - Kitchen Security Policies
# Session Security Settings
SESSION_COOKIE_SECURE = True  # Only send cookies over HTTPS
SESSION_COOKIE_HTTPONLY = True  # Prevent JavaScript access
SESSION_COOKIE_SAMESITE = 'Strict'  # Prevent CSRF attacks
SESSION_COOKIE_NAME = 'restaurantsessionid'  # Custom name for obscurity

# CSRF Protection (like having security tokens)
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SAMESITE = 'Strict'

# Additional security headers
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# Session timeout settings
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
SESSION_COOKIE_AGE = 3600  # 1 hour default
```

**Syntax Explanation:**
- `request.session.cycle_key()` generates a new session ID while preserving data
- `request.session.set_expiry()` controls when sessions expire
- `request.session.flush()` completely destroys the session
- Security headers protect against common web vulnerabilities
- IP address checking helps detect session hijacking attempts

---

## Final Quality Project: Restaurant Customer Preference System

Let's build a complete system that demonstrates all the concepts we've learned:

```python
# restaurant_system/views.py - Complete Customer Management System
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_protect
from django.contrib import messages
import json
import datetime

class RestaurantCustomerSystem:
    """
    Like having a master chef who remembers every customer perfectly
    """
    
    @staticmethod
    def welcome_customer(request):
        """
        The front door greeting - establishing customer identity
        """
        # Check if customer is returning
        if request.session.get('customer_profile'):
            profile = request.session['customer_profile']
            visit_count = request.session.get('visit_count', 0) + 1
            request.session['visit_count'] = visit_count
            
            context = {
                'returning_customer': True,
                'name': profile.get('name', 'Valued Customer'),
                'favorite_dish': profile.get('favorite_dish'),
                'visit_count': visit_count,
                'last_visit': request.session.get('last_visit')
            }
        else:
            # New customer
            context = {'returning_customer': False}
        
        # Update last visit timestamp
        request.session['last_visit'] = datetime.datetime.now().isoformat()
        
        return render(request, 'restaurant/welcome.html', context)
    
    @staticmethod
    @csrf_protect
    def create_customer_profile(request):
        """
        Like taking down a customer's preferences for the first time
        """
        if request.method == 'POST':
            # Collect customer preferences
            profile = {
                'name': request.POST.get('name'),
                'favorite_dish': request.POST.get('favorite_dish'),
                'dietary_restrictions': request.POST.getlist('dietary_restrictions'),
                'spice_level': request.POST.get('spice_level'),
                'created_date': datetime.datetime.now().isoformat()
            }
            
            # Store in session
            request.session['customer_profile'] = profile
            request.session['visit_count'] = 1
            
            # Set a welcome cookie
            response = redirect('menu')
            response.set_signed_cookie(
                'welcome_status',
                'profile_created',
                salt='restaurant-welcome',
                max_age=24*3600  # 24 hours
            )
            
            messages.success(request, f"Welcome {profile['name']}! Your preferences have been saved.")
            return response
        
        return render(request, 'restaurant/create_profile.html')
    
    @staticmethod
    def personalized_menu(request):
        """
        Like showing a customer dishes based on their preferences
        """
        profile = request.session.get('customer_profile', {})
        
        # Mock menu data (in real app, this would come from database)
        all_dishes = [
            {'name': 'Spicy Chicken Curry', 'spice_level': 'hot', 'dietary': ['gluten-free']},
            {'name': 'Margherita Pizza', 'spice_level': 'mild', 'dietary': ['vegetarian']},
            {'name': 'Beef Steak', 'spice_level': 'mild', 'dietary': []},
            {'name': 'Vegan Buddha Bowl', 'spice_level': 'medium', 'dietary': ['vegan', 'gluten-free']},
        ]
        
        # Filter based on preferences
        dietary_restrictions = profile.get('dietary_restrictions', [])
        preferred_spice = profile.get('spice_level', 'mild')
        
        recommended_dishes = []
        for dish in all_dishes:
            # Check dietary compatibility
            if any(restriction in dish['dietary'] for restriction in dietary_restrictions):
                recommended_dishes.append(dish)
            elif not dietary_restrictions:  # No restrictions
                recommended_dishes.append(dish)
        
        context = {
            'profile': profile,
            'recommended_dishes': recommended_dishes,
            'all_dishes': all_dishes
        }
        
        return render(request, 'restaurant/menu.html', context)
    
    @staticmethod
    def order_tracking(request):
        """
        Like keeping track of what a customer has ordered
        """
        if request.method == 'POST':
            # Add item to order
            dish_name = request.POST.get('dish_name')
            current_order = request.session.get('current_order', [])
            
            order_item = {
                'dish': dish_name,
                'time_ordered': datetime.datetime.now().isoformat(),
                'special_requests': request.POST.get('special_requests', '')
            }
            
            current_order.append(order_item)
            request.session['current_order'] = current_order
            request.session.modified = True
            
            return JsonResponse({'status': 'added', 'order_count': len(current_order)})
        
        # Display current order
        current_order = request.session.get('current_order', [])
        context = {'current_order': current_order}
        
        return render(request, 'restaurant/order_tracking.html', context)
    
    @staticmethod
    def customer_logout(request):
        """
        Like saying goodbye and clearing the table
        """
        customer_name = request.session.get('customer_profile', {}).get('name', 'Customer')
        
        # Clear session but preserve some data for next visit
        visit_count = request.session.get('visit_count', 0)
        profile = request.session.get('customer_profile', {})
        
        request.session.flush()  # Clear everything
        
        # Set a goodbye cookie
        response = render(request, 'restaurant/goodbye.html', {
            'customer_name': customer_name,
            'visit_count': visit_count
        })
        
        response.set_cookie(
            'last_visit_summary',
            json.dumps({
                'name': customer_name,
                'visit_count': visit_count,
                'date': datetime.datetime.now().isoformat()
            }),
            max_age=30*24*3600  # Remember for 30 days
        )
        
        return response

# URL routing
# restaurant_system/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.RestaurantCustomerSystem.welcome_customer, name='welcome'),
    path('create-profile/', views.RestaurantCustomerSystem.create_customer_profile, name='create_profile'),
    path('menu/', views.RestaurantCustomerSystem.personalized_menu, name='menu'),
    path('order/', views.RestaurantCustomerSystem.order_tracking, name='order'),
    path('logout/', views.RestaurantCustomerSystem.customer_logout, name='logout'),
]
```

```html
<!-- restaurant/templates/restaurant/welcome.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Welcome to Our Restaurant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .welcome-box { background: #f5f5f5; padding: 20px; border-radius: 8px; }
        .new-customer { background: #e8f5e9; }
        .returning-customer { background: #fff3e0; }
    </style>
</head>
<body>
    {% if returning_customer %}
        <div class="welcome-box returning-customer">
            <h1>Welcome back, {{ name }}! 🍽️</h1>
            <p>This is your visit #{{ visit_count }}</p>
            {% if favorite_dish %}
                <p>Your favorite dish: <strong>{{ favorite_dish }}</strong></p>
            {% endif %}
            <p>Last visit: {{ last_visit|date:"F j, Y g:i A" }}</p>
            <a href="{% url 'menu' %}">View Personalized Menu</a>
        </div>
    {% else %}
        <div class="welcome-box new-customer">
            <h1>Welcome to Our Restaurant! 👋</h1>
            <p>We'd love to learn about your preferences to serve you better.</p>
            <a href="{% url 'create_profile' %}">Create Your Profile</a>
        </div>
    {% endif %}
</body>
</html>
```

**Project Syntax Explanation:**
- `@staticmethod` creates methods that can be called without instantiating the class
- `request.POST.getlist()` retrieves multiple values from checkboxes
- `request.session.modified = True` ensures session changes are saved
- `JsonResponse()` returns JSON data for AJAX requests
- Template tags like `{% if %}` and `{{ variable }}` display dynamic content

---

# Django Shopping Cart Project - Sessions & Cookies

## Learning Objective
By the end of this project, you will be able to build a fully functional shopping cart system using Django's session framework, understanding how to store temporary data, manage user interactions, and create persistent cart functionality without requiring user authentication.

---

## Introduction: The Kitchen Memory System

Imagine that you're running a busy restaurant kitchen, and customers keep coming to your counter to add items to their order throughout the evening. As a chef, you need a system to remember what each customer has ordered so far, even when they step away and come back later. You can't rely on the customers to remember everything they've ordered - you need your own "memory system."

In web development, this is exactly what sessions and cookies do for us. Just like how a chef might use order tickets to track what each customer wants, Django uses sessions to remember information about each user's visit to your website. When we build a shopping cart, we're essentially creating a digital order ticket that follows the user around as they browse your online store.

Think of it this way:
- **Sessions** are like the chef's order notepad - they store temporary information about what the customer wants
- **Cookies** are like the table numbers - they help identify which order belongs to which customer
- **Shopping Cart** is like the actual order ticket - it holds all the items the customer has selected

---

##Project: Building a Shopping Cart System

Let's create a complete shopping cart system that works just like our kitchen analogy. We'll build this step by step, starting with the foundation and working our way up to a fully functional cart.

### Step 1: Setting Up Our Product Model

First, let's create a simple product model - these are like the items on our restaurant menu:

```python
# models.py
from django.db import models
from decimal import Decimal

class Product(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    image = models.ImageField(upload_to='products/', blank=True)
    stock = models.PositiveIntegerField(default=0)
    available = models.BooleanField(default=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
```

**Syntax Explanation:**
- `models.CharField(max_length=200)`: Creates a text field with maximum 200 characters
- `models.DecimalField(max_digits=10, decimal_places=2)`: Perfect for money values, stores up to 10 digits with 2 decimal places
- `models.PositiveIntegerField(default=0)`: Only allows positive numbers, defaults to 0
- `auto_now_add=True`: Automatically sets the timestamp when the record is created
- `auto_now=True`: Updates the timestamp every time the record is saved

### Step 2: Creating Our Shopping Cart Class

Now let's create our shopping cart - this is like the chef's order management system:

```python
# cart.py
from decimal import Decimal
from django.conf import settings
from .models import Product

class Cart:
    def __init__(self, request):
        """
        Initialize the cart - like starting a new order ticket
        """
        self.session = request.session
        cart = self.session.get(settings.CART_SESSION_ID)
        if not cart:
            # Create an empty cart - like getting a fresh order ticket
            cart = self.session[settings.CART_SESSION_ID] = {}
        self.cart = cart

    def add(self, product, quantity=1, override_quantity=False):
        """
        Add a product to the cart - like adding an item to the order
        """
        product_id = str(product.id)
        if product_id not in self.cart:
            self.cart[product_id] = {
                'quantity': 0,
                'price': str(product.price)
            }
        
        if override_quantity:
            self.cart[product_id]['quantity'] = quantity
        else:
            self.cart[product_id]['quantity'] += quantity
        
        self.save()

    def save(self):
        """
        Mark the session as modified - like updating the order ticket
        """
        self.session.modified = True

    def remove(self, product):
        """
        Remove a product from the cart - like crossing out an item
        """
        product_id = str(product.id)
        if product_id in self.cart:
            del self.cart[product_id]
            self.save()

    def __iter__(self):
        """
        Iterate over items in the cart - like reading through the order
        """
        product_ids = self.cart.keys()
        products = Product.objects.filter(id__in=product_ids)
        
        cart = self.cart.copy()
        for product in products:
            cart[str(product.id)]['product'] = product
        
        for item in cart.values():
            item['price'] = Decimal(item['price'])
            item['total_price'] = item['price'] * item['quantity']
            yield item

    def __len__(self):
        """
        Count all items in the cart - like counting total items on the order
        """
        return sum(item['quantity'] for item in self.cart.values())

    def get_total_price(self):
        """
        Calculate the total cost - like adding up the final bill
        """
        return sum(Decimal(item['price']) * item['quantity'] 
                  for item in self.cart.values())

    def clear(self):
        """
        Clear the cart - like starting with a fresh order ticket
        """
        del self.session[settings.CART_SESSION_ID]
        self.save()
```

**Syntax Explanation:**
- `__init__(self, request)`: Constructor method that runs when creating a new Cart instance
- `self.session = request.session`: Stores Django's session object for this user
- `str(product.id)`: Converts product ID to string (JSON keys must be strings)
- `__iter__(self)`: Makes the cart iterable (allows for loops)
- `yield item`: Returns items one by one (generator function)
- `__len__(self)`: Defines what happens when you call `len(cart)`
- `sum(item['quantity'] for item in self.cart.values())`: Generator expression that adds up all quantities

### Step 3: Adding Settings Configuration

Add this to your settings.py file:

```python
# settings.py
CART_SESSION_ID = 'cart'
```

This is like giving our order ticket system a name so Django knows where to store the cart data.

### Step 4: Creating Views for Cart Operations

Now let's create the views that handle cart operations - these are like the different actions a chef can take with orders:

```python
# views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST
from django.contrib import messages
from .models import Product
from .cart import Cart

def product_list(request):
    """
    Display all products - like showing the menu
    """
    products = Product.objects.filter(available=True)
    return render(request, 'cart/product_list.html', {'products': products})

def product_detail(request, id):
    """
    Show individual product details - like describing a menu item
    """
    product = get_object_or_404(Product, id=id, available=True)
    return render(request, 'cart/product_detail.html', {'product': product})

@require_POST
def cart_add(request, product_id):
    """
    Add product to cart - like adding an item to the order
    """
    cart = Cart(request)
    product = get_object_or_404(Product, id=product_id)
    quantity = int(request.POST.get('quantity', 1))
    
    cart.add(product=product, quantity=quantity)
    messages.success(request, f'{product.name} added to your cart!')
    return redirect('cart:cart_detail')

@require_POST
def cart_remove(request, product_id):
    """
    Remove product from cart - like removing an item from the order
    """
    cart = Cart(request)
    product = get_object_or_404(Product, id=product_id)
    cart.remove(product)
    messages.success(request, f'{product.name} removed from your cart!')
    return redirect('cart:cart_detail')

def cart_detail(request):
    """
    Show cart contents - like reviewing the current order
    """
    cart = Cart(request)
    return render(request, 'cart/cart_detail.html', {'cart': cart})
```

**Syntax Explanation:**
- `@require_POST`: Decorator that ensures the view only accepts POST requests
- `get_object_or_404(Product, id=id)`: Gets the product or returns 404 error if not found
- `int(request.POST.get('quantity', 1))`: Gets quantity from POST data, defaults to 1
- `messages.success(request, '...')`: Adds a success message to display to the user
- `redirect('cart:cart_detail')`: Redirects to another URL pattern

### Step 5: Creating Templates

Let's create the HTML templates - these are like the visual menus and order displays:

**Product List Template:**
```html
<!-- templates/cart/product_list.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Our Menu</title>
    <style>
        .product-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; padding: 20px; }
        .product-card { border: 1px solid #ddd; padding: 15px; border-radius: 8px; }
        .product-card img { width: 100%; height: 200px; object-fit: cover; }
        .price { font-size: 1.2em; font-weight: bold; color: #2c5aa0; }
        .btn { padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <h1>Welcome to Our Store</h1>
    
    <div class="product-grid">
        {% for product in products %}
        <div class="product-card">
            {% if product.image %}
                <img src="{{ product.image.url }}" alt="{{ product.name }}">
            {% endif %}
            <h3>{{ product.name }}</h3>
            <p>{{ product.description }}</p>
            <p class="price">${{ product.price }}</p>
            
            <form action="{% url 'cart:cart_add' product.id %}" method="post">
                {% csrf_token %}
                <input type="number" name="quantity" value="1" min="1" max="{{ product.stock }}">
                <button type="submit" class="btn">Add to Cart</button>
            </form>
        </div>
        {% endfor %}
    </div>
    
    <p><a href="{% url 'cart:cart_detail' %}">View Cart</a></p>
</body>
</html>
```

**Cart Detail Template:**
```html
<!-- templates/cart/cart_detail.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Your Cart</title>
    <style>
        .cart-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .cart-table th, .cart-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .cart-table th { background-color: #f8f9fa; }
        .total { font-size: 1.3em; font-weight: bold; text-align: right; margin: 20px 0; }
        .btn { padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
        .btn-danger { background: #dc3545; }
        .btn:hover { opacity: 0.8; }
        .empty-cart { text-align: center; padding: 40px; color: #666; }
    </style>
</head>
<body>
    <h1>Your Shopping Cart</h1>
    
    {% if cart %}
        <table class="cart-table">
            <thead>
                <tr>
                    <th>Product</th>
                    <th>Quantity</th>
                    <th>Price</th>
                    <th>Total</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {% for item in cart %}
                <tr>
                    <td>{{ item.product.name }}</td>
                    <td>{{ item.quantity }}</td>
                    <td>${{ item.price }}</td>
                    <td>${{ item.total_price }}</td>
                    <td>
                        <form action="{% url 'cart:cart_remove' item.product.id %}" method="post" style="display: inline;">
                            {% csrf_token %}
                            <button type="submit" class="btn btn-danger">Remove</button>
                        </form>
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <div class="total">
            Total: ${{ cart.get_total_price }}
        </div>
        
        <p>
            <a href="{% url 'cart:product_list' %}" class="btn">Continue Shopping</a>
            <a href="#" class="btn">Checkout</a>
        </p>
    {% else %}
        <div class="empty-cart">
            <h2>Your cart is empty</h2>
            <p>Looks like you haven't added anything to your cart yet.</p>
            <a href="{% url 'cart:product_list' %}" class="btn">Start Shopping</a>
        </div>
    {% endif %}
</body>
</html>
```

**Syntax Explanation:**
- `{% for product in products %}`: Django template loop
- `{% if product.image %}`: Conditional display of images
- `{{ product.name }}`: Variable output in templates
- `{% csrf_token %}`: Security token required for POST forms
- `{% url 'cart:cart_add' product.id %}`: Generates URL with parameter

### Step 6: URL Configuration

Set up the URLs to connect everything together:

```python
# urls.py (in your cart app)
from django.urls import path
from . import views

app_name = 'cart'

urlpatterns = [
    path('', views.cart_detail, name='cart_detail'),
    path('add/<int:product_id>/', views.cart_add, name='cart_add'),
    path('remove/<int:product_id>/', views.cart_remove, name='cart_remove'),
    path('products/', views.product_list, name='product_list'),
    path('products/<int:id>/', views.product_detail, name='product_detail'),
]
```

**Syntax Explanation:**
- `app_name = 'cart'`: Namespaces the URLs so we can reference them as 'cart:cart_detail'
- `<int:product_id>`: URL parameter that captures an integer and passes it to the view
- `name='cart_add'`: Gives the URL pattern a name for reverse lookup

---

You now have a complete shopping cart system that works like a professional restaurant kitchen's order management system! Here's what you've built:

### Features Implemented:
1. **Product Display** - Like showing the menu to customers
2. **Add to Cart** - Like taking orders from customers
3. **View Cart** - Like reviewing the current order
4. **Remove Items** - Like canceling items from an order
5. **Quantity Management** - Like adjusting portion sizes
6. **Total Calculation** - Like calculating the final bill
7. **Session Persistence** - Like keeping track of orders even when customers step away

### How It Works:
Just like in our kitchen analogy, when a customer (user) visits your website, Django creates a session (like starting a new order ticket). As they browse products and add items to their cart, the information is stored in the session (like writing items on the order ticket). The cart persists as they navigate around your site, and they can always return to see their cart contents (like checking their order ticket).

### Testing Your Cart:
1. Run your Django server: `python manage.py runserver`
2. Create some products in the Django admin
3. Visit the product list page
4. Add items to your cart
5. View your cart to see the items
6. Try removing items
7. Notice how the cart remembers your items even when you navigate away and come back

This shopping cart system is production-ready and handles all the essential functionality you'd expect from an e-commerce site. It's like having a professional kitchen order management system that never forgets what each customer wants!

---

## Key Concepts Mastered

Through building this shopping cart system, you've learned:

- **Session Management**: How Django stores temporary data for each user
- **Object-Oriented Design**: Creating reusable Cart class with methods
- **Template Integration**: Displaying dynamic data in HTML templates
- **Form Handling**: Processing POST requests safely
- **URL Routing**: Connecting views to URLs with parameters
- **Message Framework**: Providing user feedback
- **Security**: Using CSRF tokens and proper form validation

Just like a skilled chef who can manage multiple orders simultaneously, you can now build web applications that remember user interactions and provide seamless shopping experiences!

## Assignment: Build a Library Reading Session Tracker

**Your Task:** Create a system that tracks what books a user is reading, their reading progress, and preferences - just like our restaurant system but for a library.

**Requirements:**
1. **Session Management**: Track user's current reading list, reading progress, and favorite genres
2. **Cookie Usage**: Remember user's preferred reading settings (font size, theme) using secure cookies
3. **Security**: Implement session timeout and basic security checks
4. **Personalization**: Show book recommendations based on reading history stored in sessions

**Expected Features:**
- Welcome page that recognizes returning readers
- Book selection with progress tracking (pages read, current chapter)
- Reading preferences (stored in cookies)
- Session-based reading history
- Secure logout that preserves some data for next visit

**Deliverables:**
- Django views with session and cookie handling
- Templates for user interaction
- Settings configuration for session security
- At least 3 different types of data stored in sessions
- At least 2 secure cookies for user preferences

This assignment will test your understanding of all four concepts: Django sessions framework, cookie handling, session security, and practical application of these concepts in a real-world scenario.

---

## Summary

Today you've learned how Django sessions and cookies work together like a restaurant's customer management system. You can now:

- Use Django's session framework to store user-specific data
- Create custom session backends for specialized storage needs
- Handle cookies securely with proper security settings
- Implement session security measures to protect user data
- Build complete session-based applications

Remember: sessions are your server-side filing system, cookies are the tickets that connect users to their data, and security is the bouncer that keeps everything safe!