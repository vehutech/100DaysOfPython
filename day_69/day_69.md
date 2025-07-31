# Day 69: Django Mastery Capstone Course

## Learning Objective
By the end of this course, you will be able to analyze, refactor, document, and optimize Django applications like a master chef perfecting their signature dishes, transforming raw code ingredients into polished, production-ready applications.

---

## Introduction: Imagine That...

Imagine that you've been cooking in various kitchens for months, learning different techniques, experimenting with flavors, and creating dishes. Now, you've been invited to work in a Michelin-starred restaurant kitchen where every detail matters. The head chef hands you a complex recipe that previous cooks have worked on - it functions, but it's messy, poorly documented, and inefficient.

Your mission? Transform this kitchen chaos into a masterpiece worthy of the finest dining establishments. Just as a master chef reviews recipes, refines techniques, documents processes, and optimizes cooking methods, you'll take existing Django code and elevate it to professional standards.

---

## Lesson 1: Architecture Review - The Kitchen Audit

### Learning Objective
Master the art of analyzing Django application architecture, identifying structural weaknesses, and planning improvements like a head chef conducting a comprehensive kitchen audit.

### The Chef's Kitchen Audit Analogy
When a master chef takes over a new kitchen, they don't immediately start cooking. They first conduct a thorough audit: examining the layout, checking equipment efficiency, reviewing ingredient storage systems, and understanding workflow patterns. Similarly, reviewing Django architecture means systematically examining your application's structure, identifying bottlenecks, and understanding how all components work together.

### Key Concepts and Code Examples

#### 1. Models Architecture Review
```python
# BEFORE: Poor model design (like a cluttered pantry)
class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)  # Should be EmailField
    phone = models.CharField(max_length=20)
    address = models.TextField()
    orders_count = models.IntegerField(default=0)  # Denormalized data
    
class Order(models.Model):
    user_name = models.CharField(max_length=100)  # Redundant data
    user_email = models.CharField(max_length=100)  # Should use ForeignKey
    product_name = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=10, decimal_places=2)

# AFTER: Clean model design (like an organized kitchen)
class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)  # Proper field type and constraint
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def get_orders_count(self):
        """Dynamic property instead of stored count"""
        return self.orders.count()
    
    class Meta:
        ordering = ['name']
        indexes = [
            models.Index(fields=['email']),  # Performance optimization
        ]

class Product(models.Model):  # Separate product entity
    name = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)

class Order(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    
    @property
    def total_price(self):
        return self.product.price * self.quantity
```

#### 2. Views Architecture Analysis
```python
# BEFORE: Fat view (like a chef trying to do everything at once)
def create_order(request):
    if request.method == 'POST':
        user_name = request.POST.get('user_name')
        user_email = request.POST.get('user_email')
        product_name = request.POST.get('product_name')
        
        # Too much logic in view
        try:
            user = User.objects.get(email=user_email)
        except User.DoesNotExist:
            user = User.objects.create(name=user_name, email=user_email)
        
        product = Product.objects.get(name=product_name)
        order = Order.objects.create(user=user, product=product)
        
        # Email logic in view (should be separate)
        send_mail(
            'Order Confirmation',
            f'Your order for {product.name} has been created.',
            'from@example.com',
            [user.email],
        )
        
        return JsonResponse({'status': 'success'})

# AFTER: Lean view with service layer (like specialized kitchen stations)
class OrderCreateView(APIView):
    """Clean view that delegates to service layer"""
    
    def post(self, request):
        serializer = OrderCreateSerializer(data=request.data)
        if serializer.is_valid():
            order = OrderService.create_order(serializer.validated_data)
            return Response(OrderSerializer(order).data, status=201)
        return Response(serializer.errors, status=400)

# Service layer (like specialized kitchen stations)
class OrderService:
    @staticmethod
    def create_order(validated_data):
        """Handle order creation business logic"""
        user = UserService.get_or_create_user(
            validated_data['user_email'],
            validated_data['user_name']
        )
        
        product = Product.objects.get(id=validated_data['product_id'])
        order = Order.objects.create(user=user, product=product)
        
        # Delegate email to separate service
        NotificationService.send_order_confirmation(order)
        
        return order

class UserService:
    @staticmethod
    def get_or_create_user(email, name):
        user, created = User.objects.get_or_create(
            email=email,
            defaults={'name': name}
        )
        return user
```

### Syntax Explanation
- **`models.ForeignKey`**: Creates relationships between models, like connecting recipe ingredients to the main dish
- **`related_name='orders'`**: Allows reverse lookup (user.orders.all()) like finding all dishes that use a specific ingredient
- **`@property`**: Creates computed attributes that act like model fields but are calculated dynamically
- **`APIView`**: Django REST framework's class-based view for clean API endpoints
- **`@staticmethod`**: Methods that belong to the class but don't need instance data, like utility cooking techniques

---

## Lesson 2: Code Refactoring - Recipe Refinement

### Learning Objective
Transform messy, inefficient Django code into clean, maintainable solutions using proven refactoring techniques, just like a chef refining a recipe for consistency and excellence.

### The Recipe Refinement Analogy
A master chef doesn't just follow recipes blindly - they continuously refine them. They eliminate unnecessary steps, combine similar processes, extract reusable techniques, and ensure each ingredient serves a clear purpose. Code refactoring follows the same principles: removing duplication, extracting common functionality, and making each component serve a clear, single purpose.

### Key Refactoring Techniques with Examples

#### 1. Extract Method Pattern
```python
# BEFORE: Long method (like a recipe with too many steps in one instruction)
class OrderViewSet(viewsets.ModelViewSet):
    def create(self, request):
        # Validation logic
        if not request.data.get('user_email'):
            return Response({'error': 'Email required'}, status=400)
        if not '@' in request.data.get('user_email', ''):
            return Response({'error': 'Invalid email'}, status=400)
        if not request.data.get('product_id'):
            return Response({'error': 'Product ID required'}, status=400)
        
        # User creation logic
        email = request.data['user_email']
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            if not request.data.get('user_name'):
                return Response({'error': 'Name required for new user'}, status=400)
            user = User.objects.create(
                email=email,
                name=request.data['user_name']
            )
        
        # Order creation logic
        try:
            product = Product.objects.get(id=request.data['product_id'])
        except Product.DoesNotExist:
            return Response({'error': 'Product not found'}, status=404)
        
        order = Order.objects.create(user=user, product=product)
        return Response(OrderSerializer(order).data, status=201)

# AFTER: Extracted methods (like breaking recipe into clear steps)
class OrderViewSet(viewsets.ModelViewSet):
    def create(self, request):
        # Validate input
        validation_error = self._validate_order_data(request.data)
        if validation_error:
            return validation_error
        
        # Get or create user
        user_result = self._get_or_create_user(request.data)
        if isinstance(user_result, Response):  # Error response
            return user_result
        
        # Create order
        order_result = self._create_order(user_result, request.data['product_id'])
        if isinstance(order_result, Response):  # Error response
            return order_result
        
        return Response(OrderSerializer(order_result).data, status=201)
    
    def _validate_order_data(self, data):
        """Validate order creation data like checking ingredients quality"""
        if not data.get('user_email'):
            return Response({'error': 'Email required'}, status=400)
        if '@' not in data.get('user_email', ''):
            return Response({'error': 'Invalid email'}, status=400)
        if not data.get('product_id'):
            return Response({'error': 'Product ID required'}, status=400)
        return None
    
    def _get_or_create_user(self, data):
        """Handle user retrieval/creation like preparing ingredients"""
        try:
            return User.objects.get(email=data['user_email'])
        except User.DoesNotExist:
            if not data.get('user_name'):
                return Response({'error': 'Name required for new user'}, status=400)
            return User.objects.create(
                email=data['user_email'],
                name=data['user_name']
            )
    
    def _create_order(self, user, product_id):
        """Create order like assembling the final dish"""
        try:
            product = Product.objects.get(id=product_id)
            return Order.objects.create(user=user, product=product)
        except Product.DoesNotExist:
            return Response({'error': 'Product not found'}, status=404)
```

#### 2. Repository Pattern for Data Access
```python
# BEFORE: Direct model access scattered everywhere
class OrderService:
    def get_user_orders(self, user_id):
        return Order.objects.filter(user_id=user_id).select_related('product')
    
    def get_recent_orders(self):
        return Order.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=30)
        ).select_related('user', 'product')

# AFTER: Repository pattern (like a specialized pantry manager)
class OrderRepository:
    """Centralized data access layer like a kitchen's inventory system"""
    
    @staticmethod
    def get_by_user(user_id):
        """Get all orders for a user"""
        return Order.objects.filter(
            user_id=user_id
        ).select_related('product').prefetch_related('user')
    
    @staticmethod
    def get_recent(days=30):
        """Get recent orders within specified days"""
        cutoff_date = timezone.now() - timedelta(days=days)
        return Order.objects.filter(
            created_at__gte=cutoff_date
        ).select_related('user', 'product')
    
    @staticmethod
    def get_by_status(status):
        """Get orders by status"""
        return Order.objects.filter(status=status).select_related('user', 'product')
    
    @staticmethod
    def create_with_validation(user, product, quantity=1):
        """Create order with business rule validation"""
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if not product.is_available:
            raise ValueError("Product not available")
        
        return Order.objects.create(
            user=user,
            product=product,
            quantity=quantity
        )

# Usage in service layer
class OrderService:
    def __init__(self):
        self.repository = OrderRepository()
    
    def get_user_order_summary(self, user_id):
        """Business logic using repository"""
        orders = self.repository.get_by_user(user_id)
        return {
            'total_orders': orders.count(),
            'total_amount': sum(order.total_price for order in orders),
            'recent_orders': orders[:5]
        }
```

### Syntax Explanation
- **`select_related('product')`**: Optimizes database queries by joining related tables, like preparing multiple ingredients in one trip to the pantry
- **`prefetch_related('user')`**: Optimizes reverse foreign key lookups, like pre-gathering all ingredients you'll need
- **`@staticmethod`**: Class method that doesn't need instance data, like a universal cooking technique
- **Private methods (`_method_name`)**: Internal helper methods not meant for external use, like prep work in the kitchen

---

## Lesson 3: Documentation Writing - The Master Recipe Book

### Learning Objective
Create comprehensive, clear documentation that serves as a master recipe book for your Django application, enabling any developer to understand, maintain, and extend your code effectively.

### The Master Recipe Book Analogy
Every great restaurant has a master recipe book - detailed instructions that allow any qualified chef to recreate signature dishes perfectly. The book includes ingredient lists, step-by-step procedures, cooking tips, troubleshooting notes, and variations. Similarly, great Django applications have documentation that guides developers through setup, usage, and maintenance with clarity and precision.

### Documentation Layers and Examples

#### 1. API Documentation with Django REST Framework
```python
# models.py - Documented models (like ingredient specifications)
class Product(models.Model):
    """
    Product model representing items available for purchase.
    
    Like ingredients in our kitchen, each product has specific
    characteristics and requirements.
    """
    name = models.CharField(
        max_length=200,
        help_text="Product display name (e.g., 'Premium Coffee Beans')"
    )
    price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        help_text="Price in USD (e.g., 29.99)"
    )
    description = models.TextField(
        blank=True,
        help_text="Detailed product description for customers"
    )
    is_available = models.BooleanField(
        default=True,
        help_text="Whether product is currently available for order"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['name']
        verbose_name = "Product"
        verbose_name_plural = "Products"
    
    def __str__(self):
        return f"{self.name} - ${self.price}"
    
    def get_absolute_url(self):
        """Return the canonical URL for this product"""
        return reverse('product-detail', kwargs={'pk': self.pk})

# serializers.py - Documented serializers (like recipe formats)
class ProductSerializer(serializers.ModelSerializer):
    """
    Serializer for Product model.
    
    Transforms product data between Python objects and JSON,
    like converting a recipe between different formats.
    """
    
    class Meta:
        model = Product
        fields = ['id', 'name', 'price', 'description', 'is_available', 'created_at']
        read_only_fields = ['id', 'created_at']
    
    def validate_price(self, value):
        """
        Ensure price is positive.
        
        Like checking that ingredient quantities make sense.
        """
        if value <= 0:
            raise serializers.ValidationError("Price must be positive")
        return value

# views.py - Documented API views (like cooking instructions)
class ProductViewSet(viewsets.ModelViewSet):
    """
    API viewset for managing products.
    
    Provides CRUD operations for products, like a kitchen manager
    handling ingredient inventory.
    
    Endpoints:
    - GET /api/products/ - List all products
    - POST /api/products/ - Create new product
    - GET /api/products/{id}/ - Get specific product
    - PUT /api/products/{id}/ - Update product
    - DELETE /api/products/{id}/ - Delete product
    """
    
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter]
    filterset_fields = ['is_available']
    search_fields = ['name', 'description']
    
    @action(detail=False, methods=['get'])
    def available(self, request):
        """
        Get only available products.
        
        Returns products that are currently in stock,
        like checking what ingredients are ready to use.
        
        Returns:
            Response: JSON list of available products
        """
        available_products = self.queryset.filter(is_available=True)
        serializer = self.get_serializer(available_products, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def toggle_availability(self, request, pk=None):
        """
        Toggle product availability status.
        
        Like marking ingredients as available or out of stock.
        
        Args:
            pk: Product primary key
        
        Returns:
            Response: Updated product data
        """
        product = self.get_object()
        product.is_available = not product.is_available
        product.save()
        serializer = self.get_serializer(product)
        return Response(serializer.data)
```

#### 2. README Documentation Structure
```markdown
# Restaurant Management System

A Django-based restaurant management system that handles orders, products, and customers like a well-organized kitchen.

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [API Usage](#api-usage)
- [Development](#development)
- [Testing](#testing)

## Installation

### Prerequisites (Kitchen Requirements)
- Python 3.8+ (like having a proper stove)
- PostgreSQL 12+ (like having a reliable refrigerator)
- Redis (like having quick access to frequently used spices)

### Setup Steps (Kitchen Preparation)
```bash
# Clone the repository (get the recipe book)
git clone https://github.com/yourrepo/restaurant-system.git
cd restaurant-system

# Create virtual environment (set up your cooking space)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (stock your pantry)
pip install -r requirements.txt

# Set up database (prepare your storage)
python manage.py migrate

# Create superuser (appoint head chef)
python manage.py createsuperuser

# Start development server (fire up the kitchen)
python manage.py runserver
```

## API Usage Examples

### Creating a Product (Adding New Ingredient to Menu)
```python
import requests

# Add new product
response = requests.post('http://localhost:8000/api/products/', {
    'name': 'Artisan Bread',
    'price': '8.99',
    'description': 'Freshly baked daily',
    'is_available': True
})

print(response.json())
# Output: {'id': 1, 'name': 'Artisan Bread', 'price': '8.99', ...}
```

### Filtering Products (Finding Specific Ingredients)
```python
# Get only available products
response = requests.get('http://localhost:8000/api/products/?is_available=true')

# Search products by name
response = requests.get('http://localhost:8000/api/products/?search=bread')
```
```

#### 3. Inline Code Documentation
```python
class OrderService:
    """
    Service class for handling order operations.
    
    This service acts like a sous chef, coordinating between
    different kitchen stations to complete orders efficiently.
    """
    
    def __init__(self, repository=None):
        """
        Initialize order service.
        
        Args:
            repository: Custom repository instance (for testing)
        """
        self.repository = repository or OrderRepository()
        self.notification_service = NotificationService()
    
    def create_order(self, user_data, product_id, quantity=1):
        """
        Create a new order with full validation.
        
        Like taking a customer's order and ensuring we can fulfill it:
        1. Validate the customer information
        2. Check if the product is available
        3. Verify we have sufficient quantity
        4. Create the order record
        5. Send confirmation notification
        
        Args:
            user_data (dict): Customer information
                - email (str): Customer email address
                - name (str): Customer full name
            product_id (int): ID of the product being ordered
            quantity (int, optional): Number of items. Defaults to 1.
        
        Returns:
            Order: Created order instance
        
        Raises:
            ValidationError: If user data is invalid
            Product.DoesNotExist: If product doesn't exist
            ValueError: If quantity is invalid or product unavailable
        
        Example:
            >>> service = OrderService()
            >>> order = service.create_order(
            ...     {'email': 'customer@example.com', 'name': 'John Doe'},
            ...     product_id=1,
            ...     quantity=2
            ... )
            >>> print(f"Order {order.id} created for {order.user.name}")
        """
        # Validate inputs like checking order ticket accuracy
        self._validate_order_input(user_data, product_id, quantity)
        
        # Get or create user like greeting a new/returning customer
        user = self._get_or_create_user(user_data)
        
        # Verify product availability like checking ingredient stock
        product = self._verify_product_availability(product_id, quantity)
        
        # Create order like sending ticket to kitchen
        order = self.repository.create_order(user, product, quantity)
        
        # Send confirmation like giving customer receipt
        self.notification_service.send_order_confirmation(order)
        
        return order
    
    def _validate_order_input(self, user_data, product_id, quantity):
        """Validate order inputs like checking order ticket completeness."""
        if not isinstance(user_data, dict):
            raise ValidationError("User data must be a dictionary")
        
        required_fields = ['email', 'name']
        for field in required_fields:
            if not user_data.get(field):
                raise ValidationError(f"User {field} is required")
        
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("Quantity must be a positive integer")
```

### Syntax Explanation
- **Docstrings (`"""`)**: Multi-line documentation strings that explain what functions/classes do
- **Type hints (`user_data: dict`)**: Specify expected data types for parameters
- **`@action` decorator**: Creates custom API endpoints beyond standard CRUD operations
- **`help_text` parameter**: Provides field-level documentation for models
- **`verbose_name`**: Human-readable names for models and fields

---

## Lesson 4: Performance Optimization - Kitchen Efficiency Mastery

### Learning Objective
Optimize Django application performance using database query optimization, caching strategies, and efficient coding patterns, transforming your application from a slow short-order cook into a lightning-fast professional kitchen.

### The Kitchen Efficiency Analogy
A master chef doesn't just create delicious food - they create it efficiently. They organize ingredients for quick access, prepare components in advance, use the right tools for each task, and eliminate wasteful movements. Performance optimization in Django follows the same principles: minimize database trips, cache frequently used data, choose efficient algorithms, and eliminate bottlenecks.

### Performance Optimization Techniques

#### 1. Database Query Optimization
```python
# BEFORE: Inefficient queries (like running to the pantry for each ingredient)
def get_user_orders_slow(request, user_id):
    """Inefficient approach - multiple database hits"""
    user = User.objects.get(id=user_id)  # Query 1
    orders = Order.objects.filter(user=user)  # Query 2
    
    result = []
    for order in orders:  # N additional queries!
        result.append({
            'id': order.id,
            'product_name': order.product.name,  # Query for each order!
            'user_name': order.user.name,  # Query for each order!
            'total': order.total_price
        })
    return JsonResponse({'orders': result})

# AFTER: Optimized queries (like mise en place - everything prepared at once)
def get_user_orders_fast(request, user_id):
    """Optimized approach - minimal database hits"""
    orders = Order.objects.filter(
        user_id=user_id
    ).select_related(  # Join related tables in single query
        'user', 'product'
    ).prefetch_related(  # Optimize reverse lookups
        'orderitem_set'
    ).only(  # Fetch only needed fields
        'id', 'created_at', 'user__name', 'product__name', 'product__price'
    )
    
    # Use list comprehension for efficiency
    result = [
        {
            'id': order.id,
            'product_name': order.product.name,  # No additional query!
            'user_name': order.user.name,  # No additional query!
            'total': order.total_price,
            'created_at': order.created_at.isoformat()
        }
        for order in orders
    ]
    
    return JsonResponse({'orders': result})

# Advanced: Custom Manager for Complex Queries
class OptimizedOrderManager(models.Manager):
    """Custom manager like a specialized kitchen station"""
    
    def with_details(self):
        """Get orders with all related data in optimal way"""
        return self.select_related(
            'user', 'product'
        ).prefetch_related(
            'orderitem_set__product'
        ).annotate(
            # Calculate totals at database level
            items_count=Count('orderitem'),
            total_amount=Sum(
                F('orderitem__quantity') * F('orderitem__product__price')
            )
        )
    
    def recent_with_stats(self, days=30):
        """Get recent orders with performance statistics"""
        cutoff = timezone.now() - timedelta(days=days)
        return self.with_details().filter(
            created_at__gte=cutoff
        ).order_by('-created_at')

# Usage in models
class Order(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Add custom manager
    objects = models.Manager()  # Default manager
    optimized = OptimizedOrderManager()  # Optimized manager
    
    class Meta:
        # Database-level optimizations
        indexes = [
            models.Index(fields=['created_at']),  # For date filtering
            models.Index(fields=['user', 'created_at']),  # Composite index
        ]
```

#### 2. Caching Strategies
```python
# settings.py - Cache configuration (like setting up food warmers)
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Cache timeout settings
CACHE_TTL = 60 * 15  # 15 minutes

# Caching in views (like keeping popular dishes warm)
from django.core.cache import cache
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page

class ProductListView(APIView):
    """Cached product list like a menu board that updates periodically"""
    
    def get(self, request):
        # Try to get from cache first (like checking if dish is already prepared)
        cache_key = 'product_list_available'
        cached_products = cache.get(cache_key)
        
        if cached_products is not None:
            return Response(cached_products)
        
        # If not in cache, get from database (like preparing fresh)
        products = Product.objects.filter(is_available=True).values(
            'id', 'name', 'price', 'description'
        )
        
        serialized_data = list(products)
        
        # Store in cache (like keeping dish warm for next customer)
        cache.set(cache_key, serialized_data, timeout=CACHE_TTL)
        
        return Response(serialized_data)

# Model-level caching (like caching prep work)
class Product(models.Model):
    name = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    is_available = models.BooleanField(default=True)
    
    @property
    def expensive_calculation(self):
        """Cache expensive operations like complex sauce preparations"""
        cache_key = f'product_calculation_{self.id}'
        result = cache.get(cache_key)
        
        if result is None:
            # Simulate expensive calculation
            result = sum(
                order.total_price for order in self.order_set.all()
            )
            cache.set(cache_key, result, timeout=CACHE_TTL)
        
        return result
    
    def save(self, *args, **kwargs):
        """Clear related caches when model changes (like updating menu)"""
        super().save(*args, **kwargs)
        
        # Clear related caches
        cache.delete('product_list_available')
        cache.delete(f'product_calculation_{self.id}')
        
        # Clear pattern-based caches
        cache.delete_pattern(f'product_*_{self.id}')

# Function-based caching (like pre-made sauces)
from functools import lru_cache

@lru_cache(maxsize=128)
def get_tax_rate(location):
    """Cache tax calculations like keeping standard recipe ratios memorized"""
    # Expensive tax API call
    return calculate_tax_for_location(location)

# Usage
def calculate_order_total(order, location):
    tax_rate = get_tax_rate(location)  # Cached after first call
    return order.subtotal * (1 + tax_rate)
```

#### 3. Pagination and Lazy Loading
```python
# Efficient pagination (like serving courses rather than entire meal at once)
from rest_framework.pagination import PageNumberPagination

class OptimizedPagination(PageNumberPagination):
    """Custom pagination like serving manageable portions"""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

class OrderListView(generics.ListAPIView):
    """Paginated order list for better performance"""
    serializer_class = OrderSerializer
    pagination_class = OptimizedPagination
    
    def get_queryset(self):
        """Optimized queryset like efficient kitchen prep"""
        return Order.optimized.with_details().filter(
            user=self.request.user
        )

# Lazy loading with Django's iterator (like streaming service)
def process_large_dataset():
    """Process large datasets without loading everything into memory"""
    # Instead of loading all orders at once (like trying to cook everything simultaneously)
    # all_orders = Order.objects.all()  # Memory intensive!
    
    # Use iterator to process in chunks (like cooking in batches)
    for order in Order.objects.all().iterator(chunk_size=1000):
        # Process each order without loading all into memory
        process_single_order(order)

# Async views for I/O bound operations (like parallel cooking stations)
from django.http import JsonResponse
import asyncio
import aiohttp

async def get_external_data_async(request):
    """Handle external API calls efficiently like coordinating with suppliers"""
    async with aiohttp.ClientSession() as session:
        # Multiple API calls in parallel (like calling multiple suppliers at once)
        tasks = [
            fetch_user_data(session, user_id) 
            for user_id in request.GET.getlist('user_ids')
        ]
        
        results = await asyncio.gather(*tasks)
        return JsonResponse({'data': results})

async def fetch_user_data(session, user_id):
    """Fetch individual user data asynchronously"""
    async with session.get(f'https://api.example.com/users/{user_id}') as response:
        return await response.json()
```

### Syntax Explanation
- **`select_related()`**: Performs SQL JOIN to fetch related objects in single query, like getting all ingredients in one pantry trip
- **`prefetch_related()`**: Optimizes reverse foreign key lookups with separate queries, like getting all related items efficiently
- **`only()`**: Fetches only specified fields from database, like taking only needed ingredients
- **`annotate()`**: Adds calculated fields at database level, like having the database do the math
- **`F()` expressions**: Reference database fields in calculations, avoiding Python-level processing
- **`@lru_cache`**: Python's built-in decorator for function result caching
- **`iterator(chunk_size=1000)`**: Processes large querysets in chunks to save memory

---

## Final Project: Restaurant Order Management System

Now that you've mastered the individual techniques, it's time to create a comprehensive Django application that demonstrates all the concepts learned. This project will serve as your portfolio piece, showcasing your Django mastery skills.

### Project Overview
Create a complete restaurant order management system with the following features:

#### Core Models
```python
# models.py - Your complete kitchen ecosystem
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator
from django.urls import reverse

class Category(models.Model):
    """Menu categories like appetizers, mains, desserts"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Categories"
        ordering = ['name']
    
    def __str__(self):
        return self.name

class Product(models.Model):
    """Menu items - your signature dishes"""
    name = models.CharField(max_length=200)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2, validators=[MinValueValidator(0.01)])
    preparation_time = models.PositiveIntegerField(help_text="Minutes to prepare")
    is_available = models.BooleanField(default=True)
    featured = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['category', 'name']
        indexes = [
            models.Index(fields=['category', 'is_available']),
            models.Index(fields=['featured', 'is_available']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.category.name})"
    
    def get_absolute_url(self):
        return reverse('product-detail', kwargs={'pk': self.pk})

class Order(models.Model):
    """Customer orders - tickets to the kitchen"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('preparing', 'Preparing'),
        ('ready', 'Ready'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
    ]
    
    customer = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    notes = models.TextField(blank=True, help_text="Special instructions")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['customer', 'status']),
            models.Index(fields=['created_at']),
        ]
    
    @property
    def total_amount(self):
        """Calculate total order amount"""
        return sum(item.subtotal for item in self.items.all())
    
    @property
    def estimated_completion_time(self):
        """Calculate when order should be ready"""
        max_prep_time = max(
            (item.product.preparation_time for item in self.items.all()),
            default=0
        )
        return self.created_at + timedelta(minutes=max_prep_time)
    
    def __str__(self):
        return f"Order #{self.id} - {self.customer.username}"

class OrderItem(models.Model):
    """Individual items in an order - specific dishes ordered"""
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(validators=[MinValueValidator(1)])
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    notes = models.CharField(max_length=200, blank=True, help_text="Item-specific notes")
    
    class Meta:
        unique_together = ['order', 'product']
    
    @property
    def subtotal(self):
        """Calculate subtotal for this item"""
        return self.quantity * self.unit_price
    
    def save(self, *args, **kwargs):
        """Set unit_price from product if not provided"""
        if not self.unit_price:
            self.unit_price = self.product.price
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.quantity}x {self.product.name}"
```

#### Requirements to Implement
1. **Architecture Review**: Clean model relationships, proper separation of concerns
2. **Code Refactoring**: Use service layers, repository patterns, and extracted methods
3. **Documentation**: Comprehensive docstrings, API documentation, and README
4. **Performance Optimization**: Database query optimization, caching, and pagination

---
# Portfolio-Worthy Django Application: RecipeShare Platform

## Project Overview
Build a comprehensive recipe sharing platform that demonstrates Django mastery through advanced features, clean code architecture, and professional deployment practices.

## Core Features Implementation

### 1. User Authentication & Profiles

```python
# models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
from PIL import Image

class CustomUser(AbstractUser):
    bio = models.TextField(max_length=500, blank=True)
    profile_picture = models.ImageField(upload_to='profiles/', default='profiles/default.jpg')
    website = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Resize profile picture
        img = Image.open(self.profile_picture.path)
        if img.height > 300 or img.width > 300:
            output_size = (300, 300)
            img.thumbnail(output_size)
            img.save(self.profile_picture.path)
```

```python
# forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'password1', 'password2')
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user
```

### 2. Recipe Management System

```python
# models.py
class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    
    class Meta:
        verbose_name_plural = "Categories"
    
    def __str__(self):
        return self.name

class Recipe(models.Model):
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    ]
    
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True, blank=True)
    author = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='recipes')
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='recipes')
    description = models.TextField()
    ingredients = models.TextField()
    instructions = models.TextField()
    prep_time = models.PositiveIntegerField(help_text="Preparation time in minutes")
    cook_time = models.PositiveIntegerField(help_text="Cooking time in minutes")
    servings = models.PositiveIntegerField(default=4)
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES, default='easy')
    image = models.ImageField(upload_to='recipes/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_featured = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.title
    
    def save(self, *args, **kwargs):
        if not self.slug:
            from django.utils.text import slugify
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)
    
    @property
    def total_time(self):
        return self.prep_time + self.cook_time

class RecipeRating(models.Model):
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE, related_name='ratings')
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    rating = models.PositiveIntegerField(choices=[(i, i) for i in range(1, 6)])
    review = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('recipe', 'user')
    
    def __str__(self):
        return f"{self.recipe.title} - {self.rating} stars"
```

### 3. Advanced Views with Search & Filtering

```python
# views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from django.db.models import Q, Avg, Count
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.contrib import messages
from .models import Recipe, Category, RecipeRating
from .forms import RecipeForm, RecipeRatingForm

class RecipeListView(ListView):
    model = Recipe
    template_name = 'recipes/recipe_list.html'
    context_object_name = 'recipes'
    paginate_by = 12
    
    def get_queryset(self):
        queryset = Recipe.objects.select_related('author', 'category').annotate(
            avg_rating=Avg('ratings__rating'),
            rating_count=Count('ratings')
        )
        
        # Search functionality
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(title__icontains=search) |
                Q(description__icontains=search) |
                Q(ingredients__icontains=search)
            )
        
        # Category filter
        category = self.request.GET.get('category')
        if category:
            queryset = queryset.filter(category__slug=category)
        
        # Difficulty filter
        difficulty = self.request.GET.get('difficulty')
        if difficulty:
            queryset = queryset.filter(difficulty=difficulty)
        
        # Sorting
        sort_by = self.request.GET.get('sort', '-created_at')
        if sort_by == 'rating':
            queryset = queryset.order_by('-avg_rating', '-created_at')
        elif sort_by == 'title':
            queryset = queryset.order_by('title')
        else:
            queryset = queryset.order_by(sort_by)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        context['current_category'] = self.request.GET.get('category', '')
        context['current_difficulty'] = self.request.GET.get('difficulty', '')
        context['current_search'] = self.request.GET.get('search', '')
        context['current_sort'] = self.request.GET.get('sort', '-created_at')
        return context

class RecipeDetailView(DetailView):
    model = Recipe
    template_name = 'recipes/recipe_detail.html'
    context_object_name = 'recipe'
    
    def get_object(self):
        return get_object_or_404(
            Recipe.objects.select_related('author', 'category').annotate(
                avg_rating=Avg('ratings__rating'),
                rating_count=Count('ratings')
            ),
            slug=self.kwargs['slug']
        )
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        recipe = self.object
        
        # Get user's rating if logged in
        user_rating = None
        if self.request.user.is_authenticated:
            try:
                user_rating = RecipeRating.objects.get(recipe=recipe, user=self.request.user)
            except RecipeRating.DoesNotExist:
                pass
        
        context['user_rating'] = user_rating
        context['rating_form'] = RecipeRatingForm()
        context['recent_ratings'] = recipe.ratings.select_related('user').order_by('-created_at')[:5]
        context['related_recipes'] = Recipe.objects.filter(
            category=recipe.category
        ).exclude(id=recipe.id).annotate(
            avg_rating=Avg('ratings__rating')
        )[:4]
        
        return context

class RecipeCreateView(LoginRequiredMixin, CreateView):
    model = Recipe
    form_class = RecipeForm
    template_name = 'recipes/recipe_form.html'
    
    def form_valid(self, form):
        form.instance.author = self.request.user
        messages.success(self.request, 'Recipe created successfully!')
        return super().form_valid(form)

@login_required
def rate_recipe(request, slug):
    recipe = get_object_or_404(Recipe, slug=slug)
    
    if request.method == 'POST':
        form = RecipeRatingForm(request.POST)
        if form.is_valid():
            rating, created = RecipeRating.objects.get_or_create(
                recipe=recipe,
                user=request.user,
                defaults={
                    'rating': form.cleaned_data['rating'],
                    'review': form.cleaned_data['review']
                }
            )
            
            if not created:
                rating.rating = form.cleaned_data['rating']
                rating.review = form.cleaned_data['review']
                rating.save()
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                avg_rating = recipe.ratings.aggregate(Avg('rating'))['rating__avg']
                return JsonResponse({
                    'success': True,
                    'avg_rating': round(avg_rating, 1) if avg_rating else 0,
                    'rating_count': recipe.ratings.count()
                })
            
            messages.success(request, 'Rating submitted successfully!')
            return redirect('recipe_detail', slug=slug)
    
    return redirect('recipe_detail', slug=slug)
```

### 4. Advanced Forms with Dynamic Features

```python
# forms.py
from django import forms
from django.core.exceptions import ValidationError
from .models import Recipe, RecipeRating

class RecipeForm(forms.ModelForm):
    ingredients = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 8,
            'placeholder': 'Enter each ingredient on a new line:\n• 2 cups flour\n• 1 tsp salt\n• 3 eggs'
        }),
        help_text="List each ingredient on a separate line"
    )
    
    instructions = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 10,
            'placeholder': 'Step-by-step instructions:\n1. Preheat oven to 350°F\n2. Mix dry ingredients...'
        }),
        help_text="Numbered steps work best"
    )
    
    class Meta:
        model = Recipe
        fields = [
            'title', 'category', 'description', 'ingredients', 'instructions',
            'prep_time', 'cook_time', 'servings', 'difficulty', 'image'
        ]
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'prep_time': forms.NumberInput(attrs={'min': 1}),
            'cook_time': forms.NumberInput(attrs={'min': 1}),
            'servings': forms.NumberInput(attrs={'min': 1}),
        }
    
    def clean_title(self):
        title = self.cleaned_data['title']
        if len(title) < 5:
            raise ValidationError("Title must be at least 5 characters long.")
        return title
    
    def clean_prep_time(self):
        prep_time = self.cleaned_data['prep_time']
        if prep_time > 480:  # 8 hours
            raise ValidationError("Preparation time seems unrealistic. Please check your input.")
        return prep_time

class RecipeRatingForm(forms.ModelForm):
    class Meta:
        model = RecipeRating
        fields = ['rating', 'review']
        widgets = {
            'rating': forms.Select(attrs={'class': 'form-select'}),
            'review': forms.Textarea(attrs={'rows': 3, 'placeholder': 'Share your thoughts about this recipe...'})
        }
```

### 5. API Integration with Django REST Framework

```python
# serializers.py
from rest_framework import serializers
from .models import Recipe, Category, RecipeRating

class CategorySerializer(serializers.ModelSerializer):
    recipe_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Category
        fields = ['id', 'name', 'slug', 'description', 'recipe_count']
    
    def get_recipe_count(self, obj):
        return obj.recipes.count()

class RecipeSerializer(serializers.ModelSerializer):
    author = serializers.StringRelatedField()
    category = CategorySerializer(read_only=True)
    avg_rating = serializers.DecimalField(max_digits=3, decimal_places=2, read_only=True)
    rating_count = serializers.IntegerField(read_only=True)
    total_time = serializers.ReadOnlyField()
    
    class Meta:
        model = Recipe
        fields = [
            'id', 'title', 'slug', 'author', 'category', 'description',
            'prep_time', 'cook_time', 'total_time', 'servings', 'difficulty',
            'image', 'created_at', 'avg_rating', 'rating_count'
        ]

# api_views.py
from rest_framework import generics, filters
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Avg, Count
from .models import Recipe, Category
from .serializers import RecipeSerializer, CategorySerializer

class RecipeListAPIView(generics.ListAPIView):
    serializer_class = RecipeSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['category', 'difficulty', 'author']
    search_fields = ['title', 'description', 'ingredients']
    ordering_fields = ['created_at', 'title', 'prep_time', 'cook_time']
    ordering = ['-created_at']
    
    def get_queryset(self):
        return Recipe.objects.select_related('author', 'category').annotate(
            avg_rating=Avg('ratings__rating'),
            rating_count=Count('ratings')
        )

@api_view(['GET'])
def recipe_stats(request):
    stats = {
        'total_recipes': Recipe.objects.count(),
        'total_categories': Category.objects.count(),
        'featured_recipes': Recipe.objects.filter(is_featured=True).count(),
        'avg_prep_time': Recipe.objects.aggregate(Avg('prep_time'))['prep_time__avg'],
        'most_popular_category': Category.objects.annotate(
            recipe_count=Count('recipes')
        ).order_by('-recipe_count').first().name if Category.objects.exists() else None
    }
    return Response(stats)
```

### 6. Advanced Templates with HTMX Integration

```html
<!-- templates/recipes/recipe_list.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RecipeShare - Discover Amazing Recipes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://unpkg.com/htmx.org@1.8.4"></script>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{% url 'recipe_list' %}">
                <i class="fas fa-utensils"></i> RecipeShare
            </a>
            <div class="navbar-nav ms-auto">
                {% if user.is_authenticated %}
                    <a class="nav-link" href="{% url 'recipe_create' %}">
                        <i class="fas fa-plus"></i> Add Recipe
                    </a>
                    <a class="nav-link" href="{% url 'profile' %}">{{ user.username }}</a>
                {% else %}
                    <a class="nav-link" href="{% url 'login' %}">Login</a>
                {% endif %}
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <!-- Search and Filter Bar -->
        <div class="row mb-4">
            <div class="col-md-12">
                <form method="get" hx-get="{% url 'recipe_list' %}" hx-target="#recipe-grid" hx-push-url="true">
                    <div class="row g-3">
                        <div class="col-md-4">
                            <input type="text" name="search" class="form-control" placeholder="Search recipes..." 
                                   value="{{ current_search }}" hx-trigger="keyup changed delay:500ms">
                        </div>
                        <div class="col-md-2">
                            <select name="category" class="form-select" hx-trigger="change">
                                <option value="">All Categories</option>
                                {% for category in categories %}
                                    <option value="{{ category.slug }}" 
                                            {% if category.slug == current_category %}selected{% endif %}>
                                        {{ category.name }}
                                    </option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-2">
                            <select name="difficulty" class="form-select" hx-trigger="change">
                                <option value="">All Levels</option>
                                <option value="easy" {% if current_difficulty == 'easy' %}selected{% endif %}>Easy</option>
                                <option value="medium" {% if current_difficulty == 'medium' %}selected{% endif %}>Medium</option>
                                <option value="hard" {% if current_difficulty == 'hard' %}selected{% endif %}>Hard</option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <select name="sort" class="form-select" hx-trigger="change">
                                <option value="-created_at" {% if current_sort == '-created_at' %}selected{% endif %}>Newest</option>
                                <option value="title" {% if current_sort == 'title' %}selected{% endif %}>A-Z</option>
                                <option value="rating" {% if current_sort == 'rating' %}selected{% endif %}>Top Rated</option>
                            </select>
                        </div>
                    </div>
                </form>
            </div>
        </div>

        <!-- Recipe Grid -->
        <div id="recipe-grid">
            <div class="row">
                {% for recipe in recipes %}
                    <div class="col-md-4 mb-4">
                        <div class="card h-100 shadow-sm">
                            {% if recipe.image %}
                                <img src="{{ recipe.image.url }}" class="card-img-top" style="height: 200px; object-fit: cover;">
                            {% else %}
                                <div class="card-img-top bg-light d-flex align-items-center justify-content-center" style="height: 200px;">
                                    <i class="fas fa-image fa-3x text-muted"></i>
                                </div>
                            {% endif %}
                            
                            <div class="card-body d-flex flex-column">
                                <h5 class="card-title">{{ recipe.title }}</h5>
                                <p class="card-text text-muted small">{{ recipe.description|truncatewords:15 }}</p>
                                
                                <div class="mt-auto">
                                    <div class="d-flex justify-content-between align-items-center mb-2">
                                        <small class="text-muted">
                                            <i class="fas fa-clock"></i> {{ recipe.total_time }}min
                                        </small>
                                        {% if recipe.avg_rating %}
                                            <small class="text-warning">
                                                <i class="fas fa-star"></i> {{ recipe.avg_rating|floatformat:1 }}
                                                ({{ recipe.rating_count }})
                                            </small>
                                        {% endif %}
                                    </div>
                                    
                                    <div class="d-flex justify-content-between align-items-center">
                                        <small class="text-muted">by {{ recipe.author.username }}</small>
                                        <span class="badge bg-secondary">{{ recipe.get_difficulty_display }}</span>
                                    </div>
                                    
                                    <a href="{% url 'recipe_detail' recipe.slug %}" class="btn btn-primary btn-sm mt-2 w-100">
                                        View Recipe
                                    </a>
                                </div>
                            </div>
                        </div>
                    </div>
                {% empty %}
                    <div class="col-12 text-center">
                        <div class="alert alert-info">
                            <i class="fas fa-search fa-2x mb-3"></i>
                            <h4>No recipes found</h4>
                            <p>Try adjusting your search criteria or browse all recipes.</p>
                        </div>
                    </div>
                {% endfor %}
            </div>

            <!-- Pagination -->
            {% if is_paginated %}
                <nav aria-label="Recipe pagination">
                    <ul class="pagination justify-content-center">
                        {% if page_obj.has_previous %}
                            <li class="page-item">
                                <a class="page-link" href="?page={{ page_obj.previous_page_number }}{% if current_search %}&search={{ current_search }}{% endif %}{% if current_category %}&category={{ current_category }}{% endif %}{% if current_difficulty %}&difficulty={{ current_difficulty }}{% endif %}&sort={{ current_sort }}">
                                    Previous
                                </a>
                            </li>
                        {% endif %}
                        
                        {% for num in page_obj.paginator.page_range %}
                            {% if page_obj.number == num %}
                                <li class="page-item active">
                                    <span class="page-link">{{ num }}</span>
                                </li>
                            {% elif num > page_obj.number|add:'-3' and num < page_obj.number|add:'3' %}
                                <li class="page-item">
                                    <a class="page-link" href="?page={{ num }}{% if current_search %}&search={{ current_search }}{% endif %}{% if current_category %}&category={{ current_category }}{% endif %}{% if current_difficulty %}&difficulty={{ current_difficulty }}{% endif %}&sort={{ current_sort }}">
                                        {{ num }}
                                    </a>
                                </li>
                            {% endif %}
                        {% endfor %}
                        
                        {% if page_obj.has_next %}
                            <li class="page-item">
                                <a class="page-link" href="?page={{ page_obj.next_page_number }}{% if current_search %}&search={{ current_search }}{% endif %}{% if current_category %}&category={{ current_category }}{% endif %}{% if current_difficulty %}&difficulty={{ current_difficulty }}{% endif %}&sort={{ current_sort }}">
                                    Next
                                </a>
                            </li>
                        {% endif %}
                    </ul>
                </nav>
            {% endif %}
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

### 7. Settings Configuration

```python
# settings.py
import os
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config('SECRET_KEY', default='your-secret-key-here')
DEBUG = config('DEBUG', default=True, cast=bool)
ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='localhost,127.0.0.1', cast=lambda v: [s.strip() for s in v.split(',')])

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'django_filters',
    'corsheaders',
    'recipes',
    'accounts',
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

ROOT_URLCONF = 'recipeshare.urls'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME', default='recipeshare'),
        'USER': config('DB_USER', default='postgres'),
        'PASSWORD': config('DB_PASSWORD', default=''),
        'HOST': config('DB_HOST', default='localhost'),
        'PORT': config('DB_PORT', default='5432'),
    }
}

AUTH_USER_MODEL = 'accounts.CustomUser'

# REST Framework Configuration
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
}

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

# Email configuration
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = config('EMAIL_HOST', default='smtp.gmail.com')
EMAIL_PORT = config('EMAIL_PORT', default=587, cast=int)
EMAIL_USE_TLS = True
EMAIL_HOST_USER = config('EMAIL_HOST_USER', default='')
EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD', default='')

# Security settings for production
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_SECONDS = 31536000
    SECURE_REDIRECT_EXEMPT = []
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
```

### 8. URL Configuration

```python
# urls.py (main)
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('recipes.urls')),
    path('accounts/', include('accounts.urls')),
    path('api/', include('recipes.api_urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# recipes/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.RecipeListView.as_view(), name='recipe_list'),
    path('recipe/<slug:slug>/', views.RecipeDetailView.as_view(), name='recipe_detail'),
    path('recipe/<slug:slug>/rate/', views.rate_recipe, name='rate_recipe'),
    path('create/', views.RecipeCreateView.as_view(), name='recipe_create'),
    path('category/<slug:slug>/', views.CategoryRecipeListView.as_view(), name='category_recipes'),
]

# recipes/api_urls.py
from django.urls import path
from . import api_views

urlpatterns = [
    path('recipes/', api_views.RecipeListAPIView.as_view(), name='api_recipe_list'),
    path('recipes/<int:pk>/', api_views.RecipeDetailAPIView.as_view(), name='api_recipe_detail'),
    path('categories/', api_views.CategoryListAPIView.as_view(), name='api_category_list'),
    path('stats/', api_views.recipe_stats, name='api_recipe_stats'),
]
```

### 9. Management Commands

```python
# management/commands/populate_sample_data.py
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from recipes.models import Category, Recipe
import random

User = get_user_model()

class Command(BaseCommand):
    help = 'Populate database with sample recipe data'
    
    def handle(self, *args, **options):
        # Create sample categories
        categories_data = [
            {'name': 'Breakfast', 'description': 'Start your day right'},
            {'name': 'Lunch', 'description': 'Midday meals'},
            {'name': 'Dinner', 'description': 'Evening feasts'},
            {'name': 'Desserts', 'description': 'Sweet treats'},
            {'name': 'Appetizers', 'description': 'Small bites'},
        ]
        
        categories = []
        for cat_data in categories_data:
            category, created = Category.objects.get_or_create(
                name=cat_data['name'],
                defaults={'description': cat_data['description']}
            )
            categories.append(category)
            if created:
                self.stdout.write(f'Created category: {category.name}')
        
        # Create sample user if not exists
        user, created = User.objects.get_or_create(
            username='chef_demo',
            defaults={
                'email': 'chef@example.com',
                'first_name': 'Demo',
                'last_name': 'Chef',
                'bio': 'Passionate home cook sharing favorite recipes'
            }
        )
        if created:
            user.set_password('demo123')
            user.save()
            self.stdout.write('Created demo user')
        
        # Sample recipe data
        sample_recipes = [
            {
                'title': 'Classic Chocolate Chip Cookies',
                'category': 'Desserts',
                'description': 'Soft, chewy cookies with the perfect balance of sweetness',
                'ingredients': '''• 2¼ cups all-purpose flour
• 1 tsp baking soda
• 1 tsp salt
• 1 cup butter, softened
• ¾ cup granulated sugar
• ¾ cup brown sugar
• 2 large eggs
• 2 tsp vanilla extract
• 2 cups chocolate chips''',
                'instructions': '''1. Preheat oven to 375°F (190°C)
2. Mix flour, baking soda, and salt in a bowl
3. Cream butter and sugars until fluffy
4. Beat in eggs and vanilla
5. Gradually mix in flour mixture
6. Stir in chocolate chips
7. Drop rounded tablespoons onto ungreased cookie sheets
8. Bake 9-11 minutes until golden brown
9. Cool on baking sheet for 2 minutes before removing''',
                'prep_time': 15,
                'cook_time': 10,
                'servings': 24,
                'difficulty': 'easy'
            },
            {
                'title': 'Creamy Chicken Alfredo',
                'category': 'Dinner',
                'description': 'Rich and creamy pasta dish with tender chicken',
                'ingredients': '''• 1 lb fettuccine pasta
• 2 chicken breasts, sliced
• 2 tbsp olive oil
• 4 cloves garlic, minced
• 1 cup heavy cream
• 1 cup parmesan cheese, grated
• 2 tbsp butter
• Salt and pepper to taste
• Fresh parsley for garnish''',
                'instructions': '''1. Cook pasta according to package directions
2. Season chicken with salt and pepper
3. Heat oil in large skillet, cook chicken until done
4. Remove chicken, add garlic to pan
5. Add cream and bring to simmer
6. Stir in parmesan and butter
7. Return chicken to pan
8. Toss with cooked pasta
9. Garnish with parsley and serve''',
                'prep_time': 10,
                'cook_time': 20,
                'servings': 4,
                'difficulty': 'medium'
            },
            {
                'title': 'Perfect Pancakes',
                'category': 'Breakfast',
                'description': 'Fluffy, golden pancakes that melt in your mouth',
                'ingredients': '''• 2 cups all-purpose flour
• 2 tbsp sugar
• 2 tsp baking powder
• 1 tsp salt
• 2 cups milk
• 2 large eggs
• 4 tbsp melted butter
• 1 tsp vanilla extract''',
                'instructions': '''1. Mix dry ingredients in large bowl
2. Whisk together wet ingredients
3. Pour wet into dry ingredients, stir until just combined
4. Heat griddle or pan over medium heat
5. Pour ¼ cup batter for each pancake
6. Cook until bubbles form and edges look set
7. Flip and cook until golden
8. Serve hot with maple syrup''',
                'prep_time': 5,
                'cook_time': 15,
                'servings': 6,
                'difficulty': 'easy'
            }
        ]
        
        # Create sample recipes
        for recipe_data in sample_recipes:
            category = Category.objects.get(name=recipe_data['category'])
            recipe, created = Recipe.objects.get_or_create(
                title=recipe_data['title'],
                defaults={
                    'author': user,
                    'category': category,
                    'description': recipe_data['description'],
                    'ingredients': recipe_data['ingredients'],
                    'instructions': recipe_data['instructions'],
                    'prep_time': recipe_data['prep_time'],
                    'cook_time': recipe_data['cook_time'],
                    'servings': recipe_data['servings'],
                    'difficulty': recipe_data['difficulty']
                }
            )
            if created:
                self.stdout.write(f'Created recipe: {recipe.title}')
        
### 10. Testing Suite

```python
# tests/test_models.py
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from recipes.models import Category, Recipe, RecipeRating

User = get_user_model()

class RecipeModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.category = Category.objects.create(
            name='Test Category',
            description='Test description'
        )
    
    def test_recipe_creation(self):
        recipe = Recipe.objects.create(
            title='Test Recipe',
            author=self.user,
            category=self.category,
            description='Test description',
            ingredients='Test ingredients',
            instructions='Test instructions',
            prep_time=10,
            cook_time=20,
            servings=4
        )
        self.assertEqual(recipe.total_time, 30)
        self.assertTrue(recipe.slug)
        self.assertEqual(str(recipe), 'Test Recipe')
    
    def test_recipe_rating(self):
        recipe = Recipe.objects.create(
            title='Test Recipe',
            author=self.user,
            category=self.category,
            description='Test description',
            ingredients='Test ingredients',
            instructions='Test instructions',
            prep_time=10,
            cook_time=20
        )
        
        rating = RecipeRating.objects.create(
            recipe=recipe,
            user=self.user,
            rating=5,
            review='Excellent recipe!'
        )
        
        self.assertEqual(rating.rating, 5)
        self.assertEqual(recipe.ratings.count(), 1)

# tests/test_views.py
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from recipes.models import Category, Recipe

User = get_user_model()

class RecipeViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.category = Category.objects.create(
            name='Test Category'
        )
        self.recipe = Recipe.objects.create(
            title='Test Recipe',
            author=self.user,
            category=self.category,
            description='Test description',
            ingredients='Test ingredients',
            instructions='Test instructions',
            prep_time=10,
            cook_time=20
        )
    
    def test_recipe_list_view(self):
        response = self.client.get(reverse('recipe_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Recipe')
    
    def test_recipe_detail_view(self):
        response = self.client.get(
            reverse('recipe_detail', kwargs={'slug': self.recipe.slug})
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Recipe')
    
    def test_recipe_search(self):
        response = self.client.get(reverse('recipe_list'), {'search': 'Test'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Recipe')
    
    def test_recipe_creation_requires_login(self):
        response = self.client.get(reverse('recipe_create'))
        self.assertEqual(response.status_code, 302)  # Redirect to login
    
    def test_authenticated_recipe_creation(self):
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(reverse('recipe_create'))
        self.assertEqual(response.status_code, 200)

# tests/test_api.py
from rest_framework.test import APITestCase
from rest_framework import status
from django.contrib.auth import get_user_model
from django.urls import reverse
from recipes.models import Category, Recipe

User = get_user_model()

class RecipeAPITest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.category = Category.objects.create(name='Test Category')
        self.recipe = Recipe.objects.create(
            title='Test Recipe',
            author=self.user,
            category=self.category,
            description='Test description',
            ingredients='Test ingredients',
            instructions='Test instructions',
            prep_time=10,
            cook_time=20
        )
    
    def test_get_recipe_list(self):
        url = reverse('api_recipe_list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)
    
    def test_search_recipes(self):
        url = reverse('api_recipe_list')
        response = self.client.get(url, {'search': 'Test'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)
    
    def test_filter_by_category(self):
        url = reverse('api_recipe_list')
        response = self.client.get(url, {'category': self.category.id})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)
```

### 11. Deployment Configuration

```python
# deploy.py - Deployment script
import os
import subprocess
import sys

def run_command(command):
    """Run a command and return its output"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    return result.stdout

def deploy():
    """Deploy the application"""
    print("Starting deployment process...")
    
    # Collect static files
    print("Collecting static files...")
    run_command("python manage.py collectstatic --noinput")
    
    # Run migrations
    print("Running database migrations...")
    run_command("python manage.py migrate")
    
    # Create superuser if it doesn't exist
    print("Creating superuser...")
    run_command("""
        echo "from django.contrib.auth import get_user_model;
        User = get_user_model();
        User.objects.filter(username='admin').exists() or
        User.objects.create_superuser('admin', 'admin@example.com', 'admin123')" |
        python manage.py shell
    """)
    
    # Load sample data
    print("Loading sample data...")
    run_command("python manage.py populate_sample_data")
    
    print("Deployment completed successfully!")

if __name__ == "__main__":
    deploy()
```

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# Create static and media directories
RUN mkdir -p /app/staticfiles /app/media

# Collect static files
RUN python manage.py collectstatic --noinput

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "recipeshare.wsgi:application"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    environment:
      POSTGRES_DB: recipeshare
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"

  web:
    build: .
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/app
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    ports:
      - "8000:8000"
    depends_on:
      - db
    environment:
      - DEBUG=1
      - SECRET_KEY=your-secret-key-here
      - DB_NAME=recipeshare
      - DB_USER=postgres
      - DB_PASSWORD=postgres
      - DB_HOST=db
      - DB_PORT=5432

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    ports:
      - "80:80"
    depends_on:
      - web

volumes:
  postgres_data:
  static_volume:
  media_volume:
```

### 12. Performance Monitoring

```python
# middleware/performance.py
import time
import logging
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger('performance')

class PerformanceMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request.start_time = time.time()
    
    def process_response(self, request, response):
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            response['X-Response-Time'] = f"{duration:.3f}s"
            
            # Log slow requests
            if duration > 1.0:  # Log requests taking more than 1 second
                logger.warning(
                    f"Slow request: {request.method} {request.path} "
                    f"took {duration:.3f}s"
                )
        
        return response

# utils/cache.py
from django.core.cache import cache
from django.db.models import Avg, Count
from .models import Recipe, Category

def get_featured_recipes():
    """Get featured recipes with caching"""
    cache_key = 'featured_recipes'
    recipes = cache.get(cache_key)
    
    if recipes is None:
        recipes = Recipe.objects.filter(is_featured=True).select_related(
            'author', 'category'
        ).annotate(
            avg_rating=Avg('ratings__rating'),
            rating_count=Count('ratings')
        )[:6]
        
        # Cache for 1 hour
        cache.set(cache_key, recipes, 3600)
    
    return recipes

def get_popular_categories():
    """Get popular categories with caching"""
    cache_key = 'popular_categories'
    categories = cache.get(cache_key)
    
    if categories is None:
        categories = Category.objects.annotate(
            recipe_count=Count('recipes')
        ).filter(recipe_count__gt=0).order_by('-recipe_count')[:8]
        
        # Cache for 2 hours
        cache.set(cache_key, categories, 7200)
    
    return categories
```

### 13. Final Project Structure

```
recipeshare/
├── manage.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── deploy.py
├── .env.example
├── recipeshare/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── accounts/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── forms.py
│   └── urls.py
├── recipes/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── forms.py
│   ├── serializers.py
│   ├── api_views.py
│   ├── urls.py
│   ├── api_urls.py
│   ├── admin.py
│   ├── management/
│   │   └── commands/
│   │       └── populate_sample_data.py
│   └── migrations/
├── templates/
│   ├── base.html
│   ├── recipes/
│   │   ├── recipe_list.html
│   │   ├── recipe_detail.html
│   │   └── recipe_form.html
│   └── accounts/
│       ├── login.html
│       └── register.html
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── media/
├── tests/
│   ├── test_models.py
│   ├── test_views.py
│   └── test_api.py
└── middleware/
    └── performance.py
```

### 14. Quick Start Commands

```bash
# Initial setup
git clone <your-repo>
cd recipeshare
pip install -r requirements.txt

# Environment setup
cp .env.example .env
# Edit .env with your configuration

# Database setup
python manage.py makemigrations
python manage.py migrate
python manage.py populate_sample_data

# Create superuser
python manage.py createsuperuser

# Run development server
python manage.py runserver

# Run tests
python manage.py test

# Deploy with Docker
docker-compose up --build
```

This portfolio-worthy Django application demonstrates mastery through:

- **Advanced Model Relationships**: Custom user model, complex foreign keys, and calculated properties
- **Professional Views**: Class-based views, mixins, AJAX integration, and comprehensive filtering
- **REST API**: Full API with serializers, filtering, and pagination
- **Modern Frontend**: HTMX integration, responsive Bootstrap design, and interactive features
- **Performance**: Caching, query optimization, and performance monitoring
- **Testing**: Comprehensive test suite covering models, views, and API
- **Deployment**: Docker containerization and production-ready configuration
- **Code Quality**: Clean architecture, proper error handling, and documentation

The project showcases real-world Django development patterns and would serve as an excellent portfolio piece demonstrating full-stack Django expertise.

## Assignment: Restaurant Analytics Dashboard

**Objective**: Create a separate analytics module that demonstrates advanced Django concepts without overlapping with the main order management system.

### Assignment Description
Build a restaurant analytics dashboard that provides insights into restaurant performance using advanced Django features. This assignment focuses on data aggregation, reporting, and visualization.

### Requirements

#### 1. Create Analytics Models
```python
# analytics/models.py
class DailySales(models.Model):
    """Daily sales summary - like end-of-day kitchen reports"""
    date = models.DateField(unique=True)
    total_orders = models.PositiveIntegerField(default=0)
    total_revenue = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    average_order_value = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    peak_hour = models.TimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-date']

class ProductPerformance(models.Model):
    """Track how well each dish performs"""
    product = models.OneToOneField(Product, on_delete=models.CASCADE)
    total_ordered = models.PositiveIntegerField(default=0)
    total_revenue = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    last_ordered = models.DateTimeField(null=True, blank=True)
    popularity_score = models.FloatField(default=0.0)
    updated_at = models.DateTimeField(auto_now=True)
```

#### 2. Implement Data Aggregation Service
```python
# analytics/services.py
from django.db.models import Count, Sum, Avg, F
from datetime import datetime, timedelta

class AnalyticsService:
    """Service for generating restaurant insights like a data analyst"""
    
    @staticmethod
    def generate_daily_report(date):
        """Generate comprehensive daily sales report"""
        orders = Order.objects.filter(
            created_at__date=date,
            status__in=['delivered', 'ready']
        ).annotate(
            total=Sum(F('items__quantity') * F('items__unit_price'))
        )
        
        return {
            'date': date,
            'total_orders': orders.count(),
            'total_revenue': orders.aggregate(Sum('total'))['total__sum'] or 0,
            'average_order_value': orders.aggregate(Avg('total'))['total__avg'] or 0,
            'peak_hour': AnalyticsService._calculate_peak_hour(date),
            'top_products': AnalyticsService._get_top_products(date, limit=5)
        }
    
    @staticmethod
    def _calculate_peak_hour(date):
        """Find the hour with most orders"""
        from django.db.models import Extract
        
        hourly_orders = Order.objects.filter(
            created_at__date=date
        ).extra(
            select={'hour': "EXTRACT(hour FROM created_at)"}
        ).values('hour').annotate(
            count=Count('id')
        ).order_by('-count').first()
        
        return hourly_orders['hour'] if hourly_orders else None
    
    @staticmethod
    def _get_top_products(date, limit=5):
        """Get best-selling products for the day"""
        return OrderItem.objects.filter(
            order__created_at__date=date
        ).values(
            'product__name'
        ).annotate(
            total_quantity=Sum('quantity'),
            total_revenue=Sum(F('quantity') * F('unit_price'))
        ).order_by('-total_quantity')[:limit]
```

#### 3. Create Management Command
```python
# analytics/management/commands/generate_analytics.py
from django.core.management.base import BaseCommand
from datetime import date, timedelta
from analytics.services import AnalyticsService
from analytics.models import DailySales

class Command(BaseCommand):
    """Management command like a scheduled kitchen cleanup routine"""
    help = 'Generate daily analytics reports'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--date',
            type=str,
            help='Date to generate report for (YYYY-MM-DD format)'
        )
        
        parser.add_argument(
            '--days-back',
            type=int,
            default=1,
            help='Number of days back to generate reports for'
        )
    
    def handle(self, *args, **options):
        if options['date']:
            target_date = datetime.strptime(options['date'], '%Y-%m-%d').date()
            self.generate_report(target_date)
        else:
            # Generate reports for the last N days
            for i in range(options['days_back']):
                target_date = date.today() - timedelta(days=i+1)
                self.generate_report(target_date)
    
    def generate_report(self, target_date):
        """Generate report for specific date"""
        report_data = AnalyticsService.generate_daily_report(target_date)
        
        daily_sales, created = DailySales.objects.update_or_create(
            date=target_date,
            defaults={
                'total_orders': report_data['total_orders'],
                'total_revenue': report_data['total_revenue'],
                'average_order_value': report_data['average_order_value'],
                'peak_hour': report_data['peak_hour'],
            }
        )
        
        action = "Created" if created else "Updated"
        self.stdout.write(
            self.style.SUCCESS(
                f'{action} analytics report for {target_date}'
            )
        )
```

#### 4. API Endpoints for Dashboard
```python
# analytics/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import Sum, Count
from datetime import date, timedelta

class AnalyticsDashboardView(APIView):
    """API endpoint for dashboard data like a kitchen display system"""
    
    def get(self, request):
        """Get comprehensive analytics dashboard data"""
        days = int(request.query_params.get('days', 30))
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get daily sales trend
        daily_sales = DailySales.objects.filter(
            date__range=[start_date, end_date]
        ).values('date', 'total_revenue', 'total_orders')
        
        # Get product performance
        top_products = ProductPerformance.objects.order_by(
            '-total_revenue'
        )[:10].values('product__name', 'total_revenue', 'total_ordered')
        
        # Calculate summary statistics
        summary = DailySales.objects.filter(
            date__range=[start_date, end_date]
        ).aggregate(
            total_revenue=Sum('total_revenue'),
            total_orders=Sum('total_orders'),
            average_daily_revenue=Sum('total_revenue') / days if days > 0 else 0,
        )
        
        return Response({
            'summary': summary,
            'daily_sales': list(daily_sales),
            'top_products': list(top_products),
            'date_range': {
                'start': start_date,
                'end': end_date,
                'days': days
            }
        })
```

### Assignment Deliverables
1. **Complete Analytics Models**: DailySales and ProductPerformance models
2. **Service Layer**: AnalyticsService with data aggregation methods
3. **Management Command**: Command to generate daily reports
4. **API Endpoints**: Dashboard data endpoints with filtering
5. **Documentation**: Complete docstrings and usage examples
6. **Tests**: Unit tests for service methods and API endpoints

### Evaluation Criteria
- **Architecture**: Clean separation between analytics and main app
- **Performance**: Efficient database queries using aggregation
- **Documentation**: Clear explanations of analytics calculations
- **Code Quality**: Proper use of Django patterns and conventions

---

## Course Summary

Congratulations! You've completed the Django Mastery Capstone course. Like a chef who has progressed from prep cook to sous chef, you now have the skills to:

1. **Analyze and Review**: Audit Django applications for architectural improvements
2. **Refactor Efficiently**: Transform messy code into clean, maintainable solutions  
3. **Document Professionally**: Create comprehensive documentation that guides other developers
4. **Optimize Performance**: Implement caching, query optimization, and efficient patterns

These skills transform you from a Django developer into a Django craftsperson, capable of creating applications that are not just functional, but elegant, efficient, and maintainable. Your portfolio now demonstrates mastery of advanced Django concepts that separate professional developers from beginners.

Remember: great chefs never stop learning and refining their craft. Continue applying these principles to your Django projects, and you'll consistently create applications worthy of the finest digital restaurants!