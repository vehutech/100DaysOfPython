# Day 58: Database Optimization - From Chaos to Culinary Excellence

## Learning Objective
By the end of this lesson, you will be able to optimize database performance by implementing proper indexing strategies, writing efficient queries, managing connection pools, and executing safe production migrations - transforming your database from a chaotic kitchen into a Michelin-starred restaurant operation.

---

## Introduction: The Kitchen Analogy

Imagine that your database is like a massive restaurant kitchen during the dinner rush. Without proper organization, chefs are bumping into each other, ingredients are scattered everywhere, orders are backing up, and customers are getting frustrated. This is exactly what happens to your application when your database isn't optimized.

Today, we're going to transform your chaotic kitchen into a well-oiled culinary machine where every ingredient (data) has its place, every recipe (query) is perfected, and every chef (connection) works efficiently without stepping on each other's toes.

---

## Lesson 1: Database Indexing - Organizing Your Kitchen

### The Kitchen Story
Picture a chef frantically searching through every single spice jar in a disorganized pantry just to find oregano. That's what happens when your database scans every single row to find the data you need. Now imagine that same pantry with perfectly labeled shelves, alphabetically organized spices, and a master inventory list - that's what database indexing does for your data.

### What Are Database Indexes?
Database indexes are like the organized spice rack in our kitchen - they create shortcuts to find data quickly without scanning every single record.

### Code Example: Creating Indexes

```python
# models.py - Django Example
from django.db import models

class Recipe(models.Model):
    name = models.CharField(max_length=200)
    cuisine_type = models.CharField(max_length=100)
    difficulty_level = models.IntegerField()
    prep_time = models.IntegerField()  # in minutes
    chef = models.ForeignKey('Chef', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        # Single column index - like organizing spices alphabetically
        indexes = [
            models.Index(fields=['cuisine_type']),
            models.Index(fields=['difficulty_level']),
            models.Index(fields=['prep_time']),
            # Composite index - like organizing by cuisine AND difficulty
            models.Index(fields=['cuisine_type', 'difficulty_level']),
            # Partial index - only index recipes under 30 minutes
            models.Index(fields=['prep_time'], 
                        condition=models.Q(prep_time__lt=30))
        ]

# Raw SQL equivalent for PostgreSQL
"""
CREATE INDEX idx_recipe_cuisine ON recipes(cuisine_type);
CREATE INDEX idx_recipe_difficulty ON recipes(difficulty_level);
CREATE INDEX idx_recipe_composite ON recipes(cuisine_type, difficulty_level);
CREATE INDEX idx_quick_recipes ON recipes(prep_time) WHERE prep_time < 30;
"""
```

### When to Use Indexes
- **Frequently queried columns**: Like searching for recipes by cuisine type
- **JOIN conditions**: When connecting recipes to chefs
- **ORDER BY clauses**: When sorting recipes by prep time
- **WHERE conditions**: When filtering recipes by difficulty

### When NOT to Use Indexes
- **Rarely queried columns**: Don't index chef's favorite color
- **Frequently updated columns**: Each update requires index maintenance
- **Small tables**: A spice rack with 3 spices doesn't need organization

---

## Lesson 2: Query Optimization - Perfecting Your Recipes

### The Kitchen Story
Think of a recipe that says "add some salt" versus one that says "add 1 teaspoon of kosher salt". The first is vague and might ruin the dish, while the second is precise and efficient. Similarly, poorly written database queries are like vague recipes - they waste time and resources.

### Query Optimization Principles

```python
# Bad Query - The Vague Recipe
# This is like asking "bring me all ingredients and I'll find what I need"
def get_italian_quick_recipes_bad():
    all_recipes = Recipe.objects.all()  # Fetches ALL recipes
    italian_quick = []
    for recipe in all_recipes:
        if recipe.cuisine_type == 'Italian' and recipe.prep_time < 30:
            italian_quick.append(recipe)
    return italian_quick

# Good Query - The Precise Recipe
# This is like asking "bring me exactly what I need"
def get_italian_quick_recipes_good():
    return Recipe.objects.filter(
        cuisine_type='Italian',
        prep_time__lt=30
    ).select_related('chef')  # Joins chef data in one query

# Even Better - Using database functions
def get_recipe_stats_optimized():
    from django.db.models import Count, Avg, Q
    
    return Recipe.objects.aggregate(
        total_recipes=Count('id'),
        avg_prep_time=Avg('prep_time'),
        quick_recipes=Count('id', filter=Q(prep_time__lt=30)),
        italian_recipes=Count('id', filter=Q(cuisine_type='Italian'))
    )
```

### Advanced Query Optimization

```python
# Using database-level operations instead of Python loops
def update_recipe_difficulty_optimized():
    from django.db.models import Case, When, IntegerField
    
    Recipe.objects.update(
        difficulty_level=Case(
            When(prep_time__lt=30, then=1),  # Easy
            When(prep_time__lt=60, then=2),  # Medium
            default=3,  # Hard
            output_field=IntegerField()
        )
    )

# Batch operations - like prep work before service
def create_recipes_in_batches(recipe_data):
    recipes = [Recipe(**data) for data in recipe_data]
    Recipe.objects.bulk_create(recipes, batch_size=100)
```

### Query Performance Analysis

```python
# Analyzing query performance - like timing your cooking
from django.db import connection

def analyze_query_performance():
    with connection.cursor() as cursor:
        cursor.execute("EXPLAIN ANALYZE SELECT * FROM recipes WHERE cuisine_type = 'Italian'")
        results = cursor.fetchall()
        for row in results:
            print(row)
```

---

## Lesson 3: Database Connection Pooling - Managing Your Kitchen Staff

### The Kitchen Story
Imagine hiring a new chef every time you need to cook a single dish, then firing them immediately after. That's expensive and inefficient! Instead, successful restaurants maintain a team of chefs who work different shifts. Database connection pooling works the same way - it maintains a pool of database connections that can be reused instead of creating new ones for every request.

### Understanding Connection Pooling

```python
# settings.py - Django Configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'restaurant_db',
        'USER': 'chef_admin',
        'PASSWORD': 'secret_recipe',
        'HOST': 'localhost',
        'PORT': '5432',
        'OPTIONS': {
            'MAX_CONNS': 20,  # Maximum chefs in the kitchen
            'MIN_CONNS': 5,   # Minimum chefs always on duty
        }
    }
}

# Using connection pooling with SQLAlchemy (Python)
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

def create_database_engine():
    return create_engine(
        'postgresql://chef_admin:secret_recipe@localhost:5432/restaurant_db',
        poolclass=QueuePool,
        pool_size=10,        # Normal staff size
        max_overflow=20,     # Additional staff during rush
        pool_pre_ping=True,  # Check if chefs are still working
        pool_recycle=3600    # Rotate staff every hour
    )

# Connection pool monitoring
def monitor_connection_pool(engine):
    pool = engine.pool
    print(f"Pool size: {pool.size()}")
    print(f"Connections in use: {pool.checkedin()}")
    print(f"Available connections: {pool.checkedout()}")
```

### Best Practices for Connection Pooling

```python
# Context manager for proper connection handling
from contextlib import contextmanager

@contextmanager
def get_database_connection():
    """Like assigning a chef to a specific task"""
    conn = None
    try:
        conn = engine.connect()
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()  # Chef returns to the pool

# Usage example
def create_new_recipe(recipe_data):
    with get_database_connection() as conn:
        result = conn.execute(
            "INSERT INTO recipes (name, cuisine_type, prep_time) VALUES (%s, %s, %s)",
            (recipe_data['name'], recipe_data['cuisine_type'], recipe_data['prep_time'])
        )
        return result.lastrowid
```

---

## Lesson 4: Database Migrations in Production - Renovating a Busy Kitchen

### The Kitchen Story
Imagine trying to renovate your restaurant kitchen while it's serving 200 customers during dinner rush. You can't just shut down and rebuild - you need to carefully plan each change, test it thoroughly, and execute it with minimal disruption. That's exactly what production database migrations are like.

### Safe Migration Strategies

```python
# migrations/0001_add_nutrition_info.py - Django Migration
from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('recipes', '0001_initial'),
    ]

    operations = [
        # Safe operation - Adding nullable columns is like adding new storage
        migrations.AddField(
            model_name='recipe',
            name='calories',
            field=models.IntegerField(null=True, blank=True),
        ),
        
        # Safe operation - Adding indexes during low traffic
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY idx_recipe_calories ON recipes(calories);",
            reverse_sql="DROP INDEX idx_recipe_calories;"
        ),
    ]

# Dangerous migration example (DON'T DO THIS)
class BadMigration(migrations.Migration):
    operations = [
        # This locks the table - like blocking the kitchen entrance
        migrations.AlterField(
            model_name='recipe',
            name='name',
            field=models.CharField(max_length=100),  # Reducing from 200
        ),
    ]
```

### Blue-Green Deployment Strategy

```python
# Blue-Green migration approach
def safe_column_rename_migration():
    """
    Step 1: Add new column (like installing new equipment alongside old)
    Step 2: Dual-write to both columns
    Step 3: Backfill old data
    Step 4: Switch reads to new column
    Step 5: Remove old column
    """
    
    # Migration 1: Add new column
    operations = [
        migrations.AddField(
            model_name='recipe',
            name='recipe_title',  # New column
            field=models.CharField(max_length=200, null=True),
        ),
    ]
    
    # Migration 2: Populate new column
    operations = [
        migrations.RunSQL(
            "UPDATE recipes SET recipe_title = name WHERE recipe_title IS NULL;",
            reverse_sql="UPDATE recipes SET recipe_title = NULL;"
        ),
    ]
    
    # Migration 3: Make new column non-null
    operations = [
        migrations.AlterField(
            model_name='recipe',
            name='recipe_title',
            field=models.CharField(max_length=200),
        ),
    ]
    
    # Migration 4: Remove old column (after code is updated)
    operations = [
        migrations.RemoveField(
            model_name='recipe',
            name='name',
        ),
    ]

# Pre-migration checks
def pre_migration_checklist():
    """Like checking all equipment before service"""
    checks = [
        "Database backup completed?",
        "Migration tested on staging?",
        "Rollback plan prepared?",
        "Monitoring alerts configured?",
        "Low traffic period scheduled?",
        "Team notified of maintenance window?"
    ]
    
    for check in checks:
        print(f"☐ {check}")
```

### Migration Monitoring

```python
# Monitor migration progress
def monitor_migration_progress():
    """Like watching the kitchen during renovation"""
    import time
    import psycopg2
    
    def check_table_locks():
        conn = psycopg2.connect(database="restaurant_db")
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT relation::regclass, mode, granted 
            FROM pg_locks 
            WHERE relation::regclass::text LIKE 'recipes%'
        """)
        
        locks = cursor.fetchall()
        for lock in locks:
            print(f"Table: {lock[0]}, Mode: {lock[1]}, Granted: {lock[2]}")
        
        conn.close()
    
    def check_migration_status():
        # Check if long-running queries are blocking
        conn = psycopg2.connect(database="restaurant_db")
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
            FROM pg_stat_activity 
            WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
        """)
        
        long_queries = cursor.fetchall()
        for query in long_queries:
            print(f"Long query (PID {query[0]}): {query[1]} - {query[2][:100]}...")
        
        conn.close()
```

---

## Syntax Explanation

### Django ORM Syntax Used:
- **`models.Index(fields=['column'])`**: Creates database indexes for faster queries
- **`select_related('foreign_key')`**: Performs SQL JOIN to reduce database hits
- **`filter(condition)`**: Adds WHERE clauses to queries
- **`aggregate()`**: Performs database-level calculations (COUNT, AVG, etc.)
- **`bulk_create()`**: Inserts multiple records in a single query

### SQL Syntax Used:
- **`CREATE INDEX CONCURRENTLY`**: Creates indexes without blocking table access
- **`EXPLAIN ANALYZE`**: Shows query execution plan and performance metrics
- **`UPDATE ... SET ... WHERE`**: Modifies existing records with conditions

### PostgreSQL-Specific Features:
- **`pg_locks`**: System view showing current database locks
- **`pg_stat_activity`**: System view showing current database activity
- **Connection pooling parameters**: Manages database connection reuse

---

# Build: Performance-Optimized Models

## Learning Objective
Build a complete restaurant management system with performance-optimized Django models that can handle high-volume operations efficiently.

## Project: Restaurant Chain Management System

Imagine that you're the head chef of a rapidly growing restaurant chain. Your kitchen has evolved from a small bistro to a multi-location empire, and your old recipe management system is starting to crack under pressure. Orders are piling up, ingredient tracking is slow, and your staff is getting frustrated with the sluggish system. It's time to rebuild your kitchen's digital backbone with performance-optimized models that can handle the heat of a busy restaurant empire.

### The Kitchen Architecture

Just like a well-organized kitchen has designated stations for different tasks, our Django models need to be structured for optimal performance. Let's build a system that can handle thousands of orders, complex ingredient relationships, and real-time inventory tracking.

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db.models import Index, Q
from django.utils import timezone
import uuid

class Restaurant(models.Model):
    """The main kitchen - each restaurant location"""
    name = models.CharField(max_length=200, db_index=True)
    address = models.TextField()
    phone = models.CharField(max_length=20)
    is_active = models.BooleanField(default=True, db_index=True)
    opening_time = models.TimeField()
    closing_time = models.TimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            Index(fields=['name', 'is_active']),
            Index(fields=['created_at']),
        ]
        
    def __str__(self):
        return self.name

class CategoryManager(models.Manager):
    """Custom manager for optimized category queries"""
    def active_categories(self):
        return self.filter(is_active=True).select_related()
    
    def with_item_count(self):
        return self.annotate(
            item_count=models.Count('menuitem')
        ).filter(item_count__gt=0)

class Category(models.Model):
    """Recipe categories - like appetizers, mains, desserts"""
    name = models.CharField(max_length=100, unique=True, db_index=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True, db_index=True)
    sort_order = models.PositiveIntegerField(default=0, db_index=True)
    
    objects = CategoryManager()
    
    class Meta:
        verbose_name_plural = "Categories"
        ordering = ['sort_order', 'name']
        indexes = [
            Index(fields=['is_active', 'sort_order']),
        ]
    
    def __str__(self):
        return self.name

class IngredientManager(models.Manager):
    """Smart ingredient management like a sous chef"""
    def low_stock(self, threshold=10):
        return self.filter(
            current_stock__lte=threshold,
            is_active=True
        ).select_related('supplier')
    
    def by_supplier(self, supplier_id):
        return self.filter(
            supplier_id=supplier_id,
            is_active=True
        ).only('name', 'current_stock', 'unit_price')

class Ingredient(models.Model):
    """Individual ingredients - the building blocks of every dish"""
    UNIT_CHOICES = [
        ('kg', 'Kilogram'),
        ('g', 'Gram'),
        ('l', 'Liter'),
        ('ml', 'Milliliter'),
        ('pcs', 'Pieces'),
        ('cups', 'Cups'),
    ]
    
    name = models.CharField(max_length=200, db_index=True)
    unit = models.CharField(max_length=10, choices=UNIT_CHOICES)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    current_stock = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    minimum_stock = models.DecimalField(max_digits=10, decimal_places=2, default=10)
    supplier = models.ForeignKey('Supplier', on_delete=models.CASCADE, db_index=True)
    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    objects = IngredientManager()
    
    class Meta:
        indexes = [
            Index(fields=['name', 'is_active']),
            Index(fields=['supplier', 'is_active']),
            Index(fields=['current_stock']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.unit})"
    
    @property
    def is_low_stock(self):
        return self.current_stock <= self.minimum_stock

class Supplier(models.Model):
    """Ingredient suppliers - your trusted vendors"""
    name = models.CharField(max_length=200, unique=True, db_index=True)
    contact_person = models.CharField(max_length=100)
    phone = models.CharField(max_length=20)
    email = models.EmailField()
    address = models.TextField()
    is_active = models.BooleanField(default=True, db_index=True)
    rating = models.PositiveIntegerField(
        default=5,
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    
    class Meta:
        indexes = [
            Index(fields=['name', 'is_active']),
            Index(fields=['rating']),
        ]
    
    def __str__(self):
        return self.name

class MenuItemManager(models.Manager):
    """Optimized menu item queries like a master chef's recipe book"""
    def available_items(self):
        return self.filter(
            is_available=True,
            category__is_active=True
        ).select_related('category').prefetch_related('ingredients')
    
    def popular_items(self, limit=10):
        return self.filter(
            is_available=True
        ).annotate(
            order_count=models.Count('orderitem')
        ).order_by('-order_count')[:limit]
    
    def by_price_range(self, min_price, max_price):
        return self.filter(
            price__gte=min_price,
            price__lte=max_price,
            is_available=True
        ).select_related('category')

class MenuItem(models.Model):
    """Individual dishes - the heart of your menu"""
    name = models.CharField(max_length=200, db_index=True)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2, db_index=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, db_index=True)
    ingredients = models.ManyToManyField(Ingredient, through='Recipe')
    preparation_time = models.PositiveIntegerField(help_text="Time in minutes")
    calories = models.PositiveIntegerField(null=True, blank=True)
    is_available = models.BooleanField(default=True, db_index=True)
    is_featured = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    objects = MenuItemManager()
    
    class Meta:
        indexes = [
            Index(fields=['name', 'is_available']),
            Index(fields=['category', 'is_available']),
            Index(fields=['price']),
            Index(fields=['is_featured', 'is_available']),
        ]
    
    def __str__(self):
        return self.name
    
    @property
    def can_be_prepared(self):
        """Check if all ingredients are available"""
        for recipe in self.recipe_set.all():
            if recipe.ingredient.current_stock < recipe.quantity:
                return False
        return True

class Recipe(models.Model):
    """The bridge between dishes and ingredients - like a recipe card"""
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE)
    ingredient = models.ForeignKey(Ingredient, on_delete=models.CASCADE)
    quantity = models.DecimalField(max_digits=10, decimal_places=2)
    notes = models.TextField(blank=True)
    
    class Meta:
        unique_together = ['menu_item', 'ingredient']
        indexes = [
            Index(fields=['menu_item', 'ingredient']),
        ]
    
    def __str__(self):
        return f"{self.menu_item.name} - {self.ingredient.name}"

class OrderManager(models.Manager):
    """Optimized order management like an efficient kitchen manager"""
    def todays_orders(self):
        today = timezone.now().date()
        return self.filter(
            created_at__date=today
        ).select_related('restaurant', 'customer').prefetch_related('items__menu_item')
    
    def pending_orders(self):
        return self.filter(
            status__in=['pending', 'preparing']
        ).select_related('restaurant').prefetch_related('items__menu_item')
    
    def by_restaurant(self, restaurant_id):
        return self.filter(
            restaurant_id=restaurant_id
        ).select_related('customer').prefetch_related('items__menu_item')

class Order(models.Model):
    """Customer orders - the tickets that come into your kitchen"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('preparing', 'Preparing'),
        ('ready', 'Ready'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    order_number = models.CharField(max_length=20, unique=True, db_index=True)
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE, db_index=True)
    customer = models.ForeignKey(User, on_delete=models.CASCADE, db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', db_index=True)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    estimated_delivery = models.DateTimeField(null=True, blank=True)
    special_instructions = models.TextField(blank=True)
    
    objects = OrderManager()
    
    class Meta:
        indexes = [
            Index(fields=['restaurant', 'status']),
            Index(fields=['customer', 'created_at']),
            Index(fields=['created_at']),
            Index(fields=['status', 'created_at']),
        ]
    
    def __str__(self):
        return f"Order {self.order_number}"
    
    def save(self, *args, **kwargs):
        if not self.order_number:
            self.order_number = f"ORD{timezone.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:8].upper()}"
        super().save(*args, **kwargs)

class OrderItem(models.Model):
    """Individual items within an order - like items on a kitchen ticket"""
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    subtotal = models.DecimalField(max_digits=10, decimal_places=2)
    special_requests = models.TextField(blank=True)
    
    class Meta:
        indexes = [
            Index(fields=['order', 'menu_item']),
        ]
    
    def save(self, *args, **kwargs):
        self.subtotal = self.quantity * self.unit_price
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.quantity}x {self.menu_item.name}"

# Performance monitoring model
class KitchenMetrics(models.Model):
    """Track kitchen performance like a head chef monitoring service"""
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    date = models.DateField(db_index=True)
    orders_completed = models.PositiveIntegerField(default=0)
    average_prep_time = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    revenue = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    peak_hour_start = models.TimeField(null=True, blank=True)
    peak_hour_end = models.TimeField(null=True, blank=True)
    
    class Meta:
        unique_together = ['restaurant', 'date']
        indexes = [
            Index(fields=['restaurant', 'date']),
            Index(fields=['date']),
        ]
    
    def __str__(self):
        return f"{self.restaurant.name} - {self.date}"
```

### Views with Performance Focus

```python
# views.py
from django.shortcuts import render, get_object_or_404
from django.db.models import Count, Sum, Avg, F, Q
from django.core.paginator import Paginator
from django.views.generic import ListView
from django.utils import timezone
from .models import *

class OptimizedMenuView(ListView):
    """Display menu items with optimized queries"""
    model = MenuItem
    template_name = 'menu/menu_list.html'
    context_object_name = 'menu_items'
    paginate_by = 20
    
    def get_queryset(self):
        return MenuItem.objects.available_items().annotate(
            order_count=Count('orderitem')
        ).order_by('-order_count', 'name')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.with_item_count()
        context['featured_items'] = MenuItem.objects.filter(
            is_featured=True, 
            is_available=True
        ).select_related('category')[:6]
        return context

def kitchen_dashboard(request, restaurant_id):
    """Kitchen dashboard with performance metrics"""
    restaurant = get_object_or_404(Restaurant, id=restaurant_id)
    
    # Get today's orders with optimized queries
    today_orders = Order.objects.filter(
        restaurant=restaurant,
        created_at__date=timezone.now().date()
    ).select_related('customer').prefetch_related('items__menu_item')
    
    # Kitchen metrics
    metrics = {
        'pending_orders': today_orders.filter(status='pending').count(),
        'preparing_orders': today_orders.filter(status='preparing').count(),
        'completed_orders': today_orders.filter(status='delivered').count(),
        'total_revenue': today_orders.aggregate(
            total=Sum('total_amount')
        )['total'] or 0,
        'avg_prep_time': today_orders.filter(
            status='delivered'
        ).aggregate(
            avg_time=Avg('estimated_delivery')
        )['avg_time'],
        'low_stock_ingredients': Ingredient.objects.low_stock(threshold=5).count(),
    }
    
    # Popular items today
    popular_items = MenuItem.objects.filter(
        orderitem__order__in=today_orders
    ).annotate(
        order_count=Count('orderitem')
    ).order_by('-order_count')[:10]
    
    context = {
        'restaurant': restaurant,
        'metrics': metrics,
        'pending_orders': today_orders.filter(status='pending')[:10],
        'popular_items': popular_items,
    }
    
    return render(request, 'kitchen/dashboard.html', context)

def inventory_alert(request):
    """Alert system for low stock ingredients"""
    low_stock_ingredients = Ingredient.objects.low_stock(threshold=10)
    
    # Group by supplier for efficient ordering
    suppliers_with_low_stock = Supplier.objects.filter(
        ingredient__in=low_stock_ingredients
    ).annotate(
        low_stock_count=Count('ingredient')
    ).prefetch_related('ingredient_set')
    
    context = {
        'low_stock_ingredients': low_stock_ingredients,
        'suppliers': suppliers_with_low_stock,
    }
    
    return render(request, 'inventory/alerts.html', context)
```

### Advanced Performance Features

```python
# management/commands/optimize_kitchen.py
from django.core.management.base import BaseCommand
from django.db import connection
from django.utils import timezone
from datetime import timedelta
from myapp.models import KitchenMetrics, Order, Restaurant

class Command(BaseCommand):
    help = 'Optimize kitchen operations and generate performance reports'
    
    def handle(self, *args, **options):
        # Update kitchen metrics for all restaurants
        for restaurant in Restaurant.objects.filter(is_active=True):
            self.update_kitchen_metrics(restaurant)
        
        # Clean up old order data (older than 1 year)
        cutoff_date = timezone.now() - timedelta(days=365)
        old_orders = Order.objects.filter(created_at__lt=cutoff_date)
        deleted_count = old_orders.count()
        old_orders.delete()
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully optimized kitchen data. Deleted {deleted_count} old orders.')
        )
    
    def update_kitchen_metrics(self, restaurant):
        """Update daily metrics for a restaurant"""
        today = timezone.now().date()
        
        # Get today's completed orders
        today_orders = Order.objects.filter(
            restaurant=restaurant,
            created_at__date=today,
            status='delivered'
        )
        
        if today_orders.exists():
            metrics, created = KitchenMetrics.objects.get_or_create(
                restaurant=restaurant,
                date=today,
                defaults={
                    'orders_completed': today_orders.count(),
                    'revenue': today_orders.aggregate(total=Sum('total_amount'))['total'] or 0,
                    'average_prep_time': 25.5,  # This would be calculated from actual prep times
                }
            )
            
            if not created:
                metrics.orders_completed = today_orders.count()
                metrics.revenue = today_orders.aggregate(total=Sum('total_amount'))['total'] or 0
                metrics.save()
```

### Performance Testing Utils

```python
# utils/performance.py
from django.test import TestCase
from django.test.utils import override_settings
from django.db import connection
from django.core.management import call_command
import time

class PerformanceTestCase(TestCase):
    """Test performance of our optimized models"""
    
    def setUp(self):
        # Create test data
        self.create_test_data()
    
    def create_test_data(self):
        """Create realistic test data"""
        # Create restaurants
        restaurant = Restaurant.objects.create(
            name="Test Kitchen",
            address="123 Test St",
            phone="555-0123",
            opening_time="09:00",
            closing_time="22:00"
        )
        
        # Create categories and menu items
        category = Category.objects.create(name="Mains")
        
        # Create 1000 menu items for testing
        MenuItem.objects.bulk_create([
            MenuItem(
                name=f"Dish {i}",
                description=f"Description for dish {i}",
                price=10.99 + i,
                category=category,
                preparation_time=15 + (i % 20)
            ) for i in range(1000)
        ])
    
    def test_menu_query_performance(self):
        """Test optimized menu queries"""
        start_time = time.time()
        
        # Test the optimized query
        menu_items = list(MenuItem.objects.available_items()[:50])
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Should complete in under 0.1 seconds
        self.assertLess(query_time, 0.1, f"Query took {query_time} seconds")
        
        # Check number of queries
        with self.assertNumQueries(1):
            list(MenuItem.objects.available_items()[:10])
    
    def test_order_dashboard_performance(self):
        """Test kitchen dashboard query performance"""
        with self.assertNumQueries(5):  # Should use minimal queries
            # Simulate dashboard data retrieval
            pending_orders = Order.objects.pending_orders()[:10]
            popular_items = MenuItem.objects.popular_items(limit=5)
            low_stock = Ingredient.objects.low_stock()[:5]
```

This performance-optimized restaurant management system demonstrates how to build Django models that can handle high-volume operations efficiently. The kitchen metaphor runs throughout - from organizing ingredients like a sous chef to managing orders like an efficient kitchen manager.

The system includes optimized database queries, custom managers, strategic indexing, and performance monitoring - all essential ingredients for a scalable Django application that won't slow down when the kitchen gets busy.

## Assignment: The Restaurant Database Optimization Challenge

### Scenario
You've inherited a restaurant management system that's running slowly. The previous developer left behind a database with 50,000+ recipes, 1,000+ chefs, and no optimization. During peak hours, customers are waiting 30+ seconds for recipe searches, and the system is frequently crashing due to connection issues.

### Your Mission
Optimize the database to handle peak traffic efficiently. You need to:

1. **Analyze the current performance issues** using the provided slow queries
2. **Implement proper indexing strategy** for the most common search patterns
3. **Optimize the three worst-performing queries** provided in the starter code
4. **Set up connection pooling** to handle concurrent users
5. **Create a migration plan** to add a new "dietary_restrictions" field safely

### Starter Code

```python
# Current problematic queries that need optimization
def slow_query_1():
    """Find all Italian recipes under 30 minutes by chef experience"""
    # This query takes 15+ seconds
    return Recipe.objects.filter(
        cuisine_type='Italian',
        prep_time__lt=30
    ).select_related('chef').filter(
        chef__experience_years__gte=5
    ).order_by('prep_time')

def slow_query_2():
    """Get recipe statistics by cuisine type"""
    # This query takes 20+ seconds
    cuisines = Recipe.objects.values_list('cuisine_type', flat=True).distinct()
    stats = {}
    for cuisine in cuisines:
        recipes = Recipe.objects.filter(cuisine_type=cuisine)
        stats[cuisine] = {
            'count': recipes.count(),
            'avg_prep_time': sum(r.prep_time for r in recipes) / recipes.count(),
            'difficulty_avg': sum(r.difficulty_level for r in recipes) / recipes.count()
        }
    return stats

def slow_query_3():
    """Find recipes with similar ingredients"""
    # This query often times out
    target_recipe = Recipe.objects.get(id=1)
    similar_recipes = []
    for recipe in Recipe.objects.exclude(id=1):
        if recipe.ingredients.filter(
            name__in=target_recipe.ingredients.values_list('name', flat=True)
        ).count() >= 3:
            similar_recipes.append(recipe)
    return similar_recipes
```

### Deliverables
1. **Optimized queries** with explanations of what you changed and why
2. **Database migration file** showing your indexing strategy
3. **Connection pooling configuration** with appropriate settings
4. **Performance comparison** showing before/after query times
5. **Migration plan document** for safely adding the dietary_restrictions field

### Success Criteria
- Recipe searches complete in under 2 seconds
- System handles 100 concurrent users without connection errors
- All migrations can be safely applied during business hours
- Database performance improves by at least 80% overall

### Bonus Challenge
Implement a caching layer that works like a "mise en place" station in the kitchen - pre-prepared ingredients (cached query results) that make cooking (serving requests) faster during rush hours.

---

## Conclusion

Just like transforming a chaotic kitchen into a world-class restaurant operation, database optimization requires careful planning, the right tools, and constant monitoring. You've learned to organize your data with indexes (like a well-organized pantry), write efficient queries (like precise recipes), manage connections properly (like skilled staff management), and deploy changes safely (like renovating during off-hours).

Your database should now run like a Michelin-starred kitchen - efficient, reliable, and ready to handle any rush that comes its way!