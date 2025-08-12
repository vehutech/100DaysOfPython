# Day 44: QuerySets & Database Operations

## Learning Objective
By the end of this lesson, you will be able to construct complex database queries using Django's ORM, optimize query performance, and leverage aggregations to extract meaningful insights from your data—just like a master chef who knows exactly which ingredients to combine and how to prepare them efficiently for the perfect dish.

---

Imagine that you're the head chef at a world-renowned restaurant with an enormous pantry filled with thousands of ingredients. You need to quickly find specific combinations of ingredients, calculate nutritional information across multiple dishes, and optimize your kitchen operations to serve customers efficiently. 

Just as a skilled chef doesn't rummage through every shelf randomly but uses organized systems, precise measurements, and efficient techniques, Django's QuerySet system allows you to navigate your database with surgical precision—finding exactly what you need, when you need it, without wasting time or resources.

---

## Lesson 1: Complex Queries with ORM

### The Chef's Precision: Building Complex Queries

Think of Django's ORM as your sous chef—it speaks both your language (Python) and the database's language (SQL). When you want to find "all organic vegetables that cost less than $5 and are in season," you don't need to shout instructions in SQL; you can speak naturally, and your sous chef translates perfectly.

Let's work with a restaurant inventory system:

```python
# models.py
from django.db import models
from django.contrib.auth.models import User

class Category(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    
    def __str__(self):
        return self.name

class Supplier(models.Model):
    name = models.CharField(max_length=100)
    contact_email = models.EmailField()
    rating = models.IntegerField(default=5)
    
    def __str__(self):
        return self.name

class Ingredient(models.Model):
    name = models.CharField(max_length=100)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    supplier = models.ForeignKey(Supplier, on_delete=models.CASCADE)
    price_per_unit = models.DecimalField(max_digits=10, decimal_places=2)
    quantity_in_stock = models.IntegerField()
    is_organic = models.BooleanField(default=False)
    expiry_date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
```

**Syntax Explanation:**
- `models.ForeignKey`: Creates a relationship between tables (like connecting ingredients to their suppliers)
- `on_delete=models.CASCADE`: When a supplier is deleted, all their ingredients are also deleted
- `max_digits=10, decimal_places=2`: Allows numbers up to 99,999,999.99

### Basic Query Building Blocks

```python
# views.py - Your kitchen operations
from django.shortcuts import render
from django.db.models import Q
from .models import Ingredient, Category, Supplier

def kitchen_inventory_view(request):
    # Simple filtering - like asking "show me all tomatoes"
    tomatoes = Ingredient.objects.filter(name__icontains='tomato')
    
    # Complex filtering with multiple conditions
    # "Show me organic vegetables under $5 that expire after today"
    from datetime import date
    
    premium_ingredients = Ingredient.objects.filter(
        is_organic=True,
        price_per_unit__lt=5.00,
        expiry_date__gt=date.today(),
        category__name='Vegetables'
    )
    
    # Using Q objects for complex OR conditions
    # "Show me ingredients that are either organic OR from our top supplier"
    special_ingredients = Ingredient.objects.filter(
        Q(is_organic=True) | Q(supplier__rating__gte=4)
    )
    
    # Excluding items - like saying "everything except the expired ones"
    fresh_ingredients = Ingredient.objects.exclude(
        expiry_date__lt=date.today()
    )
    
    # Chaining filters - building complexity step by step
    premium_fresh_vegetables = Ingredient.objects.filter(
        category__name='Vegetables'
    ).filter(
        is_organic=True
    ).exclude(
        expiry_date__lt=date.today()
    ).order_by('-price_per_unit')
    
    context = {
        'tomatoes': tomatoes,
        'premium_ingredients': premium_ingredients,
        'special_ingredients': special_ingredients,
        'fresh_ingredients': fresh_ingredients,
        'premium_fresh_vegetables': premium_fresh_vegetables,
    }
    return render(request, 'kitchen_inventory.html', context)
```

**Syntax Explanation:**
- `__icontains`: Case-insensitive search (like `LIKE '%tomato%'` in SQL)
- `__lt`, `__gt`, `__gte`: Less than, greater than, greater than or equal to
- `Q()`: Allows complex logical operations (AND, OR, NOT)
- `|`: OR operator when using Q objects
- `exclude()`: Opposite of filter() - shows everything except what matches

---

## Lesson 2: Aggregations and Annotations

### The Kitchen Calculator: Aggregating Data

Just as a chef needs to calculate total costs, average cooking times, and ingredient quantities across multiple recipes, Django's aggregation system helps you perform calculations across your database records.

```python
# Advanced aggregations - your kitchen's analytical tools
from django.db.models import Count, Sum, Avg, Max, Min, F
from django.db.models.functions import Coalesce

def kitchen_analytics_view(request):
    # Basic aggregations - like counting your entire pantry
    total_ingredients = Ingredient.objects.count()
    total_inventory_value = Ingredient.objects.aggregate(
        total_value=Sum(F('price_per_unit') * F('quantity_in_stock'))
    )['total_value']
    
    # Category-wise analysis - like organizing by food type
    category_stats = Category.objects.annotate(
        ingredient_count=Count('ingredient'),
        total_value=Sum(F('ingredient__price_per_unit') * F('ingredient__quantity_in_stock')),
        avg_price=Avg('ingredient__price_per_unit'),
        most_expensive=Max('ingredient__price_per_unit')
    ).order_by('-total_value')
    
    # Supplier performance metrics
    supplier_performance = Supplier.objects.annotate(
        ingredient_count=Count('ingredient'),
        total_supply_value=Sum(
            F('ingredient__price_per_unit') * F('ingredient__quantity_in_stock')
        ),
        avg_ingredient_price=Avg('ingredient__price_per_unit'),
        organic_count=Count('ingredient', filter=Q(ingredient__is_organic=True))
    ).filter(ingredient_count__gt=0)
    
    # Complex conditional aggregations
    # "Show me categories with their organic vs non-organic breakdown"
    category_breakdown = Category.objects.annotate(
        total_ingredients=Count('ingredient'),
        organic_count=Count('ingredient', filter=Q(ingredient__is_organic=True)),
        non_organic_count=Count('ingredient', filter=Q(ingredient__is_organic=False)),
        organic_percentage=Coalesce(
            (Count('ingredient', filter=Q(ingredient__is_organic=True)) * 100.0) / 
            Count('ingredient'),
            0
        )
    ).filter(total_ingredients__gt=0)
    
    context = {
        'total_ingredients': total_ingredients,
        'total_inventory_value': total_inventory_value,
        'category_stats': category_stats,
        'supplier_performance': supplier_performance,
        'category_breakdown': category_breakdown,
    }
    return render(request, 'kitchen_analytics.html', context)
```

**Syntax Explanation:**
- `aggregate()`: Performs calculations across all records and returns a dictionary
- `annotate()`: Adds calculated fields to each record in the queryset
- `F()`: References field values in database calculations (more efficient than Python calculations)
- `Coalesce()`: Handles null values by providing a default
- `filter=Q()`: Conditional counting - like counting only organic ingredients

---

## Lesson 3: Query Optimization

### The Efficient Kitchen: Optimizing Performance

A master chef doesn't make unnecessary trips to the pantry. Similarly, efficient database queries minimize trips to the database server. Let's optimize our kitchen operations:

```python
# Query optimization - making your kitchen run like clockwork
from django.db.models import Prefetch

def optimized_kitchen_view(request):
    # Problem: N+1 queries - like making separate trips for each ingredient's category
    # BAD: This creates one query per ingredient to get its category
    # for ingredient in Ingredient.objects.all():
    #     print(ingredient.category.name)  # Each iteration hits the database!
    
    # Solution 1: select_related for ForeignKey relationships
    # Like bringing the category info with each ingredient in one trip
    ingredients_with_categories = Ingredient.objects.select_related(
        'category', 'supplier'
    ).all()
    
    # Solution 2: prefetch_related for reverse relationships and ManyToMany
    # Like bringing all related items in organized batches
    categories_with_ingredients = Category.objects.prefetch_related(
        'ingredient_set'
    ).all()
    
    # Advanced prefetching with custom querysets
    # Like organizing your pantry fetch with specific criteria
    categories_with_fresh_ingredients = Category.objects.prefetch_related(
        Prefetch(
            'ingredient_set',
            queryset=Ingredient.objects.filter(
                expiry_date__gt=date.today()
            ).select_related('supplier'),
            to_attr='fresh_ingredients'
        )
    )
    
    # Using only() and defer() - like bringing only what you need
    # only() - bring just these fields (like a shopping list)
    basic_ingredient_info = Ingredient.objects.only(
        'name', 'price_per_unit', 'quantity_in_stock'
    )
    
    # defer() - bring everything except these fields
    ingredients_without_descriptions = Ingredient.objects.defer(
        'category__description'
    ).select_related('category')
    
    # Bulk operations - like preparing multiple dishes at once
    # Instead of updating ingredients one by one
    expired_ingredients = Ingredient.objects.filter(
        expiry_date__lt=date.today()
    )
    
    # Bulk update - like marking all expired items at once
    expired_ingredients.update(quantity_in_stock=0)
    
    # Using exists() for boolean checks - like quickly checking if pantry has tomatoes
    has_tomatoes = Ingredient.objects.filter(name__icontains='tomato').exists()
    
    # Using values() for lightweight data retrieval
    # Like getting just the names and prices for a quick menu
    ingredient_prices = Ingredient.objects.values('name', 'price_per_unit')
    
    context = {
        'ingredients_with_categories': ingredients_with_categories,
        'categories_with_ingredients': categories_with_ingredients,
        'categories_with_fresh_ingredients': categories_with_fresh_ingredients,
        'basic_ingredient_info': basic_ingredient_info,
        'has_tomatoes': has_tomatoes,
        'ingredient_prices': ingredient_prices,
    }
    return render(request, 'optimized_kitchen.html', context)
```

**Syntax Explanation:**
- `select_related()`: Joins related tables in a single query (for ForeignKey/OneToOne)
- `prefetch_related()`: Fetches related objects in separate optimized queries
- `Prefetch()`: Customizes how related objects are fetched
- `only()`: Limits fields retrieved from database
- `defer()`: Excludes specific fields from retrieval
- `exists()`: Returns True/False without loading actual records
- `values()`: Returns dictionaries instead of model instances

---

## Lesson 4: Raw SQL When Needed

### The Master Chef's Secret Techniques: Raw SQL

Sometimes, even the most skilled sous chef (ORM) needs guidance from the master chef. When Django's ORM can't express your complex needs, raw SQL becomes your secret weapon.

```python
# Raw SQL - when you need to speak directly to the database
from django.db import connection

def advanced_kitchen_analysis(request):
    # When to use raw SQL:
    # 1. Complex window functions
    # 2. Database-specific features
    # 3. Performance-critical queries
    # 4. Complex joins that ORM can't handle elegantly
    
    # Example 1: Complex ranking query
    # "Rank ingredients by value within each category"
    ranking_query = """
    SELECT 
        i.name,
        c.name as category_name,
        i.price_per_unit * i.quantity_in_stock as total_value,
        ROW_NUMBER() OVER (
            PARTITION BY c.name 
            ORDER BY i.price_per_unit * i.quantity_in_stock DESC
        ) as value_rank
    FROM kitchen_ingredient i
    JOIN kitchen_category c ON i.category_id = c.id
    WHERE i.quantity_in_stock > 0
    ORDER BY c.name, value_rank;
    """
    
    # Method 1: Using raw() with model
    ranked_ingredients = Ingredient.objects.raw(ranking_query)
    
    # Method 2: Direct cursor execution for complex results
    with connection.cursor() as cursor:
        cursor.execute(ranking_query)
        ranking_results = cursor.fetchall()
    
    # Example 2: Complex aggregation with custom SQL
    # "Monthly ingredient purchase trends with moving averages"
    trend_query = """
    SELECT 
        DATE_TRUNC('month', created_at) as month,
        COUNT(*) as ingredients_added,
        AVG(price_per_unit) as avg_price,
        AVG(COUNT(*)) OVER (
            ORDER BY DATE_TRUNC('month', created_at)
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as three_month_avg
    FROM kitchen_ingredient
    WHERE created_at >= %s
    GROUP BY DATE_TRUNC('month', created_at)
    ORDER BY month;
    """
    
    from datetime import date, timedelta
    six_months_ago = date.today() - timedelta(days=180)
    
    with connection.cursor() as cursor:
        cursor.execute(trend_query, [six_months_ago])
        trend_results = cursor.fetchall()
    
    # Example 3: Safe parameterized queries
    # NEVER do this: f"SELECT * FROM table WHERE id = {user_input}"
    # ALWAYS do this:
    def get_ingredients_by_supplier_rating(min_rating):
        safe_query = """
        SELECT i.name, i.price_per_unit, s.name as supplier_name, s.rating
        FROM kitchen_ingredient i
        JOIN kitchen_supplier s ON i.supplier_id = s.id
        WHERE s.rating >= %s
        ORDER BY s.rating DESC, i.price_per_unit ASC;
        """
        
        with connection.cursor() as cursor:
            cursor.execute(safe_query, [min_rating])
            return cursor.fetchall()
    
    # Example 4: Mixing ORM and raw SQL
    # Use ORM for simple parts, raw SQL for complex parts
    high_rated_suppliers = Supplier.objects.filter(rating__gte=4)
    supplier_ids = list(high_rated_suppliers.values_list('id', flat=True))
    
    # Then use raw SQL for complex analysis
    complex_analysis_query = """
    SELECT 
        s.name as supplier_name,
        COUNT(i.id) as ingredient_count,
        SUM(i.price_per_unit * i.quantity_in_stock) as total_value,
        AVG(i.price_per_unit) as avg_price,
        STDDEV(i.price_per_unit) as price_variation
    FROM kitchen_supplier s
    JOIN kitchen_ingredient i ON s.id = i.supplier_id
    WHERE s.id = ANY(%s)
    GROUP BY s.id, s.name
    HAVING COUNT(i.id) > 5
    ORDER BY total_value DESC;
    """
    
    with connection.cursor() as cursor:
        cursor.execute(complex_analysis_query, [supplier_ids])
        supplier_analysis = cursor.fetchall()
    
    context = {
        'ranked_ingredients': ranked_ingredients,
        'ranking_results': ranking_results,
        'trend_results': trend_results,
        'supplier_analysis': supplier_analysis,
    }
    return render(request, 'advanced_kitchen_analysis.html', context)
```

**Syntax Explanation:**
- `raw()`: Executes raw SQL but returns model instances
- `connection.cursor()`: Direct database access for complex queries
- `%s`: Parameterized query placeholder (prevents SQL injection)
- `ROW_NUMBER() OVER()`: SQL window function for ranking
- `DATE_TRUNC()`: Database function to group by time periods
- `PARTITION BY`: Divides result set for window functions

---

# PROJECT: Advanced Expense Analytics with Django QuerySets

## Project Objective
By the end of this project, you will be able to build a comprehensive expense analytics system using Django's QuerySet API, implementing complex aggregations, annotations, and data visualizations to transform raw financial data into actionable business insights.

---

Imagine that you're the head chef of a bustling restaurant empire with multiple locations. Every day, thousands of ingredients flow through your kitchens - from expensive truffles to basic salt. You need to track every expense, understand spending patterns, identify waste, and optimize your food costs across all locations.

Just like a master chef needs to know exactly how much each dish costs to make, where the money is going, and which ingredients give the best value, a skilled Django developer needs to slice and dice database data to extract meaningful insights from raw expense records.

Today, we're going to build a sophisticated expense analytics system - think of it as your financial kitchen where raw data goes in, and perfectly prepared insights come out!

---

## The Kitchen Setup: Our Models

First, let's set up our "kitchen" with the basic ingredients (models) we'll need:

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from decimal import Decimal

class Category(models.Model):
    """Like different sections of our kitchen - produce, proteins, dairy, etc."""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    color = models.CharField(max_length=7, default='#3498db')  # For charts
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name_plural = "Categories"

class Vendor(models.Model):
    """Our suppliers - like the butcher, baker, and produce vendor"""
    name = models.CharField(max_length=200)
    contact_email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    
    def __str__(self):
        return self.name

class Expense(models.Model):
    """Each purchase record - like a receipt for every ingredient bought"""
    PAYMENT_METHODS = [
        ('cash', 'Cash'),
        ('card', 'Credit Card'),
        ('bank', 'Bank Transfer'),
        ('check', 'Check'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    vendor = models.ForeignKey(Vendor, on_delete=models.CASCADE, null=True, blank=True)
    
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField(default=timezone.now)
    payment_method = models.CharField(max_length=10, choices=PAYMENT_METHODS, default='card')
    
    receipt_image = models.ImageField(upload_to='receipts/', blank=True)
    is_recurring = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.title} - ${self.amount}"
    
    class Meta:
        ordering = ['-date', '-created_at']
```

**Syntax Explanation:**
- `models.DecimalField(max_digits=10, decimal_places=2)`: Perfect for currency - prevents floating point errors
- `models.ForeignKey(User, on_delete=models.CASCADE)`: Links each expense to a user, deletes expenses if user is deleted
- `choices=PAYMENT_METHODS`: Creates a dropdown with predefined options
- `default=timezone.now`: Uses Django's timezone-aware datetime
- `class Meta: ordering = ['-date', '-created_at']`: Default ordering by date (newest first)

---

## The Analytics Engine: Views & QuerySets

Now let's build our analytics "kitchen" where we'll prepare different types of insights:

```python
# views.py
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.db.models import Sum, Count, Avg, Max, Min, Q
from django.db.models.functions import TruncMonth, TruncWeek, TruncDay, Extract
from django.utils import timezone
from datetime import datetime, timedelta
import json
from .models import Expense, Category, Vendor

@login_required
def expense_analytics_dashboard(request):
    """
    The main kitchen where all our analytics dishes are prepared and served
    """
    # Get the date range (default to last 12 months)
    end_date = timezone.now().date()
    start_date = end_date - timedelta(days=365)
    
    # Filter expenses for the current user within date range
    user_expenses = Expense.objects.filter(
        user=request.user,
        date__range=[start_date, end_date]
    )
    
    # Recipe 1: Basic Stats - Like counting ingredients in your pantry
    total_expenses = user_expenses.aggregate(
        total_amount=Sum('amount'),
        expense_count=Count('id'),
        average_expense=Avg('amount'),
        largest_expense=Max('amount'),
        smallest_expense=Min('amount')
    )
    
    # Recipe 2: Monthly Trends - Like tracking seasonal ingredient costs
    monthly_data = user_expenses.annotate(
        month=TruncMonth('date')
    ).values('month').annotate(
        total=Sum('amount'),
        count=Count('id'),
        average=Avg('amount')
    ).order_by('month')
    
    # Recipe 3: Category Breakdown - Like organizing by kitchen sections
    category_analysis = user_expenses.values(
        'category__name', 'category__color'
    ).annotate(
        total_spent=Sum('amount'),
        transaction_count=Count('id'),
        average_per_transaction=Avg('amount'),
        percentage=Sum('amount') * 100.0 / (total_expenses['total_amount'] or 1)
    ).order_by('-total_spent')
    
    # Recipe 4: Vendor Performance - Like rating your suppliers
    vendor_analysis = user_expenses.filter(
        vendor__isnull=False
    ).values(
        'vendor__name'
    ).annotate(
        total_spent=Sum('amount'),
        transaction_count=Count('id'),
        average_per_transaction=Avg('amount'),
        last_purchase=Max('date')
    ).order_by('-total_spent')[:10]  # Top 10 vendors
    
    # Recipe 5: Payment Method Analysis - Like tracking how you pay suppliers
    payment_analysis = user_expenses.values('payment_method').annotate(
        total=Sum('amount'),
        count=Count('id')
    ).order_by('-total')
    
    # Recipe 6: Weekly Spending Pattern - Like understanding kitchen rush hours
    weekly_pattern = user_expenses.annotate(
        weekday=Extract('date', 'week_day')
    ).values('weekday').annotate(
        total=Sum('amount'),
        count=Count('id')
    ).order_by('weekday')
    
    # Convert weekday numbers to names for display
    weekday_names = {
        1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday',
        5: 'Thursday', 6: 'Friday', 7: 'Saturday'
    }
    
    for item in weekly_pattern:
        item['weekday_name'] = weekday_names[item['weekday']]
    
    # Recipe 7: Recent Big Expenses - Like flagging expensive ingredients
    big_expenses = user_expenses.filter(
        amount__gte=total_expenses['average_expense'] * 2
    ).order_by('-amount')[:5]
    
    # Recipe 8: Expense Growth Analysis - Like tracking ingredient cost inflation
    current_month = timezone.now().replace(day=1).date()
    previous_month = (current_month - timedelta(days=1)).replace(day=1)
    
    current_month_total = user_expenses.filter(
        date__gte=current_month
    ).aggregate(total=Sum('amount'))['total'] or 0
    
    previous_month_total = user_expenses.filter(
        date__gte=previous_month,
        date__lt=current_month
    ).aggregate(total=Sum('amount'))['total'] or 0
    
    growth_rate = 0
    if previous_month_total > 0:
        growth_rate = ((current_month_total - previous_month_total) / previous_month_total) * 100
    
    # Prepare chart data for the frontend
    chart_data = {
        'monthly_trends': {
            'labels': [item['month'].strftime('%B %Y') for item in monthly_data],
            'data': [float(item['total']) for item in monthly_data],
            'counts': [item['count'] for item in monthly_data]
        },
        'category_pie': {
            'labels': [item['category__name'] for item in category_analysis],
            'data': [float(item['total_spent']) for item in category_analysis],
            'colors': [item['category__color'] for item in category_analysis]
        },
        'weekly_pattern': {
            'labels': [item['weekday_name'] for item in weekly_pattern],
            'data': [float(item['total']) for item in weekly_pattern]
        }
    }
    
    context = {
        'total_expenses': total_expenses,
        'monthly_data': monthly_data,
        'category_analysis': category_analysis,
        'vendor_analysis': vendor_analysis,
        'payment_analysis': payment_analysis,
        'weekly_pattern': weekly_pattern,
        'big_expenses': big_expenses,
        'growth_rate': round(growth_rate, 2),
        'current_month_total': current_month_total,
        'previous_month_total': previous_month_total,
        'chart_data_json': json.dumps(chart_data),
        'date_range': f"{start_date.strftime('%B %Y')} - {end_date.strftime('%B %Y')}"
    }
    
    return render(request, 'expenses/analytics_dashboard.html', context)
```

**Syntax Explanation:**
- `aggregate(Sum('amount'))`: Calculates totals across all records
- `annotate(month=TruncMonth('date'))`: Adds a computed field grouping by month
- `values('category__name')`: Follows foreign key relationships using double underscores
- `Q` objects: For complex queries with AND/OR logic (imported but not used in this example)
- `__range=[start_date, end_date]`: Date range filtering
- `__gte`: Greater than or equal to (also `__lte`, `__lt`, `__gt`)
- `__isnull=False`: Filters out null values

---

## The Presentation Layer: Templates

Let's create a beautiful dashboard template - like plating our analytics "dishes":

```html
<!-- templates/expenses/analytics_dashboard.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Expense Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .chart-container {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .table-container {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #667eea;
            color: white;
            font-weight: bold;
        }
        .growth-positive { color: #28a745; }
        .growth-negative { color: #dc3545; }
        .growth-neutral { color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🍳 Expense Analytics Kitchen</h1>
            <p>Your financial ingredients analyzed and served fresh</p>
            <p><strong>Period:</strong> {{ date_range }}</p>
        </div>

        <!-- Key Statistics - Like your kitchen's vital signs -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${{ total_expenses.total_amount|floatformat:2 }}</div>
                <div>Total Expenses</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ total_expenses.expense_count }}</div>
                <div>Transactions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${{ total_expenses.average_expense|floatformat:2 }}</div>
                <div>Average per Transaction</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">
                    <span class="{% if growth_rate > 0 %}growth-positive{% elif growth_rate < 0 %}growth-negative{% else %}growth-neutral{% endif %}">
                        {{ growth_rate }}%
                    </span>
                </div>
                <div>Monthly Growth</div>
            </div>
        </div>

        <!-- Charts Grid -->
        <div class="chart-grid">
            <!-- Monthly Trends Chart -->
            <div class="chart-container">
                <h3>📈 Monthly Spending Trends</h3>
                <canvas id="monthlyChart"></canvas>
            </div>
            
            <!-- Category Breakdown -->
            <div class="chart-container">
                <h3>🥧 Expense Categories</h3>
                <canvas id="categoryChart"></canvas>
            </div>
        </div>

        <!-- Weekly Pattern Chart -->
        <div class="chart-container">
            <h3>📅 Weekly Spending Pattern</h3>
            <canvas id="weeklyChart"></canvas>
        </div>

        <!-- Category Analysis Table -->
        <div class="table-container">
            <h3>📊 Category Analysis</h3>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Total Spent</th>
                        <th>Transactions</th>
                        <th>Average per Transaction</th>
                        <th>Percentage of Total</th>
                    </tr>
                </thead>
                <tbody>
                    {% for category in category_analysis %}
                    <tr>
                        <td>
                            <span style="display: inline-block; width: 12px; height: 12px; background: {{ category.category__color }}; border-radius: 50%; margin-right: 8px;"></span>
                            {{ category.category__name }}
                        </td>
                        <td>${{ category.total_spent|floatformat:2 }}</td>
                        <td>{{ category.transaction_count }}</td>
                        <td>${{ category.average_per_transaction|floatformat:2 }}</td>
                        <td>{{ category.percentage|floatformat:1 }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Top Vendors -->
        <div class="table-container">
            <h3>🏪 Top Vendors</h3>
            <table>
                <thead>
                    <tr>
                        <th>Vendor</th>
                        <th>Total Spent</th>
                        <th>Transactions</th>
                        <th>Average per Transaction</th>
                        <th>Last Purchase</th>
                    </tr>
                </thead>
                <tbody>
                    {% for vendor in vendor_analysis %}
                    <tr>
                        <td>{{ vendor.vendor__name }}</td>
                        <td>${{ vendor.total_spent|floatformat:2 }}</td>
                        <td>{{ vendor.transaction_count }}</td>
                        <td>${{ vendor.average_per_transaction|floatformat:2 }}</td>
                        <td>{{ vendor.last_purchase|date:"M j, Y" }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Get chart data from Django backend
        const chartData = {{ chart_data_json|safe }};
        
        // Monthly Trends Chart
        const monthlyCtx = document.getElementById('monthlyChart').getContext('2d');
        new Chart(monthlyCtx, {
            type: 'line',
            data: {
                labels: chartData.monthly_trends.labels,
                datasets: [{
                    label: 'Monthly Expenses',
                    data: chartData.monthly_trends.data,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Like tracking seasonal ingredient costs'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });

        // Category Pie Chart
        const categoryCtx = document.getElementById('categoryChart').getContext('2d');
        new Chart(categoryCtx, {
            type: 'pie',
            data: {
                labels: chartData.category_pie.labels,
                datasets: [{
                    data: chartData.category_pie.data,
                    backgroundColor: chartData.category_pie.colors
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Like organizing your kitchen sections'
                    }
                }
            }
        });

        // Weekly Pattern Chart
        const weeklyCtx = document.getElementById('weeklyChart').getContext('2d');
        new Chart(weeklyCtx, {
            type: 'bar',
            data: {
                labels: chartData.weekly_pattern.labels,
                datasets: [{
                    label: 'Weekly Spending',
                    data: chartData.weekly_pattern.data,
                    backgroundColor: 'rgba(118, 75, 162, 0.8)',
                    borderColor: '#764ba2',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Like understanding kitchen rush hours'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
```

**Syntax Explanation:**
- `{{ total_expenses.total_amount|floatformat:2 }}`: Django template filter for formatting decimals
- `{% if growth_rate > 0 %}...{% elif %}...{% else %}...{% endif %}`: Template conditional logic
- `{% for category in category_analysis %}...{% endfor %}`: Template loop
- `{{ chart_data_json|safe }}`: The `safe` filter prevents HTML escaping for JSON data
- `|date:"M j, Y"`: Date formatting filter

---

## The URL Configuration

```python
# urls.py
from django.urls import path
from . import views

app_name = 'expenses'

urlpatterns = [
    path('analytics/', views.expense_analytics_dashboard, name='analytics_dashboard'),
]
```

---

## Assignment: Restaurant Inventory Optimization Challenge

**Scenario:** You're the new head chef at "The Gourmet Garden," a farm-to-table restaurant. The previous chef left behind a disorganized inventory system, and you need to create a comprehensive analysis system to optimize operations.

**Your Task:** Create a Django view called `inventory_optimization_dashboard` that provides the following insights:

1. **Supplier Performance Analysis:**
   - Rank suppliers by total inventory value they provide
   - Calculate the percentage of organic ingredients each supplier provides
   - Identify suppliers with ingredients expiring within the next 7 days

2. **Category Optimization:**
   - Find categories that have the highest average price per unit
   - Calculate inventory turnover potential (total value / number of items)
   - Identify categories with low stock levels (less than 10 total items)

3. **Cost Optimization:**
   - Find the top 5 most expensive ingredients per category
   - Calculate potential savings by identifying organic alternatives that cost less than non-organic equivalents
   - Create a "budget-friendly" ingredient list (under $3 per unit, in stock > 20)

**Requirements:**
- Use at least 3 different aggregation functions
- Include at least one complex Q object query
- Implement proper query optimization techniques
- Create meaningful variable names that a real chef would understand
- Add comments explaining your "kitchen logic"

**Bonus Challenge:** Write one raw SQL query that finds ingredients that are both above the average price in their category AND have above-average stock levels.

**Deliverables:**
- Complete Django view function
- Brief explanation of your optimization strategy
- Sample output data structure showing what insights your dashboard provides

This assignment tests your ability to combine all four concepts: complex ORM queries, aggregations, optimization techniques, and knowing when raw SQL might be necessary. Think like a chef who needs actionable insights to run an efficient, profitable kitchen!

---

## Key Takeaways

Like a master chef who knows their kitchen inside and out, you now understand how to:

1. **Navigate complex ingredient relationships** using Django's ORM filtering and Q objects
2. **Calculate kitchen metrics** with aggregations and annotations
3. **Optimize your database trips** like an efficient chef minimizes pantry visits
4. **Speak directly to the database** when your sous chef (ORM) needs guidance

Remember: The best chefs don't just cook—they understand their ingredients, optimize their processes, and know when to use traditional techniques alongside modern tools. Your database skills should follow the same philosophy!