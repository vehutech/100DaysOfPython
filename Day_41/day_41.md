# Day 41: Django Models & Database Design

## Learning Objective
By the end of this lesson, you will be able to design and implement Django models with proper relationships, create and run database migrations, and build a complete personal expense tracker application with a functional admin interface.

---

## Introduction: The Kitchen Blueprint

Imagine that you're the head chef of a world-class restaurant, and you've just been handed the keys to a brand new kitchen. But here's the thing - this kitchen is completely empty. No counters, no storage systems, no organization whatsoever. You have all the ingredients (your data) but nowhere to put them and no system to keep track of what goes where.

This is exactly what happens when you start a new Django project. You have the framework (the empty kitchen), but you need to design the storage systems, organization methods, and workflows that will make your restaurant run smoothly. In Django, these storage systems are called **models** - they're the blueprint for how your data will be organized, stored, and related to each other.

Just like a chef needs different containers for different ingredients - spice racks for seasonings, refrigerators for perishables, and pantries for dry goods - your Django application needs different models for different types of data, each with their own specific fields and relationships.

---

## Part 1: Django ORM Fundamentals

### The Recipe Database Concept

Think of Django's ORM (Object-Relational Mapping) as your kitchen's inventory management system. Just as a chef needs to know what ingredients are available, how much of each ingredient they have, and where everything is stored, Django's ORM helps you manage your data.

In our kitchen analogy:
- **Models** = Different types of storage containers (spice rack, refrigerator, pantry)
- **Fields** = The specific compartments or attributes of each container
- **Relationships** = How different storage systems connect (recipes use ingredients from multiple containers)

### Your First Model: The Ingredient

Let's start by creating a simple model. In Django, every model is like designing a new type of storage container for your kitchen:

```python
# models.py
from django.db import models

class Ingredient(models.Model):
    name = models.CharField(max_length=100)
    category = models.CharField(max_length=50)
    cost_per_unit = models.DecimalField(max_digits=10, decimal_places=2)
    supplier = models.CharField(max_length=100)
    date_added = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
```

This model is like designing a smart ingredient container that automatically tracks:
- What the ingredient is (`name`)
- What category it belongs to (`category`)
- How much it costs (`cost_per_unit`)
- Who supplies it (`supplier`)
- When it was added to your inventory (`date_added`)

---

## Part 2: Model Fields and Relationships

### The Spice Rack System

Just as a professional kitchen has different storage solutions for different needs, Django provides various field types for different kinds of data:

```python
from django.db import models
from django.contrib.auth.models import User

class Recipe(models.Model):
    # Text fields - like labels on containers
    title = models.CharField(max_length=200)
    description = models.TextField()
    
    # Number fields - like measuring cups
    servings = models.IntegerField()
    prep_time = models.PositiveIntegerField()  # in minutes
    difficulty_rating = models.DecimalField(max_digits=3, decimal_places=1)
    
    # Date fields - like expiration dates
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Boolean fields - like on/off switches
    is_vegetarian = models.BooleanField(default=False)
    is_published = models.BooleanField(default=True)
    
    # Choice fields - like pre-set temperature settings
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    ]
    difficulty = models.CharField(max_length=6, choices=DIFFICULTY_CHOICES)
    
    def __str__(self):
        return self.title
```

### Kitchen Relationships: How Everything Connects

In a real kitchen, recipes connect to ingredients, chefs create recipes, and customers order dishes. Django models work the same way:

```python
class Chef(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    specialty = models.CharField(max_length=100)
    years_experience = models.IntegerField()
    
    def __str__(self):
        return f"Chef {self.user.username}"

class Recipe(models.Model):
    title = models.CharField(max_length=200)
    chef = models.ForeignKey(Chef, on_delete=models.CASCADE)  # Many recipes, one chef
    ingredients = models.ManyToManyField(Ingredient, through='RecipeIngredient')
    
    def __str__(self):
        return self.title

class RecipeIngredient(models.Model):
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE)
    ingredient = models.ForeignKey(Ingredient, on_delete=models.CASCADE)
    quantity = models.DecimalField(max_digits=10, decimal_places=2)
    unit = models.CharField(max_length=20)  # cups, tbsp, etc.
    
    def __str__(self):
        return f"{self.quantity} {self.unit} of {self.ingredient.name}"
```

**Relationship Types Explained:**
- **OneToOneField**: Like a chef's personal knife set - one chef, one knife set
- **ForeignKey**: Like recipes by a chef - one chef can have many recipes
- **ManyToManyField**: Like ingredients in recipes - one recipe uses many ingredients, one ingredient can be in many recipes

---

## Part 3: Database Migrations

### The Kitchen Renovation Process

Think of database migrations as renovating your kitchen. When you decide to add a new appliance (field) or rearrange your storage (change a model), you need to plan the renovation carefully so you don't break anything or lose your existing ingredients (data).

Django's migration system is like having a professional contractor who:
1. Creates a detailed renovation plan (`makemigrations`)
2. Safely executes the renovation (`migrate`)
3. Keeps a record of all changes made

### Creating Your First Migration

```bash
# Generate migration files (create renovation plans)
python manage.py makemigrations

# Apply migrations (execute the renovation)
python manage.py migrate

# See migration status (check renovation progress)
python manage.py showmigrations
```

### Example Migration Scenario

Let's say you want to add a new field to track whether ingredients are organic:

```python
# Add this to your Ingredient model
is_organic = models.BooleanField(default=False)
```

Then run:
```bash
python manage.py makemigrations
python manage.py migrate
```

Django creates a migration file like this:
```python
# migrations/0002_ingredient_is_organic.py
from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('your_app', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='ingredient',
            name='is_organic',
            field=models.BooleanField(default=False),
        ),
    ]
```

---

## Part 4: Django Admin Interface

### The Kitchen Manager's Dashboard

Imagine having a smart kitchen manager who can instantly show you all your ingredients, recipes, and orders in an organized, easy-to-use interface. That's exactly what Django Admin provides - a pre-built, professional interface for managing your data.

### Setting Up Admin Access

```python
# admin.py
from django.contrib import admin
from .models import Ingredient, Recipe, Chef, RecipeIngredient

# Basic admin registration
admin.site.register(Ingredient)

# Advanced admin customization
@admin.register(Recipe)
class RecipeAdmin(admin.ModelAdmin):
    list_display = ['title', 'chef', 'difficulty', 'prep_time', 'is_published']
    list_filter = ['difficulty', 'is_vegetarian', 'is_published', 'created_at']
    search_fields = ['title', 'description']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'description', 'chef')
        }),
        ('Recipe Details', {
            'fields': ('servings', 'prep_time', 'difficulty')
        }),
        ('Options', {
            'fields': ('is_vegetarian', 'is_published')
        }),
    )

@admin.register(Chef)
class ChefAdmin(admin.ModelAdmin):
    list_display = ['user', 'specialty', 'years_experience']
    search_fields = ['user__username', 'specialty']
```

### Creating a Superuser (Head Chef)

```bash
python manage.py createsuperuser
```

---

## Part 5: Building the Personal Expense Tracker

### The Restaurant Budget Management System

Now let's apply everything we've learned to build a personal expense tracker - think of it as the financial management system for your restaurant. Just as a chef needs to track ingredient costs, supplier payments, and kitchen expenses, we'll create a system to track personal expenses.

### Core Models

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Category(models.Model):
    """Like different departments in a restaurant: Kitchen, Front of House, Marketing"""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    color = models.CharField(max_length=7, default='#007bff')  # Hex color for UI
    
    class Meta:
        verbose_name_plural = "Categories"
    
    def __str__(self):
        return self.name

class Expense(models.Model):
    """Individual expense items - like each ingredient purchase or equipment buy"""
    PAYMENT_METHODS = [
        ('cash', 'Cash'),
        ('card', 'Credit/Debit Card'),
        ('bank', 'Bank Transfer'),
        ('digital', 'Digital Wallet'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField(default=timezone.now)
    description = models.TextField(blank=True)
    payment_method = models.CharField(max_length=10, choices=PAYMENT_METHODS)
    receipt_image = models.ImageField(upload_to='receipts/', blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date', '-created_at']
    
    def __str__(self):
        return f"{self.title} - ${self.amount}"

class Budget(models.Model):
    """Monthly budget limits - like setting spending limits for each department"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    month = models.DateField()
    
    class Meta:
        unique_together = ['user', 'category', 'month']
    
    def __str__(self):
        return f"{self.category.name} Budget - {self.month.strftime('%B %Y')}"
    
    def get_spent_amount(self):
        """Calculate how much has been spent in this category this month"""
        return self.user.expense_set.filter(
            category=self.category,
            date__year=self.month.year,
            date__month=self.month.month
        ).aggregate(total=models.Sum('amount'))['total'] or 0
    
    def get_remaining_amount(self):
        """Calculate remaining budget"""
        return self.amount - self.get_spent_amount()
    
    def is_over_budget(self):
        """Check if spending exceeds budget"""
        return self.get_spent_amount() > self.amount
```

### Enhanced Admin Interface

```python
# admin.py
from django.contrib import admin
from django.utils.html import format_html
from .models import Category, Expense, Budget

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'colored_name', 'description']
    search_fields = ['name']
    
    def colored_name(self, obj):
        return format_html(
            '<span style="color: {};">{}</span>',
            obj.color,
            obj.name
        )
    colored_name.short_description = 'Color Preview'

@admin.register(Expense)
class ExpenseAdmin(admin.ModelAdmin):
    list_display = ['title', 'amount', 'category', 'date', 'payment_method', 'user']
    list_filter = ['category', 'payment_method', 'date', 'user']
    search_fields = ['title', 'description']
    date_hierarchy = 'date'
    ordering = ['-date']
    
    fieldsets = (
        ('Expense Details', {
            'fields': ('title', 'amount', 'category', 'date')
        }),
        ('Payment Information', {
            'fields': ('payment_method', 'description')
        }),
        ('Additional', {
            'fields': ('receipt_image',),
            'classes': ('collapse',)
        }),
    )

@admin.register(Budget)
class BudgetAdmin(admin.ModelAdmin):
    list_display = ['category', 'month', 'amount', 'spent_amount', 'remaining_amount', 'status']
    list_filter = ['category', 'month']
    ordering = ['-month']
    
    def spent_amount(self, obj):
        return f"${obj.get_spent_amount():.2f}"
    
    def remaining_amount(self, obj):
        remaining = obj.get_remaining_amount()
        color = 'red' if remaining < 0 else 'green'
        return format_html(
            '<span style="color: {};">${:.2f}</span>',
            color,
            remaining
        )
    
    def status(self, obj):
        if obj.is_over_budget():
            return format_html('<span style="color: red;">Over Budget</span>')
        return format_html('<span style="color: green;">Within Budget</span>')
```

### Views and Templates

```python
# views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.db.models import Sum
from django.utils import timezone
from .models import Expense, Category, Budget
from .forms import ExpenseForm

@login_required
def dashboard(request):
    """Main dashboard - like the kitchen's central command center"""
    current_month = timezone.now().date().replace(day=1)
    
    # Get current month's expenses
    monthly_expenses = Expense.objects.filter(
        user=request.user,
        date__year=current_month.year,
        date__month=current_month.month
    )
    
    # Calculate totals by category
    category_totals = monthly_expenses.values('category__name').annotate(
        total=Sum('amount')
    )
    
    # Get recent expenses
    recent_expenses = Expense.objects.filter(user=request.user)[:5]
    
    # Get budget status
    budgets = Budget.objects.filter(user=request.user, month=current_month)
    
    context = {
        'monthly_total': monthly_expenses.aggregate(Sum('amount'))['amount__sum'] or 0,
        'category_totals': category_totals,
        'recent_expenses': recent_expenses,
        'budgets': budgets,
    }
    
    return render(request, 'expenses/dashboard.html', context)

@login_required
def add_expense(request):
    """Add new expense - like recording a new ingredient purchase"""
    if request.method == 'POST':
        form = ExpenseForm(request.POST, request.FILES)
        if form.is_valid():
            expense = form.save(commit=False)
            expense.user = request.user
            expense.save()
            return redirect('dashboard')
    else:
        form = ExpenseForm()
    
    return render(request, 'expenses/add_expense.html', {'form': form})
```

### Simple Forms

```python
# forms.py
from django import forms
from .models import Expense

class ExpenseForm(forms.ModelForm):
    class Meta:
        model = Expense
        fields = ['title', 'amount', 'category', 'date', 'payment_method', 'description', 'receipt_image']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
            'description': forms.Textarea(attrs={'rows': 3}),
        }
```

---

## Assignment: Personal Recipe Cost Calculator

### The Challenge

Create a model that extends our expense tracker to specifically track the cost of cooking recipes at home. This combines both the kitchen analogy and the expense tracking concepts we've learned.

### Requirements

1. **Create a new model called `RecipeCost`** that tracks:
   - Recipe name
   - Ingredients used (you can simplify this as a text field listing ingredients)
   - Total cost to make the recipe
   - Cost per serving
   - Date cooked
   - Notes about the cooking experience

2. **Add proper relationships** to connect this with your existing expense tracking system

3. **Create admin interface** with proper list views and filters

4. **Add a method** to calculate cost per serving automatically

5. **Include at least 3 sample recipes** with realistic costs

### Starter Code Structure

```python
class RecipeCost(models.Model):
    # Your implementation here
    pass
```

### Bonus Challenges
- Add a photo field for the finished dish
- Create a method to compare homemade vs. restaurant costs
- Add a rating system for how much you enjoyed the recipe

---

## Course Summary

Congratulations! You've just designed and built a complete data management system for Django. Like a master chef who has organized their kitchen from scratch, you now understand how to:

- **Design models** that properly represent real-world data relationships
- **Use Django's field types** to store different kinds of information
- **Create and manage database migrations** safely
- **Build administrative interfaces** for data management
- **Implement a complete application** with proper relationships and functionality

Your personal expense tracker is now ready to help you manage your finances just like a professional kitchen management system helps chefs run their restaurants efficiently. The skills you've learned today form the foundation for building any data-driven Django application.

Remember: good database design is like good kitchen organization - it makes everything else easier and more efficient!