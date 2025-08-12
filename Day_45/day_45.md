# Django Debugging & Testing Masterclass
## Day 45: Debugging & Testing Setup

### Learning Objectives
By the end of this lesson, you will be able to:
- Implement effective Django debugging techniques to identify and fix application issues
- Write comprehensive unit tests that validate your Django application's functionality
- Create and utilize test fixtures and factories for consistent test data
- Generate and interpret coverage reports to ensure thorough testing
- Apply kitchen/chef analogies to understand debugging and testing concepts

---

Imagine that you're the head chef of a bustling restaurant kitchen. Every dish that leaves your kitchen represents a feature in your Django application. Just as a chef needs to taste-test dishes, check ingredient quality, and ensure consistent preparation methods, a Django developer needs debugging tools to identify problems and testing frameworks to ensure everything works perfectly before serving it to users.

In our culinary world of Django development, debugging is like being a food critic in your own kitchen - you need to identify what's wrong with a dish and fix it. Testing is like having a systematic quality control process where you check every recipe, every ingredient, and every cooking technique to ensure they work consistently every time.

---

## Lesson 1: Django Debugging Techniques

### The Kitchen Detective: Understanding Django Debugging

Just as a chef investigates why a soufflé collapsed or why a sauce separated, Django developers need systematic approaches to find and fix bugs in their applications.

#### Setting Up Django Debug Mode

First, let's ensure your Django kitchen has the right tools for investigation:

```python
# settings.py
DEBUG = True  # Only in development!
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# Add these for better debugging
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'debug_toolbar',  # Django Debug Toolbar
    'your_app',
]

MIDDLEWARE = [
    'debug_toolbar.middleware.DebugToolbarMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Debug toolbar configuration
INTERNAL_IPS = [
    '127.0.0.1',
]
```

**Syntax Explanation:**
- `DEBUG = True`: Enables Django's debug mode, showing detailed error pages
- `INSTALLED_APPS`: List of Django applications, including debug_toolbar for enhanced debugging
- `MIDDLEWARE`: Order matters! DebugToolbarMiddleware should be near the top
- `INTERNAL_IPS`: Tells debug toolbar which IP addresses can see the debug information

#### Using Python's Built-in Debugger

Think of `pdb` as your magnifying glass in the kitchen - it lets you examine every ingredient at each step:

```python
# views.py
import pdb
from django.shortcuts import render
from django.http import JsonResponse
from .models import Recipe

def create_recipe(request):
    if request.method == 'POST':
        recipe_name = request.POST.get('name')
        ingredients = request.POST.get('ingredients')
        
        # Set a breakpoint - like stopping mid-recipe to taste
        pdb.set_trace()
        
        # Check what we're working with
        print(f"Recipe name: {recipe_name}")
        print(f"Ingredients: {ingredients}")
        
        recipe = Recipe.objects.create(
            name=recipe_name,
            ingredients=ingredients
        )
        
        return JsonResponse({'success': True, 'recipe_id': recipe.id})
    
    return render(request, 'create_recipe.html')
```

**Syntax Explanation:**
- `import pdb`: Imports Python's debugger
- `pdb.set_trace()`: Creates a breakpoint where execution pauses
- When hit, you can inspect variables, step through code line by line
- Common pdb commands: `n` (next line), `s` (step into), `c` (continue), `l` (list code), `p variable_name` (print variable)

#### Django Logging: Your Kitchen Journal

Just as chefs keep logs of what works and what doesn't, Django logging helps track application behavior:

```python
# settings.py
import os

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'debug.log'),
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'your_app': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}
```

**Using logging in your views:**

```python
# views.py
import logging
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

logger = logging.getLogger(__name__)

@login_required
def chef_dashboard(request):
    logger.info(f"Chef {request.user.username} accessed dashboard")
    
    try:
        recipes = Recipe.objects.filter(chef=request.user)
        logger.debug(f"Found {recipes.count()} recipes for chef {request.user.username}")
    except Exception as e:
        logger.error(f"Error fetching recipes: {str(e)}")
        recipes = []
    
    return render(request, 'chef_dashboard.html', {'recipes': recipes})
```

**Syntax Explanation:**
- `LOGGING` dictionary: Configures how Django handles logging
- `formatters`: Define how log messages look
- `handlers`: Where logs go (file, console, etc.)
- `loggers`: Which parts of your app log what level of detail
- `logger.info()`, `logger.debug()`, `logger.error()`: Different log levels for different types of messages

---

## Lesson 2: Writing Unit Tests

### Quality Control in the Kitchen: Understanding Unit Tests

Just as a chef tests each component of a dish separately (the sauce, the protein, the garnish), unit tests check individual parts of your Django application in isolation.

#### Setting Up Your Test Kitchen

```python
# tests/test_models.py
from django.test import TestCase
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from decimal import Decimal
from ..models import Recipe, Ingredient

class RecipeModelTest(TestCase):
    """Test the Recipe model like testing a recipe card"""
    
    def setUp(self):
        """Prepare ingredients for testing - like mise en place"""
        self.chef = User.objects.create_user(
            username='chef_gordon',
            email='gordon@kitchen.com',
            password='seasoning123'
        )
        
        self.recipe_data = {
            'name': 'Perfect Pasta',
            'chef': self.chef,
            'prep_time': 15,
            'cook_time': 12,
            'servings': 4,
            'difficulty': 'easy'
        }
    
    def test_recipe_creation(self):
        """Test that we can create a recipe - like following a recipe card"""
        recipe = Recipe.objects.create(**self.recipe_data)
        
        self.assertEqual(recipe.name, 'Perfect Pasta')
        self.assertEqual(recipe.chef, self.chef)
        self.assertEqual(recipe.prep_time, 15)
        self.assertEqual(recipe.total_time, 27)  # prep + cook time
        self.assertTrue(recipe.slug)  # Auto-generated slug
    
    def test_recipe_string_representation(self):
        """Test the recipe displays correctly - like reading a menu"""
        recipe = Recipe.objects.create(**self.recipe_data)
        expected_string = f"Perfect Pasta by {self.chef.username}"
        self.assertEqual(str(recipe), expected_string)
    
    def test_recipe_validation(self):
        """Test that invalid recipes are rejected - like quality control"""
        invalid_data = self.recipe_data.copy()
        invalid_data['prep_time'] = -5  # Negative prep time doesn't make sense
        
        with self.assertRaises(ValidationError):
            recipe = Recipe(**invalid_data)
            recipe.full_clean()  # This triggers validation
```

**Syntax Explanation:**
- `TestCase`: Base class for Django unit tests
- `setUp()`: Runs before each test method, like preparing ingredients
- `test_*` methods: Each method starting with 'test_' is a separate test
- `assertEqual()`: Checks if two values are equal
- `assertTrue()`: Checks if a condition is True
- `assertRaises()`: Checks that a specific exception is raised
- `full_clean()`: Triggers Django model validation

#### Testing Views: The Taste Test

```python
# tests/test_views.py
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.http import JsonResponse
from ..models import Recipe

class RecipeViewTest(TestCase):
    """Test views like a food critic testing dishes"""
    
    def setUp(self):
        """Set up test kitchen"""
        self.client = Client()
        self.chef = User.objects.create_user(
            username='test_chef',
            email='test@kitchen.com',
            password='testpass123'
        )
        
        self.recipe = Recipe.objects.create(
            name='Test Recipe',
            chef=self.chef,
            prep_time=10,
            cook_time=20,
            servings=2,
            difficulty='easy'
        )
    
    def test_recipe_list_view(self):
        """Test the recipe list like reviewing a menu"""
        response = self.client.get(reverse('recipe_list'))
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Recipe')
        self.assertContains(response, self.chef.username)
    
    def test_recipe_detail_view(self):
        """Test individual recipe page like examining a dish"""
        response = self.client.get(
            reverse('recipe_detail', kwargs={'slug': self.recipe.slug})
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.recipe.name)
        self.assertEqual(response.context['recipe'], self.recipe)
    
    def test_create_recipe_authenticated(self):
        """Test recipe creation when chef is logged in"""
        self.client.login(username='test_chef', password='testpass123')
        
        recipe_data = {
            'name': 'New Recipe',
            'prep_time': 5,
            'cook_time': 15,
            'servings': 3,
            'difficulty': 'medium'
        }
        
        response = self.client.post(reverse('recipe_create'), recipe_data)
        
        self.assertEqual(response.status_code, 302)  # Redirect after creation
        self.assertTrue(Recipe.objects.filter(name='New Recipe').exists())
    
    def test_create_recipe_anonymous(self):
        """Test that anonymous users can't create recipes"""
        recipe_data = {
            'name': 'Unauthorized Recipe',
            'prep_time': 5,
            'cook_time': 15,
            'servings': 3,
            'difficulty': 'medium'
        }
        
        response = self.client.post(reverse('recipe_create'), recipe_data)
        
        # Should redirect to login
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Recipe.objects.filter(name='Unauthorized Recipe').exists())
```

**Syntax Explanation:**
- `Client()`: Django's test client for making requests
- `reverse()`: Gets URL from URL name (like `recipe_list`)
- `assertContains()`: Checks if response contains specific text
- `assertRedirects()`: Checks if response redirects to expected URL
- `response.context`: Access template context variables
- `client.login()`: Simulate user login for testing

---

## Lesson 3: Test Fixtures and Factories

### Standardizing Your Ingredients: Fixtures and Factories

Just as a chef might pre-prepare standard ingredients (mise en place) for consistent cooking, test fixtures and factories provide consistent test data.

#### Django Fixtures: Pre-made Ingredients

```python
# fixtures/test_data.json
[
    {
        "model": "auth.user",
        "pk": 1,
        "fields": {
            "username": "chef_alice",
            "email": "alice@kitchen.com",
            "first_name": "Alice",
            "last_name": "Chef",
            "is_staff": false,
            "is_active": true,
            "date_joined": "2024-01-01T00:00:00Z"
        }
    },
    {
        "model": "recipes.recipe",
        "pk": 1,
        "fields": {
            "name": "Classic Marinara",
            "chef": 1,
            "prep_time": 10,
            "cook_time": 25,
            "servings": 6,
            "difficulty": "easy",
            "ingredients": "Tomatoes, garlic, olive oil, basil",
            "instructions": "Sauté garlic, add tomatoes, simmer 25 minutes"
        }
    }
]
```

**Using fixtures in tests:**

```python
# tests/test_with_fixtures.py
from django.test import TestCase
from ..models import Recipe

class RecipeWithFixturesTest(TestCase):
    fixtures = ['test_data.json']  # Load our pre-made ingredients
    
    def test_recipe_exists(self):
        """Test that our fixture recipe exists"""
        recipe = Recipe.objects.get(name='Classic Marinara')
        self.assertEqual(recipe.chef.username, 'chef_alice')
        self.assertEqual(recipe.prep_time, 10)
```

#### Factory Boy: Your Sous Chef for Test Data

Factory Boy creates test data programmatically, like having a sous chef who can make any ingredient on demand:

```python
# factories.py
import factory
from django.contrib.auth.models import User
from factory.django import DjangoModelFactory
from .models import Recipe, Ingredient

class UserFactory(DjangoModelFactory):
    """Factory for creating test chefs"""
    class Meta:
        model = User
    
    username = factory.Sequence(lambda n: f'chef_{n}')
    email = factory.LazyAttribute(lambda obj: f'{obj.username}@kitchen.com')
    first_name = factory.Faker('first_name')
    last_name = factory.Faker('last_name')
    is_active = True

class RecipeFactory(DjangoModelFactory):
    """Factory for creating test recipes"""
    class Meta:
        model = Recipe
    
    name = factory.Faker('sentence', nb_words=2)
    chef = factory.SubFactory(UserFactory)
    prep_time = factory.Faker('random_int', min=5, max=30)
    cook_time = factory.Faker('random_int', min=10, max=60)
    servings = factory.Faker('random_int', min=1, max=8)
    difficulty = factory.Faker('random_element', elements=['easy', 'medium', 'hard'])
    ingredients = factory.Faker('text', max_nb_chars=200)
    instructions = factory.Faker('text', max_nb_chars=500)

class IngredientFactory(DjangoModelFactory):
    """Factory for creating test ingredients"""
    class Meta:
        model = Ingredient
    
    name = factory.Faker('word')
    quantity = factory.Faker('pydecimal', left_digits=2, right_digits=2, positive=True)
    unit = factory.Faker('random_element', elements=['cups', 'tbsp', 'tsp', 'lbs', 'oz'])
    recipe = factory.SubFactory(RecipeFactory)
```

**Using factories in tests:**

```python
# tests/test_with_factories.py
from django.test import TestCase
from .factories import UserFactory, RecipeFactory, IngredientFactory

class RecipeFactoryTest(TestCase):
    """Test using factories like having a sous chef prepare ingredients"""
    
    def test_single_recipe_creation(self):
        """Test creating one recipe with factory"""
        recipe = RecipeFactory()
        
        self.assertTrue(recipe.name)
        self.assertTrue(recipe.chef)
        self.assertGreater(recipe.prep_time, 0)
    
    def test_multiple_recipes_creation(self):
        """Test creating multiple recipes - like meal prep"""
        recipes = RecipeFactory.create_batch(5)
        
        self.assertEqual(len(recipes), 5)
        # Each recipe should have a unique name
        names = [recipe.name for recipe in recipes]
        self.assertEqual(len(names), len(set(names)))
    
    def test_recipe_with_custom_chef(self):
        """Test creating recipe with specific chef"""
        master_chef = UserFactory(username='master_chef')
        recipe = RecipeFactory(chef=master_chef)
        
        self.assertEqual(recipe.chef.username, 'master_chef')
    
    def test_recipe_with_ingredients(self):
        """Test creating recipe with ingredients"""
        recipe = RecipeFactory()
        ingredients = IngredientFactory.create_batch(3, recipe=recipe)
        
        self.assertEqual(recipe.ingredients.count(), 3)
        for ingredient in ingredients:
            self.assertEqual(ingredient.recipe, recipe)
```

**Syntax Explanation:**
- `DjangoModelFactory`: Base class for creating Django model factories
- `factory.Sequence()`: Creates sequential values (chef_1, chef_2, etc.)
- `factory.LazyAttribute()`: Creates values based on other attributes
- `factory.Faker()`: Uses Faker library to generate realistic fake data
- `factory.SubFactory()`: Creates related objects (like a chef for each recipe)
- `create_batch()`: Creates multiple instances at once

---

## Lesson 4: Coverage Reporting

### Kitchen Inspection: Measuring Test Coverage

Just as a health inspector checks every corner of a kitchen, coverage reporting shows which parts of your code are tested and which areas need attention.

#### Installing and Configuring Coverage

```bash
# Install coverage tool
pip install coverage

# Create .coveragerc configuration file
```

```ini
# .coveragerc
[run]
source = .
omit = 
    */venv/*
    */migrations/*
    */tests/*
    manage.py
    */settings/*
    */wsgi.py
    */asgi.py
    */urls.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:

[html]
directory = htmlcov
```

#### Running Coverage Analysis

```python
# models.py - Example model with coverage annotations
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

class Recipe(models.Model):
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    ]
    
    name = models.CharField(max_length=200)
    chef = models.ForeignKey(User, on_delete=models.CASCADE)
    prep_time = models.PositiveIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(300)]
    )
    cook_time = models.PositiveIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(480)]
    )
    servings = models.PositiveIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(20)]
    )
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def total_time(self):
        """Calculate total cooking time"""
        return self.prep_time + self.cook_time
    
    def difficulty_emoji(self):  # pragma: no cover
        """Return emoji for difficulty - not critical to test"""
        emoji_map = {
            'easy': '😊',
            'medium': '😐',
            'hard': '😰'
        }
        return emoji_map.get(self.difficulty, '🤷')
    
    def is_quick_meal(self):
        """Check if recipe is a quick meal (under 30 minutes)"""
        return self.total_time() <= 30
    
    def __str__(self):
        return f"{self.name} by {self.chef.username}"
```

**Comprehensive test with coverage in mind:**

```python
# tests/test_coverage_example.py
from django.test import TestCase
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .factories import UserFactory, RecipeFactory

class RecipeCoverageTest(TestCase):
    """Tests designed to achieve high coverage"""
    
    def setUp(self):
        self.chef = UserFactory(username='coverage_chef')
    
    def test_recipe_total_time(self):
        """Test total time calculation"""
        recipe = RecipeFactory(prep_time=15, cook_time=20)
        self.assertEqual(recipe.total_time(), 35)
    
    def test_recipe_is_quick_meal_true(self):
        """Test quick meal identification - true case"""
        recipe = RecipeFactory(prep_time=10, cook_time=15)
        self.assertTrue(recipe.is_quick_meal())
    
    def test_recipe_is_quick_meal_false(self):
        """Test quick meal identification - false case"""
        recipe = RecipeFactory(prep_time=20, cook_time=25)
        self.assertFalse(recipe.is_quick_meal())
    
    def test_recipe_string_representation(self):
        """Test string representation"""
        recipe = RecipeFactory(name="Test Recipe", chef=self.chef)
        expected = "Test Recipe by coverage_chef"
        self.assertEqual(str(recipe), expected)
    
    def test_recipe_validation_errors(self):
        """Test validation catches errors"""
        with self.assertRaises(ValidationError):
            recipe = RecipeFactory.build(prep_time=0)  # Invalid prep time
            recipe.full_clean()
        
        with self.assertRaises(ValidationError):
            recipe = RecipeFactory.build(servings=25)  # Too many servings
            recipe.full_clean()
```

**Running coverage commands:**

```bash
# Run tests with coverage
coverage run --source='.' manage.py test

# Generate coverage report
coverage report

# Generate HTML coverage report
coverage html

# Example coverage report output:
# Name                 Stmts   Miss  Cover   Missing
# --------------------------------------------------
# models.py               45      3    93%   67-69
# views.py                32      8    75%   23-25, 45-48
# forms.py                15      0   100%
# --------------------------------------------------
# TOTAL                   92     11    88%
```

**Syntax Explanation:**
- `coverage run`: Runs tests while tracking which code lines execute
- `--source='.'`: Tells coverage to track current directory
- `coverage report`: Shows text-based coverage summary
- `coverage html`: Creates detailed HTML report with line-by-line coverage
- `pragma: no cover`: Comment to exclude specific lines from coverage requirements
- `Stmts`: Total statements in file
- `Miss`: Statements not covered by tests
- `Cover`: Percentage of code covered

---

# Project: Building a Test Suite for Expense Tracker

## Learning Objective
By the end of this project, you will be able to create a comprehensive test suite for a Django expense tracker application, understanding how to test models, views, and forms while ensuring code quality and reliability.

---

Imagine that you're a master chef running a prestigious restaurant kitchen. Every dish that leaves your kitchen must meet the highest standards before reaching your customers. You wouldn't serve a soufflé without testing if it's properly risen, or send out a sauce without tasting it first. 

In the same way, as a Django developer, you're the head chef of your code kitchen. Your expense tracker application is like a complex multi-course meal - every component (models, views, forms) must be tested thoroughly before serving it to users. Just as a chef has a systematic way of testing each dish, you need a comprehensive test suite to ensure your code performs perfectly every time.

Your test suite is like your kitchen's quality control system - it catches problems before they reach your users, just like a good chef catches a oversalted dish before it leaves the kitchen.

---

## The Recipe: Building Our Test Suite

### Our Kitchen Setup (Project Structure)

First, let's set up our expense tracker "kitchen" with all the ingredients we need:

```python
# models.py - Our Core Ingredients
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator
from decimal import Decimal

class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Categories"
    
    def __str__(self):
        return self.name

class Expense(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    amount = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0.01'))]
    )
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    description = models.TextField(blank=True)
    date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date', '-created_at']
    
    def __str__(self):
        return f"{self.title} - ${self.amount}"
```

```python
# forms.py - Our Recipe Cards
from django import forms
from .models import Expense, Category

class ExpenseForm(forms.ModelForm):
    class Meta:
        model = Expense
        fields = ['title', 'amount', 'category', 'description', 'date']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
            'description': forms.Textarea(attrs={'rows': 3}),
        }
    
    def clean_amount(self):
        amount = self.cleaned_data.get('amount')
        if amount and amount <= 0:
            raise forms.ValidationError("Amount must be greater than zero.")
        return amount

class CategoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = ['name', 'description']
```

```python
# views.py - Our Cooking Methods
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Sum
from .models import Expense, Category
from .forms import ExpenseForm, CategoryForm

@login_required
def expense_list(request):
    expenses = Expense.objects.filter(user=request.user)
    total_amount = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
    
    context = {
        'expenses': expenses,
        'total_amount': total_amount,
    }
    return render(request, 'expenses/expense_list.html', context)

@login_required
def add_expense(request):
    if request.method == 'POST':
        form = ExpenseForm(request.POST)
        if form.is_valid():
            expense = form.save(commit=False)
            expense.user = request.user
            expense.save()
            messages.success(request, 'Expense added successfully!')
            return redirect('expense_list')
    else:
        form = ExpenseForm()
    
    return render(request, 'expenses/add_expense.html', {'form': form})

@login_required
def delete_expense(request, pk):
    expense = get_object_or_404(Expense, pk=pk, user=request.user)
    if request.method == 'POST':
        expense.delete()
        messages.success(request, 'Expense deleted successfully!')
        return redirect('expense_list')
    return render(request, 'expenses/delete_expense.html', {'expense': expense})
```

### Our Test Kitchen (The Test Suite)

Now, let's create our comprehensive test suite - think of this as our kitchen's quality control system:

```python
# tests.py - Our Quality Control System
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.exceptions import ValidationError
from decimal import Decimal
from datetime import date, timedelta
from .models import Expense, Category
from .forms import ExpenseForm, CategoryForm

class CategoryModelTest(TestCase):
    """Testing our Category ingredient - like testing if our base stocks are good"""
    
    def setUp(self):
        """Set up our test kitchen before each test"""
        self.category = Category.objects.create(
            name="Food",
            description="Food expenses"
        )
    
    def test_category_creation(self):
        """Test if our category ingredient is properly prepared"""
        self.assertEqual(self.category.name, "Food")
        self.assertEqual(self.category.description, "Food expenses")
        self.assertTrue(self.category.created_at)
    
    def test_category_string_representation(self):
        """Test if our category displays correctly - like checking presentation"""
        self.assertEqual(str(self.category), "Food")
    
    def test_category_unique_constraint(self):
        """Test if duplicate categories are prevented - like avoiding duplicate ingredients"""
        with self.assertRaises(Exception):
            Category.objects.create(name="Food", description="Duplicate food")

class ExpenseModelTest(TestCase):
    """Testing our main dish - the Expense model"""
    
    def setUp(self):
        """Prepare our test kitchen ingredients"""
        self.user = User.objects.create_user(
            username='testchef',
            password='testpass123'
        )
        self.category = Category.objects.create(
            name="Groceries",
            description="Grocery shopping"
        )
        self.expense = Expense.objects.create(
            user=self.user,
            title="Weekly Groceries",
            amount=Decimal('125.50'),
            category=self.category,
            description="Weekly grocery shopping",
            date=date.today()
        )
    
    def test_expense_creation(self):
        """Test if our expense dish is properly cooked"""
        self.assertEqual(self.expense.title, "Weekly Groceries")
        self.assertEqual(self.expense.amount, Decimal('125.50'))
        self.assertEqual(self.expense.category, self.category)
        self.assertEqual(self.expense.user, self.user)
        self.assertTrue(self.expense.created_at)
        self.assertTrue(self.expense.updated_at)
    
    def test_expense_string_representation(self):
        """Test the expense presentation - like plating a dish"""
        expected_str = "Weekly Groceries - $125.50"
        self.assertEqual(str(self.expense), expected_str)
    
    def test_expense_ordering(self):
        """Test if expenses are served in the right order - like course sequence"""
        # Create expenses with different dates
        older_expense = Expense.objects.create(
            user=self.user,
            title="Old Expense",
            amount=Decimal('50.00'),
            category=self.category,
            date=date.today() - timedelta(days=1)
        )
        
        expenses = Expense.objects.all()
        self.assertEqual(expenses[0], self.expense)  # Most recent first
        self.assertEqual(expenses[1], older_expense)
    
    def test_expense_user_relationship(self):
        """Test if expenses belong to the right chef"""
        user_expenses = Expense.objects.filter(user=self.user)
        self.assertIn(self.expense, user_expenses)

class ExpenseFormTest(TestCase):
    """Testing our recipe cards - making sure instructions are clear"""
    
    def setUp(self):
        """Prepare our form testing ingredients"""
        self.user = User.objects.create_user(
            username='formchef',
            password='testpass123'
        )
        self.category = Category.objects.create(
            name="Transportation",
            description="Transport costs"
        )
    
    def test_expense_form_valid_data(self):
        """Test if our recipe works with good ingredients"""
        form_data = {
            'title': 'Gas for car',
            'amount': '45.75',
            'category': self.category.id,
            'description': 'Weekly gas fill-up',
            'date': date.today()
        }
        form = ExpenseForm(data=form_data)
        self.assertTrue(form.is_valid())
    
    def test_expense_form_invalid_amount(self):
        """Test if our recipe catches bad ingredients - like spoiled food"""
        form_data = {
            'title': 'Invalid Expense',
            'amount': '-10.00',  # Negative amount - like using rotten ingredients
            'category': self.category.id,
            'description': 'This should fail',
            'date': date.today()
        }
        form = ExpenseForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('Amount must be greater than zero.', form.errors['amount'])
    
    def test_expense_form_missing_required_fields(self):
        """Test if our recipe catches missing ingredients"""
        form_data = {}  # Empty form - like trying to cook without ingredients
        form = ExpenseForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('title', form.errors)
        self.assertIn('amount', form.errors)
        self.assertIn('category', form.errors)

class CategoryFormTest(TestCase):
    """Testing our category recipe cards"""
    
    def test_category_form_valid_data(self):
        """Test if category form works with good data"""
        form_data = {
            'name': 'Entertainment',
            'description': 'Movies, games, and fun activities'
        }
        form = CategoryForm(data=form_data)
        self.assertTrue(form.is_valid())
    
    def test_category_form_missing_name(self):
        """Test if form catches missing category name"""
        form_data = {
            'description': 'Category without name'
        }
        form = CategoryForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('name', form.errors)

class ExpenseViewTest(TestCase):
    """Testing our cooking methods - the views"""
    
    def setUp(self):
        """Set up our test kitchen with a chef and ingredients"""
        self.client = Client()
        self.user = User.objects.create_user(
            username='viewchef',
            password='testpass123'
        )
        self.category = Category.objects.create(
            name="Utilities",
            description="Utility bills"
        )
        self.expense = Expense.objects.create(
            user=self.user,
            title="Electric Bill",
            amount=Decimal('89.99'),
            category=self.category,
            description="Monthly electric bill",
            date=date.today()
        )
        # Log in our chef
        self.client.login(username='viewchef', password='testpass123')
    
    def test_expense_list_view(self):
        """Test if our expense menu displays correctly"""
        response = self.client.get(reverse('expense_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Electric Bill")
        self.assertContains(response, "$89.99")
    
    def test_expense_list_view_requires_login(self):
        """Test if kitchen is locked to unauthorized users"""
        self.client.logout()
        response = self.client.get(reverse('expense_list'))
        self.assertEqual(response.status_code, 302)  # Redirects to login
    
    def test_add_expense_view_get(self):
        """Test if the add expense form loads - like setting up cooking station"""
        response = self.client.get(reverse('add_expense'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'form')
    
    def test_add_expense_view_post_valid(self):
        """Test if adding an expense works - like successfully cooking a dish"""
        form_data = {
            'title': 'Internet Bill',
            'amount': '59.99',
            'category': self.category.id,
            'description': 'Monthly internet service',
            'date': date.today()
        }
        response = self.client.post(reverse('add_expense'), data=form_data)
        self.assertEqual(response.status_code, 302)  # Redirects after success
        
        # Check if expense was created
        new_expense = Expense.objects.get(title='Internet Bill')
        self.assertEqual(new_expense.amount, Decimal('59.99'))
        self.assertEqual(new_expense.user, self.user)
    
    def test_add_expense_view_post_invalid(self):
        """Test if bad expense data is rejected - like refusing bad ingredients"""
        form_data = {
            'title': '',  # Empty title
            'amount': 'invalid',  # Invalid amount
            'category': self.category.id,
            'date': date.today()
        }
        response = self.client.post(reverse('add_expense'), data=form_data)
        self.assertEqual(response.status_code, 200)  # Stays on form page
        self.assertContains(response, 'form')
    
    def test_delete_expense_view(self):
        """Test if expense deletion works - like removing a dish from menu"""
        response = self.client.post(
            reverse('delete_expense', kwargs={'pk': self.expense.pk})
        )
        self.assertEqual(response.status_code, 302)  # Redirects after deletion
        
        # Check if expense was deleted
        with self.assertRaises(Expense.DoesNotExist):
            Expense.objects.get(pk=self.expense.pk)
    
    def test_delete_expense_security(self):
        """Test if users can only delete their own expenses - kitchen security"""
        # Create another user and their expense
        other_user = User.objects.create_user(
            username='otherchef',
            password='testpass123'
        )
        other_expense = Expense.objects.create(
            user=other_user,
            title="Other User's Expense",
            amount=Decimal('25.00'),
            category=self.category,
            date=date.today()
        )
        
        # Try to delete other user's expense
        response = self.client.post(
            reverse('delete_expense', kwargs={'pk': other_expense.pk})
        )
        self.assertEqual(response.status_code, 404)  # Should not be found
        
        # Expense should still exist
        self.assertTrue(Expense.objects.filter(pk=other_expense.pk).exists())

class ExpenseModelBusinessLogicTest(TestCase):
    """Testing our special cooking techniques - business logic"""
    
    def setUp(self):
        """Prepare advanced test ingredients"""
        self.user = User.objects.create_user(
            username='businesschef',
            password='testpass123'
        )
        self.food_category = Category.objects.create(name="Food")
        self.transport_category = Category.objects.create(name="Transport")
    
    def test_user_total_expenses(self):
        """Test if we can calculate total expenses - like calculating meal costs"""
        # Create multiple expenses
        Expense.objects.create(
            user=self.user,
            title="Lunch",
            amount=Decimal('15.50'),
            category=self.food_category,
            date=date.today()
        )
        Expense.objects.create(
            user=self.user,
            title="Bus Fare",
            amount=Decimal('2.50'),
            category=self.transport_category,
            date=date.today()
        )
        
        # Calculate total
        from django.db.models import Sum
        total = Expense.objects.filter(user=self.user).aggregate(
            Sum('amount')
        )['amount__sum']
        
        self.assertEqual(total, Decimal('18.00'))
    
    def test_expenses_by_category(self):
        """Test if we can group expenses by category - like organizing by course"""
        # Create expenses in different categories
        Expense.objects.create(
            user=self.user,
            title="Breakfast",
            amount=Decimal('8.00'),
            category=self.food_category,
            date=date.today()
        )
        Expense.objects.create(
            user=self.user,
            title="Dinner",
            amount=Decimal('25.00'),
            category=self.food_category,
            date=date.today()
        )
        
        food_expenses = Expense.objects.filter(
            user=self.user,
            category=self.food_category
        )
        
        self.assertEqual(food_expenses.count(), 2)
        total_food = sum(expense.amount for expense in food_expenses)
        self.assertEqual(total_food, Decimal('33.00'))
```

### Running Our Test Kitchen

Here's how to run your test suite - think of this as your daily kitchen inspection:

```bash
# Run all tests - complete kitchen inspection
python manage.py test

# Run specific test class - focus on one cooking station
python manage.py test expenses.tests.ExpenseModelTest

# Run with detailed output - get full inspection report
python manage.py test --verbosity=2

# Run tests with coverage - see what parts of kitchen are tested
pip install coverage
coverage run --source='.' manage.py test
coverage report
coverage html  # Creates detailed HTML report
```

---

## Syntax Explanation: Understanding Our Test Kitchen Tools

Let me explain the key testing "cooking techniques" we used:

### 1. **TestCase Class** - Our Kitchen Foundation
```python
class ExpenseModelTest(TestCase):
```
- `TestCase` is like your main kitchen workspace
- It provides tools for testing and automatically handles database setup/cleanup
- Each test method is isolated (like preparing each dish separately)

### 2. **setUp Method** - Ingredient Preparation
```python
def setUp(self):
    self.user = User.objects.create_user(...)
```
- Runs before each test method
- Like prepping ingredients before cooking each dish
- Creates test data that multiple tests can use

### 3. **Assertion Methods** - Quality Control Checks
```python
self.assertEqual(expense.title, "Weekly Groceries")  # Check if values match
self.assertTrue(expense.created_at)  # Check if something exists
self.assertIn(expense, user_expenses)  # Check if item is in collection
self.assertContains(response, "Electric Bill")  # Check if response contains text
```

### 4. **Test Client** - Our Kitchen Simulator
```python
self.client = Client()
self.client.login(username='viewchef', password='testpass123')
response = self.client.get(reverse('expense_list'))
```
- Simulates web browser requests
- Like having a customer ordering from your kitchen
- Tests the full request-response cycle

### 5. **Exception Testing** - Catching Kitchen Mistakes
```python
with self.assertRaises(Expense.DoesNotExist):
    Expense.objects.get(pk=self.expense.pk)
```
- Tests that expected errors occur
- Like ensuring your smoke alarm works

---

## Assignment: Kitchen Quality Control System

### The Challenge

You're tasked with creating a comprehensive testing and debugging setup for a Django-based restaurant management system. Your job is to ensure the kitchen runs smoothly by implementing proper debugging techniques, writing thorough tests, and measuring coverage.

### Requirements

Create a Django app called `restaurant_qc` with the following components:

1. **Models to Debug and Test:**
   - `Dish` model with name, price, category, prep_time, and chef
   - `Order` model that references multiple dishes
   - `Chef` model that extends Django's User model

2. **Debugging Implementation:**
   - Set up Django Debug Toolbar
   - Add comprehensive logging to all views
   - Include at least one view that uses `pdb` for debugging
   - Create a custom 404 and 500 error page

3. **Testing Suite:**
   - Write unit tests for all models (minimum 3 tests per model)
   - Write view tests covering both authenticated and anonymous users
   - Use both fixtures and Factory Boy for test data
   - Include tests for edge cases and error conditions

4. **Coverage Analysis:**
   - Achieve minimum 85% test coverage
   - Generate HTML coverage report
   - Document any uncovered code with explanations

5. **Documentation:**
   - Create a README explaining your debugging setup
   - Document your testing strategy
   - Include instructions for running tests and coverage

### Deliverables

1. Complete Django app with models, views, and templates
2. Comprehensive test suite in `tests/` directory
3. Factory definitions in `factories.py`
4. Coverage configuration in `.coveragerc`
5. HTML coverage report
6. README with setup and usage instructions

### Evaluation Criteria

- **Code Quality (25%)**: Clean, well-structured Django code
- **Debugging Setup (25%)**: Proper debug toolbar, logging, and error handling
- **Test Coverage (25%)**: Comprehensive tests achieving 85%+ coverage
- **Documentation (25%)**: Clear README and code comments

### Bonus Points

- Implement custom management commands for testing
- Add API endpoints and test them
- Create a test runner that automatically generates coverage reports
- Add integration tests using Django's `LiveServerTestCase`

This assignment will test your ability to create a production-ready Django application with proper debugging and testing infrastructure - just like ensuring a restaurant kitchen meets health inspection standards!

---

## Summary

In this lesson, you've learned to be a master chef of code quality by:

1. **Debugging Techniques**: Using Django's debug tools, Python debugger, and logging like a detective investigating kitchen mysteries
2. **Unit Testing**: Writing tests that check each component individually, like tasting each part of a dish
3. **Fixtures and Factories**: Creating consistent test data like preparing mise en place for cooking
4. **Coverage Reporting**: Measuring test coverage like a health inspector checking every corner of the kitchen

Remember: Just as a great chef never serves a dish without tasting it first, a great developer never deploys code without proper testing and debugging. Your Django applications should be as reliable as a perfectly executed recipe, tested and refined until they work flawlessly every time.