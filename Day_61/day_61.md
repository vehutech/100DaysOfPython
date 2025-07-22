# Day 61: Custom Management Commands in Django
*Master the Art of Kitchen Automation*

## Learning Objective
By the end of this lesson, you will be able to create custom Django management commands that automate repetitive tasks, parse command-line arguments effectively, and implement scheduled operations - just like a head chef who creates specialized kitchen tools and procedures to streamline restaurant operations.

---

## Introduction: The Kitchen Automation Vision

Imagine that you're the head chef of a bustling restaurant, and you've noticed your kitchen staff spending too much time on repetitive tasks: checking inventory, preparing daily specials, cleaning equipment, and organizing ingredients. As an innovative chef, you decide to create specialized tools and procedures - custom kitchen gadgets, automated prep schedules, and standardized cleaning protocols - that your staff can use with simple commands.

In Django, custom management commands serve the same purpose. They're your specialized kitchen tools that automate repetitive development and maintenance tasks, making your development workflow as smooth as a well-orchestrated kitchen service.

---

## Lesson 1: Writing Custom Commands
*Creating Your First Kitchen Tool*

Just as a chef designs custom kitchen tools for specific tasks, Django allows you to create custom management commands that extend beyond the built-in `migrate`, `runserver`, and `collectstatic` commands.

### The Command Structure - Your Kitchen Tool Blueprint

Every custom command in Django follows a specific structure, like a standardized kitchen tool design:

```python
# myapp/management/commands/hello_chef.py
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'A simple greeting command for our kitchen staff'
    
    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Hello from the Django kitchen!')
        )
```

**Directory Structure (Your Tool Storage):**
```
myapp/
├── management/
│   ├── __init__.py
│   └── commands/
│       ├── __init__.py
│       └── hello_chef.py
```

**Running the command:**
```bash
python manage.py hello_chef
```

### Syntax Explanation:
- `BaseCommand`: The foundation class that provides the command structure, like the basic handle all kitchen tools need
- `help`: A description that appears when you run `python manage.py help hello_chef`
- `handle()`: The main method that executes when the command runs - your tool's primary function
- `self.stdout.write()`: Sends output to the console with optional styling
- `self.style.SUCCESS()`: Colors the output green for success messages

### Enhanced Kitchen Command Example:

```python
# myapp/management/commands/kitchen_status.py
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from myapp.models import Recipe, Ingredient

class Command(BaseCommand):
    help = 'Check the current status of our Django kitchen'
    
    def handle(self, *args, **options):
        # Count our kitchen resources
        total_chefs = User.objects.count()
        total_recipes = Recipe.objects.count()
        total_ingredients = Ingredient.objects.count()
        
        # Display kitchen status
        self.stdout.write("=== Kitchen Status Report ===")
        self.stdout.write(f"👨‍🍳 Active Chefs: {total_chefs}")
        self.stdout.write(f"📝 Available Recipes: {total_recipes}")
        self.stdout.write(f"🥕 Ingredients in Stock: {total_ingredients}")
        
        if total_recipes > 0:
            self.stdout.write(
                self.style.SUCCESS("✅ Kitchen is ready for service!")
            )
        else:
            self.stdout.write(
                self.style.WARNING("⚠️ No recipes available - kitchen needs preparation!")
            )
```

---

## Lesson 2: Command-line Argument Parsing
*Customizing Your Kitchen Tools*

Just as kitchen tools can be adjusted for different tasks (like adjustable measuring spoons), Django commands can accept arguments and options to modify their behavior.

### Adding Arguments - Tool Specifications

```python
# myapp/management/commands/prep_ingredients.py
from django.core.management.base import BaseCommand
from myapp.models import Ingredient

class Command(BaseCommand):
    help = 'Prepare ingredients for kitchen service'
    
    def add_arguments(self, parser):
        # Positional argument - required ingredient type
        parser.add_argument(
            'ingredient_type',
            type=str,
            help='Type of ingredient to prepare (vegetables, proteins, spices)'
        )
        
        # Optional argument - quantity
        parser.add_argument(
            '--quantity',
            type=int,
            default=10,
            help='Number of ingredients to prepare (default: 10)'
        )
        
        # Boolean flag - detailed output
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed preparation steps'
        )
        
        # Choice argument - preparation method
        parser.add_argument(
            '--method',
            choices=['chop', 'dice', 'julienne', 'brunoise'],
            default='chop',
            help='Preparation method for ingredients'
        )
    
    def handle(self, *args, **options):
        ingredient_type = options['ingredient_type']
        quantity = options['quantity']
        detailed = options['detailed']
        method = options['method']
        
        self.stdout.write(f"🔪 Preparing {quantity} {ingredient_type} using {method} method")
        
        if detailed:
            self.stdout.write("📋 Detailed preparation steps:")
            self.stdout.write(f"  1. Select fresh {ingredient_type}")
            self.stdout.write(f"  2. Wash and clean thoroughly")
            self.stdout.write(f"  3. {method.capitalize()} into uniform pieces")
            self.stdout.write(f"  4. Store in appropriate containers")
        
        # Simulate ingredient preparation
        ingredients = Ingredient.objects.filter(category=ingredient_type)[:quantity]
        prepared_count = 0
        
        for ingredient in ingredients:
            ingredient.is_prepared = True
            ingredient.preparation_method = method
            ingredient.save()
            prepared_count += 1
        
        self.stdout.write(
            self.style.SUCCESS(f"✅ Successfully prepared {prepared_count} {ingredient_type}")
        )
```

**Usage examples:**
```bash
# Basic usage
python manage.py prep_ingredients vegetables

# With quantity
python manage.py prep_ingredients proteins --quantity 5

# With detailed output and specific method
python manage.py prep_ingredients spices --detailed --method dice

# Short form
python manage.py prep_ingredients vegetables -q 15 --detailed
```

### Syntax Explanation:
- `add_arguments(self, parser)`: Method where you define command arguments
- `parser.add_argument()`: Adds a new argument option
- `type=int`: Specifies the expected data type
- `default=10`: Sets a default value if not provided
- `action='store_true'`: Creates a boolean flag
- `choices=[...]`: Limits input to specific options
- `options['argument_name']`: Accesses the argument value in handle()

---

## Lesson 3: Scheduled Tasks
*Setting Up Kitchen Routines*

Like daily kitchen prep routines that happen at specific times, Django commands can be designed to run on schedules using external tools or internal logic.

### Time-Based Kitchen Operations

```python
# myapp/management/commands/daily_kitchen_routine.py
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
from myapp.models import Recipe, Ingredient, KitchenLog
import datetime

class Command(BaseCommand):
    help = 'Perform daily kitchen maintenance and preparation routines'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--task',
            choices=['morning_prep', 'inventory_check', 'cleanup', 'all'],
            default='all',
            help='Specific routine to run'
        )
        
        parser.add_argument(
            '--notify',
            action='store_true',
            help='Send notification email to kitchen manager'
        )
    
    def handle(self, *args, **options):
        task = options['task']
        notify = options['notify']
        
        results = []
        
        if task in ['morning_prep', 'all']:
            results.append(self.morning_prep())
        
        if task in ['inventory_check', 'all']:
            results.append(self.inventory_check())
        
        if task in ['cleanup', 'all']:
            results.append(self.cleanup_routine())
        
        # Log the routine completion
        self.log_routine_completion(task, results)
        
        if notify:
            self.send_notification_email(task, results)
        
        self.stdout.write(
            self.style.SUCCESS(f"✅ Daily routine '{task}' completed successfully!")
        )
    
    def morning_prep(self):
        """Morning preparation routine"""
        self.stdout.write("🌅 Starting morning prep routine...")
        
        # Reset daily counters
        Recipe.objects.filter(daily_special=True).update(
            orders_today=0,
            last_prepared=timezone.now()
        )
        
        # Check ingredient freshness
        expired_ingredients = Ingredient.objects.filter(
            expiry_date__lt=timezone.now().date()
        )
        
        if expired_ingredients.exists():
            self.stdout.write(
                self.style.WARNING(f"⚠️ Found {expired_ingredients.count()} expired ingredients")
            )
            expired_ingredients.update(status='expired')
        
        return f"Morning prep completed - {expired_ingredients.count()} expired ingredients handled"
    
    def inventory_check(self):
        """Check ingredient inventory levels"""
        self.stdout.write("📊 Performing inventory check...")
        
        low_stock_ingredients = Ingredient.objects.filter(
            quantity__lt=10,  # Less than 10 units
            status='active'
        )
        
        if low_stock_ingredients.exists():
            self.stdout.write(
                self.style.WARNING(f"⚠️ {low_stock_ingredients.count()} ingredients are low in stock")
            )
            for ingredient in low_stock_ingredients:
                self.stdout.write(f"  - {ingredient.name}: {ingredient.quantity} units remaining")
        
        return f"Inventory check completed - {low_stock_ingredients.count()} items need restocking"
    
    def cleanup_routine(self):
        """Daily cleanup tasks"""
        self.stdout.write("🧹 Running cleanup routine...")
        
        # Archive old logs (older than 30 days)
        thirty_days_ago = timezone.now() - datetime.timedelta(days=30)
        old_logs = KitchenLog.objects.filter(created_at__lt=thirty_days_ago)
        archived_count = old_logs.count()
        old_logs.update(archived=True)
        
        # Clean up temporary recipe drafts
        draft_recipes = Recipe.objects.filter(
            status='draft',
            created_at__lt=thirty_days_ago
        )
        deleted_drafts = draft_recipes.count()
        draft_recipes.delete()
        
        return f"Cleanup completed - {archived_count} logs archived, {deleted_drafts} draft recipes removed"
    
    def log_routine_completion(self, task, results):
        """Log the routine completion"""
        KitchenLog.objects.create(
            task_type=task,
            completed_at=timezone.now(),
            results='\n'.join(results),
            status='completed'
        )
    
    def send_notification_email(self, task, results):
        """Send notification email to kitchen manager"""
        if hasattr(settings, 'KITCHEN_MANAGER_EMAIL'):
            subject = f"Daily Kitchen Routine Report - {task.title()}"
            message = f"""
            Daily routine '{task}' has been completed.
            
            Results:
            {chr(10).join(results)}
            
            Completed at: {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            send_mail(
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,
                [settings.KITCHEN_MANAGER_EMAIL],
                fail_silently=True
            )
```

**Usage examples:**
```bash
# Run all daily routines
python manage.py daily_kitchen_routine

# Run specific routine with notification
python manage.py daily_kitchen_routine --task morning_prep --notify

# Run inventory check only
python manage.py daily_kitchen_routine --task inventory_check
```

### Setting Up with Cron (Linux/Mac)
```bash
# Add to crontab for automated scheduling
# Run morning prep at 6 AM daily
0 6 * * * cd /path/to/project && python manage.py daily_kitchen_routine --task morning_prep

# Run inventory check at 2 PM daily
0 14 * * * cd /path/to/project && python manage.py daily_kitchen_routine --task inventory_check
```

---

## Lesson 4: Data Migration Scripts
*Reorganizing Your Kitchen Storage*

Sometimes you need to reorganize your kitchen - move ingredients to new storage systems, update recipe formats, or restructure your inventory. Data migration commands help you safely transform your database, just like reorganizing a kitchen without disrupting service.

### Safe Data Migration Command

```python
# myapp/management/commands/migrate_recipe_format.py
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from myapp.models import Recipe, Ingredient, RecipeIngredient
import json

class Command(BaseCommand):
    help = 'Migrate old recipe format to new standardized format'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be changed without making actual changes'
        )
        
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Number of recipes to process in each batch'
        )
        
        parser.add_argument(
            '--recipe-id',
            type=int,
            help='Migrate specific recipe by ID'
        )
    
    def handle(self, *args, **options):
        dry_run = options['dry_run']
        batch_size = options['batch_size']
        recipe_id = options.get('recipe_id')
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING("🔍 DRY RUN MODE - No changes will be made")
            )
        
        # Get recipes to migrate
        if recipe_id:
            recipes = Recipe.objects.filter(id=recipe_id, format_version__lt=2)
        else:
            recipes = Recipe.objects.filter(format_version__lt=2)
        
        total_recipes = recipes.count()
        
        if total_recipes == 0:
            self.stdout.write(
                self.style.SUCCESS("✅ All recipes are already in the new format!")
            )
            return
        
        self.stdout.write(f"📝 Found {total_recipes} recipes to migrate")
        
        migrated_count = 0
        error_count = 0
        
        # Process recipes in batches
        for i in range(0, total_recipes, batch_size):
            batch_recipes = recipes[i:i + batch_size]
            self.stdout.write(f"Processing batch {i//batch_size + 1}...")
            
            for recipe in batch_recipes:
                try:
                    if dry_run:
                        self.preview_migration(recipe)
                    else:
                        self.migrate_recipe(recipe)
                    migrated_count += 1
                except Exception as e:
                    error_count += 1
                    self.stdout.write(
                        self.style.ERROR(f"❌ Error migrating recipe {recipe.id}: {str(e)}")
                    )
        
        # Summary
        if dry_run:
            self.stdout.write(
                self.style.SUCCESS(f"🔍 DRY RUN COMPLETE: {migrated_count} recipes ready for migration")
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(f"✅ Migration completed: {migrated_count} recipes migrated, {error_count} errors")
            )
    
    def preview_migration(self, recipe):
        """Preview what changes would be made"""
        self.stdout.write(f"  📄 Recipe: {recipe.name} (ID: {recipe.id})")
        
        # Show old ingredient format
        if recipe.old_ingredients_text:
            self.stdout.write(f"    Old format: {recipe.old_ingredients_text[:50]}...")
        
        # Show what new format would look like
        parsed_ingredients = self.parse_old_ingredients(recipe.old_ingredients_text)
        self.stdout.write(f"    Would create {len(parsed_ingredients)} ingredient relationships")
        
        for ingredient_data in parsed_ingredients:
            self.stdout.write(f"      - {ingredient_data['name']}: {ingredient_data['quantity']}")
    
    @transaction.atomic
    def migrate_recipe(self, recipe):
        """Migrate a single recipe to new format"""
        self.stdout.write(f"  🔄 Migrating: {recipe.name}")
        
        # Parse old ingredient format
        parsed_ingredients = self.parse_old_ingredients(recipe.old_ingredients_text)
        
        # Create new ingredient relationships
        for ingredient_data in parsed_ingredients:
            # Get or create ingredient
            ingredient, created = Ingredient.objects.get_or_create(
                name=ingredient_data['name'].lower(),
                defaults={
                    'category': ingredient_data.get('category', 'other'),
                    'unit': ingredient_data.get('unit', 'piece')
                }
            )
            
            # Create recipe-ingredient relationship
            RecipeIngredient.objects.create(
                recipe=recipe,
                ingredient=ingredient,
                quantity=ingredient_data['quantity'],
                unit=ingredient_data.get('unit', 'piece'),
                preparation_notes=ingredient_data.get('notes', '')
            )
        
        # Update recipe format version and clear old data
        recipe.format_version = 2
        recipe.migrated_at = timezone.now()
        recipe.old_ingredients_text = None  # Clear old format
        recipe.save()
        
        self.stdout.write(f"    ✅ Created {len(parsed_ingredients)} ingredient relationships")
    
    def parse_old_ingredients(self, ingredients_text):
        """Parse old ingredient format into structured data"""
        if not ingredients_text:
            return []
        
        ingredients = []
        lines = ingredients_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple parsing logic (adjust based on your old format)
            # Example: "2 cups flour" or "1 lb chicken breast, diced"
            parts = line.split(' ', 2)
            
            if len(parts) >= 3:
                try:
                    quantity = float(parts[0])
                    unit = parts[1]
                    name_and_notes = parts[2]
                    
                    # Split name and notes if there's a comma
                    if ',' in name_and_notes:
                        name, notes = name_and_notes.split(',', 1)
                        notes = notes.strip()
                    else:
                        name = name_and_notes
                        notes = ''
                    
                    ingredients.append({
                        'name': name.strip(),
                        'quantity': quantity,
                        'unit': unit,
                        'notes': notes,
                        'category': self.guess_category(name.strip())
                    })
                except ValueError:
                    # If parsing fails, treat as a simple ingredient
                    ingredients.append({
                        'name': line,
                        'quantity': 1,
                        'unit': 'piece',
                        'notes': '',
                        'category': 'other'
                    })
        
        return ingredients
    
    def guess_category(self, ingredient_name):
        """Guess ingredient category based on name"""
        name_lower = ingredient_name.lower()
        
        vegetables = ['onion', 'carrot', 'celery', 'tomato', 'lettuce', 'spinach']
        proteins = ['chicken', 'beef', 'pork', 'fish', 'egg', 'tofu']
        spices = ['salt', 'pepper', 'oregano', 'basil', 'thyme', 'garlic']
        grains = ['rice', 'flour', 'pasta', 'bread', 'oats']
        
        for vegetable in vegetables:
            if vegetable in name_lower:
                return 'vegetables'
        
        for protein in proteins:
            if protein in name_lower:
                return 'proteins'
        
        for spice in spices:
            if spice in name_lower:
                return 'spices'
        
        for grain in grains:
            if grain in name_lower:
                return 'grains'
        
        return 'other'
```

**Usage examples:**
```bash
# Preview migration without making changes
python manage.py migrate_recipe_format --dry-run

# Migrate all recipes in small batches
python manage.py migrate_recipe_format --batch-size 50

# Migrate specific recipe
python manage.py migrate_recipe_format --recipe-id 123

# Full migration
python manage.py migrate_recipe_format
```

### Syntax Explanation:
- `@transaction.atomic`: Ensures database operations are wrapped in a transaction
- `--dry-run`: Preview mode that shows changes without making them
- `--batch-size`: Process records in smaller groups to manage memory
- `get_or_create()`: Gets existing record or creates new one
- `timezone.now()`: Current timestamp with timezone awareness

---

## 🎯 Assignment: Kitchen Inventory Alert System

Create a custom Django management command called `inventory_alerts` that monitors your kitchen's ingredient levels and sends alerts when items are running low.

### Requirements:

1. **Command Name**: `inventory_alerts.py`

2. **Arguments to Support**:
   - `--threshold` (optional, default: 5): Minimum quantity before alert
   - `--category` (optional): Filter by ingredient category
   - `--format` (choices: 'console', 'email', 'json', default: 'console'): Output format
   - `--save-report`: Save the report to a file

3. **Functionality**:
   - Check all ingredients below the threshold
   - Calculate days until completely out of stock (based on average daily usage)
   - Categorize alerts by urgency (critical: <2 days, warning: <7 days, info: >7 days)
   - Generate different output formats

4. **Models to Create** (add to your models.py):
```python
class IngredientUsage(models.Model):
    ingredient = models.ForeignKey(Ingredient, on_delete=models.CASCADE)
    date = models.DateField()
    quantity_used = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

class InventoryAlert(models.Model):
    ingredient = models.ForeignKey(Ingredient, on_delete=models.CASCADE)
    alert_type = models.CharField(max_length=20, choices=[
        ('critical', 'Critical'),
        ('warning', 'Warning'),
        ('info', 'Info')
    ])
    current_quantity = models.FloatField()
    threshold = models.FloatField()
    days_remaining = models.IntegerField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    resolved = models.BooleanField(default=False)
```

5. **Expected Output Examples**:

**Console format:**
```
🚨 KITCHEN INVENTORY ALERTS 🚨
Generated: 2025-07-21 14:30:15

CRITICAL ALERTS (< 2 days):
❌ Olive Oil: 1.5 liters remaining (threshold: 5.0) - 1 day left
❌ Salt: 0.8 kg remaining (threshold: 2.0) - 1 day left

WARNING ALERTS (< 7 days):
⚠️ Chicken Breast: 3.2 kg remaining (threshold: 10.0) - 4 days left

INFO ALERTS (> 7 days):
ℹ️ Garlic: 4.5 kg remaining (threshold: 5.0) - 12 days left

Summary: 4 ingredients need attention
```

**JSON format:**
```json
{
  "generated_at": "2025-07-21T14:30:15Z",
  "alerts": {
    "critical": [...],
    "warning": [...],
    "info": [...]
  },
  "summary": {
    "total_alerts": 4,
    "critical_count": 2,
    "warning_count": 1,
    "info_count": 1
  }
}
```

### Bonus Features:
- Email notifications to kitchen managers
- Integration with your existing ingredient models
- Historical trend analysis
- Automatic reorder suggestions based on lead times

### Submission Guidelines:
Submit your `inventory_alerts.py` command file along with any model updates and a brief explanation of:
- How you calculated days remaining
- What design decisions you made
- Example command usage
- Any additional features you implemented

This assignment tests your understanding of argument parsing, database queries, file operations, and real-world application of management commands in a kitchen management system!

# Django Custom Management Commands - Data Import/Export Project

## Learning Objective
By the end of this lesson, you will be able to create robust Django management commands that can import data from CSV files and export Django model data to various formats, just like a chef who can both receive ingredients from suppliers and package meals for delivery.

## Imagine That...
Imagine you're the head chef of a bustling restaurant chain. Every morning, you receive fresh ingredient deliveries from various suppliers - some arrive as CSV spreadsheets, others as JSON manifests. At the end of each day, you need to export your inventory, sales data, and recipes to different formats for your accountants, suppliers, and franchise locations. Your Django management commands are like your trusted kitchen assistants who handle all these data transfers automatically, ensuring your restaurant's "digital pantry" stays organized and accessible.

## Project: Restaurant Chain Data Management System

### The Kitchen Setup
Let's create a restaurant management system where we'll build commands to import menu items from suppliers and export sales data.

First, let's set up our Django models (our recipe cards):

```python
# models.py
from django.db import models
from django.contrib.auth.models import User

class Category(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name_plural = "Categories"

class MenuItem(models.Model):
    name = models.CharField(max_length=200)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField()
    ingredients = models.TextField(help_text="Comma-separated ingredients")
    is_available = models.BooleanField(default=True)
    calories = models.IntegerField(null=True, blank=True)
    prep_time_minutes = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} - ${self.price}"

class Order(models.Model):
    customer_name = models.CharField(max_length=100)
    customer_email = models.EmailField()
    order_date = models.DateTimeField(auto_now_add=True)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    status_choices = [
        ('pending', 'Pending'),
        ('preparing', 'Preparing'),
        ('ready', 'Ready'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]
    status = models.CharField(max_length=20, choices=status_choices, default='pending')
    
    def __str__(self):
        return f"Order #{self.id} - {self.customer_name}"

class OrderItem(models.Model):
    order = models.ForeignKey(Order, related_name='items', on_delete=models.CASCADE)
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    
    def __str__(self):
        return f"{self.quantity}x {self.menu_item.name}"
```

### Command 1: The Ingredient Importer (CSV Import Command)

Create the management command directory structure:
```
myapp/
└── management/
    └── commands/
        ├── __init__.py
        ├── import_menu_items.py
        └── export_sales_data.py
```

```python
# management/commands/import_menu_items.py
import csv
import os
from decimal import Decimal
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from myapp.models import MenuItem, Category

class Command(BaseCommand):
    help = 'Import menu items from CSV file - like receiving ingredient deliveries'
    
    def add_arguments(self, parser):
        parser.add_argument(
            'csv_file',
            type=str,
            help='Path to the CSV file containing menu items'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Preview import without making changes (like tasting before serving)'
        )
        parser.add_argument(
            '--update-existing',
            action='store_true',
            help='Update existing items instead of skipping them'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Number of items to process in each batch (default: 100)'
        )
    
    def handle(self, *args, **options):
        csv_file = options['csv_file']
        dry_run = options['dry_run']
        update_existing = options['update_existing']
        batch_size = options['batch_size']
        
        # Check if file exists (like checking if delivery arrived)
        if not os.path.exists(csv_file):
            raise CommandError(f'File "{csv_file}" does not exist.')
        
        # Statistics tracking (our kitchen scoreboard)
        stats = {
            'created': 0,
            'updated': 0,
            'skipped': 0,
            'errors': 0
        }
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                # Validate required columns (checking our ingredient list)
                required_fields = ['name', 'category', 'price', 'description']
                if not all(field in reader.fieldnames for field in required_fields):
                    missing = [f for f in required_fields if f not in reader.fieldnames]
                    raise CommandError(f'Missing required columns: {", ".join(missing)}')
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f'🍽️  Starting import from {csv_file} '
                        f'{"(DRY RUN)" if dry_run else ""}'
                    )
                )
                
                menu_items_batch = []
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 for header
                    try:
                        menu_item_data = self.process_row(row, update_existing)
                        if menu_item_data:
                            menu_items_batch.append(menu_item_data)
                            
                        # Process batch when it reaches batch_size
                        if len(menu_items_batch) >= batch_size:
                            batch_stats = self.process_batch(
                                menu_items_batch, dry_run, update_existing
                            )
                            self.update_stats(stats, batch_stats)
                            menu_items_batch = []
                            
                    except Exception as e:
                        stats['errors'] += 1
                        self.stdout.write(
                            self.style.ERROR(f'❌ Error in row {row_num}: {str(e)}')
                        )
                
                # Process remaining items in final batch
                if menu_items_batch:
                    batch_stats = self.process_batch(
                        menu_items_batch, dry_run, update_existing
                    )
                    self.update_stats(stats, batch_stats)
                
        except Exception as e:
            raise CommandError(f'Error reading CSV file: {str(e)}')
        
        # Final report (like the daily kitchen report)
        self.print_final_report(stats, dry_run)
    
    def process_row(self, row, update_existing):
        """Process a single CSV row - like preparing one ingredient"""
        # Get or create category (like organizing ingredients by type)
        category_name = row['category'].strip()
        category, created = Category.objects.get_or_create(
            name=category_name,
            defaults={'description': f'Auto-created category for {category_name}'}
        )
        
        # Clean and validate data (like washing vegetables)
        name = row['name'].strip()
        if not name:
            raise ValueError("Menu item name cannot be empty")
            
        try:
            price = Decimal(str(row['price']).strip())
            if price < 0:
                raise ValueError("Price cannot be negative")
        except (ValueError, decimal.InvalidOperation):
            raise ValueError(f"Invalid price: {row['price']}")
        
        description = row['description'].strip()
        ingredients = row.get('ingredients', '').strip()
        calories = None
        prep_time = 0
        
        # Optional fields with defaults (like optional seasonings)
        if row.get('calories'):
            try:
                calories = int(row['calories'])
            except ValueError:
                pass
                
        if row.get('prep_time_minutes'):
            try:
                prep_time = int(row['prep_time_minutes'])
            except ValueError:
                prep_time = 0
        
        is_available = row.get('is_available', 'true').lower() in ['true', '1', 'yes', 'y']
        
        return {
            'name': name,
            'category': category,
            'price': price,
            'description': description,
            'ingredients': ingredients,
            'calories': calories,
            'prep_time_minutes': prep_time,
            'is_available': is_available,
        }
    
    def process_batch(self, menu_items_data, dry_run, update_existing):
        """Process a batch of menu items - like cooking multiple dishes"""
        batch_stats = {'created': 0, 'updated': 0, 'skipped': 0}
        
        if dry_run:
            # Just count what would happen (like planning the menu)
            for item_data in menu_items_data:
                exists = MenuItem.objects.filter(name=item_data['name']).exists()
                if exists and update_existing:
                    batch_stats['updated'] += 1
                elif exists:
                    batch_stats['skipped'] += 1
                else:
                    batch_stats['created'] += 1
            return batch_stats
        
        # Actual processing with database transaction (like cooking with precision)
        with transaction.atomic():
            for item_data in menu_items_data:
                menu_item, created = MenuItem.objects.get_or_create(
                    name=item_data['name'],
                    defaults=item_data
                )
                
                if created:
                    batch_stats['created'] += 1
                    self.stdout.write(f'✅ Created: {menu_item.name}')
                elif update_existing:
                    # Update existing item (like adjusting a recipe)
                    for key, value in item_data.items():
                        if key != 'name':  # Don't update the name
                            setattr(menu_item, key, value)
                    menu_item.save()
                    batch_stats['updated'] += 1
                    self.stdout.write(f'🔄 Updated: {menu_item.name}')
                else:
                    batch_stats['skipped'] += 1
                    self.stdout.write(f'⏭️  Skipped: {menu_item.name} (already exists)')
        
        return batch_stats
    
    def update_stats(self, total_stats, batch_stats):
        """Update total statistics"""
        for key in batch_stats:
            total_stats[key] += batch_stats[key]
    
    def print_final_report(self, stats, dry_run):
        """Print the final import report - like the end-of-shift summary"""
        self.stdout.write('\n' + '='*50)
        self.stdout.write(
            self.style.SUCCESS(
                f'🎉 Import Complete {"(DRY RUN)" if dry_run else ""}!'
            )
        )
        self.stdout.write('='*50)
        
        total_processed = sum(stats.values()) - stats['errors']
        
        self.stdout.write(f'📊 Total Processed: {total_processed}')
        self.stdout.write(
            self.style.SUCCESS(f'✅ Created: {stats["created"]}')
        )
        self.stdout.write(
            self.style.WARNING(f'🔄 Updated: {stats["updated"]}')
        )
        self.stdout.write(f'⏭️  Skipped: {stats["skipped"]}')
        
        if stats['errors'] > 0:
            self.stdout.write(
                self.style.ERROR(f'❌ Errors: {stats["errors"]}')
            )
```

### Command 2: The Order Exporter (Multi-format Export Command)

```python
# management/commands/export_sales_data.py
import csv
import json
import os
from datetime import datetime, date
from decimal import Decimal
from django.core.management.base import BaseCommand, CommandError
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import Sum, Count, Q
from myapp.models import Order, OrderItem, MenuItem

class Command(BaseCommand):
    help = 'Export sales data to various formats - like packaging daily reports for management'
    
    def add_arguments(self, parser):
        parser.add_argument(
            'output_file',
            type=str,
            help='Output file path (extension determines format: .csv, .json)'
        )
        parser.add_argument(
            '--start-date',
            type=str,
            help='Start date for export (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--end-date',
            type=str,
            help='End date for export (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--format',
            choices=['csv', 'json', 'summary'],
            help='Export format (overrides file extension)'
        )
        parser.add_argument(
            '--include-items',
            action='store_true',
            help='Include individual order items (detailed recipe breakdown)'
        )
        parser.add_argument(
            '--status',
            choices=['pending', 'preparing', 'ready', 'completed', 'cancelled'],
            help='Filter by order status'
        )
    
    def handle(self, *args, **options):
        output_file = options['output_file']
        start_date = self.parse_date(options.get('start_date'))
        end_date = self.parse_date(options.get('end_date'))
        export_format = options.get('format') or self.get_format_from_extension(output_file)
        include_items = options['include_items']
        status_filter = options.get('status')
        
        # Build query (like selecting ingredients for a specific dish)
        orders_query = self.build_query(start_date, end_date, status_filter)
        orders = orders_query.prefetch_related('items__menu_item')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        self.stdout.write(
            self.style.SUCCESS(
                f'🍽️  Exporting {orders.count()} orders to {output_file} '
                f'(format: {export_format})'
            )
        )
        
        # Export based on format (like choosing the right serving style)
        if export_format == 'csv':
            self.export_to_csv(orders, output_file, include_items)
        elif export_format == 'json':
            self.export_to_json(orders, output_file, include_items)
        elif export_format == 'summary':
            self.export_summary(orders_query, output_file, start_date, end_date)
        else:
            raise CommandError(f'Unsupported export format: {export_format}')
        
        self.stdout.write(
            self.style.SUCCESS(f'✅ Export completed successfully!')
        )
    
    def parse_date(self, date_str):
        """Parse date string - like reading the expiration date on ingredients"""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            raise CommandError(f'Invalid date format: {date_str}. Use YYYY-MM-DD.')
    
    def get_format_from_extension(self, filename):
        """Determine format from file extension - like identifying dish by presentation"""
        extension = os.path.splitext(filename)[1].lower()
        format_map = {
            '.csv': 'csv',
            '.json': 'json',
            '.txt': 'summary'
        }
        return format_map.get(extension, 'csv')
    
    def build_query(self, start_date, end_date, status_filter):
        """Build the database query - like selecting ingredients from pantry"""
        query = Order.objects.all()
        
        if start_date:
            query = query.filter(order_date__gte=start_date)
        if end_date:
            query = query.filter(order_date__lte=end_date)
        if status_filter:
            query = query.filter(status=status_filter)
        
        return query.order_by('-order_date')
    
    def export_to_csv(self, orders, output_file, include_items):
        """Export to CSV format - like preparing ingredients for bulk cooking"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if include_items:
                # Detailed export with individual items (full recipe breakdown)
                fieldnames = [
                    'order_id', 'order_date', 'customer_name', 'customer_email',
                    'order_status', 'order_total', 'item_name', 'item_category',
                    'item_price', 'quantity', 'item_total'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for order in orders:
                    for item in order.items.all():
                        writer.writerow({
                            'order_id': order.id,
                            'order_date': order.order_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'customer_name': order.customer_name,
                            'customer_email': order.customer_email,
                            'order_status': order.status,
                            'order_total': float(order.total_amount),
                            'item_name': item.menu_item.name,
                            'item_category': item.menu_item.category.name,
                            'item_price': float(item.unit_price),
                            'quantity': item.quantity,
                            'item_total': float(item.unit_price * item.quantity),
                        })
            else:
                # Order summary export (daily totals like cash register summary)
                fieldnames = [
                    'order_id', 'order_date', 'customer_name', 'customer_email',
                    'status', 'total_amount', 'item_count'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for order in orders:
                    writer.writerow({
                        'order_id': order.id,
                        'order_date': order.order_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'customer_name': order.customer_name,
                        'customer_email': order.customer_email,
                        'status': order.status,
                        'total_amount': float(order.total_amount),
                        'item_count': order.items.count(),
                    })
    
    def export_to_json(self, orders, output_file, include_items):
        """Export to JSON format - like creating a digital menu"""
        class DecimalEncoder(DjangoJSONEncoder):
            def default(self, obj):
                if isinstance(obj, Decimal):
                    return float(obj)
                return super().default(obj)
        
        export_data = {
            'export_info': {
                'export_date': datetime.now().isoformat(),
                'total_orders': orders.count(),
                'format': 'detailed' if include_items else 'summary'
            },
            'orders': []
        }
        
        for order in orders:
            order_data = {
                'id': order.id,
                'order_date': order.order_date.isoformat(),
                'customer_name': order.customer_name,
                'customer_email': order.customer_email,
                'status': order.status,
                'total_amount': order.total_amount,
            }
            
            if include_items:
                order_data['items'] = []
                for item in order.items.all():
                    order_data['items'].append({
                        'menu_item_name': item.menu_item.name,
                        'category': item.menu_item.category.name,
                        'unit_price': item.unit_price,
                        'quantity': item.quantity,
                        'total': item.unit_price * item.quantity,
                    })
            else:
                order_data['item_count'] = order.items.count()
            
            export_data['orders'].append(order_data)
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, cls=DecimalEncoder, indent=2)
    
    def export_summary(self, orders_query, output_file, start_date, end_date):
        """Export summary report - like the daily kitchen performance report"""
        # Aggregate statistics (like counting portions served)
        stats = orders_query.aggregate(
            total_orders=Count('id'),
            total_revenue=Sum('total_amount'),
            avg_order_value=Sum('total_amount') / Count('id')
        )
        
        # Status breakdown (like tracking order stages)
        status_stats = {}
        for status, _ in Order.status_choices:
            count = orders_query.filter(status=status).count()
            status_stats[status] = count
        
        # Top menu items (like bestselling dishes)
        top_items = (
            OrderItem.objects
            .filter(order__in=orders_query)
            .values('menu_item__name', 'menu_item__category__name')
            .annotate(
                total_quantity=Sum('quantity'),
                total_revenue=Sum('unit_price') * Sum('quantity')
            )
            .order_by('-total_quantity')[:10]
        )
        
        with open(output_file, 'w', encoding='utf-8') as txtfile:
            txtfile.write("🍽️  RESTAURANT SALES SUMMARY REPORT\n")
            txtfile.write("=" * 50 + "\n\n")
            
            # Date range
            if start_date and end_date:
                txtfile.write(f"📅 Period: {start_date} to {end_date}\n")
            elif start_date:
                txtfile.write(f"📅 From: {start_date}\n")
            elif end_date:
                txtfile.write(f"📅 Until: {end_date}\n")
            else:
                txtfile.write("📅 Period: All time\n")
            
            txtfile.write(f"📊 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            txtfile.write("💰 FINANCIAL OVERVIEW\n")
            txtfile.write("-" * 25 + "\n")
            txtfile.write(f"Total Orders: {stats['total_orders'] or 0}\n")
            txtfile.write(f"Total Revenue: ${stats['total_revenue'] or 0:.2f}\n")
            txtfile.write(f"Average Order Value: ${stats['avg_order_value'] or 0:.2f}\n\n")
            
            # Status breakdown
            txtfile.write("📋 ORDER STATUS BREAKDOWN\n")
            txtfile.write("-" * 30 + "\n")
            for status, count in status_stats.items():
                txtfile.write(f"{status.title()}: {count}\n")
            txtfile.write("\n")
            
            # Top menu items
            txtfile.write("🏆 TOP 10 MENU ITEMS\n")
            txtfile.write("-" * 25 + "\n")
            for i, item in enumerate(top_items, 1):
                txtfile.write(
                    f"{i}. {item['menu_item__name']} "
                    f"({item['menu_item__category__name']}) - "
                    f"{item['total_quantity']} sold\n"
                )
```

### Sample CSV File for Import

Create a sample CSV file called `menu_items.csv`:

```csv
name,category,price,description,ingredients,calories,prep_time_minutes,is_available
"Classic Burger","Main Course",12.99,"Juicy beef patty with lettuce and tomato","Ground beef, lettuce, tomato, bun, cheese",650,15,true
"Caesar Salad","Salad",8.99,"Fresh romaine with parmesan and croutons","Romaine lettuce, parmesan, croutons, caesar dressing",320,5,true
"Chocolate Cake","Dessert",6.99,"Rich chocolate cake with ganache","Chocolate, flour, eggs, butter, cream",450,20,true
"Fish Tacos","Main Course",14.99,"Grilled fish with fresh salsa","Fish fillet, tortillas, cabbage, salsa, lime",420,12,true
"Green Smoothie","Beverage",5.99,"Healthy blend of spinach and fruits","Spinach, banana, apple, yogurt, honey",180,3,true
```

### Running the Commands

**Import menu items:**
```bash
# Basic import
python manage.py import_menu_items menu_items.csv

# Dry run to preview
python manage.py import_menu_items menu_items.csv --dry-run

# Update existing items
python manage.py import_menu_items menu_items.csv --update-existing

# Process in smaller batches
python manage.py import_menu_items menu_items.csv --batch-size 50
```

**Export sales data:**
```bash
# Export all orders to CSV
python manage.py export_sales_data sales_report.csv

# Export with date range
python manage.py export_sales_data sales_report.csv --start-date 2024-01-01 --end-date 2024-12-31

# Export detailed JSON with items
python manage.py export_sales_data sales_data.json --include-items

# Export summary report
python manage.py export_sales_data daily_summary.txt --format summary

# Filter by status
python manage.py export_sales_data completed_orders.csv --status completed
```

### Code Syntax Explanations

**Key Django Management Command Concepts:**

1. **BaseCommand Class**: `from django.core.management.base import BaseCommand`
   - The foundation for all Django management commands
   - Like the basic recipe template that all our kitchen commands follow

2. **add_arguments() Method**: Defines command-line parameters
   - `parser.add_argument()` adds options like ingredients to a recipe
   - `action='store_true'` creates boolean flags (on/off switches)
   - `choices=[]` limits options to specific values
   - `type=int` ensures numeric input

3. **handle() Method**: The main command logic
   - Like the cooking instructions in a recipe
   - `*args, **options` receives all command-line arguments

4. **Database Operations:**
   - `get_or_create()`: Find existing record or create new one (like checking pantry before shopping)
   - `prefetch_related()`: Optimizes database queries by loading related objects
   - `transaction.atomic()`: Ensures all database operations succeed or fail together (like cooking steps that must all work)

5. **Error Handling:**
   - `CommandError`: Django-specific exception for command problems
   - `try/except`: Catches and handles errors gracefully
   - `raise ValueError`: Custom error messages for validation

6. **File Operations:**
   - `csv.DictReader()`: Reads CSV files as dictionaries (each row becomes a dict)
   - `json.dump()`: Writes Python objects to JSON format
   - Context managers (`with open()`) ensure files are properly closed

7. **Data Processing:**
   - `Decimal()`: Handles money/price values precisely
   - `.aggregate()`: Calculates summary statistics (SUM, COUNT, AVG)
   - `.annotate()`: Adds calculated fields to query results

This project demonstrates how Django management commands can handle complex data operations, from importing supplier data to generating comprehensive reports, all while maintaining data integrity and providing detailed feedback to users.