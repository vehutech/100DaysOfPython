## Day 47: Building Robust User Management Systems

### Introduction

Imagine that you're running a high-end restaurant where different staff members have different levels of access to various areas. The head chef can access all kitchen areas and modify recipes, sous chefs can prepare dishes but not change recipes, servers can view orders but not modify the kitchen inventory, and customers can only see the menu and place orders. This hierarchical system of permissions and roles is exactly what we'll be building in Django's user management system.

Just as a restaurant needs clear staff roles and access levels to function smoothly, web applications need robust user profiles and permission systems to maintain security and organize functionality effectively.

---

## Learning Objectives

By the end of this lesson, you will be able to:
- Extend Django's built-in User model to include additional profile information
- Create custom user models when the default User model isn't sufficient
- Implement Django's groups and permissions system for role-based access control
- Use decorators to control access to views based on user permissions
- Apply these concepts to build a realistic user management system

---

## Lesson 1: Extending the User Model

Think of Django's built-in User model as a basic chef's uniform that comes with just a name tag and basic identification. While functional, you often need to add more details like the chef's specialization, years of experience, favorite cooking style, or emergency contact information. Extending the User model is like adding custom patches and accessories to make the uniform more informative and useful.

### Why Extend the User Model?
Django's default User model includes:
- username
- first_name
- last_name  
- email
- password
- is_staff
- is_active
- is_superuser
- date_joined
- last_login

But what if you need to store a user's profile picture, bio, phone number, or preferences? This is where extending the User model becomes essential.

### Method 1: One-to-One Profile Model (Recommended)

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class UserProfile(models.Model):
    # Create a one-to-one relationship with User
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    
    # Additional fields for our "chef profile"
    bio = models.TextField(max_length=500, blank=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', blank=True)
    phone_number = models.CharField(max_length=15, blank=True)
    specialization = models.CharField(max_length=100, blank=True)
    years_experience = models.IntegerField(default=0)
    
    # Preference fields
    preferred_shift = models.CharField(
        max_length=20,
        choices=[
            ('morning', 'Morning Shift'),
            ('afternoon', 'Afternoon Shift'),
            ('evening', 'Evening Shift'),
            ('night', 'Night Shift'),
        ],
        default='morning'
    )
    
    def __str__(self):
        return f"{self.user.username}'s Profile"

# Signal to automatically create UserProfile when User is created
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.userprofile.save()
```

### Syntax Explanation:
- `models.OneToOneField(User, on_delete=models.CASCADE)`: Creates a one-to-one relationship with the User model. When a User is deleted, their profile is also deleted.
- `@receiver(post_save, sender=User)`: A Django signal decorator that automatically triggers when a User object is saved.
- `blank=True`: Allows the field to be empty in forms.
- `upload_to='profile_pics/'`: Specifies where uploaded images should be stored.

### Using the Extended Profile in Views

```python
# views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import UserProfile
from .forms import UserProfileForm

@login_required
def profile_view(request):
    """View to display user profile - like showing a chef's complete information"""
    try:
        profile = request.user.userprofile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)
    
    context = {
        'user': request.user,
        'profile': profile,
    }
    return render(request, 'users/profile.html', context)

@login_required
def edit_profile(request):
    """Edit profile view - like updating a chef's information"""
    profile = request.user.userprofile
    
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            return redirect('profile_view')
    else:
        form = UserProfileForm(instance=profile)
    
    return render(request, 'users/edit_profile.html', {'form': form})
```

### Syntax Explanation:
- `@login_required`: Decorator that ensures only authenticated users can access the view.
- `request.user.userprofile`: Accesses the related UserProfile through the one-to-one relationship.
- `request.FILES`: Handles file uploads (like profile pictures).

---

## Lesson 2: Custom User Models

Sometimes the basic chef uniform just won't work for your restaurant. Maybe you run a specialized sushi restaurant where you need to track each chef's sushi certification level, knife preferences, and rice preparation expertise. In these cases, you need to design a completely custom uniform system from scratch rather than just adding accessories to the standard one.

### When to Use Custom User Models
Create a custom user model when:
- You want to use email as the username field instead of username
- You need to add required fields that can't be null
- You want to remove fields from the default User model
- You need significantly different authentication logic

### Creating a Custom User Model

```python
# models.py
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models

class ChefUserManager(BaseUserManager):
    """Custom manager for our Chef User model"""
    
    def create_user(self, email, first_name, last_name, password=None):
        """Create a regular chef user"""
        if not email:
            raise ValueError('Chef must have an email address')
        
        email = self.normalize_email(email)
        user = self.model(
            email=email,
            first_name=first_name,
            last_name=last_name,
        )
        user.set_password(password)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, first_name, last_name, password):
        """Create a head chef (superuser)"""
        user = self.create_user(
            email=email,
            first_name=first_name,
            last_name=last_name,
            password=password,
        )
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user

class ChefUser(AbstractBaseUser, PermissionsMixin):
    """Custom user model for our restaurant system"""
    
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    
    # Chef-specific fields
    chef_id = models.CharField(max_length=10, unique=True)
    specialization = models.CharField(max_length=100)
    certification_level = models.CharField(
        max_length=20,
        choices=[
            ('apprentice', 'Apprentice Chef'),
            ('line_cook', 'Line Cook'),
            ('sous_chef', 'Sous Chef'),
            ('head_chef', 'Head Chef'),
            ('executive', 'Executive Chef'),
        ],
        default='apprentice'
    )
    
    hire_date = models.DateField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    
    objects = ChefUserManager()
    
    USERNAME_FIELD = 'email'  # Use email to login instead of username
    REQUIRED_FIELDS = ['first_name', 'last_name']
    
    def __str__(self):
        return f"{self.first_name} {self.last_name} - {self.certification_level}"
    
    def get_full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    def get_short_name(self):
        return self.first_name
```

### Settings Configuration

```python
# settings.py
AUTH_USER_MODEL = 'your_app.ChefUser'  # Replace 'your_app' with your actual app name
```

### Syntax Explanation:
- `AbstractBaseUser`: Provides the core implementation of a user model with authentication features.
- `PermissionsMixin`: Adds permission-related fields and methods.
- `BaseUserManager`: Provides helper methods for creating users.
- `USERNAME_FIELD = 'email'`: Specifies which field is used for authentication.
- `REQUIRED_FIELDS`: List of fields required when creating a user via createsuperuser command.

---

## Lesson 3: Groups and Permissions

In our restaurant, we have different types of staff with different responsibilities. Head chefs can modify recipes and manage inventory, sous chefs can prepare dishes and view recipes, servers can take orders and view the menu, and dishwashers can access cleaning supplies. Django's groups and permissions system works exactly like this - you create groups (job roles) and assign specific permissions (what they can do) to each group.

### Understanding Django Permissions

Django automatically creates four permissions for each model:
- `add_modelname`: Can create new instances
- `change_modelname`: Can modify existing instances  
- `delete_modelname`: Can delete instances
- `view_modelname`: Can view instances

### Creating Custom Permissions

```python
# models.py
from django.db import models
from django.contrib.auth.models import User

class Recipe(models.Model):
    name = models.CharField(max_length=200)
    ingredients = models.TextField()
    instructions = models.TextField()
    difficulty_level = models.CharField(
        max_length=20,
        choices=[
            ('easy', 'Easy'),
            ('medium', 'Medium'),
            ('hard', 'Hard'),
            ('expert', 'Expert'),
        ]
    )
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    
    class Meta:
        permissions = [
            ("can_approve_recipe", "Can approve recipes for publication"),
            ("can_modify_any_recipe", "Can modify any recipe regardless of creator"),
            ("can_view_secret_recipes", "Can view secret/premium recipes"),
        ]
    
    def __str__(self):
        return self.name

class KitchenInventory(models.Model):
    item_name = models.CharField(max_length=100)
    quantity = models.IntegerField()
    unit = models.CharField(max_length=20)
    reorder_level = models.IntegerField()
    
    class Meta:
        permissions = [
            ("can_manage_inventory", "Can add/remove inventory items"),
            ("can_order_supplies", "Can place orders for supplies"),
        ]
```

### Setting Up Groups and Permissions

```python
# management/commands/setup_groups.py
from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType
from myapp.models import Recipe, KitchenInventory

class Command(BaseCommand):
    help = 'Create user groups and assign permissions'
    
    def handle(self, *args, **options):
        # Create groups
        head_chef_group, created = Group.objects.get_or_create(name='Head Chef')
        sous_chef_group, created = Group.objects.get_or_create(name='Sous Chef')
        line_cook_group, created = Group.objects.get_or_create(name='Line Cook')
        server_group, created = Group.objects.get_or_create(name='Server')
        
        # Get content types
        recipe_ct = ContentType.objects.get_for_model(Recipe)
        inventory_ct = ContentType.objects.get_for_model(KitchenInventory)
        
        # Get permissions
        recipe_perms = Permission.objects.filter(content_type=recipe_ct)
        inventory_perms = Permission.objects.filter(content_type=inventory_ct)
        
        # Assign permissions to Head Chef (can do everything)
        head_chef_group.permissions.set(recipe_perms)
        head_chef_group.permissions.set(inventory_perms)
        
        # Assign permissions to Sous Chef
        sous_chef_permissions = [
            'view_recipe', 'add_recipe', 'change_recipe',
            'can_approve_recipe', 'view_kitcheninventory'
        ]
        for perm_name in sous_chef_permissions:
            try:
                perm = Permission.objects.get(codename=perm_name)
                sous_chef_group.permissions.add(perm)
            except Permission.DoesNotExist:
                pass
        
        # Assign permissions to Line Cook
        line_cook_permissions = ['view_recipe', 'add_recipe', 'view_kitcheninventory']
        for perm_name in line_cook_permissions:
            try:
                perm = Permission.objects.get(codename=perm_name)
                line_cook_group.permissions.add(perm)
            except Permission.DoesNotExist:
                pass
        
        # Assign permissions to Server
        server_permissions = ['view_recipe']
        for perm_name in server_permissions:
            try:
                perm = Permission.objects.get(codename=perm_name)
                server_group.permissions.add(perm)
            except Permission.DoesNotExist:
                pass
        
        self.stdout.write(self.style.SUCCESS('Successfully created groups and permissions'))
```

### Syntax Explanation:
- `Group.objects.get_or_create(name='Head Chef')`: Creates a group if it doesn't exist, or retrieves it if it does.
- `ContentType.objects.get_for_model(Recipe)`: Gets the content type for the Recipe model, needed for permission queries.
- `group.permissions.set(permissions)`: Assigns a set of permissions to a group.
- `group.permissions.add(permission)`: Adds a single permission to a group.

---

## Lesson 4: Decorators for Access Control

Think of decorators as security guards at different stations in your restaurant. Just as a security guard at the wine cellar checks if someone has the proper credentials before letting them in, decorators check if a user has the right permissions before allowing them to access certain views or functions.

### Built-in Permission Decorators

```python
# views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, permission_required, user_passes_test
from django.contrib.auth.models import Group
from django.http import HttpResponseForbidden
from django.contrib import messages
from .models import Recipe, KitchenInventory

@login_required
@permission_required('myapp.add_recipe', raise_exception=True)
def create_recipe(request):
    """Only users with add_recipe permission can create recipes"""
    if request.method == 'POST':
        # Recipe creation logic here
        pass
    return render(request, 'recipes/create_recipe.html')

@login_required
@permission_required('myapp.can_approve_recipe', raise_exception=True)
def approve_recipe(request, recipe_id):
    """Only users who can approve recipes can access this view"""
    recipe = get_object_or_404(Recipe, id=recipe_id)
    
    if request.method == 'POST':
        # Approval logic here
        recipe.is_approved = True
        recipe.save()
        messages.success(request, f'Recipe "{recipe.name}" has been approved!')
        return redirect('recipe_list')
    
    return render(request, 'recipes/approve_recipe.html', {'recipe': recipe})

@login_required
@permission_required(['myapp.can_manage_inventory', 'myapp.can_order_supplies'], raise_exception=True)
def manage_inventory(request):
    """Requires multiple permissions - like needing both keys to nuclear launch"""
    inventory_items = KitchenInventory.objects.all()
    return render(request, 'inventory/manage.html', {'items': inventory_items})
```

### Custom Decorators for Groups

```python
# decorators.py
from functools import wraps
from django.http import HttpResponseForbidden
from django.contrib.auth.decorators import user_passes_test
from django.core.exceptions import PermissionDenied

def group_required(group_name):
    """Decorator to check if user belongs to a specific group"""
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return HttpResponseForbidden("You must be logged in.")
            
            if not request.user.groups.filter(name=group_name).exists():
                return HttpResponseForbidden(f"You must be a {group_name} to access this page.")
            
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator

def chef_level_required(min_level):
    """Custom decorator to check chef certification level"""
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return HttpResponseForbidden("You must be logged in.")
            
            # Define hierarchy levels
            level_hierarchy = {
                'apprentice': 1,
                'line_cook': 2,
                'sous_chef': 3,
                'head_chef': 4,
                'executive': 5
            }
            
            # Check if user has required level (assuming custom user model)
            user_level = getattr(request.user, 'certification_level', 'apprentice')
            
            if level_hierarchy.get(user_level, 0) < level_hierarchy.get(min_level, 0):
                return HttpResponseForbidden(f"You need at least {min_level} certification to access this page.")
            
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator

# Usage examples
@group_required('Head Chef')
def secret_recipes(request):
    """Only Head Chefs can view secret recipes"""
    recipes = Recipe.objects.filter(is_secret=True)
    return render(request, 'recipes/secret_recipes.html', {'recipes': recipes})

@chef_level_required('sous_chef')
def advanced_techniques(request):
    """Only Sous Chef level and above can view advanced techniques"""
    return render(request, 'training/advanced_techniques.html')
```

### Class-Based View Decorators

```python
# views.py
from django.contrib.auth.mixins import LoginRequiredMixin, PermissionRequiredMixin
from django.views.generic import ListView, CreateView, UpdateView
from .models import Recipe

class RecipeListView(LoginRequiredMixin, ListView):
    """List all recipes - requires login"""
    model = Recipe
    template_name = 'recipes/recipe_list.html'
    context_object_name = 'recipes'

class RecipeCreateView(LoginRequiredMixin, PermissionRequiredMixin, CreateView):
    """Create new recipe - requires login and add_recipe permission"""
    model = Recipe
    fields = ['name', 'ingredients', 'instructions', 'difficulty_level']
    template_name = 'recipes/create_recipe.html'
    permission_required = 'myapp.add_recipe'
    
    def form_valid(self, form):
        form.instance.created_by = self.request.user
        return super().form_valid(form)

class RecipeUpdateView(LoginRequiredMixin, PermissionRequiredMixin, UpdateView):
    """Update recipe - requires change permission"""
    model = Recipe
    fields = ['name', 'ingredients', 'instructions', 'difficulty_level']
    template_name = 'recipes/update_recipe.html'
    permission_required = 'myapp.change_recipe'
    
    def get_object(self):
        obj = super().get_object()
        # Only allow users to edit their own recipes unless they have special permission
        if obj.created_by != self.request.user and not self.request.user.has_perm('myapp.can_modify_any_recipe'):
            raise PermissionDenied("You can only edit your own recipes.")
        return obj
```

### Syntax Explanation:
- `@wraps(view_func)`: Preserves the original function's metadata when creating decorators.
- `raise_exception=True`: Raises a PermissionDenied exception instead of redirecting to login.
- `user_passes_test`: A flexible decorator that takes a function to test user conditions.
- `LoginRequiredMixin`: Class-based view mixin that requires user authentication.
- `PermissionRequiredMixin`: Class-based view mixin that requires specific permissions.

---

## Code Examples Summary

Throughout this course, we've used several key Django concepts:

### Models and Relationships
- **OneToOneField**: Creates a one-to-one relationship between User and UserProfile
- **ForeignKey**: Creates many-to-one relationships
- **Meta class**: Defines model metadata including custom permissions

### Authentication and Authorization
- **User model extension**: Adding profile information without changing core User model
- **Custom User models**: Complete replacement of default User model
- **Groups and Permissions**: Role-based access control system
- **Decorators**: Function-based access control mechanisms

### Django Signals
- **post_save**: Automatically triggers actions when models are saved
- **@receiver**: Decorator for connecting signal handlers

### Management Commands
- **BaseCommand**: Base class for creating custom Django management commands
- **handle method**: Main execution method for management commands

---
# Project: Building a User Profiles with Role-Based Access

## Project Objective
By the end of this lesson, you will be able to create a comprehensive user profile system with role-based access control, implementing different permission levels for various user types in a Django web application.

---

## Introduction: The Restaurant Management System

Imagine that you're building a management system for a high-end restaurant chain. Just like how a restaurant has different roles - head chef, sous chef, line cook, waiter, and manager - your web application needs different user types with varying levels of access and permissions.

In our analogy:
- **Head Chef** (Admin): Has access to everything - can modify recipes, manage staff, view finances
- **Sous Chef** (Manager): Can modify recipes, manage kitchen staff, but can't access financial data
- **Line Cook** (Staff): Can view recipes and update inventory, but can't modify recipes
- **Waiter** (Basic User): Can only view menu items and take orders

Let's build this step by step, creating a user profile system that manages these different roles and their permissions.

---

## Building User Profiles with Role-Based Access

### Step 1: Setting Up the Project Structure

First, let's create our Django project structure:

```python
# models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.contrib.auth.models import Group, Permission

class CustomUser(AbstractUser):
    """
    Custom user model extending Django's AbstractUser
    Think of this as the employee ID card that contains basic info
    """
    ROLE_CHOICES = [
        ('admin', 'Head Chef'),
        ('manager', 'Sous Chef'), 
        ('staff', 'Line Cook'),
        ('basic', 'Waiter'),
    ]
    
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='basic')
    phone_number = models.CharField(max_length=15, blank=True)
    hire_date = models.DateField(auto_now_add=True)
    is_active_employee = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.username} - {self.get_role_display()}"

class UserProfile(models.Model):
    """
    Extended profile information for each user
    Like the detailed employee file in HR
    """
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
    bio = models.TextField(max_length=500, blank=True)
    location = models.CharField(max_length=100, blank=True)
    birth_date = models.DateField(null=True, blank=True)
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    
    # Role-specific fields
    years_experience = models.IntegerField(default=0)
    specialization = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
```

**Syntax Explanation:**
- `AbstractUser`: Django's built-in user model that we extend rather than creating from scratch
- `OneToOneField`: Creates a one-to-one relationship (each user has exactly one profile)
- `choices`: Provides predefined options for a field
- `get_role_display()`: Django method that returns the human-readable version of choice fields

### Step 2: Creating the Views with Role-Based Access

```python
# views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib import messages
from django.http import HttpResponseForbidden
from .models import CustomUser, UserProfile
from .forms import UserProfileForm, UserRegistrationForm

def role_required(allowed_roles):
    """
    Custom decorator - like a bouncer at the restaurant
    Only lets certain roles enter specific areas
    """
    def decorator(view_func):
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('login')
            
            if request.user.role not in allowed_roles:
                return HttpResponseForbidden("You don't have permission to access this area.")
            
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator

@login_required
def dashboard(request):
    """
    Main dashboard - like the kitchen's command center
    Different views based on user role
    """
    context = {
        'user': request.user,
        'user_profile': getattr(request.user, 'userprofile', None),
    }
    
    # Different dashboard content based on role
    if request.user.role == 'admin':
        context['can_manage_users'] = True
        context['can_view_reports'] = True
        template = 'dashboard/admin_dashboard.html'
    elif request.user.role == 'manager':
        context['can_manage_staff'] = True
        context['can_view_inventory'] = True
        template = 'dashboard/manager_dashboard.html'
    elif request.user.role == 'staff':
        context['can_update_inventory'] = True
        context['can_view_recipes'] = True
        template = 'dashboard/staff_dashboard.html'
    else:  # basic user
        context['can_view_menu'] = True
        template = 'dashboard/basic_dashboard.html'
    
    return render(request, template, context)

@login_required
def profile_view(request):
    """
    View user's own profile - like checking your employee file
    """
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    context = {
        'user': request.user,
        'profile': profile,
        'is_own_profile': True,
    }
    
    return render(request, 'profiles/profile_detail.html', context)

@login_required
def profile_edit(request):
    """
    Edit user's own profile - like updating your employee information
    """
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile_view')
    else:
        form = UserProfileForm(instance=profile)
    
    context = {
        'form': form,
        'profile': profile,
    }
    
    return render(request, 'profiles/profile_edit.html', context)

@role_required(['admin', 'manager'])
def user_management(request):
    """
    Manage other users - only for head chef and sous chef
    """
    users = CustomUser.objects.all().select_related('userprofile')
    
    context = {
        'users': users,
        'can_delete_users': request.user.role == 'admin',
    }
    
    return render(request, 'management/user_list.html', context)

@role_required(['admin'])
def user_detail(request, user_id):
    """
    View detailed user information - only for head chef
    """
    user = get_object_or_404(CustomUser, id=user_id)
    profile = getattr(user, 'userprofile', None)
    
    context = {
        'viewed_user': user,
        'viewed_profile': profile,
    }
    
    return render(request, 'management/user_detail.html', context)

def register(request):
    """
    New employee registration - like hiring process
    """
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Create profile automatically
            UserProfile.objects.create(user=user)
            
            login(request, user)
            messages.success(request, 'Welcome to the team! Please complete your profile.')
            return redirect('profile_edit')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'registration/register.html', {'form': form})
```

**Syntax Explanation:**
- `@login_required`: Decorator ensuring only logged-in users can access the view
- `get_object_or_404`: Returns object or 404 error if not found
- `getattr(obj, 'attribute', default)`: Safely gets attribute, returns default if doesn't exist
- `select_related()`: Optimizes database queries by joining related tables
- `HttpResponseForbidden`: Returns 403 Forbidden HTTP response

### Step 3: Creating Forms

```python
# forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser, UserProfile

class UserRegistrationForm(UserCreationForm):
    """
    Registration form - like the job application
    """
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    phone_number = forms.CharField(max_length=15, required=False)
    
    class Meta:
        model = CustomUser
        fields = ('username', 'first_name', 'last_name', 'email', 
                 'phone_number', 'role', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add CSS classes for styling
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'

class UserProfileForm(forms.ModelForm):
    """
    Profile editing form - like updating your employee file
    """
    birth_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        required=False
    )
    
    class Meta:
        model = UserProfile
        fields = ('bio', 'location', 'birth_date', 'avatar', 
                 'years_experience', 'specialization')
        widgets = {
            'bio': forms.Textarea(attrs={'rows': 4, 'class': 'form-control'}),
            'location': forms.TextInput(attrs={'class': 'form-control'}),
            'years_experience': forms.NumberInput(attrs={'class': 'form-control'}),
            'specialization': forms.TextInput(attrs={'class': 'form-control'}),
        }
```

### Step 4: Templates

```html
<!-- templates/dashboard/admin_dashboard.html -->
{% extends 'base.html' %}

{% block title %}Head Chef Dashboard{% endblock %}

{% block content %}
<div class="container mt-4">
    <div class="row">
        <div class="col-md-12">
            <h1>Welcome, Head Chef {{ user.first_name }}!</h1>
            <p class="lead">You have full access to the restaurant management system.</p>
        </div>
    </div>
    
    <div class="row mt-4">
        <div class="col-md-3">
            <div class="card text-white bg-primary">
                <div class="card-body">
                    <h5 class="card-title">Manage Staff</h5>
                    <p class="card-text">Add, edit, or remove restaurant staff</p>
                    <a href="{% url 'user_management' %}" class="btn btn-light">Manage Users</a>
                </div>
            </div>
        </div>
        
        <div class="col-md-3">
            <div class="card text-white bg-success">
                <div class="card-body">
                    <h5 class="card-title">View Reports</h5>
                    <p class="card-text">Access financial and operational reports</p>
                    <a href="#" class="btn btn-light">View Reports</a>
                </div>
            </div>
        </div>
        
        <div class="col-md-3">
            <div class="card text-white bg-info">
                <div class="card-body">
                    <h5 class="card-title">Recipe Management</h5>
                    <p class="card-text">Create and modify recipes</p>
                    <a href="#" class="btn btn-light">Manage Recipes</a>
                </div>
            </div>
        </div>
        
        <div class="col-md-3">
            <div class="card text-white bg-warning">
                <div class="card-body">
                    <h5 class="card-title">My Profile</h5>
                    <p class="card-text">Update your personal information</p>
                    <a href="{% url 'profile_view' %}" class="btn btn-light">View Profile</a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

```html
<!-- templates/profiles/profile_detail.html -->
{% extends 'base.html' %}

{% block title %}{{ user.first_name }}'s Profile{% endblock %}

{% block content %}
<div class="container mt-4">
    <div class="row">
        <div class="col-md-4">
            <div class="card">
                <div class="card-body text-center">
                    {% if profile.avatar %}
                        <img src="{{ profile.avatar.url }}" alt="Avatar" class="img-fluid rounded-circle mb-3" style="width: 150px; height: 150px;">
                    {% else %}
                        <div class="bg-secondary rounded-circle mx-auto mb-3" style="width: 150px; height: 150px; display: flex; align-items: center; justify-content: center;">
                            <i class="fas fa-user fa-3x text-white"></i>
                        </div>
                    {% endif %}
                    
                    <h4>{{ user.first_name }} {{ user.last_name }}</h4>
                    <p class="text-muted">{{ user.get_role_display }}</p>
                    
                    {% if is_own_profile %}
                        <a href="{% url 'profile_edit' %}" class="btn btn-primary">Edit Profile</a>
                    {% endif %}
                </div>
            </div>
        </div>
        
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <h5>Profile Information</h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <strong>Username:</strong> {{ user.username }}
                        </div>
                        <div class="col-md-6">
                            <strong>Email:</strong> {{ user.email }}
                        </div>
                    </div>
                    <hr>
                    <div class="row">
                        <div class="col-md-6">
                            <strong>Phone:</strong> {{ user.phone_number|default:"Not provided" }}
                        </div>
                        <div class="col-md-6">
                            <strong>Hire Date:</strong> {{ user.hire_date }}
                        </div>
                    </div>
                    <hr>
                    <div class="row">
                        <div class="col-md-6">
                            <strong>Experience:</strong> {{ profile.years_experience }} years
                        </div>
                        <div class="col-md-6">
                            <strong>Specialization:</strong> {{ profile.specialization|default:"Not specified" }}
                        </div>
                    </div>
                    <hr>
                    <div class="row">
                        <div class="col-md-12">
                            <strong>Bio:</strong>
                            <p>{{ profile.bio|default:"No bio provided" }}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

### Step 5: URL Configuration

```python
# urls.py
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Authentication URLs
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('register/', views.register, name='register'),
    
    # Dashboard URLs
    path('', views.dashboard, name='dashboard'),
    
    # Profile URLs
    path('profile/', views.profile_view, name='profile_view'),
    path('profile/edit/', views.profile_edit, name='profile_edit'),
    
    # Management URLs (restricted access)
    path('management/users/', views.user_management, name='user_management'),
    path('management/users/<int:user_id>/', views.user_detail, name='user_detail'),
]
```

### Step 6: Settings Configuration

```python
# settings.py
import os

# Custom user model
AUTH_USER_MODEL = 'your_app.CustomUser'

# Login/logout redirects
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/login/'

# Media files (for avatar uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Add to INSTALLED_APPS
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'your_app',  # Replace with your app name
]
```

---

## Final Quality Project: Restaurant Management System

Now let's put it all together into a complete, working system that demonstrates role-based access control:

```python
# Complete working example - management/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.core.paginator import Paginator
from .models import CustomUser, UserProfile

@role_required(['admin', 'manager'])
def staff_dashboard(request):
    """
    Staff management dashboard - like the restaurant's staff board
    """
    # Get all users with their profiles
    users = CustomUser.objects.select_related('userprofile').all()
    
    # Filter by role if specified
    role_filter = request.GET.get('role', 'all')
    if role_filter != 'all':
        users = users.filter(role=role_filter)
    
    # Search functionality
    search_query = request.GET.get('search', '')
    if search_query:
        users = users.filter(
            models.Q(username__icontains=search_query) |
            models.Q(first_name__icontains=search_query) |
            models.Q(last_name__icontains=search_query)
        )
    
    # Pagination - show 10 users per page
    paginator = Paginator(users, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Statistics for the dashboard
    stats = {
        'total_users': CustomUser.objects.count(),
        'admin_count': CustomUser.objects.filter(role='admin').count(),
        'manager_count': CustomUser.objects.filter(role='manager').count(),
        'staff_count': CustomUser.objects.filter(role='staff').count(),
        'basic_count': CustomUser.objects.filter(role='basic').count(),
    }
    
    context = {
        'page_obj': page_obj,
        'stats': stats,
        'role_filter': role_filter,
        'search_query': search_query,
        'role_choices': CustomUser.ROLE_CHOICES,
    }
    
    return render(request, 'management/staff_dashboard.html', context)

@role_required(['admin'])
def toggle_user_status(request, user_id):
    """
    Toggle user active status - like hiring/firing in the restaurant
    """
    if request.method == 'POST':
        user = get_object_or_404(CustomUser, id=user_id)
        user.is_active_employee = not user.is_active_employee
        user.save()
        
        status = "activated" if user.is_active_employee else "deactivated"
        messages.success(request, f"User {user.username} has been {status}.")
        
        return JsonResponse({'status': 'success', 'is_active': user.is_active_employee})
    
    return JsonResponse({'status': 'error'})
```

**Syntax Explanation:**
- `select_related()`: Optimizes queries by joining related tables in one query
- `Q objects`: Allows complex database queries with OR conditions
- `Paginator`: Breaks large datasets into pages for better performance
- `JsonResponse`: Returns JSON data for AJAX requests
- `models.Q`: Django's query object for complex database lookups

---

## Key Concepts Demonstrated

- **Custom User Model**: Employee ID system with different roles
- **User Profiles**: Detailed employee files with personal information
- **Role-Based Access**: Different kitchen areas for different staff levels
- **Decorators**: Security guards checking permissions before entry
- **Dashboard Views**: Different control panels for each role level

**Technical Concepts:**
- Extending Django's user model with custom fields
- One-to-one relationships between models
- Custom decorators for access control
- Role-based view rendering
- Form handling with file uploads
- Database query optimization
- Template inheritance and conditional rendering

This complete system demonstrates how to build a robust user management system with role-based permissions, just like organizing a professional kitchen where everyone has their specific responsibilities and access levels!


## Assignment: Restaurant Staff Management System

### Objective
Create a staff management system for a restaurant that demonstrates all the concepts learned in this course.

### Requirements

1. **User Profile Extension**: Extend the default User model with a StaffProfile that includes:
   - Profile picture
   - Phone number
   - Emergency contact
   - Department (Kitchen, Service, Management)
   - Hire date
   - Hourly wage

2. **Groups and Permissions Setup**: Create the following groups with appropriate permissions:
   - **Restaurant Manager**: Can manage all staff, view all reports, modify any profile
   - **Kitchen Manager**: Can manage kitchen staff, view kitchen reports, approve time-off requests
   - **Server Manager**: Can manage service staff, view service reports
   - **Staff Member**: Can view their own profile, submit time-off requests

3. **Protected Views**: Create the following views with proper permission decorators:
   - **Staff Directory**: Only managers can view all staff information
   - **Payroll Report**: Only Restaurant Managers can access
   - **Time-off Requests**: Kitchen/Server Managers can approve for their departments
   - **Profile Edit**: Staff can edit their own profiles, managers can edit any profile

4. **Custom Decorators**: Create a custom decorator called `@manager_required` that checks if a user belongs to any manager group.

### Deliverables

1. **Models file** (`models.py`) with StaffProfile model and custom permissions
2. **Views file** (`views.py`) with all required views and proper decorators
3. **Management command** (`setup_staff_groups.py`) to create groups and assign permissions
4. **Custom decorators file** (`decorators.py`) with the manager_required decorator
5. **Brief documentation** explaining how to set up and use the system

### Success Criteria
- All views properly protected with appropriate permissions
- Groups and permissions correctly configured
- Custom decorator works as expected
- Staff can only access features appropriate to their role
- Code is well-commented and follows Django best practices

### Bonus Points
- Add email notifications when time-off requests are approved/denied
- Create a dashboard showing different content based on user role
- Add unit tests for your custom decorators and permissions

This assignment will solidify your understanding of Django's user management system and prepare you for building robust, secure web applications with proper access control.