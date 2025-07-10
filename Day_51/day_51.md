# Day 51: Class-Based Views Mastery Course

## Learning Objective
By the end of this lesson, you will be able to create and customize Django class-based views using generic views, implement ListView and DetailView patterns, build custom mixins for reusable functionality, and understand when to use view decorators versus mixins in your Django applications.

---

## Introduction: The Master Chef's Kitchen

Imagine that you're stepping into a world-class restaurant kitchen where everything runs like clockwork. The head chef doesn't personally prepare every single dish from scratch - instead, they have specialized stations, pre-made sauces, and trained sous chefs who handle specific tasks. This is exactly what Django's Class-Based Views (CBVs) offer us: a professional kitchen setup where we can leverage pre-built "cooking stations" (generic views) and create our own "signature sauces" (custom mixins) to efficiently serve up web applications.

Just as a master chef builds upon classic French techniques while adding their own flair, we'll learn to use Django's built-in view classes while customizing them to meet our specific needs.

---

## Lesson 1: Generic Class-Based Views - The Foundation Recipes

### The Chef's Analogy
Think of generic class-based views as your basic mother sauces in French cuisine - béchamel, velouté, espagnole, hollandaise, and tomato. These five sauces form the foundation for hundreds of other sauces. Similarly, Django's generic views provide the foundation for most web application patterns.

### What Are Generic Class-Based Views?

Generic class-based views are pre-built view classes that handle common web development patterns. Instead of writing the same code repeatedly, we inherit from these base classes and customize them.

### Key Generic Views and Their Kitchen Equivalents

```python
# views.py
from django.views.generic import (
    TemplateView,    # Like a display plate - shows content
    ListView,        # Like a buffet line - displays multiple items
    DetailView,      # Like a featured dish - shows one item in detail
    CreateView,      # Like a prep station - creates new items
    UpdateView,      # Like a finishing station - modifies existing items
    DeleteView,      # Like waste disposal - removes items
)
from django.urls import reverse_lazy
from .models import Recipe

# Basic TemplateView - The Display Plate
class HomeView(TemplateView):
    template_name = 'home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['restaurant_name'] = "Django Chef's Kitchen"
        context['featured_dish'] = "Today's Special"
        return context
```

### Syntax Explanation:
- **`TemplateView`**: Inherits from Django's base template view
- **`template_name`**: Specifies which HTML template to render
- **`get_context_data()`**: Method to pass additional data to the template
- **`super()`**: Calls the parent class method to maintain inheritance chain
- **`**kwargs`**: Accepts any keyword arguments passed to the method

---

## Lesson 2: ListView, DetailView, CreateView - The Specialized Stations

### The Chef's Analogy
In our kitchen, we have specialized stations: the salad station (ListView) displays all available ingredients, the presentation station (DetailView) focuses on plating one perfect dish, and the prep station (CreateView) is where new dishes are born.

### ListView - The Buffet Line

```python
# views.py
from django.views.generic import ListView
from .models import Recipe

class RecipeListView(ListView):
    model = Recipe
    template_name = 'recipes/recipe_list.html'
    context_object_name = 'recipes'
    paginate_by = 10
    ordering = ['-created_at']
    
    def get_queryset(self):
        """Filter recipes like a chef selecting ingredients"""
        queryset = super().get_queryset()
        category = self.request.GET.get('category')
        if category:
            queryset = queryset.filter(category=category)
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Recipe.CATEGORY_CHOICES
        context['current_category'] = self.request.GET.get('category', '')
        return context
```

### DetailView - The Featured Dish

```python
# views.py
from django.views.generic import DetailView
from django.shortcuts import get_object_or_404

class RecipeDetailView(DetailView):
    model = Recipe
    template_name = 'recipes/recipe_detail.html'
    context_object_name = 'recipe'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        recipe = self.get_object()
        
        # Add related data like garnishes to a main dish
        context['similar_recipes'] = Recipe.objects.filter(
            category=recipe.category
        ).exclude(id=recipe.id)[:3]
        
        context['ingredients_count'] = recipe.ingredients.count()
        return context
```

### CreateView - The Prep Station

```python
# views.py
from django.views.generic import CreateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from .forms import RecipeForm

class RecipeCreateView(LoginRequiredMixin, CreateView):
    model = Recipe
    form_class = RecipeForm
    template_name = 'recipes/recipe_create.html'
    success_url = reverse_lazy('recipe:list')
    
    def form_valid(self, form):
        """Like a chef signing their dish"""
        form.instance.chef = self.request.user
        messages.success(self.request, 'Recipe created successfully!')
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['action'] = 'Create'
        context['button_text'] = 'Add Recipe to Kitchen'
        return context
```

### Syntax Explanation:
- **`model`**: Specifies which model this view operates on
- **`context_object_name`**: Custom name for the object in templates (default is 'object')
- **`paginate_by`**: Number of items per page
- **`ordering`**: How to sort the queryset
- **`get_queryset()`**: Custom method to filter or modify the data
- **`LoginRequiredMixin`**: Requires user authentication (must be first in inheritance)
- **`form_valid()`**: Called when form submission is successful
- **`reverse_lazy()`**: Lazy evaluation of URL reversal (important for class-based views)

---

## Lesson 3: Custom Mixins - Your Signature Sauces

### The Chef's Analogy
Just as a chef creates signature sauces that can enhance multiple dishes, mixins are reusable pieces of functionality that can be added to multiple views. They're like having your own collection of special seasonings that you can sprinkle on any dish.

### Creating Custom Mixins

```python
# mixins.py
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.shortcuts import redirect
from django.core.exceptions import PermissionDenied

class ChefOwnerMixin:
    """Ensure only the chef who created the recipe can modify it"""
    
    def dispatch(self, request, *args, **kwargs):
        obj = self.get_object()
        if obj.chef != request.user:
            raise PermissionDenied("You can only modify your own recipes!")
        return super().dispatch(request, *args, **kwargs)

class SuccessMessageMixin:
    """Add success messages like a chef announcing a completed dish"""
    success_message = "Operation completed successfully!"
    
    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, self.success_message)
        return response

class KitchenContextMixin:
    """Add common kitchen data to all views"""
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['kitchen_stats'] = {
            'total_recipes': Recipe.objects.count(),
            'active_chefs': User.objects.filter(is_active=True).count(),
            'featured_category': 'Italian',
        }
        return context

# Using multiple mixins together
class RecipeUpdateView(LoginRequiredMixin, ChefOwnerMixin, 
                       SuccessMessageMixin, KitchenContextMixin, UpdateView):
    model = Recipe
    form_class = RecipeForm
    template_name = 'recipes/recipe_form.html'
    success_message = "Recipe updated successfully! Your dish is ready to serve."
    
    def get_success_url(self):
        return reverse_lazy('recipe:detail', kwargs={'pk': self.object.pk})
```

### Advanced Mixin Example

```python
# mixins.py
class AjaxResponseMixin:
    """Handle AJAX requests like a chef handling express orders"""
    
    def dispatch(self, request, *args, **kwargs):
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return self.ajax_dispatch(request, *args, **kwargs)
        return super().dispatch(request, *args, **kwargs)
    
    def ajax_dispatch(self, request, *args, **kwargs):
        """Handle AJAX requests differently"""
        from django.http import JsonResponse
        
        try:
            response = super().dispatch(request, *args, **kwargs)
            if hasattr(response, 'context_data'):
                return JsonResponse({
                    'success': True,
                    'data': response.context_data
                })
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
```

### Syntax Explanation:
- **Mixin Order**: Mixins are processed left to right, so order matters
- **`dispatch()`**: First method called on every request
- **`get_object()`**: Retrieves the model instance
- **`*args, **kwargs`**: Captures all positional and keyword arguments
- **Method Resolution Order (MRO)**: Python's system for determining which method to call in multiple inheritance

---

## Lesson 4: View Decorators vs Mixins - Choosing Your Tools

### The Chef's Analogy
Think of decorators as quick garnishes you can add to any dish at the last moment, while mixins are like marinating ingredients - they need to be integrated into the cooking process from the beginning. Both have their place in a well-organized kitchen.

### View Decorators Approach

```python
# Using decorators (traditional approach)
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

@method_decorator(login_required, name='dispatch')
@method_decorator(require_POST, name='post')
class RecipeQuickActionView(View):
    """Like a chef's quick prep station"""
    
    def post(self, request, *args, **kwargs):
        # Handle quick actions
        action = request.POST.get('action')
        recipe_id = request.POST.get('recipe_id')
        
        if action == 'favorite':
            # Quick favorite action
            recipe = get_object_or_404(Recipe, id=recipe_id)
            # Add to favorites logic
            return JsonResponse({'success': True, 'message': 'Added to favorites!'})
        
        return JsonResponse({'success': False, 'message': 'Invalid action'})
```

### Mixins Approach (Recommended)

```python
# Using mixins (modern approach)
class RecipeActionView(LoginRequiredMixin, View):
    """Using mixins for better organization"""
    
    def post(self, request, *args, **kwargs):
        # Same logic as above
        pass

# Comparison: Decorator vs Mixin for complex scenarios
class RecipeManagementView(LoginRequiredMixin, ChefOwnerMixin, View):
    """Mixins allow for complex combinations"""
    
    def get(self, request, *args, **kwargs):
        # Multiple mixins work together seamlessly
        return render(request, 'recipes/manage.html', {
            'user_recipes': Recipe.objects.filter(chef=request.user),
            'can_edit': True,  # ChefOwnerMixin ensures this
        })

# When to use decorators vs mixins
class HybridView(LoginRequiredMixin, View):
    """Sometimes you need both approaches"""
    
    @method_decorator(csrf_exempt)  # Specific decorator for this method
    def post(self, request, *args, **kwargs):
        # LoginRequiredMixin handles authentication
        # csrf_exempt decorator handles CSRF for this specific method
        return JsonResponse({'status': 'success'})
```

### Decision Guide: Decorators vs Mixins

```python
# Use DECORATORS when:
# 1. Simple, one-off functionality
# 2. Method-specific requirements
# 3. Third-party decorators that don't have mixin equivalents

@method_decorator(cache_page(60 * 5), name='get')  # Cache for 5 minutes
class FastRecipeListView(ListView):
    model = Recipe
    template_name = 'recipes/fast_list.html'

# Use MIXINS when:
# 1. Complex, reusable functionality
# 2. Multiple related methods need the same behavior
# 3. You want to combine multiple behaviors
# 4. Object-oriented approach fits better

class AdminRequiredMixin:
    """Reusable admin check - like head chef privileges"""
    
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_staff:
            raise PermissionDenied("Head chef access required!")
        return super().dispatch(request, *args, **kwargs)

class KitchenAdminView(AdminRequiredMixin, TemplateView):
    template_name = 'admin/kitchen_dashboard.html'
```

### Syntax Explanation:
- **`@method_decorator()`**: Applies function decorators to class methods
- **`name='dispatch'`**: Specifies which method to decorate
- **`csrf_exempt`**: Disables CSRF protection for specific views
- **`cache_page()`**: Caches the view response for specified time
- **Multiple decorators**: Can be stacked, applied from bottom to top

---

# Day 51: Class-Based Views Mastery - Digital Art Gallery Project

## **Build**: Digital Art Gallery with Complete CRUD Operations

Imagine that you're a head chef running an exclusive culinary exhibition where food artists showcase their signature dishes. Just like how a master chef organizes their kitchen with specialized stations - prep, cooking, plating, and presentation - we'll build a Digital Art Gallery where artists can showcase their masterpieces using Django's Class-Based Views as our specialized kitchen stations.

### The Project: ArtVault - Digital Art Gallery

We'll create a sophisticated art gallery platform where artists can upload, display, and manage their digital artwork. Think of it as the MasterChef kitchen of the digital art world!

### Project Structure

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone

class Artist(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(max_length=500, blank=True)
    profile_picture = models.ImageField(upload_to='artists/', blank=True)
    website = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.get_full_name()} - {self.user.username}"
    
    def get_absolute_url(self):
        return reverse('artist-detail', kwargs={'pk': self.pk})

class ArtCategory(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    
    class Meta:
        verbose_name_plural = "Art Categories"
    
    def __str__(self):
        return self.name

class Artwork(models.Model):
    MEDIUM_CHOICES = [
        ('digital', 'Digital Art'),
        ('painting', 'Painting'),
        ('photography', 'Photography'),
        ('sculpture', 'Sculpture'),
        ('mixed', 'Mixed Media'),
    ]
    
    title = models.CharField(max_length=200)
    artist = models.ForeignKey(Artist, on_delete=models.CASCADE, related_name='artworks')
    category = models.ForeignKey(ArtCategory, on_delete=models.SET_NULL, null=True)
    description = models.TextField()
    medium = models.CharField(max_length=20, choices=MEDIUM_CHOICES)
    image = models.ImageField(upload_to='artworks/')
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    is_for_sale = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    featured = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} by {self.artist.user.get_full_name()}"
    
    def get_absolute_url(self):
        return reverse('artwork-detail', kwargs={'pk': self.pk})

class ArtworkView(models.Model):
    artwork = models.ForeignKey(Artwork, on_delete=models.CASCADE, related_name='views')
    viewer_ip = models.GenericIPAddressField()
    viewed_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['artwork', 'viewer_ip']
```

### Views - Our Kitchen Stations

```python
# views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Count
from django.core.paginator import Paginator
from django.urls import reverse_lazy
from .models import Artwork, Artist, ArtCategory, ArtworkView
from .forms import ArtworkForm, ArtistForm

# Gallery Station - Display all artworks
class ArtworkListView(ListView):
    model = Artwork
    template_name = 'gallery/artwork_list.html'
    context_object_name = 'artworks'
    paginate_by = 12
    
    def get_queryset(self):
        queryset = Artwork.objects.select_related('artist__user', 'category')
        
        # Search functionality
        search_query = self.request.GET.get('search')
        if search_query:
            queryset = queryset.filter(
                Q(title__icontains=search_query) |
                Q(artist__user__first_name__icontains=search_query) |
                Q(artist__user__last_name__icontains=search_query) |
                Q(description__icontains=search_query)
            )
        
        # Filter by category
        category_id = self.request.GET.get('category')
        if category_id:
            queryset = queryset.filter(category_id=category_id)
        
        # Filter by medium
        medium = self.request.GET.get('medium')
        if medium:
            queryset = queryset.filter(medium=medium)
        
        # Filter for sale items
        for_sale = self.request.GET.get('for_sale')
        if for_sale == 'true':
            queryset = queryset.filter(is_for_sale=True)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = ArtCategory.objects.all()
        context['medium_choices'] = Artwork.MEDIUM_CHOICES
        context['featured_artworks'] = Artwork.objects.filter(featured=True)[:6]
        context['search_query'] = self.request.GET.get('search', '')
        context['selected_category'] = self.request.GET.get('category')
        context['selected_medium'] = self.request.GET.get('medium')
        return context

# Showcase Station - Display single artwork
class ArtworkDetailView(DetailView):
    model = Artwork
    template_name = 'gallery/artwork_detail.html'
    context_object_name = 'artwork'
    
    def get_object(self):
        artwork = super().get_object()
        # Track artwork views
        viewer_ip = self.get_client_ip()
        ArtworkView.objects.get_or_create(
            artwork=artwork,
            viewer_ip=viewer_ip
        )
        return artwork
    
    def get_client_ip(self):
        x_forwarded_for = self.request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = self.request.META.get('REMOTE_ADDR')
        return ip
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        artwork = self.get_object()
        context['related_artworks'] = Artwork.objects.filter(
            category=artwork.category
        ).exclude(id=artwork.id)[:4]
        context['view_count'] = artwork.views.count()
        context['other_works'] = Artwork.objects.filter(
            artist=artwork.artist
        ).exclude(id=artwork.id)[:3]
        return context

# Creation Station - Add new artwork
class ArtworkCreateView(LoginRequiredMixin, CreateView):
    model = Artwork
    form_class = ArtworkForm
    template_name = 'gallery/artwork_form.html'
    
    def form_valid(self, form):
        # Get or create artist profile for the user
        artist, created = Artist.objects.get_or_create(
            user=self.request.user,
            defaults={'bio': 'Artist biography coming soon...'}
        )
        form.instance.artist = artist
        messages.success(self.request, 'Your artwork has been uploaded successfully!')
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Upload New Artwork'
        return context

# Update Station - Edit artwork
class ArtworkUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Artwork
    form_class = ArtworkForm
    template_name = 'gallery/artwork_form.html'
    
    def test_func(self):
        artwork = self.get_object()
        return self.request.user == artwork.artist.user
    
    def form_valid(self, form):
        messages.success(self.request, 'Your artwork has been updated successfully!')
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Edit Artwork'
        return context

# Removal Station - Delete artwork
class ArtworkDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Artwork
    template_name = 'gallery/artwork_confirm_delete.html'
    success_url = reverse_lazy('artwork-list')
    
    def test_func(self):
        artwork = self.get_object()
        return self.request.user == artwork.artist.user
    
    def delete(self, request, *args, **kwargs):
        messages.success(self.request, 'Your artwork has been deleted successfully!')
        return super().delete(request, *args, **kwargs)

# Artist Profile Station
class ArtistDetailView(DetailView):
    model = Artist
    template_name = 'gallery/artist_detail.html'
    context_object_name = 'artist'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        artist = self.get_object()
        artworks = artist.artworks.all()
        context['artworks'] = artworks
        context['artwork_count'] = artworks.count()
        context['total_views'] = sum(artwork.views.count() for artwork in artworks)
        context['featured_works'] = artworks.filter(featured=True)[:3]
        return context

# Artist Profile Update
class ArtistUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Artist
    form_class = ArtistForm
    template_name = 'gallery/artist_form.html'
    
    def test_func(self):
        artist = self.get_object()
        return self.request.user == artist.user
    
    def form_valid(self, form):
        messages.success(self.request, 'Your profile has been updated successfully!')
        return super().form_valid(form)

# Dashboard - Artist's personal kitchen
class ArtistDashboardView(LoginRequiredMixin, ListView):
    model = Artwork
    template_name = 'gallery/artist_dashboard.html'
    context_object_name = 'artworks'
    
    def get_queryset(self):
        artist, created = Artist.objects.get_or_create(
            user=self.request.user,
            defaults={'bio': 'Artist biography coming soon...'}
        )
        return Artwork.objects.filter(artist=artist).order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        artist, created = Artist.objects.get_or_create(
            user=self.request.user,
            defaults={'bio': 'Artist biography coming soon...'}
        )
        artworks = self.get_queryset()
        context['artist'] = artist
        context['total_artworks'] = artworks.count()
        context['total_views'] = sum(artwork.views.count() for artwork in artworks)
        context['featured_count'] = artworks.filter(featured=True).count()
        context['for_sale_count'] = artworks.filter(is_for_sale=True).count()
        context['recent_artworks'] = artworks[:5]
        return context
```

### Forms - Our Recipe Cards

```python
# forms.py
from django import forms
from .models import Artwork, Artist, ArtCategory

class ArtworkForm(forms.ModelForm):
    class Meta:
        model = Artwork
        fields = ['title', 'category', 'description', 'medium', 'image', 'price', 'is_for_sale']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter artwork title'
            }),
            'category': forms.Select(attrs={
                'class': 'form-select'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe your artwork, inspiration, technique...'
            }),
            'medium': forms.Select(attrs={
                'class': 'form-select'
            }),
            'image': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
            'price': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': '0.00',
                'step': '0.01'
            }),
            'is_for_sale': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            })
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['price'].required = False
        self.fields['category'].empty_label = "Select a category"

class ArtistForm(forms.ModelForm):
    first_name = forms.CharField(max_length=30)
    last_name = forms.CharField(max_length=30)
    
    class Meta:
        model = Artist
        fields = ['first_name', 'last_name', 'bio', 'profile_picture', 'website']
        widgets = {
            'first_name': forms.TextInput(attrs={
                'class': 'form-control'
            }),
            'last_name': forms.TextInput(attrs={
                'class': 'form-control'
            }),
            'bio': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Tell us about yourself, your artistic journey...'
            }),
            'profile_picture': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
            'website': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://your-website.com'
            })
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.user:
            self.fields['first_name'].initial = self.instance.user.first_name
            self.fields['last_name'].initial = self.instance.user.last_name
    
    def save(self, commit=True):
        artist = super().save(commit=False)
        if commit:
            artist.user.first_name = self.cleaned_data['first_name']
            artist.user.last_name = self.cleaned_data['last_name']
            artist.user.save()
            artist.save()
        return artist
```

### URLs - Our Kitchen Layout

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Gallery routes
    path('', views.ArtworkListView.as_view(), name='artwork-list'),
    path('artwork/<int:pk>/', views.ArtworkDetailView.as_view(), name='artwork-detail'),
    
    # Artist routes
    path('artist/<int:pk>/', views.ArtistDetailView.as_view(), name='artist-detail'),
    path('artist/<int:pk>/edit/', views.ArtistUpdateView.as_view(), name='artist-update'),
    path('dashboard/', views.ArtistDashboardView.as_view(), name='artist-dashboard'),
    
    # Artwork management routes
    path('artwork/new/', views.ArtworkCreateView.as_view(), name='artwork-create'),
    path('artwork/<int:pk>/edit/', views.ArtworkUpdateView.as_view(), name='artwork-update'),
    path('artwork/<int:pk>/delete/', views.ArtworkDeleteView.as_view(), name='artwork-delete'),
]
```

### Key Templates

```html
<!-- gallery/artwork_list.html -->
{% extends 'base.html' %}
{% load static %}

{% block content %}
<div class="container-fluid py-4">
    <!-- Search and Filter Bar -->
    <div class="row mb-4">
        <div class="col-md-8">
            <form method="GET" class="d-flex">
                <input type="text" name="search" class="form-control me-2" 
                       placeholder="Search artworks, artists..." value="{{ search_query }}">
                <select name="category" class="form-select me-2">
                    <option value="">All Categories</option>
                    {% for category in categories %}
                        <option value="{{ category.id }}" 
                                {% if category.id|stringformat:"s" == selected_category %}selected{% endif %}>
                            {{ category.name }}
                        </option>
                    {% endfor %}
                </select>
                <select name="medium" class="form-select me-2">
                    <option value="">All Mediums</option>
                    {% for value, label in medium_choices %}
                        <option value="{{ value }}" 
                                {% if value == selected_medium %}selected{% endif %}>
                            {{ label }}
                        </option>
                    {% endfor %}
                </select>
                <button type="submit" class="btn btn-primary">Filter</button>
            </form>
        </div>
        <div class="col-md-4 text-end">
            <a href="{% url 'artwork-create' %}" class="btn btn-success">
                <i class="fas fa-plus"></i> Upload Artwork
            </a>
        </div>
    </div>

    <!-- Featured Artworks -->
    {% if featured_artworks and not search_query %}
    <div class="row mb-5">
        <div class="col-12">
            <h2 class="mb-4">Featured Artworks</h2>
            <div class="row">
                {% for artwork in featured_artworks %}
                <div class="col-md-2 mb-3">
                    <div class="card artwork-card">
                        <img src="{{ artwork.image.url }}" class="card-img-top" alt="{{ artwork.title }}">
                        <div class="card-body p-2">
                            <h6 class="card-title">{{ artwork.title }}</h6>
                            <small class="text-muted">{{ artwork.artist.user.get_full_name }}</small>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
    </div>
    {% endif %}

    <!-- Artwork Grid -->
    <div class="row">
        {% for artwork in artworks %}
        <div class="col-lg-3 col-md-4 col-sm-6 mb-4">
            <div class="card artwork-card h-100">
                <img src="{{ artwork.image.url }}" class="card-img-top" alt="{{ artwork.title }}">
                <div class="card-body">
                    <h5 class="card-title">{{ artwork.title }}</h5>
                    <p class="card-text">
                        <small class="text-muted">by {{ artwork.artist.user.get_full_name }}</small>
                    </p>
                    <p class="card-text">{{ artwork.description|truncatewords:15 }}</p>
                    {% if artwork.is_for_sale and artwork.price %}
                        <p class="card-text"><strong>${{ artwork.price }}</strong></p>
                    {% endif %}
                    <a href="{% url 'artwork-detail' artwork.pk %}" class="btn btn-primary btn-sm">View Details</a>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>

    <!-- Pagination -->
    {% if is_paginated %}
    <nav aria-label="Artwork pagination">
        <ul class="pagination justify-content-center">
            {% if page_obj.has_previous %}
                <li class="page-item">
                    <a class="page-link" href="?page=1">First</a>
                </li>
                <li class="page-item">
                    <a class="page-link" href="?page={{ page_obj.previous_page_number }}">Previous</a>
                </li>
            {% endif %}
            
            <li class="page-item active">
                <span class="page-link">{{ page_obj.number }} of {{ page_obj.paginator.num_pages }}</span>
            </li>
            
            {% if page_obj.has_next %}
                <li class="page-item">
                    <a class="page-link" href="?page={{ page_obj.next_page_number }}">Next</a>
                </li>
                <li class="page-item">
                    <a class="page-link" href="?page={{ page_obj.paginator.num_pages }}">Last</a>
                </li>
            {% endif %}
        </ul>
    </nav>
    {% endif %}
</div>
{% endblock %}
```

```html
<!-- gallery/artist_dashboard.html -->
{% extends 'base.html' %}
{% load static %}

{% block content %}
<div class="container py-4">
    <div class="row">
        <div class="col-md-3">
            <div class="card">
                <div class="card-body text-center">
                    {% if artist.profile_picture %}
                        <img src="{{ artist.profile_picture.url }}" class="rounded-circle mb-3" 
                             width="100" height="100" alt="Profile Picture">
                    {% else %}
                        <div class="rounded-circle bg-light d-inline-flex align-items-center justify-content-center mb-3" 
                             style="width: 100px; height: 100px;">
                            <i class="fas fa-user fa-2x text-muted"></i>
                        </div>
                    {% endif %}
                    <h5>{{ user.get_full_name }}</h5>
                    <p class="text-muted">{{ artist.bio|truncatewords:20 }}</p>
                    <a href="{% url 'artist-update' artist.pk %}" class="btn btn-outline-primary btn-sm">
                        Edit Profile
                    </a>
                </div>
            </div>
            
            <div class="card mt-3">
                <div class="card-body">
                    <h6 class="card-title">Statistics</h6>
                    <p class="mb-2">Artworks: <strong>{{ total_artworks }}</strong></p>
                    <p class="mb-2">Total Views: <strong>{{ total_views }}</strong></p>
                    <p class="mb-2">Featured: <strong>{{ featured_count }}</strong></p>
                    <p class="mb-0">For Sale: <strong>{{ for_sale_count }}</strong></p>
                </div>
            </div>
        </div>
        
        <div class="col-md-9">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>My Artworks</h2>
                <a href="{% url 'artwork-create' %}" class="btn btn-primary">
                    <i class="fas fa-plus"></i> Upload New Artwork
                </a>
            </div>
            
            {% if artworks %}
                <div class="row">
                    {% for artwork in artworks %}
                    <div class="col-md-4 mb-4">
                        <div class="card">
                            <img src="{{ artwork.image.url }}" class="card-img-top" alt="{{ artwork.title }}">
                            <div class="card-body">
                                <h6 class="card-title">{{ artwork.title }}</h6>
                                <p class="card-text"><small class="text-muted">{{ artwork.created_at|date:"M d, Y" }}</small></p>
                                <p class="card-text">Views: {{ artwork.views.count }}</p>
                                {% if artwork.is_for_sale %}
                                    <span class="badge bg-success">For Sale</span>
                                {% endif %}
                                {% if artwork.featured %}
                                    <span class="badge bg-warning">Featured</span>
                                {% endif %}
                                <div class="mt-2">
                                    <a href="{% url 'artwork-detail' artwork.pk %}" class="btn btn-sm btn-outline-primary">View</a>
                                    <a href="{% url 'artwork-update' artwork.pk %}" class="btn btn-sm btn-outline-secondary">Edit</a>
                                    <a href="{% url 'artwork-delete' artwork.pk %}" class="btn btn-sm btn-outline-danger">Delete</a>
                                </div>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="text-center py-5">
                    <i class="fas fa-paint-brush fa-3x text-muted mb-3"></i>
                    <h4>No artworks yet</h4>
                    <p class="text-muted">Start building your gallery by uploading your first artwork!</p>
                    <a href="{% url 'artwork-create' %}" class="btn btn-primary">Upload First Artwork</a>
                </div>
            {% endif %}
        </div>
    </div>
</div>
{% endblock %}
```

### Settings Configuration

```python
# settings.py additions
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# For image handling
INSTALLED_APPS = [
    # ... other apps
    'PIL',  # Python Imaging Library
]

# Optional: Image optimization
# pip install Pillow
```

### Admin Configuration

```python
# admin.py
from django.contrib import admin
from .models import Artist, Artwork, ArtCategory, ArtworkView

@admin.register(ArtCategory)
class ArtCategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'description']
    search_fields = ['name']

@admin.register(Artist)
class ArtistAdmin(admin.ModelAdmin):
    list_display = ['user', 'created_at']
    search_fields = ['user__first_name', 'user__last_name', 'user__username']
    list_filter = ['created_at']

@admin.register(Artwork)
class ArtworkAdmin(admin.ModelAdmin):
    list_display = ['title', 'artist', 'category', 'medium', 'is_for_sale', 'featured', 'created_at']
    list_filter = ['medium', 'category', 'is_for_sale', 'featured', 'created_at']
    search_fields = ['title', 'artist__user__first_name', 'artist__user__last_name']
    list_editable = ['featured', 'is_for_sale']
    
@admin.register(ArtworkView)
class ArtworkViewAdmin(admin.ModelAdmin):
    list_display = ['artwork', 'viewer_ip', 'viewed_at']
    list_filter = ['viewed_at']
    readonly_fields = ['artwork', 'viewer_ip', 'viewed_at']
```

## Project Features Implemented

✅ **Complete CRUD Operations**
- Create: Upload new artworks
- Read: Browse gallery, view artwork details
- Update: Edit artwork information and artist profiles
- Delete: Remove artworks from gallery

✅ **Advanced Features**
- Image upload and handling
- Search and filtering system
- User authentication and permissions
- Artist profiles and dashboards
- View tracking for artworks
- Featured artwork system
- Marketplace functionality (for sale items)

✅ **Class-Based Views Used**
- `ListView` for gallery and dashboard
- `DetailView` for artwork and artist details
- `CreateView` for artwork upload
- `UpdateView` for editing artworks and profiles
- `DeleteView` for artwork removal

✅ **Security & Permissions**
- Login required mixins
- User ownership validation
- Proper permission testing

This digital art gallery demonstrates the power of Django's Class-Based Views in creating a sophisticated, feature-rich web application. Just like a master chef's kitchen where each station has its specific purpose, each CBV handles its designated responsibility efficiently and elegantly!

## Assignment: The Master Chef's Challenge

### Project: Recipe Management System

Create a comprehensive recipe management system that demonstrates mastery of class-based views. Your system should include:

**Requirements:**

1. **Recipe List View** (ListView)
   - Display all recipes with pagination
   - Filter by category and difficulty level
   - Search functionality
   - Show recipe ratings and chef information

2. **Recipe Detail View** (DetailView)
   - Show complete recipe information
   - Display similar recipes
   - Show chef profile information
   - Include rating and review system

3. **Recipe Creation View** (CreateView)
   - Form for adding new recipes
   - Automatic chef assignment
   - Success message using custom mixin
   - Redirect to recipe detail page

4. **Custom Mixins Implementation**
   - `ChefOwnerMixin`: Ensure only recipe owners can edit
   - `SuccessMessageMixin`: Add success messages to all form views
   - `KitchenContextMixin`: Add common kitchen data to all views

**Code Structure:**

```python
# models.py (provided structure)
class Recipe(models.Model):
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    ]
    
    CATEGORY_CHOICES = [
        ('appetizer', 'Appetizer'),
        ('main', 'Main Course'),
        ('dessert', 'Dessert'),
        ('beverage', 'Beverage'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField()
    ingredients = models.TextField()
    instructions = models.TextField()
    prep_time = models.IntegerField()  # in minutes
    cook_time = models.IntegerField()  # in minutes
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    chef = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse('recipe:detail', kwargs={'pk': self.pk})
```

**Your Task:**
Implement the views, mixins, and URL configuration that brings this recipe management system to life. Focus on clean, reusable code that demonstrates your understanding of:

- Generic class-based views
- Custom mixins for reusable functionality
- Proper use of context data
- Form handling with success messages
- Permission checking and user authentication

**Evaluation Criteria:**
- Proper use of ListView, DetailView, and CreateView
- Implementation of at least two custom mixins
- Clean, well-documented code
- Proper error handling and user feedback
- Demonstration of CBV best practices

**Deliverables:**
1. Complete `views.py` with all required views
2. `mixins.py` with custom mixins
3. `urls.py` with proper URL configuration
4. Brief explanation of your design choices

This assignment will test your ability to orchestrate class-based views like a master chef orchestrates a kitchen - with efficiency, elegance, and expertise!

---

## Summary

You've now learned to work with Django's class-based views like a master chef running a professional kitchen. You understand how to use generic views as your foundation recipes, customize them with mixins like signature sauces, and choose the right tools for each situation. Remember: just as a great chef builds upon classic techniques while adding their own creativity, great Django developers master the fundamentals of CBVs while crafting elegant, reusable solutions.

The key to mastery is practice - start with simple views and gradually build more complex combinations. Your code should be as elegant and efficient as a well-run kitchen, where every component has its place and purpose.