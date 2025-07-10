# Day 51: Class-Based Views CRUD Operations Course

## Project Objective
By the end of this project, you will be able to build a complete CRUD (Create, Read, Update, Delete) application using Django's Class-Based Views, understanding how to structure views like a professional chef organizes their kitchen stations.

---

## Imagine That...

Imagine that you're the head chef of a bustling restaurant, and your kitchen is your Django application. In your kitchen, you have different stations - one for appetizers, one for main courses, one for desserts, and one for plating. Each station has its own specialized tools and workflows, but they all work together to create a complete dining experience.

Class-Based Views (CBVs) are like having specialized kitchen stations. Instead of having one cook (function-based view) trying to handle everything, you have dedicated stations (class-based views) that excel at specific tasks. Today, we're going to build a complete recipe management system where each CRUD operation is handled by its own specialized "kitchen station."

---

## Final Project: Restaurant Recipe Management System

You'll build a complete recipe management system where chefs can manage their recipes with full CRUD operations using Class-Based Views. Think of it as the digital recipe book for your restaurant kitchen.

### Project Structure Setup

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class Recipe(models.Model):
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField()
    ingredients = models.TextField()
    instructions = models.TextField()
    cook_time = models.IntegerField(help_text="Time in minutes")
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES)
    chef = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse('recipe-detail', kwargs={'pk': self.pk})
```

### Complete CRUD Views

```python
# views.py
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from .models import Recipe

class RecipeListView(ListView):
    model = Recipe
    template_name = 'recipes/recipe_list.html'
    context_object_name = 'recipes'
    paginate_by = 9
    
    def get_queryset(self):
        difficulty = self.request.GET.get('difficulty')
        if difficulty:
            return Recipe.objects.filter(difficulty=difficulty)
        return Recipe.objects.all()
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['difficulty_choices'] = Recipe.DIFFICULTY_CHOICES
        return context

class RecipeDetailView(DetailView):
    model = Recipe
    template_name = 'recipes/recipe_detail.html'
    context_object_name = 'recipe'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        recipe = self.get_object()
        
        hours = recipe.cook_time // 60
        minutes = recipe.cook_time % 60
        context['formatted_time'] = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        
        context['chef_other_recipes'] = Recipe.objects.filter(
            chef=recipe.chef
        ).exclude(pk=recipe.pk)[:3]
        
        return context

class RecipeCreateView(LoginRequiredMixin, CreateView):
    model = Recipe
    template_name = 'recipes/recipe_form.html'
    fields = ['title', 'description', 'ingredients', 'instructions', 'cook_time', 'difficulty']
    success_url = reverse_lazy('recipe-list')
    
    def form_valid(self, form):
        form.instance.chef = self.request.user
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form_title'] = 'Create New Recipe'
        context['button_text'] = 'Add to Menu'
        return context

class RecipeUpdateView(LoginRequiredMixin, UpdateView):
    model = Recipe
    template_name = 'recipes/recipe_form.html'
    fields = ['title', 'description', 'ingredients', 'instructions', 'cook_time', 'difficulty']
    
    def get_queryset(self):
        return Recipe.objects.filter(chef=self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form_title'] = f'Update Recipe: {self.object.title}'
        context['button_text'] = 'Update Recipe'
        return context

class RecipeDeleteView(LoginRequiredMixin, DeleteView):
    model = Recipe
    template_name = 'recipes/recipe_confirm_delete.html'
    success_url = reverse_lazy('recipe-list')
    
    def get_queryset(self):
        return Recipe.objects.filter(chef=self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['warning_message'] = "This action cannot be undone. Are you sure you want to remove this recipe from your collection?"
        return context
```

### URL Configuration

```python
# urls.py
from django.urls import path
from .views import (
    RecipeListView, RecipeDetailView, RecipeCreateView, 
    RecipeUpdateView, RecipeDeleteView
)

urlpatterns = [
    path('', RecipeListView.as_view(), name='recipe-list'),
    path('recipe/<int:pk>/', RecipeDetailView.as_view(), name='recipe-detail'),
    path('recipe/new/', RecipeCreateView.as_view(), name='recipe-create'),
    path('recipe/<int:pk>/update/', RecipeUpdateView.as_view(), name='recipe-update'),
    path('recipe/<int:pk>/delete/', RecipeDeleteView.as_view(), name='recipe-delete'),
]
```

### Templates

#### Recipe List Template
```html
<!-- templates/recipes/recipe_list.html -->
{% extends 'base.html' %}

{% block content %}
<div class="container">
    <div class="row">
        <div class="col-md-12">
            <h1>🍳 Kitchen Recipe Collection</h1>
            <a href="{% url 'recipe-create' %}" class="btn btn-success mb-3">
                <i class="fas fa-plus"></i> Add New Recipe
            </a>
            
            <div class="mb-3">
                <a href="{% url 'recipe-list' %}" class="btn btn-outline-primary">All Recipes</a>
                {% for choice in difficulty_choices %}
                    <a href="?difficulty={{ choice.0 }}" class="btn btn-outline-{{ choice.0 }}">
                        {{ choice.1 }}
                    </a>
                {% endfor %}
            </div>
        </div>
    </div>
    
    <div class="row">
        {% for recipe in recipes %}
        <div class="col-md-4 mb-4">
            <div class="card h-100">
                <div class="card-body">
                    <h5 class="card-title">{{ recipe.title }}</h5>
                    <p class="card-text">{{ recipe.description|truncatewords:15 }}</p>
                    <div class="recipe-meta">
                        <small class="text-muted">
                            👨‍🍳 Chef: {{ recipe.chef.username }}<br>
                            ⏱️ Cook Time: {{ recipe.cook_time }} min<br>
                            📊 Difficulty: {{ recipe.get_difficulty_display }}
                        </small>
                    </div>
                </div>
                <div class="card-footer">
                    <a href="{% url 'recipe-detail' recipe.pk %}" class="btn btn-primary btn-sm">
                        View Recipe
                    </a>
                    {% if recipe.chef == request.user %}
                        <a href="{% url 'recipe-update' recipe.pk %}" class="btn btn-warning btn-sm">
                            Edit
                        </a>
                        <a href="{% url 'recipe-delete' recipe.pk %}" class="btn btn-danger btn-sm">
                            Delete
                        </a>
                    {% endif %}
                </div>
            </div>
        </div>
        {% empty %}
        <div class="col-12">
            <div class="alert alert-info">
                <h4>No recipes yet in our kitchen!</h4>
                <p>Time to create your first culinary masterpiece!</p>
                <a href="{% url 'recipe-create' %}" class="btn btn-primary">Start Cooking</a>
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
                    <a class="page-link" href="?page={{ page_obj.previous_page_number }}">Previous</a>
                </li>
            {% endif %}
            
            <li class="page-item active">
                <span class="page-link">
                    Page {{ page_obj.number }} of {{ page_obj.paginator.num_pages }}
                </span>
            </li>
            
            {% if page_obj.has_next %}
                <li class="page-item">
                    <a class="page-link" href="?page={{ page_obj.next_page_number }}">Next</a>
                </li>
            {% endif %}
        </ul>
    </nav>
    {% endif %}
</div>
{% endblock %}
```

#### Recipe Detail Template
```html
<!-- templates/recipes/recipe_detail.html -->
{% extends 'base.html' %}

{% block content %}
<div class="container">
    <div class="row">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <h1>{{ recipe.title }}</h1>
                    <div class="recipe-meta">
                        <span class="badge badge-{{ recipe.difficulty }}">{{ recipe.get_difficulty_display }}</span>
                        <span class="badge badge-secondary">⏱️ {{ formatted_time }}</span>
                        <span class="badge badge-info">👨‍🍳 Chef {{ recipe.chef.username }}</span>
                    </div>
                </div>
                <div class="card-body">
                    <h5>Description</h5>
                    <p class="lead">{{ recipe.description }}</p>
                    
                    <h5>Ingredients</h5>
                    <div class="ingredients-section">
                        {{ recipe.ingredients|linebreaks }}
                    </div>
                    
                    <h5>Instructions</h5>
                    <div class="instructions-section">
                        {{ recipe.instructions|linebreaks }}
                    </div>
                </div>
                <div class="card-footer">
                    <small class="text-muted">
                        Created: {{ recipe.created_at|date:"F d, Y" }}
                        {% if recipe.updated_at != recipe.created_at %}
                            | Updated: {{ recipe.updated_at|date:"F d, Y" }}
                        {% endif %}
                    </small>
                </div>
            </div>
            
            {% if recipe.chef == request.user %}
            <div class="mt-3">
                <a href="{% url 'recipe-update' recipe.pk %}" class="btn btn-warning">
                    <i class="fas fa-edit"></i> Edit Recipe
                </a>
                <a href="{% url 'recipe-delete' recipe.pk %}" class="btn btn-danger">
                    <i class="fas fa-trash"></i> Delete Recipe
                </a>
            </div>
            {% endif %}
        </div>
        
        <div class="col-md-4">
            <div class="card">
                <div class="card-header">
                    <h6>More from Chef {{ recipe.chef.username }}</h6>
                </div>
                <div class="card-body">
                    {% for other_recipe in chef_other_recipes %}
                        <div class="mb-2">
                            <a href="{% url 'recipe-detail' other_recipe.pk %}" class="text-decoration-none">
                                <strong>{{ other_recipe.title }}</strong>
                            </a>
                            <br>
                            <small class="text-muted">{{ other_recipe.cook_time }} min</small>
                        </div>
                    {% empty %}
                        <p>This chef's first recipe!</p>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
    
    <div class="mt-3">
        <a href="{% url 'recipe-list' %}" class="btn btn-secondary">
            <i class="fas fa-arrow-left"></i> Back to Recipe Collection
        </a>
    </div>
</div>
{% endblock %}
```

#### Recipe Form Template
```html
<!-- templates/recipes/recipe_form.html -->
{% extends 'base.html' %}

{% block content %}
<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <h2>{{ form_title }}</h2>
                </div>
                <div class="card-body">
                    <form method="post">
                        {% csrf_token %}
                        
                        <div class="form-group">
                            <label for="{{ form.title.id_for_label }}">Recipe Title</label>
                            {{ form.title }}
                            {% if form.title.errors %}
                                <div class="text-danger">{{ form.title.errors }}</div>
                            {% endif %}
                        </div>
                        
                        <div class="form-group">
                            <label for="{{ form.description.id_for_label }}">Description</label>
                            {{ form.description }}
                            {% if form.description.errors %}
                                <div class="text-danger">{{ form.description.errors }}</div>
                            {% endif %}
                        </div>
                        
                        <div class="form-group">
                            <label for="{{ form.ingredients.id_for_label }}">Ingredients</label>
                            {{ form.ingredients }}
                            <small class="form-text text-muted">List each ingredient on a new line</small>
                            {% if form.ingredients.errors %}
                                <div class="text-danger">{{ form.ingredients.errors }}</div>
                            {% endif %}
                        </div>
                        
                        <div class="form-group">
                            <label for="{{ form.instructions.id_for_label }}">Cooking Instructions</label>
                            {{ form.instructions }}
                            <small class="form-text text-muted">Step-by-step cooking instructions</small>
                            {% if form.instructions.errors %}
                                <div class="text-danger">{{ form.instructions.errors }}</div>
                            {% endif %}
                        </div>
                        
                        <div class="form-row">
                            <div class="form-group col-md-6">
                                <label for="{{ form.cook_time.id_for_label }}">Cook Time (minutes)</label>
                                {{ form.cook_time }}
                                {% if form.cook_time.errors %}
                                    <div class="text-danger">{{ form.cook_time.errors }}</div>
                                {% endif %}
                            </div>
                            
                            <div class="form-group col-md-6">
                                <label for="{{ form.difficulty.id_for_label }}">Difficulty Level</label>
                                {{ form.difficulty }}
                                {% if form.difficulty.errors %}
                                    <div class="text-danger">{{ form.difficulty.errors }}</div>
                                {% endif %}
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <button type="submit" class="btn btn-success">
                                <i class="fas fa-save"></i> {{ button_text }}
                            </button>
                            <a href="{% url 'recipe-list' %}" class="btn btn-secondary">
                                <i class="fas fa-times"></i> Cancel
                            </a>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

#### Recipe Delete Confirmation Template
```html
<!-- templates/recipes/recipe_confirm_delete.html -->
{% extends 'base.html' %}

{% block content %}
<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-6">
            <div class="card">
                <div class="card-header bg-danger text-white">
                    <h3><i class="fas fa-exclamation-triangle"></i> Confirm Recipe Deletion</h3>
                </div>
                <div class="card-body">
                    <h5>Are you sure you want to delete "{{ recipe.title }}"?</h5>
                    <div class="alert alert-warning">
                        <strong>{{ warning_message }}</strong>
                    </div>
                    
                    <div class="recipe-preview">
                        <h6>Recipe Details:</h6>
                        <ul>
                            <li><strong>Title:</strong> {{ recipe.title }}</li>
                            <li><strong>Cook Time:</strong> {{ recipe.cook_time }} minutes</li>
                            <li><strong>Difficulty:</strong> {{ recipe.get_difficulty_display }}</li>
                            <li><strong>Created:</strong> {{ recipe.created_at|date:"F d, Y" }}</li>
                        </ul>
                    </div>
                    
                    <form method="post">
                        {% csrf_token %}
                        <button type="submit" class="btn btn-danger">
                            <i class="fas fa-trash"></i> Yes, Delete Recipe
                        </button>
                        <a href="{% url 'recipe-detail' recipe.pk %}" class="btn btn-secondary">
                            <i class="fas fa-arrow-left"></i> Cancel
                        </a>
                    </form>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

### CSS Styling
```css
/* static/css/recipes.css */
.card {
    transition: transform 0.2s;
}

.card:hover {
    transform: translateY(-5px);
}

.recipe-meta {
    margin-top: 10px;
}

.badge {
    margin-right: 5px;
}

.ingredients-section, .instructions-section {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 20px;
}

.form-control {
    margin-bottom: 10px;
}

.recipe-preview {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 5px;
    margin: 15px 0;
}

.btn-outline-easy { border-color: #28a745; color: #28a745; }
.btn-outline-medium { border-color: #ffc107; color: #ffc107; }
.btn-outline-hard { border-color: #dc3545; color: #dc3545; }
```

### Project Features Implemented

1. **Complete CRUD Operations**: Create, Read, Update, Delete recipes
2. **User Authentication**: Only logged-in users can create/edit/delete recipes
3. **Chef Ownership**: Users can only edit/delete their own recipes
4. **Filtering**: Filter recipes by difficulty level
5. **Pagination**: Handle large numbers of recipes efficiently
6. **Responsive Design**: Works on mobile and desktop
7. **Rich Templates**: Beautiful, professional-looking interface
8. **Security**: Proper permission checks and CSRF protection

This project demonstrates the power of Django's Class-Based Views by creating a complete, functional recipe management system with minimal code. Each view handles its specific responsibility, just like specialized kitchen stations in a professional restaurant.