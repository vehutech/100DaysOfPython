# Day 42: Django Forms & User Input - Complete Course

## Learning Objective
By the end of this lesson, you will be able to create secure, validated forms in Django that handle user input professionally, implement custom validation logic, and protect against common web vulnerabilities like CSRF attacks.

---

## Course Introduction

**Imagine that...** you're the head chef of a world-class restaurant, and every day, customers give you special requests - dietary restrictions, cooking preferences, special occasions they're celebrating. But here's the thing: you can't just accept any request without checking it first. Is the customer asking for something you can actually make? Are they allergic to ingredients they're requesting? Do they have the authority to make changes to the order?

In the digital kitchen of web development, forms are your **order-taking system**. Just like a professional restaurant has standardized order forms and verification processes, Django provides a powerful forms framework that helps you collect, validate, and process user input safely and efficiently.

---

## Lesson 1: Understanding Django Forms Framework

### The Kitchen Analogy
Think of Django forms like a restaurant's order system:
- **Raw Form**: Like a blank order pad - you can write anything, but there's no structure
- **Django Form**: Like a standardized order form with checkboxes, dropdowns, and validation rules
- **ModelForm**: Like a specialized order form that automatically knows your menu items and prices

### Basic Form Structure

```python
# forms.py
from django import forms

class ContactForm(forms.Form):
    """
    A basic contact form - like a customer feedback card
    """
    name = forms.CharField(
        max_length=100,
        label="Your Name",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your full name'
        })
    )
    
    email = forms.EmailField(
        label="Email Address",
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'your.email@example.com'
        })
    )
    
    subject = forms.CharField(
        max_length=200,
        label="Subject",
        widget=forms.TextInput(attrs={
            'class': 'form-control'
        })
    )
    
    message = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': 'Tell us what you think...'
        }),
        label="Your Message"
    )
    
    priority = forms.ChoiceField(
        choices=[
            ('low', 'Low Priority'),
            ('medium', 'Medium Priority'),
            ('high', 'High Priority'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
```

### Syntax Explanation:
- `forms.CharField()`: Creates a text input field, like a line on an order form
- `max_length=100`: Sets character limit (like limiting order notes to fit on the form)
- `widget=forms.TextInput()`: Specifies the HTML input type (like choosing between checkbox vs text field)
- `attrs={'class': 'form-control'}`: Adds HTML attributes for styling (like formatting the order form)

---

## Lesson 2: Form Validation and Cleaning

### The Kitchen Analogy
Validation is like a head chef reviewing orders before they go to the kitchen:
- **Field validation**: Checking if each item is available (like verifying we have the ingredients)
- **Form validation**: Checking if the whole order makes sense (like ensuring the wine pairs with the food)
- **Cleaning**: Standardizing the order format (like converting "medium rare" to "MR" for the kitchen)

### Custom Validation Methods

```python
# forms.py
from django import forms
from django.core.exceptions import ValidationError
import re

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100)
    email = forms.EmailField()
    phone = forms.CharField(max_length=15, required=False)
    subject = forms.CharField(max_length=200)
    message = forms.CharField(widget=forms.Textarea)
    
    def clean_name(self):
        """
        Custom validation for name field - like checking if a customer 
        name is on the VIP list
        """
        name = self.cleaned_data['name']
        
        # Check if name contains only letters and spaces
        if not re.match(r'^[a-zA-Z\s]+$', name):
            raise ValidationError(
                "Name can only contain letters and spaces."
            )
        
        # Check for minimum length
        if len(name.strip()) < 2:
            raise ValidationError(
                "Name must be at least 2 characters long."
            )
        
        return name.title()  # Return cleaned data (proper case)
    
    def clean_phone(self):
        """
        Clean phone number - like standardizing how we write phone numbers
        """
        phone = self.cleaned_data.get('phone')
        
        if phone:
            # Remove all non-digit characters
            cleaned_phone = re.sub(r'\D', '', phone)
            
            # Check if it's a valid length
            if len(cleaned_phone) not in [10, 11]:
                raise ValidationError(
                    "Phone number must be 10 or 11 digits."
                )
            
            # Format the phone number
            if len(cleaned_phone) == 10:
                return f"({cleaned_phone[:3]}) {cleaned_phone[3:6]}-{cleaned_phone[6:]}"
            else:
                return f"+{cleaned_phone[0]} ({cleaned_phone[1:4]}) {cleaned_phone[4:7]}-{cleaned_phone[7:]}"
        
        return phone
    
    def clean(self):
        """
        Form-wide validation - like checking if the entire order makes sense
        """
        cleaned_data = super().clean()
        subject = cleaned_data.get('subject')
        message = cleaned_data.get('message')
        
        # Check if urgent subject has detailed message
        if subject and 'urgent' in subject.lower():
            if not message or len(message.strip()) < 20:
                raise ValidationError(
                    "Urgent messages must include detailed information (at least 20 characters)."
                )
        
        return cleaned_data
```

### Syntax Explanation:
- `clean_fieldname()`: Method for validating specific fields (like checking individual menu items)
- `self.cleaned_data['fieldname']`: Gets the cleaned value of a field
- `ValidationError()`: Raises an error if validation fails (like rejecting an invalid order)
- `clean()`: Method for form-wide validation (like checking if the entire order is valid)

---

## Lesson 3: ModelForms - The Smart Order System

### The Kitchen Analogy
ModelForms are like having an intelligent order system that already knows your entire menu, prices, and what combinations are possible. Instead of manually creating every field, it reads your menu (model) and creates the form automatically.

### Creating ModelForms

```python
# models.py
from django.db import models
from django.contrib.auth.models import User

class Restaurant(models.Model):
    name = models.CharField(max_length=100)
    cuisine_type = models.CharField(max_length=50)
    location = models.CharField(max_length=200)
    phone = models.CharField(max_length=15)
    email = models.EmailField()
    website = models.URLField(blank=True)
    description = models.TextField()
    rating = models.DecimalField(max_digits=3, decimal_places=2, default=0.00)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class Review(models.Model):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    reviewer = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])
    title = models.CharField(max_length=100)
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['restaurant', 'reviewer']
```

```python
# forms.py
from django import forms
from .models import Restaurant, Review

class RestaurantForm(forms.ModelForm):
    """
    Smart form that automatically creates fields based on Restaurant model
    Like having an order form that knows your entire menu
    """
    class Meta:
        model = Restaurant
        fields = ['name', 'cuisine_type', 'location', 'phone', 'email', 'website', 'description']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Restaurant Name'
            }),
            'cuisine_type': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Italian, Chinese, Mexican'
            }),
            'location': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Full Address'
            }),
            'phone': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '(555) 123-4567'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'contact@restaurant.com'
            }),
            'website': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://restaurant.com'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Tell us about your restaurant...'
            }),
        }
        labels = {
            'cuisine_type': 'Cuisine Type',
            'location': 'Address',
            'phone': 'Phone Number',
            'email': 'Email Address',
            'website': 'Website (Optional)',
            'description': 'Description'
        }
    
    def clean_phone(self):
        """Custom validation for phone - like standardizing order format"""
        phone = self.cleaned_data.get('phone')
        # Remove all non-digit characters
        cleaned_phone = re.sub(r'\D', '', phone)
        
        if len(cleaned_phone) != 10:
            raise forms.ValidationError("Phone number must be 10 digits.")
        
        return f"({cleaned_phone[:3]}) {cleaned_phone[3:6]}-{cleaned_phone[6:]}"

class ReviewForm(forms.ModelForm):
    """
    Review form - like a customer feedback card
    """
    class Meta:
        model = Review
        fields = ['rating', 'title', 'comment']
        widgets = {
            'rating': forms.Select(attrs={'class': 'form-control'}),
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Summary of your experience'
            }),
            'comment': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Share your detailed review...'
            }),
        }
```

### Syntax Explanation:
- `forms.ModelForm`: Creates a form automatically from a model (like auto-generating an order form from your menu)
- `class Meta`: Configuration class that tells Django how to build the form
- `model = Restaurant`: Specifies which model to base the form on
- `fields = [...]`: Lists which model fields to include in the form
- `widgets = {...}`: Customizes how each field appears in HTML

---

## Lesson 4: CSRF Protection - The Security Bouncer

### The Kitchen Analogy
CSRF protection is like having a security bouncer at your restaurant who checks that every order actually came from a real customer sitting at a table, not from someone trying to sneak in fake orders from outside.

### Understanding CSRF

```python
# views.py
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST
from .forms import ContactForm, RestaurantForm, ReviewForm
from .models import Restaurant, Review

@csrf_protect
def contact_view(request):
    """
    Contact form view with CSRF protection
    Like having a bouncer check every customer's ID
    """
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Process the form data
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            subject = form.cleaned_data['subject']
            message = form.cleaned_data['message']
            
            # In a real app, you'd send an email or save to database
            # For now, we'll just show a success message
            messages.success(request, f'Thank you {name}! Your message has been sent.')
            return redirect('contact_success')
    else:
        form = ContactForm()
    
    return render(request, 'contact.html', {'form': form})

def add_restaurant_view(request):
    """
    Add restaurant form - like adding a new restaurant to your directory
    """
    if request.method == 'POST':
        form = RestaurantForm(request.POST)
        if form.is_valid():
            restaurant = form.save()
            messages.success(request, f'{restaurant.name} has been added successfully!')
            return redirect('restaurant_detail', pk=restaurant.pk)
    else:
        form = RestaurantForm()
    
    return render(request, 'add_restaurant.html', {'form': form})

def add_review_view(request, restaurant_id):
    """
    Add review form - like letting customers leave feedback
    """
    restaurant = Restaurant.objects.get(id=restaurant_id)
    
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review = form.save(commit=False)
            review.restaurant = restaurant
            review.reviewer = request.user
            review.save()
            messages.success(request, 'Your review has been posted!')
            return redirect('restaurant_detail', pk=restaurant.pk)
    else:
        form = ReviewForm()
    
    return render(request, 'add_review.html', {
        'form': form,
        'restaurant': restaurant
    })
```

### HTML Templates with CSRF

```html
<!-- templates/contact.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Contact Us</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h2>Contact Us</h2>
        
        <!-- Display messages (success/error) -->
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }}">{{ message }}</div>
            {% endfor %}
        {% endif %}
        
        <form method="post">
            <!-- CSRF token - like the security bouncer's stamp -->
            {% csrf_token %}
            
            <div class="mb-3">
                <label for="{{ form.name.id_for_label }}" class="form-label">{{ form.name.label }}</label>
                {{ form.name }}
                {% if form.name.errors %}
                    <div class="text-danger">{{ form.name.errors }}</div>
                {% endif %}
            </div>
            
            <div class="mb-3">
                <label for="{{ form.email.id_for_label }}" class="form-label">{{ form.email.label }}</label>
                {{ form.email }}
                {% if form.email.errors %}
                    <div class="text-danger">{{ form.email.errors }}</div>
                {% endif %}
            </div>
            
            <div class="mb-3">
                <label for="{{ form.subject.id_for_label }}" class="form-label">{{ form.subject.label }}</label>
                {{ form.subject }}
                {% if form.subject.errors %}
                    <div class="text-danger">{{ form.subject.errors }}</div>
                {% endif %}
            </div>
            
            <div class="mb-3">
                <label for="{{ form.message.id_for_label }}" class="form-label">{{ form.message.label }}</label>
                {{ form.message }}
                {% if form.message.errors %}
                    <div class="text-danger">{{ form.message.errors }}</div>
                {% endif %}
            </div>
            
            <button type="submit" class="btn btn-primary">Send Message</button>
        </form>
    </div>
</body>
</html>
```

### Syntax Explanation:
- `{% csrf_token %}`: Adds CSRF protection token to forms (like a security stamp)
- `@csrf_protect`: Decorator that ensures CSRF validation (like hiring a bouncer)
- `form.is_valid()`: Checks if form data passes all validation rules
- `form.cleaned_data`: Gets the cleaned, validated data from the form
- `form.save()`: Saves ModelForm data to the database (like confirming an order)

---

## Final Quality Project: Restaurant Review System

### The Kitchen Analogy
We're building a complete restaurant review system - like creating a comprehensive dining guide where customers can discover restaurants, read reviews, and leave their own feedback. This combines all our form concepts into one cohesive system.

```python
# Complete project structure
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

class Restaurant(models.Model):
    CUISINE_CHOICES = [
        ('italian', 'Italian'),
        ('chinese', 'Chinese'),
        ('mexican', 'Mexican'),
        ('american', 'American'),
        ('indian', 'Indian'),
        ('french', 'French'),
        ('japanese', 'Japanese'),
        ('thai', 'Thai'),
        ('other', 'Other'),
    ]
    
    name = models.CharField(max_length=100)
    cuisine_type = models.CharField(max_length=20, choices=CUISINE_CHOICES)
    location = models.CharField(max_length=200)
    phone = models.CharField(max_length=15)
    email = models.EmailField()
    website = models.URLField(blank=True)
    description = models.TextField()
    price_range = models.CharField(max_length=10, choices=[
        ('$', 'Budget ($)'),
        ('$$', 'Moderate ($$)'),
        ('$$$', 'Expensive ($$$)'),
        ('$$$$', 'Very Expensive ($$$$)'),
    ])
    average_rating = models.DecimalField(max_digits=3, decimal_places=2, default=0.00)
    total_reviews = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def update_rating(self):
        """Update average rating based on reviews"""
        reviews = self.review_set.all()
        if reviews:
            self.average_rating = sum(r.rating for r in reviews) / len(reviews)
            self.total_reviews = len(reviews)
        else:
            self.average_rating = 0.00
            self.total_reviews = 0
        self.save()
    
    def __str__(self):
        return self.name

class Review(models.Model):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    reviewer = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)])
    title = models.CharField(max_length=100)
    comment = models.TextField()
    visit_date = models.DateField()
    would_recommend = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['restaurant', 'reviewer']
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.restaurant.update_rating()
    
    def __str__(self):
        return f"{self.title} - {self.restaurant.name}"

# forms.py
from django import forms
from django.core.exceptions import ValidationError
from .models import Restaurant, Review
import re
from datetime import date

class RestaurantSearchForm(forms.Form):
    """Search form for finding restaurants"""
    search = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search restaurants...'
        })
    )
    
    cuisine = forms.ChoiceField(
        choices=[('', 'All Cuisines')] + Restaurant.CUISINE_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    price_range = forms.ChoiceField(
        choices=[('', 'Any Price')] + [
            ('$', 'Budget ($)'),
            ('$$', 'Moderate ($$)'),
            ('$$$', 'Expensive ($$$)'),
            ('$$$$', 'Very Expensive ($$$$)'),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

class RestaurantForm(forms.ModelForm):
    """Form for adding/editing restaurants"""
    class Meta:
        model = Restaurant
        fields = ['name', 'cuisine_type', 'location', 'phone', 'email', 'website', 'description', 'price_range']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'cuisine_type': forms.Select(attrs={'class': 'form-control'}),
            'location': forms.TextInput(attrs={'class': 'form-control'}),
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'website': forms.URLInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'price_range': forms.Select(attrs={'class': 'form-control'}),
        }
    
    def clean_phone(self):
        phone = self.cleaned_data.get('phone')
        cleaned_phone = re.sub(r'\D', '', phone)
        
        if len(cleaned_phone) != 10:
            raise ValidationError("Phone number must be 10 digits.")
        
        return f"({cleaned_phone[:3]}) {cleaned_phone[3:6]}-{cleaned_phone[6:]}"

class ReviewForm(forms.ModelForm):
    """Form for adding restaurant reviews"""
    class Meta:
        model = Review
        fields = ['rating', 'title', 'comment', 'visit_date', 'would_recommend']
        widgets = {
            'rating': forms.Select(choices=[(i, f'{i} Star{"s" if i != 1 else ""}') for i in range(1, 6)], 
                                 attrs={'class': 'form-control'}),
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Summary of your experience'}),
            'comment': forms.Textarea(attrs={'class': 'form-control', 'rows': 5}),
            'visit_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'would_recommend': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def clean_visit_date(self):
        visit_date = self.cleaned_data.get('visit_date')
        if visit_date and visit_date > date.today():
            raise ValidationError("Visit date cannot be in the future.")
        return visit_date

# views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from .models import Restaurant, Review
from .forms import RestaurantSearchForm, RestaurantForm, ReviewForm

def restaurant_list(request):
    """List all restaurants with search functionality"""
    restaurants = Restaurant.objects.filter(is_active=True)
    form = RestaurantSearchForm(request.GET)
    
    if form.is_valid():
        search = form.cleaned_data.get('search')
        cuisine = form.cleaned_data.get('cuisine')
        price_range = form.cleaned_data.get('price_range')
        
        if search:
            restaurants = restaurants.filter(
                Q(name__icontains=search) | 
                Q(description__icontains=search) |
                Q(location__icontains=search)
            )
        
        if cuisine:
            restaurants = restaurants.filter(cuisine_type=cuisine)
        
        if price_range:
            restaurants = restaurants.filter(price_range=price_range)
    
    return render(request, 'restaurant_list.html', {
        'restaurants': restaurants,
        'form': form
    })

def restaurant_detail(request, pk):
    """Show restaurant details and reviews"""
    restaurant = get_object_or_404(Restaurant, pk=pk)
    reviews = restaurant.review_set.all().order_by('-created_at')
    
    return render(request, 'restaurant_detail.html', {
        'restaurant': restaurant,
        'reviews': reviews
    })

@login_required
def add_restaurant(request):
    """Add a new restaurant"""
    if request.method == 'POST':
        form = RestaurantForm(request.POST)
        if form.is_valid():
            restaurant = form.save()
            messages.success(request, f'{restaurant.name} has been added successfully!')
            return redirect('restaurant_detail', pk=restaurant.pk)
    else:
        form = RestaurantForm()
    
    return render(request, 'add_restaurant.html', {'form': form})

@login_required
def add_review(request, restaurant_id):
    """Add a review for a restaurant"""
    restaurant = get_object_or_404(Restaurant, id=restaurant_id)
    
    # Check if user already reviewed this restaurant
    if Review.objects.filter(restaurant=restaurant, reviewer=request.user).exists():
        messages.error(request, 'You have already reviewed this restaurant.')
        return redirect('restaurant_detail', pk=restaurant.pk)
    
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review = form.save(commit=False)
            review.restaurant = restaurant
            review.reviewer = request.user
            review.save()
            messages.success(request, 'Your review has been posted!')
            return redirect('restaurant_detail', pk=restaurant.pk)
    else:
        form = ReviewForm()
    
    return render(request, 'add_review.html', {
        'form': form,
        'restaurant': restaurant
    })
```

### Key Features Implemented:
1. **Search functionality** with multiple filters
2. **Restaurant management** with full CRUD operations
3. **Review system** with validation and user restrictions
4. **Rating calculations** that automatically update
5. **CSRF protection** on all forms
6. **Custom validation** for phone numbers and dates
7. **User authentication** integration

---

## Assignment: Restaurant Reservation System

### Task Description
Create a reservation system for restaurants that allows customers to book tables online. This assignment combines all the concepts we've learned about forms, validation, and user input handling.

### Requirements:

1. **Create a Reservation Model** with the following fields:
   - Customer name, email, phone
   - Restaurant (foreign key)
   - Date and time
   - Number of guests
   - Special requests (optional)
   - Status (pending, confirmed, cancelled)

2. **Create a ReservationForm** that includes:
   - Custom validation to prevent booking in the past
   - Validation to ensure reasonable party sizes (1-20 people)
   - Phone number formatting
   - Special requests limited to 500 characters

3. **Implement Views** for:
   - Making a reservation
   - Viewing reservation details
   - Cancelling a reservation (if user is authenticated)

4. **Add Business Logic**:
   - Restaurants can't accept reservations less than 2 hours in advance
   - No more than 5 reservations per time slot
   - Email confirmation (mock implementation)

### Success Criteria:
- All forms must have CSRF protection
- Implement proper error handling and user feedback
- Use ModelForms where appropriate
- Include custom validation methods
- Create clean, user-friendly templates
- Add proper form styling with Bootstrap

### Bonus Points:
- Add a calendar widget for date selection
- Implement time slot availability checking
- Add email notifications (mock or real)
- Create an admin interface for restaurant managers

This assignment will test your understanding of Django forms, validation, CSRF protection, and user experience design while building a practical, real-world application.

---

## Course Summary

In this course, you've learned to be a master chef of user input handling:

1. **Forms Framework**: Creating structured, professional forms like standardized order systems
2. **Validation**: Implementing quality control like a head chef reviewing orders
3. **ModelForms**: Using intelligent forms that know your database structure
4. **CSRF Protection**: Securing your application like having a security bouncer
5. **Real-world Application**: Building a complete restaurant review system

You now have the skills to handle user input safely, validate data properly, and create professional web applications that users can trust with their information.