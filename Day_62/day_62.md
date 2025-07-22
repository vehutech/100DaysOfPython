# Day 62: Django Internationalization (i18n) - Going Global with Your Kitchen

## Learning Objective
By the end of this lesson, you will be able to create Django applications that speak multiple languages, format content for different locales, and handle time zones effectively - turning your single-kitchen restaurant into an international chain that serves customers worldwide.

---

## Introduction: Imagine That...

Imagine that you're a successful chef who started with a small local restaurant. Your signature dishes were so amazing that food lovers from around the world started visiting. Soon, you realize you need to expand globally - but here's the challenge: your French customers want menus in French, your Japanese customers prefer different date formats, and your customers in Tokyo are confused when you display New York time for reservations.

Just like a chef adapting recipes and service for different cultures while maintaining the essence of their cuisine, Django's internationalization (i18n) allows your web application to adapt to different languages, cultural conventions, and time zones while keeping your core functionality intact.

---

## Lesson 1: Multi-language Support - Your Multilingual Menu

### The Chef Analogy
Think of multi-language support like having multiple versions of your restaurant menu. The same delicious "Grilled Salmon" exists, but it appears as "Salmón a la Parrilla" for Spanish speakers and "Saumon Grillé" for French speakers. The dish remains the same - only the presentation changes.

### Setting Up Internationalization

First, let's prepare our Django kitchen for international service:

```python
# settings.py
import os
from django.utils.translation import gettext_lazy as _

# Enable internationalization
USE_I18N = True
USE_L10N = True  # For locale-specific formatting
USE_TZ = True    # For timezone support

# Supported languages (like menu languages in your restaurant)
LANGUAGES = [
    ('en', _('English')),
    ('es', _('Spanish')),
    ('fr', _('French')),
    ('de', _('German')),
]

# Default language (your restaurant's primary language)
LANGUAGE_CODE = 'en'

# Where Django looks for translation files (your menu translations folder)
LOCALE_PATHS = [
    os.path.join(BASE_DIR, 'locale'),
]

# Middleware to detect customer's preferred language
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',  # Add this line
    'django.middleware.common.CommonMiddleware',
    # ... other middleware
]
```

### Making Your Views Multilingual

```python
# views.py
from django.shortcuts import render
from django.utils.translation import gettext as _, get_language
from django.http import HttpResponse

def restaurant_menu(request):
    # Like a chef greeting customers in their language
    current_language = get_language()
    
    context = {
        'welcome_message': _('Welcome to our restaurant!'),
        'today_special': _('Today\'s special: Grilled Salmon'),
        'price': _('Price'),
        'current_language': current_language,
    }
    return render(request, 'menu.html', context)

def dish_detail(request, dish_id):
    # Even dynamic content can be translated
    dish_name = _('Grilled Salmon')
    description = _('Fresh Atlantic salmon grilled to perfection with herbs.')
    
    context = {
        'dish_name': dish_name,
        'description': description,
        'preparation_time': _('15 minutes'),
    }
    return render(request, 'dish_detail.html', context)
```

### Syntax Explanation:
- `gettext_lazy as _`: Creates lazy translations (like having a menu template that gets filled in based on customer preference)
- `gettext as _`: Creates immediate translations (like translating on the spot)
- `get_language()`: Returns current active language (like asking "what language is this customer speaking?")

---

## Lesson 2: Translation Files - Your Recipe Translation Book

### The Chef Analogy
Translation files are like having a comprehensive translation book in your kitchen. When a Spanish-speaking customer orders, you look up "Grilled Salmon" and find "Salmón a la Parrilla". Each language has its own section in your book.

### Creating Translation Messages

First, we extract all translatable strings from our code:

```bash
# In your terminal (like compiling your recipe book)
python manage.py makemessages -l es  # For Spanish
python manage.py makemessages -l fr  # For French
```

This creates files like `locale/es/LC_MESSAGES/django.po`:

```po
# locale/es/LC_MESSAGES/django.po
msgid "Welcome to our restaurant!"
msgstr "¡Bienvenidos a nuestro restaurante!"

msgid "Today's special: Grilled Salmon"
msgstr "Especial de hoy: Salmón a la Parrilla"

msgid "Price"
msgstr "Precio"

msgid "Fresh Atlantic salmon grilled to perfection with herbs."
msgstr "Salmón atlántico fresco a la parrilla con hierbas."
```

### Using Translations in Templates

```html
<!-- menu.html -->
{% load i18n %}
<!DOCTYPE html>
<html lang="{{ LANGUAGE_CODE }}">
<head>
    <title>{% trans "Restaurant Menu" %}</title>
</head>
<body>
    <h1>{{ welcome_message }}</h1>
    
    <!-- Like displaying the daily special in customer's language -->
    <div class="special">
        <h2>{{ today_special }}</h2>
        <p>{% trans "Preparation time" %}: {% trans "15 minutes" %}</p>
    </div>
    
    <!-- Language switcher (like having language options on your menu) -->
    <div class="language-switcher">
        {% get_current_language as LANGUAGE_CODE %}
        <form action="{% url 'set_language' %}" method="post">
            {% csrf_token %}
            <select name="language" onchange="this.form.submit();">
                {% get_available_languages as LANGUAGES %}
                {% for lang_code, lang_name in LANGUAGES %}
                    <option value="{{ lang_code }}"{% if lang_code == LANGUAGE_CODE %} selected{% endif %}>
                        {{ lang_name }}
                    </option>
                {% endfor %}
            </select>
        </form>
    </div>
</body>
</html>
```

### Compiling Translations

```bash
# Compile your translation book (make it ready for use)
python manage.py compilemessages
```

### Syntax Explanation:
- `{% load i18n %}`: Loads internationalization template tags (like getting your translation tools ready)
- `{% trans "text" %}`: Translates text in templates (like looking up a translation on the spot)
- `{% get_current_language %}`: Gets the current language (like checking what language you're currently serving)

---

## Lesson 3: Locale-Specific Formatting - Cultural Table Manners

### The Chef Analogy
Different cultures have different ways of presenting information. Americans write dates as MM/DD/YYYY, while Europeans prefer DD/MM/YYYY. It's like knowing that French diners expect their salad after the main course, while Americans prefer it before. Same content, different presentation.

### Number and Currency Formatting

```python
# views.py
from django.shortcuts import render
from django.utils.translation import gettext as _
from django.utils import formats
from decimal import Decimal
import locale

def menu_prices(request):
    # Like pricing dishes according to local customs
    salmon_price = Decimal('24.99')
    wine_price = Decimal('45.50')
    
    context = {
        'salmon_price': salmon_price,
        'wine_price': wine_price,
        # Django automatically formats these according to current locale
        'formatted_salmon': formats.localize(salmon_price),
        'formatted_wine': formats.localize(wine_price),
    }
    return render(request, 'prices.html', context)

def reservation_details(request):
    from datetime import datetime
    
    # Current time (like showing reservation time in local format)
    reservation_time = datetime.now()
    
    context = {
        'reservation_time': reservation_time,
        # Django formats datetime according to locale settings
        'formatted_time': formats.date_format(reservation_time, 'DATETIME_FORMAT'),
        'formatted_date': formats.date_format(reservation_time, 'DATE_FORMAT'),
    }
    return render(request, 'reservation.html', context)
```

### Custom Locale Settings

```python
# settings.py

# Locale-specific formatting settings
FORMAT_MODULE_PATH = [
    'myrestaurant.formats',  # Custom format definitions
]

# In myrestaurant/formats/es/formats.py (Spanish formatting)
DATE_FORMAT = 'd/m/Y'  # European style: 25/12/2024
TIME_FORMAT = 'H:i'    # 24-hour format: 14:30
DATETIME_FORMAT = 'd/m/Y H:i'
DECIMAL_SEPARATOR = ','
THOUSAND_SEPARATOR = '.'
NUMBER_GROUPING = 3
```

### Template Usage

```html
<!-- prices.html -->
{% load i18n l10n %}
<!DOCTYPE html>
<html>
<head>
    <title>{% trans "Menu Prices" %}</title>
</head>
<body>
    <h1>{% trans "Today's Prices" %}</h1>
    
    <!-- Like displaying prices in local currency format -->
    <div class="menu-item">
        <h3>{% trans "Grilled Salmon" %}</h3>
        <p>{% trans "Price" %}: {{ salmon_price|localize }}</p>
    </div>
    
    <div class="menu-item">
        <h3>{% trans "House Wine" %}</h3>
        <p>{% trans "Price" %}: {{ wine_price|localize }}</p>
    </div>
    
    <!-- Unlocalized version (like showing raw price for staff) -->
    <div class="debug-info">
        <p>Raw salmon price: {{ salmon_price|unlocalize }}</p>
    </div>
</body>
</html>
```

### Syntax Explanation:
- `formats.localize()`: Formats numbers according to current locale (like converting prices to local format)
- `|localize` filter: Template filter for localization (like automatic local formatting)
- `|unlocalize` filter: Removes localization (like showing raw data)

---

## Lesson 4: Time Zone Handling - Serving Customers Across Time Zones

### The Chef Analogy
Imagine you're a chef with restaurants in New York, London, and Tokyo. When you advertise "Fresh bread baked at 6 AM daily," each location needs to know what 6 AM means in their local time. Django's timezone handling is like having a master clock that automatically adjusts to show the right time for each customer.

### Setting Up Time Zones

```python
# settings.py
USE_TZ = True
TIME_ZONE = 'UTC'  # Your kitchen's master clock (server time)

# In your views.py
from django.shortcuts import render
from django.utils import timezone
from django.contrib.auth.decorators import login_required
import pytz

def restaurant_hours(request):
    # Like showing opening hours in customer's local time
    utc_now = timezone.now()  # Master kitchen time
    
    # Define restaurant hours in restaurant's local time
    ny_tz = pytz.timezone('America/New_York')
    london_tz = pytz.timezone('Europe/London')
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    
    context = {
        'utc_time': utc_now,
        'ny_time': utc_now.astimezone(ny_tz),
        'london_time': utc_now.astimezone(london_tz),
        'tokyo_time': utc_now.astimezone(tokyo_tz),
    }
    return render(request, 'hours.html', context)

@login_required
def make_reservation(request):
    from datetime import datetime
    
    if request.method == 'POST':
        # Customer selects time in their timezone
        reservation_date = request.POST.get('date')
        reservation_time = request.POST.get('time')
        
        # Convert to restaurant's timezone for kitchen scheduling
        user_timezone = request.user.profile.timezone  # Assume user has timezone in profile
        user_tz = pytz.timezone(user_timezone)
        
        # Create datetime in user's timezone
        naive_datetime = datetime.strptime(f"{reservation_date} {reservation_time}", "%Y-%m-%d %H:%M")
        user_datetime = user_tz.localize(naive_datetime)
        
        # Convert to UTC for database storage (like converting to master kitchen time)
        utc_datetime = user_datetime.astimezone(pytz.UTC)
        
        # Save reservation with UTC time
        reservation = Reservation.objects.create(
            user=request.user,
            datetime=utc_datetime,
            timezone=user_timezone
        )
        
    return render(request, 'reservation_form.html')
```

### Working with Timezones in Models

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    timezone = models.CharField(max_length=50, default='UTC')
    # Like storing each customer's preferred time zone

class Reservation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    datetime = models.DateTimeField()  # Always stored in UTC (master time)
    timezone = models.CharField(max_length=50)  # Customer's timezone
    special_requests = models.TextField(blank=True)
    
    def local_datetime(self):
        # Convert stored UTC time back to customer's timezone
        user_tz = pytz.timezone(self.timezone)
        return self.datetime.astimezone(user_tz)

class KitchenSchedule(models.Model):
    prep_start_time = models.DateTimeField()  # In UTC
    dish_name = models.CharField(max_length=100)
    estimated_completion = models.DateTimeField()  # In UTC
    
    class Meta:
        ordering = ['prep_start_time']
```

### Template Usage with Timezones

```html
<!-- hours.html -->
{% load i18n tz %}
<!DOCTYPE html>
<html>
<head>
    <title>{% trans "Restaurant Hours Worldwide" %}</title>
</head>
<body>
    <h1>{% trans "Current Time at Our Locations" %}</h1>
    
    <!-- Like displaying time at each restaurant location -->
    <div class="time-zones">
        <div class="location">
            <h3>{% trans "New York" %}</h3>
            {% timezone "America/New_York" %}
                <p>{{ utc_time|date:"F j, Y, P" }}</p>
            {% endtimezone %}
        </div>
        
        <div class="location">
            <h3>{% trans "London" %}</h3>
            {% timezone "Europe/London" %}
                <p>{{ utc_time|date:"F j, Y, P" }}</p>
            {% endtimezone %}
        </div>
        
        <div class="location">
            <h3>{% trans "Tokyo" %}</h3>
            {% timezone "Asia/Tokyo" %}
                <p>{{ utc_time|date:"F j, Y, P" }}</p>
            {% endtimezone %}
        </div>
    </div>
    
    <!-- User's local time (if logged in) -->
    {% if user.is_authenticated %}
        <div class="user-time">
            <h3>{% trans "Your Local Time" %}</h3>
            {% timezone user.profile.timezone %}
                <p>{{ utc_time|date:"F j, Y, P" }}</p>
            {% endtimezone %}
        </div>
    {% endif %}
</body>
</html>
```

### Syntax Explanation:
- `timezone.now()`: Gets current UTC time (like checking the master kitchen clock)
- `astimezone()`: Converts timezone (like adjusting clock for different locations)
- `{% timezone %}` tag: Temporarily sets timezone context in templates
- `pytz.timezone()`: Creates timezone objects (like setting up regional clocks)

---

## Assignment: International Recipe Blog

Create a Django application for a recipe blog that supports multiple languages and handles international users. Your blog should be like a chef's international cookbook that adapts to different cultures.

### Requirements:

1. **Multi-language Support**:
   - Support at least English and Spanish
   - Translate all user interface elements (navigation, buttons, labels)
   - Allow users to switch languages dynamically

2. **Recipe Model with Translation**:
   ```python
   class Recipe(models.Model):
       title = models.CharField(max_length=200)
       ingredients = models.TextField()
       instructions = models.TextField()
       prep_time = models.IntegerField()  # in minutes
       created_at = models.DateTimeField(auto_now_add=True)
       author = models.ForeignKey(User, on_delete=models.CASCADE)
   ```

3. **Locale-Specific Formatting**:
   - Display cooking times in appropriate format (12/24 hour)
   - Show recipe creation dates in local date format
   - Handle any numeric measurements properly

4. **Timezone Features**:
   - Show recipe creation time in user's local timezone
   - Display "posted X hours ago" correctly for different timezones
   - Allow users to set their preferred timezone

5. **Required Views**:
   - Recipe list page (with language switching)
   - Recipe detail page (fully translated)
   - Add recipe form (with proper timezone handling)

### Deliverables:
- Django project with proper i18n settings
- At least 2 translation files (English and Spanish)
- Templates using Django's i18n template tags
- Views that handle timezone conversion
- Working language switcher

### Success Criteria:
Your application should work seamlessly whether accessed by a user in Madrid (Spanish, European date format, CET timezone) or Los Angeles (English, American date format, PST timezone) - just like how a good chef adapts their service to different customers while maintaining the quality of their food.

---

## Summary

Today you learned how to make your Django applications truly international - like transforming a local restaurant into a global chain that feels native to customers anywhere in the world. You mastered:

- **Multi-language support**: Making your app speak multiple languages fluently
- **Translation files**: Organizing and managing your translations systematically  
- **Locale-specific formatting**: Presenting information according to local customs
- **Time zone handling**: Serving customers across different time zones accurately

Just as a successful international chef maintains the essence of their cuisine while adapting to local tastes, Django's i18n features let you reach global audiences while keeping your core application logic intact. Your digital kitchen is now ready to serve customers from Tokyo to São Paulo, each feeling like the app was made specifically for them!

# Project: Multi-language Expense Tracker

## Project Structure
```
expense_tracker/
├── manage.py
├── expense_tracker/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── expenses/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── migrations/
├── templates/
│   └── expenses/
│       ├── base.html
│       ├── expense_list.html
│       └── add_expense.html
├── locale/
│   ├── es/
│   │   └── LC_MESSAGES/
│   │       ├── django.po
│   │       └── django.mo
│   └── fr/
│       └── LC_MESSAGES/
│           ├── django.po
│           └── django.mo
└── static/
    └── css/
        └── style.css
```

## 1. Settings Configuration

**expense_tracker/settings.py**
```python
import os
from django.utils.translation import gettext_lazy as _

BASE_DIR = Path(__file__).resolve().parent.parent

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Available languages
LANGUAGES = [
    ('en', _('English')),
    ('es', _('Spanish')),
    ('fr', _('French')),
]

# Locale paths
LOCALE_PATHS = [
    BASE_DIR / 'locale',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',  # Add this for i18n
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'expense_tracker.urls'

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'expenses',
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
```

## 2. Main URL Configuration

**expense_tracker/urls.py**
```python
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.i18n import i18n_patterns

urlpatterns = [
    path('admin/', admin.site.urls),
]

# Add i18n patterns for multi-language URLs
urlpatterns += i18n_patterns(
    path('', include('expenses.urls')),
    prefix_default_language=False
)
```

## 3. Models

**expenses/models.py**
```python
from django.db import models
from django.contrib.auth.models import User
from django.utils.translation import gettext_lazy as _
from django.utils import timezone

class Category(models.Model):
    name = models.CharField(_('Name'), max_length=100)
    description = models.TextField(_('Description'), blank=True)
    created_at = models.DateTimeField(_('Created at'), auto_now_add=True)
    
    class Meta:
        verbose_name = _('Category')
        verbose_name_plural = _('Categories')
    
    def __str__(self):
        return self.name

class Expense(models.Model):
    CURRENCY_CHOICES = [
        ('USD', 'USD ($)'),
        ('EUR', 'EUR (€)'),
        ('GBP', 'GBP (£)'),
        ('JPY', 'JPY (¥)'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name=_('User'))
    title = models.CharField(_('Title'), max_length=200)
    amount = models.DecimalField(_('Amount'), max_digits=10, decimal_places=2)
    currency = models.CharField(_('Currency'), max_length=3, choices=CURRENCY_CHOICES, default='USD')
    category = models.ForeignKey(Category, on_delete=models.CASCADE, verbose_name=_('Category'))
    description = models.TextField(_('Description'), blank=True)
    date = models.DateTimeField(_('Date'), default=timezone.now)
    created_at = models.DateTimeField(_('Created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('Updated at'), auto_now=True)
    
    class Meta:
        verbose_name = _('Expense')
        verbose_name_plural = _('Expenses')
        ordering = ['-date']
    
    def __str__(self):
        return f"{self.title} - {self.amount} {self.currency}"
```

## 4. Forms

**expenses/forms.py**
```python
from django import forms
from django.utils.translation import gettext_lazy as _
from .models import Expense, Category

class ExpenseForm(forms.ModelForm):
    class Meta:
        model = Expense
        fields = ['title', 'amount', 'currency', 'category', 'description', 'date']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': _('Enter expense title')
            }),
            'amount': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': _('0.00'),
                'step': '0.01'
            }),
            'currency': forms.Select(attrs={'class': 'form-control'}),
            'category': forms.Select(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': _('Optional description')
            }),
            'date': forms.DateTimeInput(attrs={
                'class': 'form-control',
                'type': 'datetime-local'
            }),
        }
        labels = {
            'title': _('Expense Title'),
            'amount': _('Amount'),
            'currency': _('Currency'),
            'category': _('Category'),
            'description': _('Description'),
            'date': _('Date'),
        }

class CategoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = ['name', 'description']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': _('Category name')
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 2,
                'placeholder': _('Category description')
            }),
        }
        labels = {
            'name': _('Category Name'),
            'description': _('Description'),
        }
```

## 5. Views

**expenses/views.py**
```python
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils.translation import gettext as _
from django.utils import timezone
from django.db.models import Sum, Count
from django.http import JsonResponse
from .models import Expense, Category
from .forms import ExpenseForm, CategoryForm
import json
from decimal import Decimal

@login_required
def expense_list(request):
    expenses = Expense.objects.filter(user=request.user)
    
    # Calculate statistics
    total_expenses = expenses.aggregate(
        total=Sum('amount'),
        count=Count('id')
    )
    
    # Monthly breakdown
    monthly_data = expenses.extra(
        select={'month': "strftime('%%Y-%%m', date)"}
    ).values('month').annotate(
        total=Sum('amount'),
        count=Count('id')
    ).order_by('month')
    
    # Category breakdown
    category_data = expenses.values('category__name').annotate(
        total=Sum('amount'),
        count=Count('id')
    ).order_by('-total')
    
    context = {
        'expenses': expenses[:10],  # Show latest 10
        'total_amount': total_expenses['total'] or 0,
        'total_count': total_expenses['count'] or 0,
        'monthly_data': list(monthly_data),
        'category_data': list(category_data),
        'page_title': _('Expense Dashboard')
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
            messages.success(request, _('Expense added successfully!'))
            return redirect('expense_list')
    else:
        form = ExpenseForm()
    
    return render(request, 'expenses/add_expense.html', {
        'form': form,
        'page_title': _('Add New Expense')
    })

@login_required
def edit_expense(request, expense_id):
    expense = get_object_or_404(Expense, id=expense_id, user=request.user)
    
    if request.method == 'POST':
        form = ExpenseForm(request.POST, instance=expense)
        if form.is_valid():
            form.save()
            messages.success(request, _('Expense updated successfully!'))
            return redirect('expense_list')
    else:
        form = ExpenseForm(instance=expense)
    
    return render(request, 'expenses/edit_expense.html', {
        'form': form,
        'expense': expense,
        'page_title': _('Edit Expense')
    })

@login_required
def delete_expense(request, expense_id):
    expense = get_object_or_404(Expense, id=expense_id, user=request.user)
    
    if request.method == 'POST':
        expense.delete()
        messages.success(request, _('Expense deleted successfully!'))
        return redirect('expense_list')
    
    return render(request, 'expenses/confirm_delete.html', {
        'expense': expense,
        'page_title': _('Delete Expense')
    })

@login_required
def expense_analytics(request):
    expenses = Expense.objects.filter(user=request.user)
    
    # Analytics data for charts
    analytics_data = {
        'currency_breakdown': list(expenses.values('currency').annotate(
            total=Sum('amount'),
            count=Count('id')
        )),
        'daily_spending': list(expenses.extra(
            select={'day': "date(date)"}
        ).values('day').annotate(
            total=Sum('amount')
        ).order_by('day')[-30:])  # Last 30 days
    }
    
    return JsonResponse(analytics_data)
```

## 6. URLs

**expenses/urls.py**
```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.expense_list, name='expense_list'),
    path('add/', views.add_expense, name='add_expense'),
    path('edit/<int:expense_id>/', views.edit_expense, name='edit_expense'),
    path('delete/<int:expense_id>/', views.delete_expense, name='delete_expense'),
    path('analytics/', views.expense_analytics, name='expense_analytics'),
]
```

## 7. Base Template

**templates/expenses/base.html**
```html
{% load static %}
{% load i18n %}
<!DOCTYPE html>
<html lang="{{ LANGUAGE_CODE }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}{{ page_title }}{% endblock %} - {% trans "Expense Tracker" %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="{% static 'css/style.css' %}" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{% url 'expense_list' %}">
                💰 {% trans "Expense Tracker" %}
            </a>
            
            <div class="navbar-nav ms-auto">
                <div class="nav-item dropdown">
                    <a class="nav-link dropdown-toggle" href="#" id="languageDropdown" role="button" data-bs-toggle="dropdown">
                        🌐 {% get_current_language as LANGUAGE_CODE %}
                        {% if LANGUAGE_CODE == 'en' %}{% trans "English" %}
                        {% elif LANGUAGE_CODE == 'es' %}{% trans "Spanish" %}
                        {% elif LANGUAGE_CODE == 'fr' %}{% trans "French" %}
                        {% endif %}
                    </a>
                    <ul class="dropdown-menu">
                        {% get_available_languages as LANGUAGES %}
                        {% for lang_code, lang_name in LANGUAGES %}
                            <li>
                                <form action="{% url 'set_language' %}" method="post" class="d-inline">
                                    {% csrf_token %}
                                    <input name="next" type="hidden" value="{{ request.get_full_path }}" />
                                    <input name="language" type="hidden" value="{{ lang_code }}" />
                                    <button type="submit" class="dropdown-item">
                                        {{ lang_name }}
                                    </button>
                                </form>
                            </li>
                        {% endfor %}
                    </ul>
                </div>
                {% if user.is_authenticated %}
                    <span class="navbar-text">
                        {% trans "Welcome" %}, {{ user.username }}!
                    </span>
                {% endif %}
            </div>
        </div>
    </nav>

    <main class="container mt-4">
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}

        {% block content %}
        {% endblock %}
    </main>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
```

## 8. Expense List Template

**templates/expenses/expense_list.html**
```html
{% extends 'expenses/base.html' %}
{% load static %}
{% load i18n %}
{% load humanize %}

{% block content %}
<div class="row">
    <div class="col-md-12">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h1>{% trans "Expense Dashboard" %}</h1>
            <a href="{% url 'add_expense' %}" class="btn btn-primary">
                ➕ {% trans "Add New Expense" %}
            </a>
        </div>
    </div>
</div>

<!-- Statistics Cards -->
<div class="row mb-4">
    <div class="col-md-6">
        <div class="card bg-primary text-white">
            <div class="card-body">
                <h5 class="card-title">{% trans "Total Expenses" %}</h5>
                <h2 class="card-text">{{ total_count }}</h2>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card bg-success text-white">
            <div class="card-body">
                <h5 class="card-title">{% trans "Total Amount" %}</h5>
                <h2 class="card-text">${{ total_amount|floatformat:2|default:"0.00" }}</h2>
            </div>
        </div>
    </div>
</div>

<!-- Charts Row -->
<div class="row mb-4">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5>{% trans "Monthly Spending" %}</h5>
            </div>
            <div class="card-body">
                <canvas id="monthlyChart" height="200"></canvas>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5>{% trans "Category Breakdown" %}</h5>
            </div>
            <div class="card-body">
                <canvas id="categoryChart" height="200"></canvas>
            </div>
        </div>
    </div>
</div>

<!-- Recent Expenses -->
<div class="row">
    <div class="col-md-12">
        <div class="card">
            <div class="card-header">
                <h5>{% trans "Recent Expenses" %}</h5>
            </div>
            <div class="card-body">
                {% if expenses %}
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>{% trans "Title" %}</th>
                                    <th>{% trans "Amount" %}</th>
                                    <th>{% trans "Category" %}</th>
                                    <th>{% trans "Date" %}</th>
                                    <th>{% trans "Actions" %}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for expense in expenses %}
                                    <tr>
                                        <td>{{ expense.title }}</td>
                                        <td>{{ expense.amount }} {{ expense.currency }}</td>
                                        <td>{{ expense.category.name }}</td>
                                        <td>{{ expense.date|date:"Y-m-d H:i" }}</td>
                                        <td>
                                            <a href="{% url 'edit_expense' expense.id %}" class="btn btn-sm btn-warning">
                                                ✏️ {% trans "Edit" %}
                                            </a>
                                            <a href="{% url 'delete_expense' expense.id %}" class="btn btn-sm btn-danger">
                                                🗑️ {% trans "Delete" %}
                                            </a>
                                        </td>
                                    </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                {% else %}
                    <div class="text-center py-4">
                        <p class="text-muted">{% trans "No expenses found. Start by adding your first expense!" %}</p>
                        <a href="{% url 'add_expense' %}" class="btn btn-primary">
                            {% trans "Add Your First Expense" %}
                        </a>
                    </div>
                {% endif %}
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
    // Monthly Chart
    const monthlyCtx = document.getElementById('monthlyChart').getContext('2d');
    const monthlyData = {{ monthly_data|safe }};
    
    new Chart(monthlyCtx, {
        type: 'line',
        data: {
            labels: monthlyData.map(item => item.month),
            datasets: [{
                label: '{% trans "Monthly Spending" %}',
                data: monthlyData.map(item => parseFloat(item.total)),
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });

    // Category Chart
    const categoryCtx = document.getElementById('categoryChart').getContext('2d');
    const categoryData = {{ category_data|safe }};
    
    new Chart(categoryCtx, {
        type: 'doughnut',
        data: {
            labels: categoryData.map(item => item.category__name),
            datasets: [{
                data: categoryData.map(item => parseFloat(item.total)),
                backgroundColor: [
                    '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
                    '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': $' + context.parsed.toFixed(2);
                        }
                    }
                }
            }
        }
    });
</script>
{% endblock %}
```

## 9. Add Expense Template

**templates/expenses/add_expense.html**
```html
{% extends 'expenses/base.html' %}
{% load i18n %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h3>➕ {% trans "Add New Expense" %}</h3>
            </div>
            <div class="card-body">
                <form method="post" class="needs-validation" novalidate>
                    {% csrf_token %}
                    
                    <div class="row">
                        <div class="col-md-8">
                            <div class="mb-3">
                                <label class="form-label">{{ form.title.label }}</label>
                                {{ form.title }}
                                {% if form.title.errors %}
                                    <div class="invalid-feedback d-block">
                                        {{ form.title.errors.0 }}
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="mb-3">
                                <label class="form-label">{{ form.currency.label }}</label>
                                {{ form.currency }}
                            </div>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label class="form-label">{{ form.amount.label }}</label>
                                {{ form.amount }}
                                {% if form.amount.errors %}
                                    <div class="invalid-feedback d-block">
                                        {{ form.amount.errors.0 }}
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label class="form-label">{{ form.category.label }}</label>
                                {{ form.category }}
                                {% if form.category.errors %}
                                    <div class="invalid-feedback d-block">
                                        {{ form.category.errors.0 }}
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label class="form-label">{{ form.date.label }}</label>
                        {{ form.date }}
                        {% if form.date.errors %}
                            <div class="invalid-feedback d-block">
                                {{ form.date.errors.0 }}
                            </div>
                        {% endif %}
                    </div>
                    
                    <div class="mb-3">
                        <label class="form-label">{{ form.description.label }}</label>
                        {{ form.description }}
                        {% if form.description.errors %}
                            <div class="invalid-feedback d-block">
                                {{ form.description.errors.0 }}
                            </div>
                        {% endif %}
                    </div>
                    
                    <div class="d-flex justify-content-between">
                        <a href="{% url 'expense_list' %}" class="btn btn-secondary">
                            ⬅️ {% trans "Back to Dashboard" %}
                        </a>
                        <button type="submit" class="btn btn-primary">
                            💾 {% trans "Save Expense" %}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

## 10. CSS Styling

**static/css/style.css**
```css
/* Multi-language Expense Tracker Styles */
body {
    background-color: #f8f9fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.navbar-brand {
    font-weight: bold;
    font-size: 1.5rem;
}

.card {
    border: none;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}

.card:hover {
    transform: translateY(-2px);
}

.card-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px 10px 0 0 !important;
    border: none;
}

.btn {
    border-radius: 25px;
    padding: 10px 20px;
    font-weight: 500;
    transition: all 0.3s;
}

.btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.form-control {
    border-radius: 8px;
    border: 2px solid #e9ecef;
    transition: border-color 0.3s;
}

.form-control:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
}

.table {
    border-radius: 10px;
    overflow: hidden;
    background: white;
}

.table th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    font-weight: 600;
}

.table td {
    vertical-align: middle;
    border-color: #f1f3f4;
}

.alert {
    border: none;
    border-radius: 10px;
    padding: 15px 20px;
}

.bg-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.bg-success {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
}

.dropdown-menu {
    border-radius: 10px;
    border: none;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.dropdown-item {
    border: none;
    background: none;
    width: 100%;
    text-align: left;
    padding: 8px 20px;
    transition: background-color 0.2s;
}

.dropdown-item:hover {
    background-color: #f8f9fa;
}

/* RTL Support for Arabic/Hebrew */
[dir="rtl"] .navbar-nav {
    margin-left: auto;
    margin-right: 0;
}

/* Chart containers */
canvas {
    max-height: 300px !important;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .container {
        padding: 0 15px;
    }
    
    .card {
        margin-bottom: 20px;
    }
    
    .btn {
        padding: 8px 16px;
        font-size: 0.9rem;
    }
}

/* Language-specific adjustments */
html[lang="ar"], html[lang="he"] {
    direction: rtl;
}

html[lang="ar"] body, html[lang="he"] body {
    font-family: 'Arial', 'Helvetica', sans-serif;
}
```

## 11. Translation Files Setup

Create Spanish translation file: **locale/es/LC_MESSAGES/django.po**
```po
# Spanish translation for Expense Tracker
msgid ""
msgstr ""
"Language: es\n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=UTF-8\n"

msgid "English"
msgstr "Inglés"

msgid "Spanish"
msgstr "Español"

msgid "French"
msgstr "Francés"

msgid "Expense Tracker"
msgstr "Rastreador de Gastos"

msgid "Welcome"
msgstr "Bienvenido"

msgid "Expense Dashboard"
msgstr "Panel de Gastos"

msgid "Add New Expense"
msgstr "Agregar Nuevo Gasto"

msgid "Total Expenses"
msgstr "Total de Gastos"

msgid "Total Amount"
msgstr "Monto Total"

msgid "Monthly Spending"
msgstr "Gasto Mensual"

msgid "Category Breakdown"
msgstr "Desglose por Categoría"

msgid "Recent Expenses"
msgstr "Gastos Recientes"

msgid "Title"
msgstr "Título"

msgid "Amount"
msgstr "Monto"

msgid "Category"
msgstr "Categoría"

msgid "Date"
msgstr "Fecha"

msgid "Actions"
msgstr "Acciones"

msgid "Edit"
msgstr "Editar"

msgid "Delete"
msgstr "Eliminar"

msgid "No expenses found. Start by adding your first expense!"
msgstr "No se encontraron gastos. ¡Comienza agregando tu primer gasto!"

msgid "Add Your First Expense"
msgstr "Agregar Tu Primer Gasto"

msgid "Enter expense title"
msgstr "Ingrese el título del gasto"

msgid "Optional description"
msgstr "Descripción opcional"

msgid "Expense Title"
msgstr "Título del Gasto"

msgid "Currency"
msgstr "Moneda"

msgid "Description"
msgstr "Descripción"

msgid "Back to Dashboard"
msgstr "Volver al Panel"

msgid "Save Expense"
msgstr "Guardar Gasto"

msgid "Expense added successfully!"
msgstr "¡Gasto agregado exitosamente!"

msgid "Expense updated successfully!"
msgstr "¡Gasto actualizado exitosamente!"

msgid "Expense deleted successfully!"
msgstr "¡Gasto eliminado exitosamente!"

msgid "Edit Expense"
msgstr "Editar Gasto"

msgid "Delete Expense"
msgstr "Eliminar Gasto"

msgid "Name"
msgstr "Nombre"

msgid "Created at"
msgstr "Creado en"

msgid "Categories"
msgstr "Categorías"

msgid "User"
msgstr "Usuario"

msgid "Updated at"
msgstr "Actualizado en"

msgid "Expenses"
msgstr "Gastos"

msgid "Category name"
msgstr "Nombre de la categoría"

msgid "Category description"
msgstr "Descripción de la categoría"

msgid "Category Name"
msgstr "Nombre de la Categoría"
```

Create French translation file: **locale/fr/LC_MESSAGES/django.po**
```po
# French translation for Expense Tracker
msgid ""
msgstr ""
"Language: fr\n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=UTF-8\n"

msgid "English"
msgstr "Anglais"

msgid "Spanish"
msgstr "Espagnol"

msgid "French"
msgstr "Français"

msgid "Expense Tracker"
msgstr "Suivi des Dépenses"

msgid "Welcome"
msgstr "Bienvenue"

msgid "Expense Dashboard"
msgstr "Tableau de Bord des Dépenses"

msgid "Add New Expense"
msgstr "Ajouter une Nouvelle Dépense"

msgid "Total Expenses"
msgstr "Total des Dépenses"

msgid "Total Amount"
msgstr "Montant Total"

msgid "Monthly Spending"
msgstr "Dépenses Mensuelles"

msgid "Category Breakdown"
msgstr "Répartition par Catégorie"

msgid "Recent Expenses"
msgstr "Dépenses Récentes"

msgid "Title"
msgstr "Titre"

msgid "Amount"
msgstr "Montant"

msgid "Category"
msgstr "Catégorie"

msgid "Date"
msgstr "Date"

msgid "Actions"
msgstr "Actions"

msgid "Edit"
msgstr "Modifier"

msgid "Delete"
msgstr "Supprimer"

msgid "No expenses found. Start by adding your first expense!"
msgstr "Aucune dépense trouvée. Commencez par ajouter votre première dépense!"

msgid "Add Your First Expense"
msgstr "Ajouter Votre Première Dépense"

msgid "Enter expense title"
msgstr "Saisir le titre de la dépense"

msgid "Optional description"
msgstr "Description facultative"

msgid "Expense Title"
msgstr "Titre de la Dépense"

msgid "Currency"
msgstr "Devise"

msgid "Description"
msgstr "Description"

msgid "Back to Dashboard"
msgstr "Retour au Tableau de Bord"

msgid "Save Expense"
msgstr "Enregistrer la Dépense"

msgid "Expense added successfully!"
msgstr "Dépense ajoutée avec succès!"

msgid "Expense updated successfully!"
msgstr "Dépense mise à jour avec succès!"

msgid "Expense deleted successfully!"
msgstr "Dépense supprimée avec succès!"

msgid "Edit Expense"
msgstr "Modifier la Dépense"

msgid "Delete Expense"
msgstr "Supprimer la Dépense"

msgid "Name"
msgstr "Nom"

msgid "Created at"
msgstr "Créé le"

msgid "Categories"
msgstr "Catégories"

msgid "User"
msgstr "Utilisateur"

msgid "Updated at"
msgstr "Mis à jour le"

msgid "Expenses"
msgstr "Dépenses"

msgid "Category name"
msgstr "Nom de la catégorie"

msgid "Category description"
msgstr "Description de la catégorie"

msgid "Category Name"
msgstr "Nom de la Catégorie"
```

## 12. Admin Configuration

**expenses/admin.py**
```python
from django.contrib import admin
from django.utils.translation import gettext_lazy as _
from .models import Category, Expense

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'description', 'created_at']
    search_fields = ['name', 'description']
    list_filter = ['created_at']
    ordering = ['name']

@admin.register(Expense)
class ExpenseAdmin(admin.ModelAdmin):
    list_display = ['title', 'amount', 'currency', 'category', 'user', 'date']
    list_filter = ['currency', 'category', 'date', 'created_at']
    search_fields = ['title', 'description', 'user__username']
    date_hierarchy = 'date'
    ordering = ['-date']
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': ('title', 'amount', 'currency', 'category')
        }),
        (_('Details'), {
            'fields': ('description', 'date')
        }),
        (_('System'), {
            'fields': ('user',),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        return qs.filter(user=request.user)
```

## 13. Delete Confirmation Template

**templates/expenses/confirm_delete.html**
```html
{% extends 'expenses/base.html' %}
{% load i18n %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header bg-danger text-white">
                <h3>🗑️ {% trans "Delete Expense" %}</h3>
            </div>
            <div class="card-body">
                <div class="alert alert-warning">
                    <strong>{% trans "Warning!" %}</strong>
                    {% trans "Are you sure you want to delete this expense? This action cannot be undone." %}
                </div>
                
                <div class="expense-details bg-light p-3 rounded">
                    <h5>{{ expense.title }}</h5>
                    <p><strong>{% trans "Amount" %}:</strong> {{ expense.amount }} {{ expense.currency }}</p>
                    <p><strong>{% trans "Category" %}:</strong> {{ expense.category.name }}</p>
                    <p><strong>{% trans "Date" %}:</strong> {{ expense.date|date:"Y-m-d H:i" }}</p>
                    {% if expense.description %}
                        <p><strong>{% trans "Description" %}:</strong> {{ expense.description }}</p>
                    {% endif %}
                </div>
                
                <div class="d-flex justify-content-between mt-4">
                    <a href="{% url 'expense_list' %}" class="btn btn-secondary">
                        ⬅️ {% trans "Cancel" %}
                    </a>
                    <form method="post" class="d-inline">
                        {% csrf_token %}
                        <button type="submit" class="btn btn-danger">
                            🗑️ {% trans "Yes, Delete" %}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

## 14. Edit Expense Template

**templates/expenses/edit_expense.html**
```html
{% extends 'expenses/base.html' %}
{% load i18n %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h3>✏️ {% trans "Edit Expense" %}</h3>
            </div>
            <div class="card-body">
                <form method="post" class="needs-validation" novalidate>
                    {% csrf_token %}
                    
                    <div class="row">
                        <div class="col-md-8">
                            <div class="mb-3">
                                <label class="form-label">{{ form.title.label }}</label>
                                {{ form.title }}
                                {% if form.title.errors %}
                                    <div class="invalid-feedback d-block">
                                        {{ form.title.errors.0 }}
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="mb-3">
                                <label class="form-label">{{ form.currency.label }}</label>
                                {{ form.currency }}
                            </div>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label class="form-label">{{ form.amount.label }}</label>
                                {{ form.amount }}
                                {% if form.amount.errors %}
                                    <div class="invalid-feedback d-block">
                                        {{ form.amount.errors.0 }}
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label class="form-label">{{ form.category.label }}</label>
                                {{ form.category }}
                                {% if form.category.errors %}
                                    <div class="invalid-feedback d-block">
                                        {{ form.category.errors.0 }}
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label class="form-label">{{ form.date.label }}</label>
                        {{ form.date }}
                        {% if form.date.errors %}
                            <div class="invalid-feedback d-block">
                                {{ form.date.errors.0 }}
                            </div>
                        {% endif %}
                    </div>
                    
                    <div class="mb-3">
                        <label class="form-label">{{ form.description.label }}</label>
                        {{ form.description }}
                        {% if form.description.errors %}
                            <div class="invalid-feedback d-block">
                                {{ form.description.errors.0 }}
                            </div>
                        {% endif %}
                    </div>
                    
                    <div class="d-flex justify-content-between">
                        <a href="{% url 'expense_list' %}" class="btn btn-secondary">
                            ⬅️ {% trans "Back to Dashboard" %}
                        </a>
                        <button type="submit" class="btn btn-primary">
                            💾 {% trans "Update Expense" %}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

## 15. Setup Instructions

### Database Migration
```bash
# Create and apply migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

### Translation Setup
```bash
# Create translation files
python manage.py makemessages -l es
python manage.py makemessages -l fr

# Compile translation files
python manage.py compilemessages
```

### Initial Data
```python
# Create initial categories via Django shell
python manage.py shell

from expenses.models import Category

categories = [
    {'name': 'Food & Dining', 'description': 'Restaurant meals, groceries, food delivery'},
    {'name': 'Transportation', 'description': 'Gas, public transport, taxi, car maintenance'},
    {'name': 'Entertainment', 'description': 'Movies, games, streaming services, hobbies'},
    {'name': 'Shopping', 'description': 'Clothes, electronics, household items'},
    {'name': 'Health & Medical', 'description': 'Doctor visits, medications, health insurance'},
    {'name': 'Utilities', 'description': 'Electricity, water, internet, phone bills'},
]

for cat_data in categories:
    Category.objects.get_or_create(**cat_data)
```

### Run the Application
```bash
python manage.py runserver
```

## Features Implemented

✅ **Multi-language Support**: English, Spanish, French with easy language switching
✅ **Currency Support**: Multiple currencies (USD, EUR, GBP, JPY)
✅ **Time Zone Awareness**: Proper datetime handling
✅ **Interactive Dashboard**: Charts showing monthly spending and category breakdown
✅ **Responsive Design**: Mobile-friendly Bootstrap interface
✅ **User Authentication**: Expense tracking per user
✅ **CRUD Operations**: Create, Read, Update, Delete expenses
✅ **Data Visualization**: Chart.js integration for spending analytics
✅ **Admin Interface**: Django admin with internationalization
✅ **Form Validation**: Client and server-side validation
✅ **Success Messages**: User feedback for all actions