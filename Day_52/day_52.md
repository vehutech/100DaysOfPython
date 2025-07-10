# Day 52: Advanced Forms & Formsets

## Learning Objective
By the end of this lesson, you will be able to create dynamic forms that adapt to user input, manage multiple related forms simultaneously, build custom form widgets for enhanced user experience, and implement multi-step form wizards - just like a master chef who can prepare multiple courses simultaneously while adapting recipes based on available ingredients.

---

## Introduction

Imagine that you're running a high-end restaurant kitchen where orders come in with varying complexity. Sometimes you need just a simple dish (basic form), but other times you're preparing a multi-course meal for a large party (formsets), customizing presentations for dietary restrictions (custom widgets), or guiding customers through a wine-pairing experience step by step (form wizards). 

Django's advanced form features are like having a sophisticated kitchen brigade system - each tool serves a specific purpose, but together they create seamless, complex user experiences.

---

## 1. Dynamic Forms

### The Chef's Adaptive Menu Concept

Think of dynamic forms like a chef's daily specials menu that changes based on what ingredients are available in the kitchen. The form structure adapts based on user selections or external conditions.

### Code Example: Dynamic Restaurant Order Form

```python
# forms.py
from django import forms
from django.forms import ModelForm
from .models import Restaurant, MenuItem, OrderItem

class DynamicOrderForm(forms.Form):
    restaurant = forms.ModelChoiceField(
        queryset=Restaurant.objects.all(),
        empty_label="Select a restaurant",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    def __init__(self, *args, **kwargs):
        # Extract restaurant_id if passed during form initialization
        restaurant_id = kwargs.pop('restaurant_id', None)
        super().__init__(*args, **kwargs)
        
        # If restaurant is selected, populate menu items dynamically
        if restaurant_id:
            self.fields['menu_items'] = forms.ModelMultipleChoiceField(
                queryset=MenuItem.objects.filter(restaurant_id=restaurant_id),
                widget=forms.CheckboxSelectMultiple,
                required=False
            )
            
            # Add dynamic fields based on restaurant type
            restaurant = Restaurant.objects.get(id=restaurant_id)
            if restaurant.cuisine_type == 'italian':
                self.fields['pasta_preference'] = forms.ChoiceField(
                    choices=[('al_dente', 'Al Dente'), ('soft', 'Soft')],
                    required=False
                )
            elif restaurant.cuisine_type == 'asian':
                self.fields['spice_level'] = forms.ChoiceField(
                    choices=[('mild', 'Mild'), ('medium', 'Medium'), ('hot', 'Hot')],
                    required=False
                )

# views.py
from django.shortcuts import render
from django.http import JsonResponse
from .forms import DynamicOrderForm

def order_form_view(request):
    if request.method == 'POST':
        form = DynamicOrderForm(request.POST)
        if form.is_valid():
            # Process the order
            return render(request, 'order_success.html')
    else:
        form = DynamicOrderForm()
    
    return render(request, 'order_form.html', {'form': form})

def get_menu_items(request):
    """AJAX endpoint to fetch menu items based on restaurant selection"""
    restaurant_id = request.GET.get('restaurant_id')
    if restaurant_id:
        menu_items = MenuItem.objects.filter(restaurant_id=restaurant_id)
        data = [{'id': item.id, 'name': item.name, 'price': str(item.price)} 
                for item in menu_items]
        return JsonResponse({'menu_items': data})
    return JsonResponse({'menu_items': []})
```

### JavaScript for Dynamic Updates

```javascript
// static/js/dynamic_form.js
document.addEventListener('DOMContentLoaded', function() {
    const restaurantSelect = document.getElementById('id_restaurant');
    const menuItemsContainer = document.getElementById('menu-items-container');
    
    restaurantSelect.addEventListener('change', function() {
        const restaurantId = this.value;
        
        if (restaurantId) {
            fetch(`/get-menu-items/?restaurant_id=${restaurantId}`)
                .then(response => response.json())
                .then(data => {
                    // Clear existing menu items
                    menuItemsContainer.innerHTML = '';
                    
                    // Add new menu items
                    data.menu_items.forEach(item => {
                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox';
                        checkbox.name = 'menu_items';
                        checkbox.value = item.id;
                        checkbox.id = `menu_item_${item.id}`;
                        
                        const label = document.createElement('label');
                        label.htmlFor = checkbox.id;
                        label.textContent = `${item.name} - $${item.price}`;
                        
                        const div = document.createElement('div');
                        div.appendChild(checkbox);
                        div.appendChild(label);
                        
                        menuItemsContainer.appendChild(div);
                    });
                });
        }
    });
});
```

**Syntax Explanation:**
- `kwargs.pop('restaurant_id', None)`: Removes and returns the restaurant_id from kwargs, defaulting to None if not found
- `super().__init__(*args, **kwargs)`: Calls the parent class constructor with remaining arguments
- `ModelMultipleChoiceField`: Creates a field that allows multiple selections from a model queryset
- `CheckboxSelectMultiple`: Widget that renders multiple checkboxes for selection

---

## 2. Formsets and Inline Formsets

### The Kitchen Brigade System

Imagine formsets as coordinating multiple chefs working on different parts of the same meal. Each chef (form) handles their specialty, but they all work together to create one cohesive dining experience.

### Code Example: Order with Multiple Items

```python
# forms.py
from django.forms import modelformset_factory, inlineformset_factory
from .models import Order, OrderItem, Customer

class OrderForm(forms.ModelForm):
    class Meta:
        model = Order
        fields = ['customer', 'table_number', 'special_instructions']
        widgets = {
            'special_instructions': forms.Textarea(attrs={'rows': 3}),
        }

class OrderItemForm(forms.ModelForm):
    class Meta:
        model = OrderItem
        fields = ['menu_item', 'quantity', 'special_requests']
        widgets = {
            'special_requests': forms.TextInput(attrs={'placeholder': 'No onions, extra cheese, etc.'})
        }

# Create formset for multiple order items
OrderItemFormSet = inlineformset_factory(
    Order, 
    OrderItem,
    form=OrderItemForm,
    extra=3,  # Number of empty forms to display
    can_delete=True,  # Allow deletion of items
    min_num=1,  # Minimum required forms
    validate_min=True
)

# Alternative: Regular formset (not tied to a parent model)
OrderItemRegularFormSet = modelformset_factory(
    OrderItem,
    form=OrderItemForm,
    extra=2,
    can_delete=True
)

# views.py
from django.shortcuts import render, redirect
from django.contrib import messages
from django.db import transaction

def create_order_view(request):
    if request.method == 'POST':
        order_form = OrderForm(request.POST)
        formset = OrderItemFormSet(request.POST)
        
        if order_form.is_valid() and formset.is_valid():
            with transaction.atomic():  # Ensure data consistency
                # Save the main order
                order = order_form.save()
                
                # Save all order items
                instances = formset.save(commit=False)
                for instance in instances:
                    instance.order = order
                    instance.save()
                
                # Handle deletions
                for obj in formset.deleted_objects:
                    obj.delete()
                
                messages.success(request, 'Order created successfully!')
                return redirect('order_detail', order_id=order.id)
    else:
        order_form = OrderForm()
        formset = OrderItemFormSet()
    
    return render(request, 'create_order.html', {
        'order_form': order_form,
        'formset': formset
    })

def edit_order_view(request, order_id):
    order = Order.objects.get(id=order_id)
    
    if request.method == 'POST':
        order_form = OrderForm(request.POST, instance=order)
        formset = OrderItemFormSet(request.POST, instance=order)
        
        if order_form.is_valid() and formset.is_valid():
            with transaction.atomic():
                order_form.save()
                formset.save()
                messages.success(request, 'Order updated successfully!')
                return redirect('order_detail', order_id=order.id)
    else:
        order_form = OrderForm(instance=order)
        formset = OrderItemFormSet(instance=order)
    
    return render(request, 'edit_order.html', {
        'order_form': order_form,
        'formset': formset,
        'order': order
    })
```

### Template for Formsets

```html
<!-- templates/create_order.html -->
<form method="post">
    {% csrf_token %}
    
    <!-- Main Order Form -->
    <div class="order-details">
        <h3>Order Details</h3>
        {{ order_form.as_p }}
    </div>
    
    <!-- Order Items Formset -->
    <div class="order-items">
        <h3>Order Items</h3>
        {{ formset.management_form }}
        
        <div id="formset-container">
            {% for form in formset %}
                <div class="formset-form">
                    {{ form.as_p }}
                    {% if form.instance.pk %}
                        {{ form.DELETE }}
                        <label for="{{ form.DELETE.id_for_label }}">Delete this item</label>
                    {% endif %}
                </div>
            {% endfor %}
        </div>
        
        <button type="button" id="add-form">Add Another Item</button>
    </div>
    
    <button type="submit">Save Order</button>
</form>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const addButton = document.getElementById('add-form');
    const formsetContainer = document.getElementById('formset-container');
    let formCount = {{ formset.total_form_count }};
    
    addButton.addEventListener('click', function() {
        // Clone the last form and update field names/ids
        const lastForm = formsetContainer.lastElementChild;
        const newForm = lastForm.cloneNode(true);
        
        // Update form index in field names and IDs
        newForm.innerHTML = newForm.innerHTML.replace(
            /orderitem_set-\d+-/g, 
            `orderitem_set-${formCount}-`
        );
        
        // Clear values in new form
        newForm.querySelectorAll('input, select, textarea').forEach(field => {
            if (field.type !== 'hidden') {
                field.value = '';
            }
        });
        
        formsetContainer.appendChild(newForm);
        formCount++;
        
        // Update management form
        document.getElementById('id_orderitem_set-TOTAL_FORMS').value = formCount;
    });
});
</script>
```

**Syntax Explanation:**
- `inlineformset_factory()`: Creates a formset for models with a foreign key relationship
- `modelformset_factory()`: Creates a formset for a single model without parent-child relationship
- `commit=False`: Saves model instances without committing to database, allowing modifications
- `transaction.atomic()`: Ensures all database operations succeed or fail together
- `formset.management_form`: Hidden fields that track formset state

---

## 3. Custom Form Widgets

### The Specialized Kitchen Tools

Think of custom widgets as specialized kitchen tools - just like a mandoline slicer creates perfect, uniform cuts that enhance both presentation and functionality, custom widgets create better user interfaces that improve the dining (user) experience.

### Code Example: Custom Rating Widget and Date Picker

```python
# widgets.py
from django import forms
from django.forms.widgets import Widget
from django.html import format_html
from django.utils.safestring import mark_safe

class StarRatingWidget(forms.Widget):
    """Custom widget for star ratings like restaurant reviews"""
    
    def __init__(self, max_stars=5, attrs=None):
        self.max_stars = max_stars
        super().__init__(attrs)
    
    def render(self, name, value, attrs=None, renderer=None):
        if value is None:
            value = 0
        
        html = '<div class="star-rating" data-rating="{}">\n'.format(value)
        
        for i in range(1, self.max_stars + 1):
            checked = 'checked' if i <= int(value) else ''
            html += format_html(
                '<input type="radio" name="{}" value="{}" id="star{}-{}" {} />\n'
                '<label for="star{}-{}" class="star">★</label>\n',
                name, i, name, i, checked, name, i
            )
        
        html += '</div>\n'
        html += '''
        <style>
            .star-rating { display: inline-block; }
            .star-rating input[type="radio"] { display: none; }
            .star-rating label.star { 
                color: #ddd; 
                cursor: pointer; 
                font-size: 24px;
                transition: color 0.2s;
            }
            .star-rating label.star:hover,
            .star-rating label.star:hover ~ label.star,
            .star-rating input[type="radio"]:checked ~ label.star { 
                color: #ffc107; 
            }
        </style>
        '''
        
        return mark_safe(html)

class TimeSlotWidget(forms.Select):
    """Custom widget for restaurant reservation time slots"""
    
    def __init__(self, attrs=None):
        # Generate time slots (e.g., 5:00 PM, 5:30 PM, etc.)
        choices = []
        for hour in range(17, 23):  # 5 PM to 10 PM
            for minute in [0, 30]:
                time_str = f"{hour:02d}:{minute:02d}"
                display = f"{hour if hour <= 12 else hour-12}:{minute:02d} {'AM' if hour < 12 else 'PM'}"
                choices.append((time_str, display))
        
        super().__init__(attrs, choices=choices)

class IngredientCheckboxWidget(forms.CheckboxSelectMultiple):
    """Custom widget for selecting pizza toppings with images"""
    
    def __init__(self, attrs=None):
        super().__init__(attrs)
        self.attrs.update({'class': 'ingredient-selector'})
    
    def create_option(self, name, value, label, selected, index, subindex=None, attrs=None):
        option = super().create_option(name, value, label, selected, index, subindex, attrs)
        
        # Add image path based on ingredient name
        ingredient_images = {
            'pepperoni': '/static/images/pepperoni.png',
            'mushrooms': '/static/images/mushrooms.png',
            'cheese': '/static/images/cheese.png',
        }
        
        if label.lower() in ingredient_images:
            option['attrs']['data-image'] = ingredient_images[label.lower()]
        
        return option

# forms.py using custom widgets
class RestaurantReviewForm(forms.ModelForm):
    class Meta:
        model = Review
        fields = ['restaurant', 'rating', 'comment', 'visit_date']
        widgets = {
            'rating': StarRatingWidget(max_stars=5),
            'comment': forms.Textarea(attrs={
                'rows': 4, 
                'placeholder': 'Share your dining experience...'
            }),
            'visit_date': forms.DateInput(attrs={
                'type': 'date',
                'class': 'form-control'
            })
        }

class ReservationForm(forms.Form):
    party_size = forms.IntegerField(
        min_value=1, 
        max_value=12,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    preferred_time = forms.CharField(
        widget=TimeSlotWidget(attrs={'class': 'form-control'})
    )
    special_requests = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'rows': 3,
            'placeholder': 'Birthday celebration, wheelchair access, etc.'
        })
    )

class PizzaOrderForm(forms.Form):
    TOPPING_CHOICES = [
        ('pepperoni', 'Pepperoni'),
        ('mushrooms', 'Mushrooms'),
        ('cheese', 'Extra Cheese'),
        ('olives', 'Olives'),
        ('peppers', 'Bell Peppers'),
    ]
    
    size = forms.ChoiceField(
        choices=[('small', 'Small'), ('medium', 'Medium'), ('large', 'Large')],
        widget=forms.RadioSelect(attrs={'class': 'size-selector'})
    )
    toppings = forms.MultipleChoiceField(
        choices=TOPPING_CHOICES,
        widget=IngredientCheckboxWidget(),
        required=False
    )
```

### Enhanced Template with Custom Widgets

```html
<!-- templates/pizza_order.html -->
<form method="post" class="pizza-order-form">
    {% csrf_token %}
    
    <div class="form-group">
        <label>Pizza Size:</label>
        {{ form.size }}
    </div>
    
    <div class="form-group">
        <label>Toppings:</label>
        {{ form.toppings }}
    </div>
    
    <button type="submit" class="btn btn-primary">Order Pizza</button>
</form>

<style>
.ingredient-selector {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 10px;
    margin-top: 10px;
}

.ingredient-selector label {
    display: block;
    padding: 10px;
    border: 2px solid #ddd;
    border-radius: 8px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
}

.ingredient-selector input[type="checkbox"] {
    display: none;
}

.ingredient-selector input[type="checkbox"]:checked + label {
    border-color: #007bff;
    background-color: #e3f2fd;
}

.ingredient-selector label:hover {
    border-color: #007bff;
}
</style>
```

**Syntax Explanation:**
- `mark_safe()`: Tells Django that the HTML string is safe and shouldn't be escaped
- `format_html()`: Safely formats HTML strings with variable substitution
- `super().__init__()`: Calls parent class constructor to inherit base functionality
- `create_option()`: Method to customize how individual options in select widgets are rendered

---

## 4. Form Wizards

### The Multi-Course Meal Experience

Form wizards are like serving a multi-course meal - each course (step) builds upon the previous one, creating a complete dining experience. The chef doesn't serve everything at once but guides diners through a carefully orchestrated sequence.

### Code Example: Multi-Step Restaurant Registration

```python
# forms.py
from django import forms
from django.contrib.formtools.wizard.views import SessionWizardView
from django.core.files.storage import FileSystemStorage
import os

# Step 1: Basic Restaurant Information
class RestaurantBasicForm(forms.Form):
    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    cuisine_type = forms.ChoiceField(
        choices=[
            ('italian', 'Italian'),
            ('asian', 'Asian'),
            ('american', 'American'),
            ('mexican', 'Mexican'),
            ('french', 'French'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 4, 'class': 'form-control'})
    )

# Step 2: Location and Contact
class RestaurantLocationForm(forms.Form):
    address = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    city = forms.CharField(
        max_length=50,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    phone = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    website = forms.URLField(
        required=False,
        widget=forms.URLInput(attrs={'class': 'form-control'})
    )

# Step 3: Operating Hours
class RestaurantHoursForm(forms.Form):
    DAYS = [
        ('monday', 'Monday'),
        ('tuesday', 'Tuesday'),
        ('wednesday', 'Wednesday'),
        ('thursday', 'Thursday'),
        ('friday', 'Friday'),
        ('saturday', 'Saturday'),
        ('sunday', 'Sunday'),
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Dynamically create fields for each day
        for day_code, day_name in self.DAYS:
            self.fields[f'{day_code}_open'] = forms.TimeField(
                required=False,
                widget=forms.TimeInput(attrs={'type': 'time', 'class': 'form-control'})
            )
            self.fields[f'{day_code}_close'] = forms.TimeField(
                required=False,
                widget=forms.TimeInput(attrs={'type': 'time', 'class': 'form-control'})
            )
            self.fields[f'{day_code}_closed'] = forms.BooleanField(
                required=False,
                label=f'Closed on {day_name}'
            )

# Step 4: Media and Final Details
class RestaurantMediaForm(forms.Form):
    logo = forms.ImageField(
        required=False,
        widget=forms.ClearableFileInput(attrs={'class': 'form-control'})
    )
    photos = forms.FileField(
        required=False,
        widget=forms.ClearableFileInput(attrs={'multiple': True, 'class': 'form-control'})
    )
    menu_pdf = forms.FileField(
        required=False,
        widget=forms.ClearableFileInput(attrs={'class': 'form-control'})
    )
    average_price_range = forms.ChoiceField(
        choices=[
            ('$', 'Budget ($)'),
            ('$$', 'Moderate ($$)'),
            ('$$$', 'Upscale ($$$)'),
            ('$$$$', 'Fine Dining ($$$$)'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )

# views.py
from django.shortcuts import render, redirect
from django.contrib.formtools.wizard.views import SessionWizardView
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import Restaurant, OperatingHours

class RestaurantRegistrationWizard(SessionWizardView):
    # Define the forms for each step
    form_list = [
        ('basic', RestaurantBasicForm),
        ('location', RestaurantLocationForm),
        ('hours', RestaurantHoursForm),
        ('media', RestaurantMediaForm),
    ]
    
    # Templates for each step
    template_name = 'restaurant_registration_wizard.html'
    
    # File storage for uploaded files
    file_storage = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'temp'))
    
    def get_template_names(self):
        """Return different templates for different steps if needed"""
        return [f'wizard_step_{self.steps.current}.html', self.template_name]
    
    def get_context_data(self, form, **kwargs):
        context = super().get_context_data(form=form, **kwargs)
        context.update({
            'wizard_steps': self.steps,
            'step_titles': {
                'basic': 'Restaurant Information',
                'location': 'Location & Contact',
                'hours': 'Operating Hours',
                'media': 'Media & Pricing'
            }
        })
        return context
    
    def get_form_initial(self, step):
        """Pre-populate form data if needed"""
        initial = super().get_form_initial(step)
        
        if step == 'location':
            # Pre-populate with data from previous step if available
            basic_data = self.get_cleaned_data_for_step('basic')
            if basic_data:
                # Could pre-populate based on cuisine type, etc.
                pass
        
        return initial
    
    def process_step(self, form):
        """Process each step - useful for validation across steps"""
        step_data = self.get_form_step_data(form)
        
        if self.steps.current == 'hours':
            # Custom validation for operating hours
            cleaned_data = form.cleaned_data
            for day_code, day_name in RestaurantHoursForm.DAYS:
                is_closed = cleaned_data.get(f'{day_code}_closed', False)
                open_time = cleaned_data.get(f'{day_code}_open')
                close_time = cleaned_data.get(f'{day_code}_close')
                
                if not is_closed and (not open_time or not close_time):
                    form.add_error(
                        f'{day_code}_open',
                        f'Please provide opening hours for {day_name} or mark as closed'
                    )
        
        return step_data
    
    def done(self, form_list, **kwargs):
        """Process all form data when wizard is complete"""
        # Collect all cleaned data
        all_data = {}
        for form in form_list:
            all_data.update(form.cleaned_data)
        
        # Create restaurant instance
        restaurant = Restaurant.objects.create(
            name=all_data['name'],
            cuisine_type=all_data['cuisine_type'],
            description=all_data['description'],
            address=all_data['address'],
            city=all_data['city'],
            phone=all_data['phone'],
            email=all_data['email'],
            website=all_data.get('website', ''),
            logo=all_data.get('logo'),
            menu_pdf=all_data.get('menu_pdf'),
            average_price_range=all_data['average_price_range'],
        )
        
        # Create operating hours
        for day_code, day_name in RestaurantHoursForm.DAYS:
            is_closed = all_data.get(f'{day_code}_closed', False)
            if not is_closed:
                OperatingHours.objects.create(
                    restaurant=restaurant,
                    day_of_week=day_code,
                    opening_time=all_data[f'{day_code}_open'],
                    closing_time=all_data[f'{day_code}_close']
                )
        
        return redirect('restaurant_success', restaurant_id=restaurant.id)

# urls.py
from django.urls import path
from .views import RestaurantRegistrationWizard

urlpatterns = [
    path('register-restaurant/', 
         RestaurantRegistrationWizard.as_view(), 
         name='restaurant_registration'),
]
```

### Wizard Template

```html
<!-- templates/restaurant_registration_wizard.html -->
<div class="wizard-container">
    <div class="wizard-header">
        <h2>Restaurant Registration</h2>
        
        <!-- Progress Bar -->
        <div class="progress-bar">
            {% for step, title in step_titles.items %}
                <div class="progress-step {% if step == wizard.steps.current %}active{% elif step in wizard.steps.prev %}completed{% endif %}">
                    <div class="step-number">{{ forloop.counter }}</div>
                    <div class="step-title">{{ title }}</div>
                </div>
            {% endfor %}
        </div>
    </div>
    
    <div class="wizard-content">
        <form method="post" enctype="multipart/form-data">
            {% csrf_token %}
            {{ wizard.management_form }}
            
            <div class="form-step">
                <h3>{{ step_titles|get_item:wizard.steps.current }}</h3>
                
                {% if wizard.form.errors %}
                    <div class="alert alert-danger">
                        Please correct the errors below.
                    </div>
                {% endif %}
                
                {{ wizard.form.as_p }}
            </div>
            
            <div class="wizard-buttons">
                {% if wizard.steps.prev %}
                    <button type="submit" name="wizard_goto_step" value="{{ wizard.steps.prev }}" class="btn btn-secondary">
                        Previous
                    </button>
                {% endif %}
                
                <button type="submit" class="btn btn-primary">
                    {% if wizard.steps.last %}
                        Complete Registration
                    {% else %}
                        Next Step
                    {% endif %}
                </button>
            </div>
        </form>
    </div>
</div>

<style>
.progress-bar {
    display: flex;
    justify-content: space-between;
    margin: 20px 0;
}

.progress-step {
    flex: 1;
    text-align: center;
    padding: 10px;
    position: relative;
}

.progress-step.active .step-number {
    background-color: #007bff;
    color: white;
}

.progress-step.completed .step-number {
    background-color: #28a745;
    color: white;
}

.step-number {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background-color: #ddd;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 5px;
}

.wizard-buttons {
    margin-top: 20px;
    display: flex;
    justify-content: space-between;
}
</style>
```

**Syntax Explanation:**
- `SessionWizardView`: Base class for multi-step forms using session storage
- `form_list`: Ordered list of forms for each step
- `get_cleaned_data_for_step()`: Retrieves validated data from a specific step
- `get_form_step_data()`: Gets raw form data for the current step
- `wizard.management_form`: Hidden fields that track wizard state
- `wizard.steps.current`: Current step identifier
- `wizard.steps.prev/next`: Navigation between steps

---
# Day 52: Advanced Forms & Formsets - Multi-Step Order Form Project

## Learning Objective
By the end of this project, you will be able to create a sophisticated multi-step order form that seamlessly guides users through a complete ordering process, combining dynamic forms, formsets, custom widgets, and form wizards to build a professional e-commerce checkout experience.

---

##Project: **Build**: Multi-Step Order Form

Imagine that you're the head chef of a premium restaurant, and you want to create an online ordering system that guides customers through selecting their meal, customizing their order, and completing their purchase. Just like how a chef carefully orchestrates each course of a meal, we'll orchestrate each step of our order form to create a smooth, intuitive experience.

Think of this multi-step form as preparing a gourmet meal - each step builds upon the previous one, and only when all steps are perfectly executed do we get our final masterpiece: a complete order ready for processing.

### Project Structure Overview

Our multi-step order form will consist of:
1. **Menu Selection** (Appetizers, Mains, Desserts)
2. **Customization** (Special instructions, dietary requirements)
3. **Customer Information** (Contact details, delivery preferences)
4. **Payment & Confirmation** (Order summary, payment processing)

### Step 1: Project Setup and Models

First, let's create our restaurant models that will serve as the foundation of our ordering system:

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

class Category(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    
    class Meta:
        verbose_name_plural = "Categories"
    
    def __str__(self):
        return self.name

class MenuItem(models.Model):
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    image = models.ImageField(upload_to='menu_items/', blank=True, null=True)
    is_available = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.name} - ${self.price}"

class Order(models.Model):
    DELIVERY_CHOICES = [
        ('pickup', 'Pickup'),
        ('delivery', 'Delivery'),
        ('dine_in', 'Dine In'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('preparing', 'Preparing'),
        ('ready', 'Ready'),
        ('completed', 'Completed'),
    ]
    
    customer_name = models.CharField(max_length=200)
    customer_email = models.EmailField()
    customer_phone = models.CharField(max_length=20)
    delivery_method = models.CharField(max_length=20, choices=DELIVERY_CHOICES)
    delivery_address = models.TextField(blank=True)
    special_instructions = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Order #{self.id} - {self.customer_name}"

class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1, validators=[MinValueValidator(1)])
    special_notes = models.CharField(max_length=500, blank=True)
    
    def get_total_price(self):
        return self.quantity * self.menu_item.price
    
    def __str__(self):
        return f"{self.quantity}x {self.menu_item.name}"
```

### Step 2: Form Wizard Setup

Now, let's create our form wizard. Think of this as the chef's recipe book - each step has its own specific ingredients (forms) and instructions:

```python
# forms.py
from django import forms
from django.forms import formset_factory, BaseFormSet
from .models import MenuItem, Category, Order, OrderItem

class MenuSelectionForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        categories = Category.objects.all()
        
        for category in categories:
            items = MenuItem.objects.filter(category=category, is_available=True)
            for item in items:
                field_name = f'item_{item.id}'
                self.fields[field_name] = forms.IntegerField(
                    min_value=0,
                    max_value=10,
                    initial=0,
                    required=False,
                    widget=forms.NumberInput(attrs={
                        'class': 'form-control quantity-input',
                        'data-price': str(item.price),
                        'data-name': item.name,
                    }),
                    label=f"{item.name} - ${item.price}"
                )

class CustomizationForm(forms.Form):
    dietary_requirements = forms.MultipleChoiceField(
        choices=[
            ('vegetarian', 'Vegetarian'),
            ('vegan', 'Vegan'),
            ('gluten_free', 'Gluten Free'),
            ('dairy_free', 'Dairy Free'),
            ('nut_free', 'Nut Free'),
        ],
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}),
        required=False,
        label="Dietary Requirements"
    )
    
    special_instructions = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 4,
            'class': 'form-control',
            'placeholder': 'Any special instructions for the chef? (e.g., "Extra spicy", "No onions", etc.)'
        }),
        required=False,
        label="Special Instructions"
    )
    
    preferred_cooking_level = forms.ChoiceField(
        choices=[
            ('', 'No preference'),
            ('rare', 'Rare'),
            ('medium_rare', 'Medium Rare'),
            ('medium', 'Medium'),
            ('medium_well', 'Medium Well'),
            ('well_done', 'Well Done'),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label="Preferred Cooking Level (for meat dishes)"
    )

class CustomerInfoForm(forms.Form):
    customer_name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Your full name'}),
        label="Full Name"
    )
    
    customer_email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'your@email.com'}),
        label="Email Address"
    )
    
    customer_phone = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '+1 (555) 123-4567'}),
        label="Phone Number"
    )
    
    delivery_method = forms.ChoiceField(
        choices=Order.DELIVERY_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label="How would you like to receive your order?"
    )
    
    delivery_address = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 3,
            'class': 'form-control',
            'placeholder': 'Street address, city, state, zip code'
        }),
        required=False,
        label="Delivery Address"
    )

class ConfirmationForm(forms.Form):
    terms_accepted = forms.BooleanField(
        required=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        label="I accept the terms and conditions"
    )
    
    marketing_emails = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        label="I'd like to receive promotional emails about new menu items and special offers"
    )
```

### Step 3: The Form Wizard View

Here's where the magic happens - our master chef (the view) coordinates all the steps:

```python
# views.py
from django.shortcuts import render, redirect
from django.contrib.formtools.wizard.views import SessionWizardView
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib import messages
from django.urls import reverse_lazy
from .forms import MenuSelectionForm, CustomizationForm, CustomerInfoForm, ConfirmationForm
from .models import MenuItem, Order, OrderItem
from decimal import Decimal
import json

class OrderWizardView(SessionWizardView):
    template_name = 'orders/order_wizard.html'
    form_list = [
        ('menu', MenuSelectionForm),
        ('customization', CustomizationForm),
        ('customer', CustomerInfoForm),
        ('confirmation', ConfirmationForm),
    ]
    
    def get_template_names(self):
        return [f'orders/order_wizard_step_{self.steps.current}.html']
    
    def get_context_data(self, form, **kwargs):
        context = super().get_context_data(form=form, **kwargs)
        context['categories'] = Category.objects.all()
        context['menu_items'] = MenuItem.objects.filter(is_available=True)
        
        # Add step-specific context
        if self.steps.current == 'menu':
            context['step_title'] = 'Select Your Dishes'
            context['step_description'] = 'Choose from our delicious menu items'
            
        elif self.steps.current == 'customization':
            context['step_title'] = 'Customize Your Order'
            context['step_description'] = 'Tell us about your preferences'
            context['selected_items'] = self.get_selected_items()
            
        elif self.steps.current == 'customer':
            context['step_title'] = 'Your Information'
            context['step_description'] = 'We need your details to complete the order'
            
        elif self.steps.current == 'confirmation':
            context['step_title'] = 'Confirm Your Order'
            context['step_description'] = 'Review everything before we start cooking'
            context['order_summary'] = self.get_order_summary()
            
        return context
    
    def get_selected_items(self):
        """Get items selected in the menu step"""
        menu_data = self.get_cleaned_data_for_step('menu')
        selected_items = []
        
        if menu_data:
            for field_name, quantity in menu_data.items():
                if field_name.startswith('item_') and quantity > 0:
                    item_id = int(field_name.split('_')[1])
                    try:
                        menu_item = MenuItem.objects.get(id=item_id)
                        selected_items.append({
                            'item': menu_item,
                            'quantity': quantity,
                            'total_price': menu_item.price * quantity
                        })
                    except MenuItem.DoesNotExist:
                        continue
        
        return selected_items
    
    def get_order_summary(self):
        """Generate complete order summary"""
        selected_items = self.get_selected_items()
        customization_data = self.get_cleaned_data_for_step('customization')
        customer_data = self.get_cleaned_data_for_step('customer')
        
        total_amount = sum(item['total_price'] for item in selected_items)
        
        return {
            'items': selected_items,
            'customization': customization_data,
            'customer': customer_data,
            'total_amount': total_amount,
            'tax_amount': total_amount * Decimal('0.08'),  # 8% tax
            'final_total': total_amount * Decimal('1.08'),
        }
    
    def done(self, form_list, **kwargs):
        """Process the completed order"""
        # Get all cleaned data
        menu_data = self.get_cleaned_data_for_step('menu')
        customization_data = self.get_cleaned_data_for_step('customization')
        customer_data = self.get_cleaned_data_for_step('customer')
        confirmation_data = self.get_cleaned_data_for_step('confirmation')
        
        # Create the order
        order = Order.objects.create(
            customer_name=customer_data['customer_name'],
            customer_email=customer_data['customer_email'],
            customer_phone=customer_data['customer_phone'],
            delivery_method=customer_data['delivery_method'],
            delivery_address=customer_data.get('delivery_address', ''),
            special_instructions=customization_data.get('special_instructions', ''),
        )
        
        # Create order items
        total_amount = Decimal('0')
        for field_name, quantity in menu_data.items():
            if field_name.startswith('item_') and quantity > 0:
                item_id = int(field_name.split('_')[1])
                try:
                    menu_item = MenuItem.objects.get(id=item_id)
                    order_item = OrderItem.objects.create(
                        order=order,
                        menu_item=menu_item,
                        quantity=quantity,
                        special_notes=f"Dietary: {', '.join(customization_data.get('dietary_requirements', []))}"
                    )
                    total_amount += order_item.get_total_price()
                except MenuItem.DoesNotExist:
                    continue
        
        # Update order total (including tax)
        order.total_amount = total_amount * Decimal('1.08')  # 8% tax
        order.save()
        
        # Add success message
        messages.success(
            self.request,
            f'Thank you {order.customer_name}! Your order #{order.id} has been received and is being prepared.'
        )
        
        # Redirect to success page
        return redirect('order_success', order_id=order.id)

def order_success(request, order_id):
    """Display order success page"""
    try:
        order = Order.objects.get(id=order_id)
        return render(request, 'orders/order_success.html', {'order': order})
    except Order.DoesNotExist:
        messages.error(request, 'Order not found.')
        return redirect('order_wizard')

# AJAX view for dynamic price calculation
@csrf_exempt
def calculate_order_total(request):
    """Calculate order total via AJAX"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            total = Decimal('0')
            
            for item_id, quantity in data.items():
                if quantity > 0:
                    try:
                        menu_item = MenuItem.objects.get(id=int(item_id))
                        total += menu_item.price * quantity
                    except MenuItem.DoesNotExist:
                        continue
            
            return JsonResponse({
                'subtotal': str(total),
                'tax': str(total * Decimal('0.08')),
                'total': str(total * Decimal('1.08'))
            })
        except (ValueError, TypeError):
            return JsonResponse({'error': 'Invalid data'}, status=400)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
```

### Step 4: Templates

Let's create our templates. First, the base wizard template:

```html
<!-- templates/orders/order_wizard.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ step_title }} - Gourmet Kitchen</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .order-progress {
            margin-bottom: 2rem;
        }
        .progress-step {
            position: relative;
            flex: 1;
            text-align: center;
        }
        .progress-step.active .step-circle {
            background-color: #28a745;
            color: white;
        }
        .progress-step.completed .step-circle {
            background-color: #007bff;
            color: white;
        }
        .step-circle {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: #e9ecef;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 0.5rem;
            font-weight: bold;
        }
        .quantity-input {
            width: 80px;
        }
        .menu-item-card {
            transition: transform 0.2s;
        }
        .menu-item-card:hover {
            transform: translateY(-2px);
        }
        .order-total {
            position: sticky;
            top: 20px;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <div class="row">
            <div class="col-md-12">
                <h1 class="text-center mb-4">
                    <i class="fas fa-utensils"></i> Gourmet Kitchen Online Order
                </h1>
                
                <!-- Progress indicator -->
                <div class="order-progress">
                    <div class="d-flex justify-content-between align-items-center">
                        <div class="progress-step {% if wizard.steps.current == 'menu' %}active{% endif %}">
                            <div class="step-circle">1</div>
                            <small>Menu</small>
                        </div>
                        <div class="progress-step {% if wizard.steps.current == 'customization' %}active{% endif %}">
                            <div class="step-circle">2</div>
                            <small>Customize</small>
                        </div>
                        <div class="progress-step {% if wizard.steps.current == 'customer' %}active{% endif %}">
                            <div class="step-circle">3</div>
                            <small>Info</small>
                        </div>
                        <div class="progress-step {% if wizard.steps.current == 'confirmation' %}active{% endif %}">
                            <div class="step-circle">4</div>
                            <small>Confirm</small>
                        </div>
                    </div>
                    <div class="progress mt-3">
                        <div class="progress-bar" role="progressbar" 
                             style="width: {% if wizard.steps.current == 'menu' %}25%{% elif wizard.steps.current == 'customization' %}50%{% elif wizard.steps.current == 'customer' %}75%{% else %}100%{% endif %}">
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header">
                                <h3>{{ step_title }}</h3>
                                <p class="text-muted mb-0">{{ step_description }}</p>
                            </div>
                            <div class="card-body">
                                {% block step_content %}
                                <form method="post">
                                    {% csrf_token %}
                                    {{ wizard.management_form }}
                                    {{ form.as_p }}
                                    
                                    <div class="d-flex justify-content-between mt-4">
                                        {% if wizard.steps.prev %}
                                            <button type="submit" name="wizard_goto_step" 
                                                    value="{{ wizard.steps.prev }}" 
                                                    class="btn btn-secondary">
                                                <i class="fas fa-arrow-left"></i> Previous
                                            </button>
                                        {% endif %}
                                        
                                        <button type="submit" class="btn btn-primary">
                                            {% if wizard.steps.last %}
                                                Complete Order <i class="fas fa-check"></i>
                                            {% else %}
                                                Next <i class="fas fa-arrow-right"></i>
                                            {% endif %}
                                        </button>
                                    </div>
                                </form>
                                {% endblock %}
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-4">
                        <div class="order-total">
                            <div class="card">
                                <div class="card-header">
                                    <h5><i class="fas fa-receipt"></i> Order Summary</h5>
                                </div>
                                <div class="card-body">
                                    <div id="order-summary">
                                        <p class="text-muted">Select items to see your order total</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Dynamic order total calculation
        document.addEventListener('DOMContentLoaded', function() {
            const quantityInputs = document.querySelectorAll('.quantity-input');
            
            quantityInputs.forEach(input => {
                input.addEventListener('change', updateOrderTotal);
            });
            
            function updateOrderTotal() {
                const orderData = {};
                quantityInputs.forEach(input => {
                    const itemId = input.name.split('_')[1];
                    const quantity = parseInt(input.value) || 0;
                    if (quantity > 0) {
                        orderData[itemId] = quantity;
                    }
                });
                
                fetch('/calculate-order-total/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(orderData)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error('Error:', data.error);
                        return;
                    }
                    
                    const summaryDiv = document.getElementById('order-summary');
                    summaryDiv.innerHTML = `
                        <div class="d-flex justify-content-between">
                            <span>Subtotal:</span>
                            <span>$${data.subtotal}</span>
                        </div>
                        <div class="d-flex justify-content-between">
                            <span>Tax (8%):</span>
                            <span>$${data.tax}</span>
                        </div>
                        <hr>
                        <div class="d-flex justify-content-between fw-bold">
                            <span>Total:</span>
                            <span>$${data.total}</span>
                        </div>
                    `;
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
        });
    </script>
</body>
</html>
```

### Step 5: URL Configuration

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('order/', views.OrderWizardView.as_view(), name='order_wizard'),
    path('order/success/<int:order_id>/', views.order_success, name='order_success'),
    path('calculate-order-total/', views.calculate_order_total, name='calculate_order_total'),
]
```

### Step 6: Settings Configuration

Add these to your Django settings:

```python
# settings.py
INSTALLED_APPS = [
    # ... other apps
    'formtools',
    'orders',  # your app name
]

# Session configuration for wizard
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_AGE = 1800  # 30 minutes

# File upload settings
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

## Code Syntax Explanation

### Key Django Concepts Used:

1. **SessionWizardView**: Django's built-in class for creating multi-step forms that persist data between steps using sessions.

2. **FormSet Factory**: Creates multiple instances of the same form, perfect for handling multiple order items.

3. **Dynamic Form Fields**: We create form fields programmatically based on available menu items.

4. **AJAX Integration**: Real-time order total calculation without page refresh.

5. **Custom Widgets**: Enhanced form controls with CSS classes and data attributes.

### Python Syntax Breakdown:

- **List Comprehensions**: Used for efficient data processing
- **Dictionary Comprehensions**: For transforming form data
- **F-strings**: Modern Python string formatting
- **Try/Except Blocks**: Error handling for database operations
- **Decorators**: `@csrf_exempt` for AJAX views
- **Class-based Views**: Inheritance and method overriding

### JavaScript/AJAX Syntax:

- **Fetch API**: Modern way to make HTTP requests
- **Event Listeners**: Handling user interactions
- **DOM Manipulation**: Updating content dynamically
- **JSON Handling**: Data exchange between frontend and backend

This multi-step order form demonstrates professional-level Django development, combining multiple advanced concepts into a cohesive, user-friendly application. The project showcases real-world e-commerce functionality while maintaining clean, maintainable code structure.

## Key Learning Outcomes

After completing this project, you will have mastered:

- Building complex multi-step forms with Django
- Implementing dynamic form generation
- Creating smooth user experiences with AJAX
- Managing form state across multiple steps
- Integrating multiple Django advanced features into a cohesive application

This project represents the culmination of advanced Django form handling techniques, providing you with the skills to build sophisticated web applications that handle complex user interactions gracefully.

## Assignment: Restaurant Menu Builder

**Objective:** Create a dynamic form system for restaurant owners to build their menu with categories and items.

**Requirements:**
1. **Dynamic Category Form**: Create a form that allows adding/removing menu categories (Appetizers, Main Courses, Desserts, etc.)

2. **Formsets for Menu Items**: Use inline formsets to manage multiple menu items within each category

3. **Custom Widget**: Implement a custom widget for difficulty level (Easy, Medium, Hard to prepare) using visual indicators (like chef hats: 👨‍🍳, 👨‍🍳👨‍🍳, 👨‍🍳👨‍🍳👨‍🍳)

4. **Form Wizard**: Create a 3-step wizard:
   - Step 1: Restaurant basic info (name, cuisine type)
   - Step 2: Menu categories setup  
   - Step 3: Add menu items to categories

**Models to Create:**
```python
# models.py
class Restaurant(models.Model):
    name = models.CharField(max_length=100)
    cuisine_type = models.CharField(max_length=50)
    
class MenuCategory(models.Model):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    name = models.CharField(max_length=50)
    display_order = models.IntegerField(default=0)
    
class MenuItem(models.Model):
    DIFFICULTY_CHOICES = [
        (1, 'Easy'),
        (2, 'Medium'), 
        (3, 'Hard')
    ]
    
    category = models.ForeignKey(MenuCategory, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    description = models.TextField()
    price = models.DecimalField(max_digits=6, decimal_places=2)
    difficulty = models.IntegerField(choices=DIFFICULTY_CHOICES, default=1)
    is_vegetarian = models.BooleanField(default=False)
    is_available = models.BooleanField(default=True)
```

**Deliverables:**
- Complete Django forms with all required functionality
- Templates with JavaScript for dynamic form handling
- Views that process the wizard and formsets
- CSS styling for the custom difficulty widget

**Success Criteria:**
- Restaurant owners can dynamically add/remove menu categories
- Each category can have multiple menu items managed through formsets
- The difficulty widget displays chef hat icons instead of radio buttons
- The wizard smoothly guides users through the 3-step process
- All form validations work correctly
- Clean, professional styling throughout

**Bonus Points:**
- Add AJAX functionality to save form data between wizard steps
- Implement drag-and-drop reordering for menu categories
- Add image upload capability for menu items
- Create a preview mode showing how the menu will look to customers

This assignment combines all four concepts we've covered: dynamic forms adapt based on user input, formsets manage multiple related items, custom widgets enhance user experience, and form wizards break complex processes into manageable steps - just like a master chef orchestrating a complex kitchen operation!

---

## Summary

In this lesson, we've explored Django's advanced form features like a master chef learning sophisticated kitchen techniques:

**Dynamic Forms** act like adaptive menus that change based on available ingredients (user selections), allowing forms to respond intelligently to user input.

**Formsets** work like a coordinated kitchen brigade, managing multiple related forms simultaneously while maintaining data consistency and relationships.

**Custom Widgets** are specialized tools that enhance the user experience, much like how specialized kitchen equipment makes cooking more efficient and results more professional.

**Form Wizards** guide users through complex processes step-by-step, like serving a multi-course meal where each course builds upon the previous one.

These tools work together to create sophisticated, user-friendly web applications that can handle complex data entry scenarios while maintaining clean, maintainable code. Master these concepts, and you'll be able to build forms that are both powerful and elegant - the hallmark of a skilled Django developer.