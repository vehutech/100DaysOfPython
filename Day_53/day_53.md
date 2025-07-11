# Django Email Integration Course - Day 53

## Learning Objective
By the end of this lesson, you will be able to configure Django's email framework, create dynamic email templates, implement asynchronous email sending, and configure different email backends to handle various email delivery scenarios in your Django applications.

---

## Introduction

Imagine that you're running a bustling restaurant kitchen where communication is everything. Just as a head chef needs to coordinate with different stations - sending orders to the grill, updates to the pastry section, and notifications to the wait staff - your Django application needs to communicate with users through emails. Today, we'll transform you into a master chef of email communication, learning how to set up your "email kitchen," create beautiful "email recipes" (templates), and efficiently deliver these messages to your customers.

---

## Lesson 1: Django Email Framework

### The Foundation - Setting Up Your Email Kitchen

Just like a chef needs proper equipment and ingredients before cooking, Django needs proper email configuration before sending emails.

#### Basic Email Configuration

First, let's set up our email "kitchen" in `settings.py`:

```python
# settings.py - Your Email Kitchen Configuration

# SMTP Configuration (Like choosing your cooking method)
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'  # Your email server (like choosing gas vs electric stove)
EMAIL_PORT = 587  # Port number (like setting the right temperature)
EMAIL_USE_TLS = True  # Security protocol (like using proper food safety)
EMAIL_HOST_USER = 'your_email@gmail.com'  # Your email address
EMAIL_HOST_PASSWORD = 'your_app_password'  # Your email password

# Default "From" address (Your restaurant's signature)
DEFAULT_FROM_EMAIL = 'noreply@yourrestaurant.com'
```

#### Basic Email Sending

Now let's create our first "email recipe":

```python
# views.py - Your Email Preparation Area

from django.core.mail import send_mail
from django.shortcuts import render
from django.contrib import messages

def send_welcome_email(request):
    """
    Send a welcome email - like preparing a welcome appetizer
    """
    if request.method == 'POST':
        user_email = request.POST.get('email')
        user_name = request.POST.get('name')
        
        # Preparing the email ingredients
        subject = f'Welcome to our Restaurant, {user_name}!'
        message = f'''
        Dear {user_name},
        
        Welcome to our culinary family! We're excited to have you join us.
        
        Best regards,
        The Kitchen Team
        '''
        
        try:
            # Sending the email (like serving the dish)
            send_mail(
                subject=subject,
                message=message,
                from_email='chef@restaurant.com',
                recipient_list=[user_email],
                fail_silently=False,  # Like quality control - we want to know if something fails
            )
            messages.success(request, 'Welcome email sent successfully!')
        except Exception as e:
            messages.error(request, f'Failed to send email: {str(e)}')
    
    return render(request, 'email_form.html')
```

#### Multiple Email Recipients

Sometimes you need to send the same message to multiple people - like announcing today's special to all VIP customers:

```python
# bulk_email.py - Preparing for a Banquet

from django.core.mail import send_mass_mail

def send_daily_special_announcement():
    """
    Send daily special to multiple customers - like announcing today's menu
    """
    # List of VIP customers
    vip_customers = ['customer1@email.com', 'customer2@email.com', 'customer3@email.com']
    
    # Preparing the same message for everyone
    subject = "Today's Special: Grilled Salmon with Django Sauce"
    message = """
    Dear Valued Customer,
    
    Today's special is our signature Grilled Salmon with Django Sauce,
    prepared with the finest ingredients and served with love.
    
    Reserve your table now!
    
    Chef's Team
    """
    
    # Creating individual messages (like plating individual dishes)
    messages = []
    for customer_email in vip_customers:
        email_tuple = (
            subject,
            message,
            'chef@restaurant.com',
            [customer_email]
        )
        messages.append(email_tuple)
    
    # Send all emails at once (like serving a banquet)
    send_mass_mail(messages, fail_silently=False)
```

---

## Lesson 2: Email Templates

### Creating Your Recipe Book - HTML Email Templates

Just as a chef has recipe cards for consistent dish preparation, we need email templates for consistent, beautiful emails.

#### Creating HTML Email Templates

Create a new directory structure:
```
templates/
    emails/
        base_email.html
        welcome_email.html
        order_confirmation.html
```

**Base Email Template** (`templates/emails/base_email.html`):

```html
<!-- Base Email Template - Your Standard Kitchen Setup -->
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Restaurant Email{% endblock %}</title>
    <style>
        /* Email-safe CSS - like basic cooking techniques that work everywhere */
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .content {
            padding: 20px;
            background-color: #f9f9f9;
        }
        .footer {
            background-color: #34495e;
            color: white;
            padding: 15px;
            text-align: center;
            font-size: 14px;
        }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🍽️ Django Restaurant</h1>
        <p>{% block header_subtitle %}Serving Excellence Since 2024{% endblock %}</p>
    </div>
    
    <div class="content">
        {% block content %}
        <!-- Your email content goes here -->
        {% endblock %}
    </div>
    
    <div class="footer">
        <p>© 2024 Django Restaurant | 123 Framework Street | contact@djangorestaurant.com</p>
        <p>{% block footer_extra %}{% endblock %}</p>
    </div>
</body>
</html>
```

**Welcome Email Template** (`templates/emails/welcome_email.html`):

```html
<!-- Welcome Email - Like a Special Welcome Appetizer -->
{% extends 'emails/base_email.html' %}

{% block title %}Welcome to Django Restaurant{% endblock %}

{% block header_subtitle %}Welcome to Our Culinary Family!{% endblock %}

{% block content %}
    <h2>Hello {{ customer_name }}! 👋</h2>
    
    <p>We're absolutely delighted to welcome you to Django Restaurant! Just like how we carefully select each ingredient for our dishes, we're excited to have carefully selected you as part of our community.</p>
    
    <div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3>🎉 Welcome Bonus!</h3>
        <p>Use code <strong>WELCOME20</strong> for 20% off your first order!</p>
    </div>
    
    <p>Here's what you can expect from us:</p>
    <ul>
        <li>🍽️ Fresh, quality meals prepared with Django magic</li>
        <li>📧 Weekly updates on new menu items and specials</li>
        <li>🎁 Exclusive offers and member-only events</li>
        <li>👨‍🍳 Tips and recipes from our head chef</li>
    </ul>
    
    <p style="text-align: center; margin: 30px 0;">
        <a href="{{ restaurant_url }}" class="btn">Explore Our Menu</a>
    </p>
    
    <p>If you have any questions, just reply to this email - we're here to help!</p>
    
    <p>Happy dining!<br>
    <strong>The Django Restaurant Team</strong></p>
{% endblock %}

{% block footer_extra %}
    <p>Don't want to receive these emails? <a href="#" style="color: #bdc3c7;">Unsubscribe here</a></p>
{% endblock %}
```

#### Using Templates in Views

Now let's use our templates like a chef following a recipe:

```python
# views.py - Your Email Preparation Kitchen

from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.shortcuts import render, redirect
from django.contrib import messages

def send_welcome_email_with_template(request):
    """
    Send a templated welcome email - like preparing a signature dish
    """
    if request.method == 'POST':
        customer_name = request.POST.get('name')
        customer_email = request.POST.get('email')
        
        # Preparing the email context (like gathering ingredients)
        context = {
            'customer_name': customer_name,
            'restaurant_url': 'https://www.djangorestaurant.com',
            'welcome_code': 'WELCOME20',
        }
        
        # Render the HTML template (like following the recipe)
        html_content = render_to_string('emails/welcome_email.html', context)
        
        # Create plain text version (like having a simple version of the dish)
        text_content = strip_tags(html_content)
        
        # Prepare the email (like plating the dish)
        subject = f'Welcome to Django Restaurant, {customer_name}!'
        from_email = 'chef@djangorestaurant.com'
        to_email = [customer_email]
        
        # Create the email message
        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,  # Plain text version
            from_email=from_email,
            to=to_email
        )
        
        # Attach the HTML version (like adding garnish)
        email.attach_alternative(html_content, "text/html")
        
        try:
            # Send the email (like serving the dish)
            email.send()
            messages.success(request, f'Welcome email sent to {customer_name}!')
        except Exception as e:
            messages.error(request, f'Failed to send email: {str(e)}')
        
        return redirect('email_success')
    
    return render(request, 'email_form.html')

def order_confirmation_email(request, order_id):
    """
    Send order confirmation - like giving a receipt with style
    """
    # Get order details (like getting the order from the kitchen)
    order = get_object_or_404(Order, id=order_id)
    
    context = {
        'customer_name': order.customer.name,
        'order_number': order.order_number,
        'order_items': order.items.all(),
        'total_amount': order.total_amount,
        'estimated_delivery': order.estimated_delivery_time,
        'restaurant_url': 'https://www.djangorestaurant.com',
    }
    
    # Render email template
    html_content = render_to_string('emails/order_confirmation.html', context)
    text_content = strip_tags(html_content)
    
    # Send email
    subject = f'Order Confirmation #{order.order_number}'
    email = EmailMultiAlternatives(
        subject=subject,
        body=text_content,
        from_email='orders@djangorestaurant.com',
        to=[order.customer.email]
    )
    email.attach_alternative(html_content, "text/html")
    email.send()
```

---

## Lesson 3: Asynchronous Email Sending

### Working Smart, Not Hard - The Efficient Kitchen

Just as a smart chef doesn't make customers wait while bread bakes, we shouldn't make users wait while emails send. Let's implement asynchronous email sending.

#### Using Django's Built-in Threading

```python
# async_email.py - Your Efficient Kitchen Operations

import threading
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
import logging

logger = logging.getLogger(__name__)

def send_email_async(subject, template_name, context, recipient_list, from_email=None):
    """
    Send email asynchronously - like having a sous chef handle the plating
    while you start the next dish
    """
    def send_email():
        try:
            # Render the email template
            html_content = render_to_string(template_name, context)
            text_content = strip_tags(html_content)
            
            # Create and send email
            email = EmailMultiAlternatives(
                subject=subject,
                body=text_content,
                from_email=from_email or 'noreply@djangorestaurant.com',
                to=recipient_list
            )
            email.attach_alternative(html_content, "text/html")
            email.send()
            
            logger.info(f"Email sent successfully to {recipient_list}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
    
    # Start the email sending in a separate thread (like delegating to sous chef)
    email_thread = threading.Thread(target=send_email)
    email_thread.daemon = True  # Thread will die when main program exits
    email_thread.start()

# Usage in views
def quick_order_confirmation(request, order_id):
    """
    Send order confirmation without making user wait
    """
    order = get_object_or_404(Order, id=order_id)
    
    # Prepare email context
    context = {
        'customer_name': order.customer.name,
        'order_number': order.order_number,
        'order_items': order.items.all(),
        'total_amount': order.total_amount,
    }
    
    # Send email asynchronously (customer doesn't wait)
    send_email_async(
        subject=f'Order Confirmation #{order.order_number}',
        template_name='emails/order_confirmation.html',
        context=context,
        recipient_list=[order.customer.email]
    )
    
    # User gets immediate response while email sends in background
    messages.success(request, 'Order confirmed! Confirmation email is being sent.')
    return redirect('order_success')
```

#### Using Celery for Production (Advanced Kitchen Setup)

For production environments, use Celery - like having a dedicated pastry chef for complex desserts:

```python
# tasks.py - Your Specialized Kitchen Station

from celery import shared_task
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
import logging

logger = logging.getLogger(__name__)

@shared_task
def send_templated_email(subject, template_name, context, recipient_list, from_email=None):
    """
    Celery task for sending emails - like having a dedicated email chef
    """
    try:
        # Render templates
        html_content = render_to_string(template_name, context)
        text_content = strip_tags(html_content)
        
        # Create and send email
        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=from_email or 'noreply@djangorestaurant.com',
            to=recipient_list
        )
        email.attach_alternative(html_content, "text/html")
        email.send()
        
        logger.info(f"Email sent successfully to {recipient_list}")
        return f"Email sent to {recipient_list}"
        
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        raise

# Usage in views
def celery_order_confirmation(request, order_id):
    """
    Send order confirmation using Celery
    """
    order = get_object_or_404(Order, id=order_id)
    
    context = {
        'customer_name': order.customer.name,
        'order_number': order.order_number,
        'order_items': order.items.all(),
        'total_amount': order.total_amount,
    }
    
    # Queue the email task (like putting an order in the kitchen queue)
    send_templated_email.delay(
        subject=f'Order Confirmation #{order.order_number}',
        template_name='emails/order_confirmation.html',
        context=context,
        recipient_list=[order.customer.email]
    )
    
    messages.success(request, 'Order confirmed! Confirmation email is being sent.')
    return redirect('order_success')
```

---

## Lesson 4: Email Backends

### Choosing Your Cooking Method - Different Email Backends

Just as a chef might use different cooking methods (grilling, baking, steaming) for different dishes, Django offers different email backends for different scenarios.

#### Console Backend (For Development)

Perfect for testing - like tasting your dish before serving:

```python
# settings.py - Development Kitchen Setup

# For development - emails will be displayed in console
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
```

#### File Backend (For Testing)

Saves emails to files - like keeping recipe notes:

```python
# settings.py - Testing Kitchen Setup

EMAIL_BACKEND = 'django.core.mail.backends.filebased.EmailBackend'
EMAIL_FILE_PATH = '/tmp/app-messages'  # Directory to save emails
```

#### SMTP Backend (For Production)

Real email delivery - like serving to actual customers:

```python
# settings.py - Production Kitchen Setup

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'your_email@gmail.com'
EMAIL_HOST_PASSWORD = 'your_app_password'
```

#### Custom Backend Class

Create your own backend - like inventing a new cooking technique:

```python
# custom_backends.py - Your Signature Cooking Style

from django.core.mail.backends.base import BaseEmailBackend
from django.core.mail import EmailMessage
import logging

logger = logging.getLogger(__name__)

class LoggingEmailBackend(BaseEmailBackend):
    """
    Custom email backend that logs all emails - like keeping a kitchen journal
    """
    
    def send_messages(self, email_messages):
        """
        Send messages and log them - like documenting each dish served
        """
        if not email_messages:
            return 0
        
        sent_count = 0
        for message in email_messages:
            try:
                # Log the email details
                logger.info(f"Sending email to: {message.to}")
                logger.info(f"Subject: {message.subject}")
                logger.info(f"From: {message.from_email}")
                
                # Here you could add logic to actually send the email
                # For now, we'll just log it
                
                sent_count += 1
                logger.info(f"Email sent successfully!")
                
            except Exception as e:
                logger.error(f"Failed to send email: {str(e)}")
        
        return sent_count

# Usage in settings.py
EMAIL_BACKEND = 'myapp.custom_backends.LoggingEmailBackend'
```

#### Environment-Specific Backend Configuration

Smart configuration - like having different kitchen setups for different occasions:

```python
# settings.py - Adaptive Kitchen Setup

import os

# Choose email backend based on environment
if os.environ.get('DJANGO_ENV') == 'production':
    # Production - real email delivery
    EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
    EMAIL_HOST = os.environ.get('EMAIL_HOST')
    EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
    EMAIL_USE_TLS = True
    EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
    EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
    
elif os.environ.get('DJANGO_ENV') == 'testing':
    # Testing - save to files
    EMAIL_BACKEND = 'django.core.mail.backends.filebased.EmailBackend'
    EMAIL_FILE_PATH = '/tmp/test-emails'
    
else:
    # Development - show in console
    EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Common settings for all environments
DEFAULT_FROM_EMAIL = 'noreply@djangorestaurant.com'
```

---

## Code Syntax Explanation

Let me explain the key coding concepts we used, like explaining cooking techniques to a new chef:

### 1. **Import Statements**
```python
from django.core.mail import send_mail, EmailMultiAlternatives
```
- Like gathering your cooking tools before starting
- Imports bring in pre-built functionality we need

### 2. **Function Parameters**
```python
def send_welcome_email(request):
```
- `request` is like the customer's order - it contains all the information from the user
- Functions are like recipes - they take ingredients (parameters) and produce a result

### 3. **Dictionary Context**
```python
context = {
    'customer_name': customer_name,
    'restaurant_url': 'https://www.djangorestaurant.com',
}
```
- Like preparing ingredients in separate bowls before cooking
- Dictionaries store key-value pairs for easy access

### 4. **Template Rendering**
```python
html_content = render_to_string('emails/welcome_email.html', context)
```
- Like following a recipe and substituting ingredients
- Takes a template file and fills in the variables with actual values

### 5. **Exception Handling**
```python
try:
    email.send()
except Exception as e:
    messages.error(request, f'Failed to send email: {str(e)}')
```
- Like having a backup plan if something goes wrong in the kitchen
- `try` attempts the action, `except` handles any errors

### 6. **Threading**
```python
email_thread = threading.Thread(target=send_email)
email_thread.start()
```
- Like having multiple chefs working simultaneously
- Allows the program to do multiple things at once

### 7. **Decorators**
```python
@shared_task
def send_templated_email():
```
- Like putting a special label on a recipe box
- Decorators add extra functionality to functions

---

# Django Automated Email Notifications

## Project Objective
Building a complete automated email notification system that sends different types of emails based on user actions and system events.

## Project: Restaurant Order Notification System

Imagine that you're the head chef of a busy restaurant, and you need to keep everyone informed about what's happening in the kitchen. Just like how a chef coordinates with waiters, managers, and customers about order status, our Django application will automatically send emails to keep users updated about their activities.

### Core Features to Build

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Order(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('preparing', 'Preparing'),
        ('ready', 'Ready'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
    ]
    
    customer = models.ForeignKey(User, on_delete=models.CASCADE)
    items = models.TextField()
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Order #{self.id} - {self.customer.username}"

class EmailNotification(models.Model):
    NOTIFICATION_TYPES = [
        ('order_confirmation', 'Order Confirmation'),
        ('status_update', 'Status Update'),
        ('delivery_notification', 'Delivery Notification'),
        ('welcome', 'Welcome Email'),
        ('password_reset', 'Password Reset'),
    ]
    
    recipient = models.ForeignKey(User, on_delete=models.CASCADE)
    notification_type = models.CharField(max_length=30, choices=NOTIFICATION_TYPES)
    subject = models.CharField(max_length=200)
    sent_at = models.DateTimeField(auto_now_add=True)
    is_sent = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.notification_type} to {self.recipient.username}"
```

```python
# email_service.py
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.conf import settings
from django.contrib.auth.models import User
from .models import Order, EmailNotification
import logging

logger = logging.getLogger(__name__)

class EmailNotificationService:
    """
    Like a chef's assistant who knows exactly when to send out notifications
    about each dish's progress to the right people
    """
    
    @staticmethod
    def send_order_confirmation(order):
        """Send order confirmation email - like telling the customer their order is received"""
        try:
            subject = f'Order Confirmation #{order.id}'
            html_message = render_to_string('emails/order_confirmation.html', {
                'customer_name': order.customer.first_name or order.customer.username,
                'order': order,
                'order_items': order.items.split(','),
            })
            plain_message = strip_tags(html_message)
            
            send_mail(
                subject=subject,
                message=plain_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[order.customer.email],
                html_message=html_message,
                fail_silently=False,
            )
            
            # Log the notification
            EmailNotification.objects.create(
                recipient=order.customer,
                notification_type='order_confirmation',
                subject=subject,
                is_sent=True
            )
            
            logger.info(f"Order confirmation sent to {order.customer.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send order confirmation: {str(e)}")
            return False
    
    @staticmethod
    def send_status_update(order):
        """Send status update email - like updating the customer about their dish progress"""
        try:
            status_messages = {
                'confirmed': 'Your order has been confirmed and is being prepared!',
                'preparing': 'Our chef is now preparing your delicious meal!',
                'ready': 'Your order is ready for pickup/delivery!',
                'delivered': 'Your order has been delivered. Enjoy your meal!',
                'cancelled': 'Your order has been cancelled. We apologize for any inconvenience.',
            }
            
            subject = f'Order #{order.id} Status Update: {order.get_status_display()}'
            html_message = render_to_string('emails/status_update.html', {
                'customer_name': order.customer.first_name or order.customer.username,
                'order': order,
                'status_message': status_messages.get(order.status, 'Status updated'),
            })
            plain_message = strip_tags(html_message)
            
            send_mail(
                subject=subject,
                message=plain_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[order.customer.email],
                html_message=html_message,
                fail_silently=False,
            )
            
            EmailNotification.objects.create(
                recipient=order.customer,
                notification_type='status_update',
                subject=subject,
                is_sent=True
            )
            
            logger.info(f"Status update sent to {order.customer.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send status update: {str(e)}")
            return False
    
    @staticmethod
    def send_welcome_email(user):
        """Send welcome email to new users - like greeting new customers"""
        try:
            subject = 'Welcome to Our Restaurant!'
            html_message = render_to_string('emails/welcome.html', {
                'user_name': user.first_name or user.username,
            })
            plain_message = strip_tags(html_message)
            
            send_mail(
                subject=subject,
                message=plain_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                html_message=html_message,
                fail_silently=False,
            )
            
            EmailNotification.objects.create(
                recipient=user,
                notification_type='welcome',
                subject=subject,
                is_sent=True
            )
            
            logger.info(f"Welcome email sent to {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send welcome email: {str(e)}")
            return False
    
    @staticmethod
    def send_bulk_notifications(users, notification_type, subject, template_name, context=None):
        """Send bulk emails - like announcing today's special to all customers"""
        if context is None:
            context = {}
            
        success_count = 0
        
        for user in users:
            try:
                context['user_name'] = user.first_name or user.username
                html_message = render_to_string(template_name, context)
                plain_message = strip_tags(html_message)
                
                send_mail(
                    subject=subject,
                    message=plain_message,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[user.email],
                    html_message=html_message,
                    fail_silently=False,
                )
                
                EmailNotification.objects.create(
                    recipient=user,
                    notification_type=notification_type,
                    subject=subject,
                    is_sent=True
                )
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to send bulk email to {user.email}: {str(e)}")
        
        logger.info(f"Bulk notification sent to {success_count}/{len(users)} users")
        return success_count
```

```python
# signals.py
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import Order
from .email_service import EmailNotificationService

@receiver(post_save, sender=User)
def send_welcome_email(sender, instance, created, **kwargs):
    """
    Automatically send welcome email when new user registers
    Like greeting every new customer who walks into the restaurant
    """
    if created and instance.email:
        EmailNotificationService.send_welcome_email(instance)

@receiver(post_save, sender=Order)
def handle_order_notifications(sender, instance, created, **kwargs):
    """
    Handle order-related email notifications
    Like a chef calling out order updates to the front of house
    """
    if created:
        # Send confirmation email for new orders
        EmailNotificationService.send_order_confirmation(instance)
    else:
        # Send status update for existing orders
        EmailNotificationService.send_status_update(instance)
```

```python
# views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.contrib.auth.models import User
from .models import Order, EmailNotification
from .email_service import EmailNotificationService
from .forms import OrderForm, BulkEmailForm

@login_required
def create_order(request):
    """Create a new order - like taking a customer's order"""
    if request.method == 'POST':
        form = OrderForm(request.POST)
        if form.is_valid():
            order = form.save(commit=False)
            order.customer = request.user
            order.save()
            messages.success(request, 'Order created successfully! Confirmation email sent.')
            return redirect('order_detail', order_id=order.id)
    else:
        form = OrderForm()
    
    return render(request, 'orders/create_order.html', {'form': form})

@login_required
def update_order_status(request, order_id):
    """Update order status - like updating the kitchen board"""
    order = get_object_or_404(Order, id=order_id)
    
    if request.method == 'POST':
        new_status = request.POST.get('status')
        if new_status in dict(Order.STATUS_CHOICES):
            order.status = new_status
            order.save()
            messages.success(request, f'Order status updated to {order.get_status_display()}')
    
    return redirect('order_detail', order_id=order.id)

@login_required
def send_bulk_email(request):
    """Send bulk emails - like announcing the daily special"""
    if request.method == 'POST':
        form = BulkEmailForm(request.POST)
        if form.is_valid():
            users = User.objects.filter(is_active=True)
            success_count = EmailNotificationService.send_bulk_notifications(
                users=users,
                notification_type='bulk_notification',
                subject=form.cleaned_data['subject'],
                template_name='emails/bulk_notification.html',
                context={'message': form.cleaned_data['message']}
            )
            messages.success(request, f'Bulk email sent to {success_count} users')
            return redirect('admin_dashboard')
    else:
        form = BulkEmailForm()
    
    return render(request, 'admin/bulk_email.html', {'form': form})

def notification_history(request):
    """View notification history - like checking the order log"""
    notifications = EmailNotification.objects.filter(
        recipient=request.user
    ).order_by('-sent_at')
    
    return render(request, 'emails/notification_history.html', {
        'notifications': notifications
    })
```

```python
# management/commands/send_daily_summary.py
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
from myapp.models import Order
from myapp.email_service import EmailNotificationService

class Command(BaseCommand):
    help = 'Send daily summary emails'
    
    def handle(self, *args, **options):
        """Send daily summary - like the chef's end-of-day report"""
        yesterday = timezone.now() - timedelta(days=1)
        
        # Get users with orders in the last 24 hours
        users_with_orders = User.objects.filter(
            order__created_at__gte=yesterday
        ).distinct()
        
        for user in users_with_orders:
            recent_orders = Order.objects.filter(
                customer=user,
                created_at__gte=yesterday
            )
            
            context = {
                'user': user,
                'orders': recent_orders,
                'total_orders': recent_orders.count(),
            }
            
            EmailNotificationService.send_bulk_notifications(
                users=[user],
                notification_type='daily_summary',
                subject='Your Daily Order Summary',
                template_name='emails/daily_summary.html',
                context=context
            )
        
        self.stdout.write(
            self.style.SUCCESS(f'Daily summary sent to {users_with_orders.count()} users')
        )
```

```python
# celery_tasks.py
from celery import shared_task
from django.contrib.auth.models import User
from .email_service import EmailNotificationService
from .models import Order

@shared_task
def send_delayed_order_reminder(order_id):
    """
    Send reminder email for pending orders
    Like a chef checking on orders that have been waiting too long
    """
    try:
        order = Order.objects.get(id=order_id)
        if order.status == 'pending':
            EmailNotificationService.send_status_update(order)
            return f"Reminder sent for order #{order_id}"
    except Order.DoesNotExist:
        return f"Order #{order_id} not found"

@shared_task
def send_promotional_email(user_ids, subject, template_name, context):
    """Send promotional emails asynchronously"""
    users = User.objects.filter(id__in=user_ids)
    success_count = EmailNotificationService.send_bulk_notifications(
        users=users,
        notification_type='promotional',
        subject=subject,
        template_name=template_name,
        context=context
    )
    return f"Promotional email sent to {success_count} users"
```

```html
<!-- templates/emails/order_confirmation.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Order Confirmation</title>
    <style>
        .container { max-width: 600px; margin: 0 auto; font-family: Arial, sans-serif; }
        .header { background-color: #f8f9fa; padding: 20px; text-align: center; }
        .content { padding: 20px; }
        .order-details { background-color: #fff3cd; padding: 15px; border-radius: 5px; }
        .footer { background-color: #f8f9fa; padding: 20px; text-align: center; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Order Confirmation</h1>
        </div>
        <div class="content">
            <p>Dear {{ customer_name }},</p>
            <p>Thank you for your order! We're excited to prepare your delicious meal.</p>
            
            <div class="order-details">
                <h3>Order Details:</h3>
                <p><strong>Order #:</strong> {{ order.id }}</p>
                <p><strong>Items:</strong></p>
                <ul>
                    {% for item in order_items %}
                        <li>{{ item|trim }}</li>
                    {% endfor %}
                </ul>
                <p><strong>Total:</strong> ${{ order.total_amount }}</p>
                <p><strong>Status:</strong> {{ order.get_status_display }}</p>
            </div>
            
            <p>We'll keep you updated on your order's progress. Expected preparation time is 20-30 minutes.</p>
        </div>
        <div class="footer">
            <p>Thank you for choosing our restaurant!</p>
        </div>
    </div>
</body>
</html>
```

```python
# forms.py
from django import forms
from .models import Order

class OrderForm(forms.ModelForm):
    class Meta:
        model = Order
        fields = ['items', 'total_amount']
        widgets = {
            'items': forms.Textarea(attrs={'placeholder': 'Enter items separated by commas'}),
            'total_amount': forms.NumberInput(attrs={'step': '0.01'}),
        }

class BulkEmailForm(forms.Form):
    subject = forms.CharField(max_length=200, widget=forms.TextInput(attrs={'class': 'form-control'}))
    message = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 5}))
```

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('orders/create/', views.create_order, name='create_order'),
    path('orders/<int:order_id>/', views.order_detail, name='order_detail'),
    path('orders/<int:order_id>/update-status/', views.update_order_status, name='update_order_status'),
    path('admin/bulk-email/', views.send_bulk_email, name='send_bulk_email'),
    path('notifications/', views.notification_history, name='notification_history'),
]
```

```python
# settings.py additions
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'your-email@gmail.com'
EMAIL_HOST_PASSWORD = 'your-app-password'
DEFAULT_FROM_EMAIL = 'Restaurant Notifications <noreply@restaurant.com>'

# Celery settings for async email sending
CELERY_BROKER_URL = 'redis://localhost:6379'
CELERY_RESULT_BACKEND = 'redis://localhost:6379'

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'email_notifications.log',
        },
    },
    'loggers': {
        'myapp.email_service': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
```

### Assignment: Customer Review Email System

Create an automated email system that sends review requests to customers 24 hours after their order is marked as "delivered". The system should:

1. **Create a CustomerReview model** with fields: customer, order, rating, comment, created_at
2. **Build a Celery task** that automatically sends review request emails
3. **Create an email template** for review requests with a link to submit reviews
4. **Add a view** to handle review submissions
5. **Implement email tracking** to prevent sending duplicate review requests

**Bonus**: Add a follow-up email if no review is submitted within 7 days.

### Testing Your Implementation

```python
# test_email_notifications.py
from django.test import TestCase
from django.core import mail
from django.contrib.auth.models import User
from .models import Order, EmailNotification
from .email_service import EmailNotificationService

class EmailNotificationTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.order = Order.objects.create(
            customer=self.user,
            items='Pizza, Salad',
            total_amount=25.99,
            status='pending'
        )
    
    def test_order_confirmation_email(self):
        """Test that order confirmation emails are sent"""
        result = EmailNotificationService.send_order_confirmation(self.order)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('Order Confirmation', mail.outbox[0].subject)
        self.assertEqual(mail.outbox[0].to, [self.user.email])
    
    def test_status_update_email(self):
        """Test that status update emails are sent"""
        self.order.status = 'confirmed'
        result = EmailNotificationService.send_status_update(self.order)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('Status Update', mail.outbox[0].subject)
    
    def test_welcome_email(self):
        """Test that welcome emails are sent"""
        result = EmailNotificationService.send_welcome_email(self.user)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('Welcome', mail.outbox[0].subject)
    
    def test_email_notification_logging(self):
        """Test that email notifications are logged"""
        EmailNotificationService.send_order_confirmation(self.order)
        
        notification = EmailNotification.objects.get(
            recipient=self.user,
            notification_type='order_confirmation'
        )
        self.assertTrue(notification.is_sent)
```

This project demonstrates a complete automated email notification system that handles various scenarios in a restaurant ordering system, just like how a well-organized kitchen keeps everyone informed about what's happening with each order.

## Assignment: Build a Restaurant Newsletter System

### The Challenge
Create a newsletter system for Django Restaurant that demonstrates all four concepts we've learned. Think of it as creating a complete communication system for your restaurant.

### Requirements

1. **Create a Newsletter Model** (Your subscriber list)
```python
# models.py
class NewsletterSubscriber(models.Model):
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=100)
    subscribed_date = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    preferences = models.JSONField(default=dict)  # For storing email preferences
```

2. **Build Three Email Templates**:
   - Welcome email for new subscribers
   - Weekly newsletter template
   - Unsubscribe confirmation

3. **Create Views That**:
   - Handle newsletter subscription
   - Send welcome emails asynchronously
   - Allow users to unsubscribe
   - Send weekly newsletters to all subscribers

4. **Implementation Requirements**:
   - Use at least 2 different email backends (console for development, SMTP for production)
   - Implement asynchronous email sending for the welcome email
   - Create beautiful HTML email templates with CSS
   - Add proper error handling and user feedback

### Starter Code Structure
```
newsletter/
    models.py
    views.py
    forms.py
    tasks.py (if using Celery)
    templates/
        newsletter/
            subscribe.html
            success.html
        emails/
            newsletter_welcome.html
            weekly_newsletter.html
            unsubscribe_confirmation.html
```

### Success Criteria
- Users can subscribe and receive a welcome email
- Newsletter templates are visually appealing
- Emails send asynchronously without blocking the user
- System works with different email backends
- Proper error handling and user feedback

This assignment combines all four concepts into a real-world scenario that any restaurant (or business) would actually use!

---

## Summary

Today we've transformed you from a novice cook into a master chef of Django email communication! You've learned to:

1. **Set up Django's email framework** - Like organizing your kitchen with the right tools
2. **Create beautiful email templates** - Like developing signature recipes
3. **Send emails asynchronously** - Like running an efficient kitchen with multiple chefs
4. **Configure different email backends** - Like choosing the right cooking method for each dish

Just as a great chef knows that presentation is as important as taste, you now know that email communication is as important as the functionality of your Django application. Your users will appreciate the thoughtful, well-crafted emails just as much as diners appreciate a beautifully presented meal!