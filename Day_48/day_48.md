# Day 48: File Uploads & Media Handling

## Learning Objective
By the end of this lesson, you will be able to implement secure file upload functionality in Django, configure media settings, validate uploaded files, and process images using Pillow - just like a head chef who knows how to safely receive, inspect, and prepare ingredients from suppliers.

---

## Lesson Introduction

Imagine that you're the head chef of a bustling restaurant, and every day suppliers deliver fresh ingredients to your kitchen. Some bring vegetables, others bring meats, and some deliver specialty items like exotic spices or delicate pastries. As a responsible chef, you can't just accept everything blindly - you need to inspect each delivery, verify the quality, store items properly, and sometimes even process them before they're ready for your kitchen.

File uploads in web applications work exactly the same way. Your Django application is like your restaurant kitchen, and users are like suppliers bringing you "ingredients" (files). You need to safely receive these files, validate them, store them securely, and sometimes process them before they're ready to be served to other users.

Today, we'll learn how to be the master chef of file handling!

---

## 1. File and Image Uploads

### The Kitchen Receiving Area

Think of file uploads as your restaurant's receiving area - the place where deliveries arrive. Just as you need a proper loading dock and procedures for accepting deliveries, Django needs specific configurations and views to handle file uploads.

### Setting Up the Basic Upload Infrastructure

First, let's create the "receiving area" in our Django kitchen:

**models.py** - Creating storage containers
```python
from django.db import models
from django.contrib.auth.models import User
import os

def user_directory_path(instance, filename):
    """
    Like organizing ingredients in labeled containers by supplier.
    Files will be uploaded to MEDIA_ROOT/user_<id>/<filename>
    """
    return f'user_{instance.user.id}/{filename}'

class Recipe(models.Model):
    """Our recipe model - like a recipe card with an attached photo"""
    title = models.CharField(max_length=200)
    description = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # This is our "photo storage" - like a recipe card with picture attachment
    recipe_image = models.ImageField(
        upload_to=user_directory_path,
        null=True,
        blank=True,
        help_text="Upload a mouth-watering photo of your dish!"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
```

**Syntax Explanation:**
- `ImageField`: A special Django field for handling image uploads, like a specialized storage container for photos
- `upload_to`: Defines where files are stored, like deciding which shelf in your pantry
- `user_directory_path`: A function that creates organized storage paths, like labeling shelves by supplier
- `null=True, blank=True`: Makes the field optional, like having a recipe that doesn't require a photo

### Creating the Upload Form

**forms.py** - The order form for deliveries
```python
from django import forms
from .models import Recipe

class RecipeUploadForm(forms.ModelForm):
    """
    Like a delivery order form - specifies what we're expecting
    and how it should be packaged for delivery
    """
    class Meta:
        model = Recipe
        fields = ['title', 'description', 'recipe_image']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Name your delicious creation...'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe your recipe...'
            }),
            'recipe_image': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'  # Only accept image files
            })
        }
```

**Syntax Explanation:**
- `ModelForm`: Automatically creates a form based on a model, like a pre-printed order form
- `ClearableFileInput`: A widget that allows file selection and deletion, like a delivery slip with a checkbox
- `accept='image/*'`: HTML attribute that filters file selection to images only

### Creating the Upload View

**views.py** - The receiving dock manager
```python
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import RecipeUploadForm
from .models import Recipe

@login_required
def upload_recipe(request):
    """
    Like a receiving dock manager who processes incoming deliveries.
    Checks credentials, inspects packages, and stores them properly.
    """
    if request.method == 'POST':
        form = RecipeUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            # Create the recipe but don't save yet (like preparing the storage space)
            recipe = form.save(commit=False)
            
            # Assign the current user as the author (like signing the delivery receipt)
            recipe.author = request.user
            
            # Now save everything to the database (like filing the completed order)
            recipe.save()
            
            messages.success(request, 'Recipe uploaded successfully! Your dish looks delicious!')
            return redirect('recipe_detail', pk=recipe.pk)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = RecipeUploadForm()
    
    return render(request, 'recipes/upload.html', {'form': form})

def recipe_detail(request, pk):
    """Display a single recipe - like presenting a finished dish"""
    recipe = Recipe.objects.get(pk=pk)
    return render(request, 'recipes/detail.html', {'recipe': recipe})
```

**Syntax Explanation:**
- `request.FILES`: Contains uploaded files, like the actual delivery packages
- `commit=False`: Prevents immediate saving, like preparing a storage space before moving items
- `@login_required`: Decorator ensuring only authenticated users can upload, like checking ID at the receiving dock

### Template for Upload

**templates/recipes/upload.html** - The delivery interface
```html
<!DOCTYPE html>
<html>
<head>
    <title>Upload Recipe</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h3>📸 Share Your Culinary Creation</h3>
                        <p class="text-muted">Upload a photo and details of your delicious recipe!</p>
                    </div>
                    <div class="card-body">
                        <!-- Very important: enctype for file uploads -->
                        <form method="post" enctype="multipart/form-data">
                            {% csrf_token %}
                            
                            <div class="mb-3">
                                <label for="{{ form.title.id_for_label }}" class="form-label">Recipe Title</label>
                                {{ form.title }}
                            </div>
                            
                            <div class="mb-3">
                                <label for="{{ form.description.id_for_label }}" class="form-label">Description</label>
                                {{ form.description }}
                            </div>
                            
                            <div class="mb-3">
                                <label for="{{ form.recipe_image.id_for_label }}" class="form-label">Recipe Photo</label>
                                {{ form.recipe_image }}
                                <div class="form-text">Choose a high-quality image (JPG, PNG, etc.)</div>
                            </div>
                            
                            <button type="submit" class="btn btn-primary">🍳 Upload Recipe</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
```

**Syntax Explanation:**
- `enctype="multipart/form-data"`: Essential for file uploads, like using the right type of delivery truck
- `{% csrf_token %}`: Security token, like a security seal on deliveries

---

## 2. Media Files Configuration

### Setting Up the Kitchen Storage System

Just as a restaurant needs proper storage areas, refrigerators, and pantries, Django needs to know where to store uploaded files and how to serve them.

**settings.py** - Kitchen storage configuration
```python
import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Media files configuration - like setting up your restaurant's storage areas
MEDIA_URL = '/media/'  # The URL path to access files (like the address of your storage room)
MEDIA_ROOT = BASE_DIR / 'media'  # Physical location where files are stored (like the actual storage room)

# Static files (CSS, JavaScript, Images)
# These are like your restaurant's permanent fixtures and equipment
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# During development, we might want to store media files in a specific location
if not os.path.exists(MEDIA_ROOT):
    os.makedirs(MEDIA_ROOT)  # Create the storage room if it doesn't exist
```

### URL Configuration for Media Files

**urls.py (main project)** - Setting up delivery routes
```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('recipes/', include('recipes.urls')),
]

# During development, serve media files
# Like having a delivery route that brings stored ingredients to the kitchen
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

**Syntax Explanation:**
- `MEDIA_URL`: The URL prefix for accessing uploaded files, like a street address
- `MEDIA_ROOT`: The filesystem path where files are stored, like GPS coordinates
- `static()`: A helper function that creates URL patterns for serving files during development

### App-level URLs

**recipes/urls.py** - Kitchen department routes
```python
from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_recipe, name='upload_recipe'),
    path('<int:pk>/', views.recipe_detail, name='recipe_detail'),
]
```

---

## 3. File Validation and Security

### The Quality Control Inspector

Just as a chef inspects every delivery for quality and safety, we need to validate uploaded files. You wouldn't want spoiled ingredients in your kitchen, and you don't want malicious files in your application!

**validators.py** - Your quality control department
```python
from django.core.exceptions import ValidationError
from django.core.files.images import get_image_dimensions
import os

def validate_image_size(image):
    """
    Like a quality inspector checking if ingredients meet size standards.
    Ensures images aren't too large for our 'storage containers'.
    """
    # Maximum file size: 5MB (like checking if a delivery truck can fit in the loading dock)
    max_size = 5 * 1024 * 1024  # 5 MB in bytes
    
    if image.size > max_size:
        raise ValidationError(
            f'Image file too large. Maximum size is 5MB. '
            f'Your file is {image.size / (1024*1024):.1f}MB.'
        )

def validate_image_dimensions(image):
    """
    Like checking if ingredients will fit in your standard containers.
    Ensures images have reasonable dimensions.
    """
    width, height = get_image_dimensions(image)
    
    # Maximum dimensions (like checking if a cake will fit in your display case)
    max_width, max_height = 4000, 4000
    
    if width > max_width or height > max_height:
        raise ValidationError(
            f'Image dimensions too large. Maximum is {max_width}x{max_height}px. '
            f'Your image is {width}x{height}px.'
        )
    
    # Minimum dimensions (like ensuring a photo is clear enough to display)
    min_width, min_height = 100, 100
    
    if width < min_width or height < min_height:
        raise ValidationError(
            f'Image too small. Minimum dimensions are {min_width}x{min_height}px. '
            f'Your image is {width}x{height}px.'
        )

def validate_file_extension(file):
    """
    Like checking if delivered ingredients are the right type.
    Only allows specific file types in our kitchen.
    """
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    
    # Get the file extension (like checking the label on a delivery box)
    file_extension = os.path.splitext(file.name)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise ValidationError(
            f'File type not allowed. Allowed types: {", ".join(allowed_extensions)}. '
            f'Your file type: {file_extension}'
        )
```

### Updated Model with Validation

**models.py** - Enhanced storage with quality control
```python
from django.db import models
from django.contrib.auth.models import User
from .validators import validate_image_size, validate_image_dimensions, validate_file_extension

def user_directory_path(instance, filename):
    """Organize files like a well-structured pantry"""
    return f'user_{instance.user.id}/{filename}'

class Recipe(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Enhanced image field with validation - like a high-security storage area
    recipe_image = models.ImageField(
        upload_to=user_directory_path,
        null=True,
        blank=True,
        validators=[
            validate_image_size,
            validate_image_dimensions,
            validate_file_extension
        ],
        help_text="Upload a high-quality image (JPG, PNG, GIF, WebP). Max 5MB, minimum 100x100px."
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
```

### Security Middleware

**middleware.py** - Security checkpoint for all deliveries
```python
from django.http import HttpResponseForbidden
from django.core.files.storage import default_storage
import os

class FileUploadSecurityMiddleware:
    """
    Like a security guard at the loading dock who checks every delivery.
    Prevents dangerous files from entering our kitchen.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        
        # Dangerous file types - like ingredients that could poison customers
        self.dangerous_extensions = [
            '.exe', '.bat', '.com', '.cmd', '.scr', '.pif',
            '.php', '.jsp', '.asp', '.js', '.vbs', '.jar'
        ]
    
    def __call__(self, request):
        # Check file uploads before processing
        if request.method == 'POST' and request.FILES:
            for field_name, uploaded_file in request.FILES.items():
                if not self.is_safe_file(uploaded_file):
                    return HttpResponseForbidden(
                        "File type not allowed for security reasons. "
                        "Please upload only image files."
                    )
        
        response = self.get_response(request)
        return response
    
    def is_safe_file(self, uploaded_file):
        """Check if a file is safe to store in our kitchen"""
        # Check file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension in self.dangerous_extensions:
            return False
        
        # Additional check: verify file content matches extension
        # (like making sure a box labeled "flour" actually contains flour)
        try:
            from PIL import Image
            Image.open(uploaded_file).verify()
            return True
        except Exception:
            return False
```

**Syntax Explanation:**
- `ValidationError`: Django's way of reporting problems, like a rejection slip for bad deliveries
- `get_image_dimensions()`: Function to check image size, like measuring ingredients
- `os.path.splitext()`: Separates filename from extension, like reading the label on a package

---

## 4. Image Processing with Pillow

### The Food Preparation Station

Just as a chef prepares ingredients before cooking - washing vegetables, cutting meat, or garnishing dishes - we often need to process images before storing them. Pillow is like having a skilled sous chef who specializes in image preparation.

### Installing and Setting Up Pillow

First, let's add our "sous chef" to the kitchen:

```bash
pip install Pillow
```

### Basic Image Processing

**image_processing.py** - Your image preparation station
```python
from PIL import Image, ImageFilter, ImageEnhance
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import io
import os

class ImageProcessor:
    """
    Like a skilled sous chef who specializes in food presentation.
    Handles all image preparation tasks.
    """
    
    def __init__(self):
        # Standard sizes for different purposes (like standard plate sizes)
        self.thumbnail_size = (150, 150)
        self.medium_size = (500, 500)
        self.large_size = (1200, 1200)
    
    def create_thumbnail(self, image_file, size=None):
        """
        Like creating a small garnish version of a dish for display.
        Creates a small version of the image for quick loading.
        """
        if size is None:
            size = self.thumbnail_size
        
        # Open the image (like unwrapping an ingredient)
        with Image.open(image_file) as img:
            # Convert to RGB if necessary (like preparing ingredients for cooking)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create thumbnail while maintaining aspect ratio (like plating food proportionally)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Save to memory buffer (like preparing a dish on a serving plate)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            
            return ContentFile(buffer.read())
    
    def resize_image(self, image_file, max_size=None):
        """
        Like cutting ingredients to the right size for cooking.
        Resizes images to fit within specified dimensions.
        """
        if max_size is None:
            max_size = self.large_size
        
        with Image.open(image_file) as img:
            # Keep original aspect ratio (like maintaining food's natural proportions)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Enhance the image slightly (like adding a light seasoning)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.1)  # Slightly sharper
            
            # Save the processed image
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=90, optimize=True)
            buffer.seek(0)
            
            return ContentFile(buffer.read())
    
    def add_watermark(self, image_file, watermark_text="YourKitchen"):
        """
        Like adding a chef's signature to a dish.
        Adds a subtle watermark to protect images.
        """
        with Image.open(image_file) as img:
            # Create a transparent overlay (like a clear garnish sheet)
            overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
            
            # You would normally import ImageDraw and ImageFont here
            # For simplicity, we'll just return the original image
            # In a real implementation, you'd draw text on the overlay
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            
            return ContentFile(buffer.read())
    
    def optimize_for_web(self, image_file):
        """
        Like preparing food for efficient service.
        Optimizes images for fast web loading.
        """
        with Image.open(image_file) as img:
            # Convert to RGB for web compatibility
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (like portioning food appropriately)
            if img.size[0] > 1200 or img.size[1] > 1200:
                img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            
            # Save with web optimization
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True, progressive=True)
            buffer.seek(0)
            
            return ContentFile(buffer.read())
```

### Enhanced Model with Image Processing

**models.py** - Kitchen with a preparation station
```python
from django.db import models
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from .validators import validate_image_size, validate_image_dimensions, validate_file_extension
from .image_processing import ImageProcessor
import os

def user_directory_path(instance, filename):
    return f'user_{instance.user.id}/{filename}'

def user_thumbnail_path(instance, filename):
    """Separate storage for thumbnails - like a display case for appetizers"""
    name, ext = os.path.splitext(filename)
    return f'user_{instance.user.id}/thumbnails/{name}_thumb{ext}'

class Recipe(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Main image - like the hero shot of your dish
    recipe_image = models.ImageField(
        upload_to=user_directory_path,
        null=True,
        blank=True,
        validators=[validate_image_size, validate_image_dimensions, validate_file_extension],
        help_text="Upload a high-quality image of your recipe"
    )
    
    # Thumbnail - like a small preview photo on the menu
    thumbnail = models.ImageField(
        upload_to=user_thumbnail_path,
        null=True,
        blank=True,
        help_text="Automatically generated thumbnail"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def save(self, *args, **kwargs):
        """
        Like a head chef who oversees food preparation.
        Automatically processes images when a recipe is saved.
        """
        # First, save the model to get an ID
        super().save(*args, **kwargs)
        
        # If we have a recipe image, process it
        if self.recipe_image:
            processor = ImageProcessor()
            
            # Create optimized version of the main image
            optimized_image = processor.optimize_for_web(self.recipe_image)
            
            # Save the optimized image back
            self.recipe_image.save(
                self.recipe_image.name,
                optimized_image,
                save=False  # Don't trigger save() again
            )
            
            # Create and save thumbnail
            thumbnail_image = processor.create_thumbnail(self.recipe_image)
            thumbnail_name = f"thumb_{self.recipe_image.name}"
            
            self.thumbnail.save(
                thumbnail_name,
                thumbnail_image,
                save=False
            )
            
            # Save the model again with processed images
            super().save(*args, **kwargs)
    
    def __str__(self):
        return self.title
```

### Enhanced View with Processing

**views.py** - Kitchen manager with preparation oversight
```python
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.exceptions import ValidationError
from .forms import RecipeUploadForm
from .models import Recipe
from .image_processing import ImageProcessor

@login_required
def upload_recipe(request):
    """
    Enhanced receiving dock with quality control and preparation.
    """
    if request.method == 'POST':
        form = RecipeUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                # Create recipe instance
                recipe = form.save(commit=False)
                recipe.author = request.user
                
                # If there's an image, do a final validation
                if recipe.recipe_image:
                    processor = ImageProcessor()
                    
                    # Additional processing validation
                    # (like a final quality check before storing)
                    try:
                        # Test that we can process the image
                        processor.optimize_for_web(recipe.recipe_image)
                        
                        messages.success(
                            request, 
                            'Recipe uploaded successfully! Your image has been optimized for web display.'
                        )
                    except Exception as e:
                        messages.warning(
                            request,
                            f'Recipe saved, but image processing had an issue: {str(e)}'
                        )
                
                # Save the recipe (this will trigger image processing)
                recipe.save()
                
                return redirect('recipe_detail', pk=recipe.pk)
                
            except ValidationError as e:
                messages.error(request, f'Upload failed: {str(e)}')
            except Exception as e:
                messages.error(request, f'An unexpected error occurred: {str(e)}')
        else:
            messages.error(request, 'Please correct the errors in the form.')
    
    else:
        form = RecipeUploadForm()
    
    return render(request, 'recipes/upload.html', {'form': form})
```

**Syntax Explanation:**
- `PIL.Image`: The main image processing class, like your primary cooking tool
- `ImageFilter`, `ImageEnhance`: Tools for image improvement, like seasoning and garnishes
- `io.BytesIO()`: A memory buffer for image data, like a temporary prep bowl
- `ContentFile()`: Django's way of handling file content, like a standardized storage container
- `save=False`: Prevents recursive saving, like finishing one prep step before starting another

---

# Project: File Uploads & Media Handling - Profile Picture Uploads

## Project Objective
By the end of this project, you will be able to implement a complete profile picture upload system in Django, including file handling, validation, and display functionality.

---

## Introduction

Imagine that you're running a bustling restaurant kitchen, and each of your chefs wants to display their professional headshot on their station. Just like how a head chef needs to ensure that only proper, high-quality photos are displayed (not blurry snapshots or inappropriate images), Django needs a system to handle, validate, and serve profile pictures safely and efficiently.

In our analogy, think of Django as the head chef who:
- **Receives** the photos (file uploads)
- **Inspects** them for quality and appropriateness (validation)
- **Stores** them in the right location (media handling)
- **Displays** them at each chef's station (serving files)

---

## Build: Profile Picture Uploads

# models.py
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
from PIL import Image
import os

def user_profile_pic_path(instance, filename):
    """
    Custom upload path function - like assigning each chef their own photo frame location
    This creates a path like: profile_pics/user_123/filename.jpg
    """
    return f'profile_pics/user_{instance.user.id}/{filename}'

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(
        upload_to=user_profile_pic_path,
        null=True,
        blank=True,
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png'])],
        help_text="Upload a profile picture (JPG, JPEG, or PNG format)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
    
    def save(self, *args, **kwargs):
        """
        Override save method - like the head chef's final quality check
        This resizes images to maintain consistency
        """
        super().save(*args, **kwargs)
        
        if self.profile_picture:
            # Open the uploaded image
            img = Image.open(self.profile_picture.path)
            
            # Resize if image is too large (like standardizing photo sizes in our kitchen)
            if img.height > 300 or img.width > 300:
                output_size = (300, 300)
                img.thumbnail(output_size)
                img.save(self.profile_picture.path)

# forms.py
from django import forms
from .models import UserProfile

class ProfilePictureForm(forms.ModelForm):
    """
    Form for profile picture upload - like the application form chefs fill out
    """
    class Meta:
        model = UserProfile
        fields = ['profile_picture']
        widgets = {
            'profile_picture': forms.FileInput(attrs={
                'class': 'form-control-file',
                'accept': 'image/*',
                'id': 'profile-pic-input'
            })
        }
    
    def clean_profile_picture(self):
        """
        Custom validation - like the head chef checking photo quality
        """
        picture = self.cleaned_data.get('profile_picture')
        
        if picture:
            # Check file size (max 5MB)
            if picture.size > 5 * 1024 * 1024:  # 5MB in bytes
                raise forms.ValidationError("Image file too large (max 5MB)")
            
            # Check image dimensions
            img = Image.open(picture)
            if img.width < 50 or img.height < 50:
                raise forms.ValidationError("Image too small (minimum 50x50 pixels)")
        
        return picture

# views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from .models import UserProfile
from .forms import ProfilePictureForm
import json

@login_required
def upload_profile_picture(request):
    """
    Handle profile picture upload - like the main photo submission process
    """
    try:
        user_profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        user_profile = UserProfile.objects.create(user=request.user)
    
    if request.method == 'POST':
        form = ProfilePictureForm(request.POST, request.FILES, instance=user_profile)
        
        if form.is_valid():
            # Delete old profile picture if exists (like removing the old photo from frame)
            if user_profile.profile_picture:
                old_pic_path = user_profile.profile_picture.path
                if os.path.exists(old_pic_path):
                    os.remove(old_pic_path)
            
            form.save()
            messages.success(request, 'Profile picture updated successfully!')
            
            # If it's an AJAX request, return JSON response
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'image_url': user_profile.profile_picture.url,
                    'message': 'Profile picture updated successfully!'
                })
            
            return redirect('profile')
        else:
            messages.error(request, 'Please correct the errors below.')
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'errors': form.errors
                })
    else:
        form = ProfilePictureForm(instance=user_profile)
    
    return render(request, 'profiles/upload_picture.html', {
        'form': form,
        'user_profile': user_profile
    })

@login_required
def remove_profile_picture(request):
    """
    Remove profile picture - like taking down the photo from the chef's station
    """
    if request.method == 'POST':
        try:
            user_profile = UserProfile.objects.get(user=request.user)
            
            if user_profile.profile_picture:
                # Delete the file from storage
                pic_path = user_profile.profile_picture.path
                if os.path.exists(pic_path):
                    os.remove(pic_path)
                
                # Clear the field
                user_profile.profile_picture = None
                user_profile.save()
                
                messages.success(request, 'Profile picture removed successfully!')
                
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'success': True})
            
        except UserProfile.DoesNotExist:
            messages.error(request, 'Profile not found.')
    
    return redirect('profile')

def user_profile_view(request, username):
    """
    Display user profile with picture - like showing the chef's station with their photo
    """
    user = get_object_or_404(User, username=username)
    try:
        user_profile = UserProfile.objects.get(user=user)
    except UserProfile.DoesNotExist:
        user_profile = None
    
    return render(request, 'profiles/profile.html', {
        'profile_user': user,
        'user_profile': user_profile,
        'is_own_profile': request.user == user
    })

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('upload-picture/', views.upload_profile_picture, name='upload_profile_picture'),
    path('remove-picture/', views.remove_profile_picture, name='remove_profile_picture'),
    path('profile/<str:username>/', views.user_profile_view, name='user_profile'),
]

# Template: profiles/upload_picture.html
"""
<!DOCTYPE html>
<html>
<head>
    <title>Upload Profile Picture</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .profile-pic-preview {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            object-fit: cover;
            border: 3px solid #ddd;
        }
        .upload-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 20px;
        }
        .preview-container {
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <div class="upload-container">
            <h2 class="text-center mb-4">Upload Profile Picture</h2>
            
            <!-- Current Profile Picture Preview -->
            <div class="preview-container">
                {% if user_profile.profile_picture %}
                    <img src="{{ user_profile.profile_picture.url }}" 
                         alt="Current Profile Picture" 
                         class="profile-pic-preview" 
                         id="current-pic">
                {% else %}
                    <div class="profile-pic-preview bg-light d-flex align-items-center justify-content-center" 
                         id="current-pic">
                        <span class="text-muted">No Picture</span>
                    </div>
                {% endif %}
            </div>
            
            <!-- Upload Form -->
            <form method="post" enctype="multipart/form-data" id="upload-form">
                {% csrf_token %}
                <div class="mb-3">
                    {{ form.profile_picture }}
                    <small class="form-text text-muted">
                        {{ form.profile_picture.help_text }}
                    </small>
                </div>
                
                <div class="d-grid gap-2">
                    <button type="submit" class="btn btn-primary">Upload Picture</button>
                    {% if user_profile.profile_picture %}
                        <button type="button" class="btn btn-danger" id="remove-btn">Remove Picture</button>
                    {% endif %}
                </div>
            </form>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Preview image before upload - like showing the chef their photo before framing it
        document.getElementById('profile-pic-input').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('current-pic').src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Handle remove button
        document.getElementById('remove-btn')?.addEventListener('click', function() {
            if (confirm('Are you sure you want to remove your profile picture?')) {
                fetch('{% url "remove_profile_picture" %}', {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': '{{ csrf_token }}',
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        location.reload();
                    }
                });
            }
        });
    </script>
</body>
</html>
"""

# Template: profiles/profile.html
"""
<!DOCTYPE html>
<html>
<head>
    <title>{{ profile_user.username }}'s Profile</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .profile-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 0;
        }
        .profile-pic {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            object-fit: cover;
            border: 4px solid white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .profile-info {
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="profile-header">
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-md-8">
                    <div class="profile-info">
                        {% if user_profile.profile_picture %}
                            <img src="{{ user_profile.profile_picture.url }}" 
                                 alt="{{ profile_user.username }}'s Profile Picture" 
                                 class="profile-pic mb-3">
                        {% else %}
                            <div class="profile-pic bg-light d-flex align-items-center justify-content-center mb-3">
                                <span class="text-muted">{{ profile_user.username.0|upper }}</span>
                            </div>
                        {% endif %}
                        
                        <h2>{{ profile_user.first_name }} {{ profile_user.last_name }}</h2>
                        <p class="lead">@{{ profile_user.username }}</p>
                        
                        {% if is_own_profile %}
                            <a href="{% url 'upload_profile_picture' %}" class="btn btn-light">
                                Change Profile Picture
                            </a>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="container mt-5">
        <div class="row">
            <div class="col-md-8 offset-md-2">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Profile Information</h5>
                        <p><strong>Username:</strong> {{ profile_user.username }}</p>
                        <p><strong>Email:</strong> {{ profile_user.email }}</p>
                        <p><strong>Joined:</strong> {{ profile_user.date_joined|date:"F d, Y" }}</p>
                        {% if user_profile %}
                            <p><strong>Profile Updated:</strong> {{ user_profile.updated_at|date:"F d, Y" }}</p>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

#settings configuration

# settings.py
import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-your-secret-key-here-change-in-production'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'your_app_name',  # Replace with your actual app name (e.g., 'profiles')
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'your_project_name.urls'  # Replace with your project name

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  # Add this if you have a project-level templates folder
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'your_project_name.wsgi.application'  # Replace with your project name

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

# Media files configuration - like setting up the photo storage room in our kitchen
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Login/Logout redirect URLs (useful for authentication)
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'

# File upload settings for security
FILE_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB

# Additional security settings for production (commented out for development)
# SECURE_BROWSER_XSS_FILTER = True
# SECURE_CONTENT_TYPE_NOSNIFF = True
# X_FRAME_OPTIONS = 'DENY'

# Main project urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('your_app_name.urls')),  # Replace with your app name
]

# This is crucial - it tells Django how to serve media files during development
# Like setting up the photo display system in our kitchen
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# admin.py - Register the model so chefs can manage profiles from the admin kitchen
from django.contrib import admin
from .models import UserProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'created_at', 'updated_at']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['created_at', 'updated_at']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')

### Code Syntax Explanation

### 1. **Model Field Types**
```python
profile_picture = models.ImageField(upload_to=user_profile_pic_path, ...)
```
- `ImageField`: A specialized file field that validates uploaded files as images
- `upload_to`: Defines where files are stored (can be a string or callable function)
- `null=True, blank=True`: Allows the field to be empty in database and forms

### 2. **Custom Upload Path Function**
```python
def user_profile_pic_path(instance, filename):
    return f'profile_pics/user_{instance.user.id}/{filename}'
```
- This function receives the model instance and original filename
- Returns a custom path for file storage
- Like assigning each chef their own designated photo display area

### 3. **Model Method Override**
```python
def save(self, *args, **kwargs):
    super().save(*args, **kwargs)  # Call parent save method first
    # Custom logic after saving
```
- Overrides the default save behavior
- Always call `super().save()` to maintain Django's built-in functionality
- Used here to automatically resize uploaded images

### 4. **Form Validation**
```python
def clean_profile_picture(self):
    picture = self.cleaned_data.get('profile_picture')
    # Custom validation logic
    return picture
```
- `clean_<fieldname>()`: Custom validation method for specific fields
- Must return the cleaned data
- Like the head chef's quality inspection process

### 5. **File Handling**
```python
if os.path.exists(old_pic_path):
    os.remove(old_pic_path)
```
- Django doesn't automatically delete old files when new ones are uploaded
- Manual file deletion prevents storage buildup
- Like clearing out old photos before hanging new ones

### 6. **AJAX Detection**
```python
if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
    return JsonResponse({'success': True})
```
- Detects if request came from JavaScript (AJAX)
- Returns JSON response for dynamic updates
- Like having a special communication system between kitchen stations

### 7. **Template Context**
```python
return render(request, 'template.html', {
    'form': form,
    'user_profile': user_profile
})
```
- Passes data from view to template
- Dictionary keys become template variables
- Like preparing ingredients for the chef to use

### 8. **Media URL Configuration**
```python
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```
- `MEDIA_URL`: URL prefix for serving media files
- `MEDIA_ROOT`: Filesystem path where media files are stored
- Like setting up the kitchen's photo display system

This system creates a complete, production-ready profile picture feature that handles uploads, validation, storage, and display - just like a well-organized kitchen photo management system where each chef has their professional headshot properly displayed and maintained!

## Assignment: Recipe Gallery with Smart Uploads

### The Challenge

Create a "Recipe Gallery" feature for a cooking website that demonstrates all the concepts we've learned. You'll build a system that allows users to upload recipe photos with smart processing, validation, and display.

### Requirements

1. **User Authentication**: Only registered users can upload recipes
2. **Smart Upload System**: 
   - Validate file types (images only)
   - Limit file size to 5MB
   - Automatically resize large images
   - Generate thumbnails for gallery view
3. **Gallery Display**: Show all recipes in a responsive grid with thumbnails
4. **Individual Recipe View**: Full-size image with recipe details
5. **User Management**: Users can only edit/delete their own recipes

### Starter Code Structure

You'll need to create:
- A User model extension for profile management
- Enhanced Recipe model with image processing
- Upload form with validation
- Gallery view with pagination
- Individual recipe detail view

### Expected Features

1. **Upload Page**: Clean form with drag-and-drop file selection
2. **Gallery Page**: Responsive grid of recipe thumbnails
3. **Recipe Detail Page**: Full image with recipe information
4. **User Dashboard**: Show user's own recipes with edit/delete options
5. **Error Handling**: Proper feedback for upload failures

### Bonus Points

- Add image filters (black & white, sepia, etc.)
- Implement image cropping during upload
- Add a "featured recipe" system
- Include recipe rating system
- Add search functionality by recipe name or ingredients

### Evaluation Criteria

- **Functionality** (40%): All upload and display features work correctly
- **Security** (20%): Proper file validation and user authentication
- **Code Quality** (20%): Clean, well-commented code following Django best practices
- **User Experience** (20%): Intuitive interface with proper error handling

### Submission Instructions

1. Create a GitHub repository with your complete Django project
2. Include a README.md with setup instructions
3. Add sample images for testing
4. Include a requirements.txt file
5. Write a brief explanation of your image processing choices

This assignment will test your understanding of file uploads, image processing, security validation, and Django best practices - all while creating a practical, real-world application that could be deployed to production!

---

## Key Takeaways

Today you learned to be a master chef of file handling:

1. **File Uploads**: Set up proper receiving areas for user files
2. **Media Configuration**: Organize your storage system efficiently
3. **Security & Validation**: Implement quality control for all uploads
4. **Image Processing**: Use Pillow to prepare images for optimal display

Just like running a successful kitchen, handling file uploads requires attention to security, organization, and quality. With these skills, you can now safely accept, process, and serve user-generated content in your Django applications!

Remember: Always validate, never trust user input completely, and process images to ensure optimal performance. Your users will thank you for the smooth, secure experience!