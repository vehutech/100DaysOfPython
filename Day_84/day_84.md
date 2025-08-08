# AI Mastery Course: Day 84 - Generative Models Introduction

## Learning Objective
By the end of this lesson, you will understand the fundamental concepts of generative models, implement basic autoencoders and GANs using Python and Django, and be able to explore latent spaces for creative AI applications.

---

## Introduction

Imagine that you're the head chef in a revolutionary restaurant where instead of following traditional recipes, you've discovered the secret to creating entirely new dishes by understanding the essence of flavors and reconstructing them from scratch. Your kitchen has two magical stations: one that can break down any dish into its core flavor components (encoding), and another that can recreate dishes from these components (decoding). But here's where it gets interesting - you've also hired a creative assistant chef who constantly challenges you to make your recreated dishes so realistic that even the most discerning food critic can't tell if it's an original or your recreation.

This is exactly what we're exploring today in the world of generative models - sophisticated AI systems that learn to understand and recreate data patterns with remarkable creativity and precision.

---

## 1. Autoencoders and Variational Autoencoders

### Understanding Autoencoders

Think of an autoencoder as your master recipe compression system. Just as a skilled chef can identify the essential ingredients and techniques that define a signature dish, an autoencoder learns to compress data into its most important features and then reconstruct it.

```python
# Django project structure for our AI models
# models/autoencoder.py

import torch
import torch.nn as nn
import torch.optim as optim
from django.db import models
from django.core.files.storage import default_storage
import numpy as np

class AutoencoderModel(nn.Module):
    """
    Basic Autoencoder - like a recipe that compresses a complex dish 
    into its essential components and reconstructs it
    """
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        super(AutoencoderModel, self).__init__()
        
        # Encoder: Breaks down input into essential components
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),    # First compression layer
            nn.ReLU(True),                       # Activation function
            nn.Linear(hidden_dim, latent_dim)    # Final compression to essence
        )
        
        # Decoder: Reconstructs from essential components
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),   # Expand from essence
            nn.ReLU(True),                       # Activation function
            nn.Linear(hidden_dim, input_dim),    # Full reconstruction
            nn.Sigmoid()                         # Output activation
        )
    
    def forward(self, x):
        # Encode: Extract the essence
        latent = self.encoder(x)
        # Decode: Reconstruct from essence
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Django model to store our trained autoencoders
class AutoencoderRecord(models.Model):
    name = models.CharField(max_length=100)
    model_file = models.FileField(upload_to='ai_models/')
    input_dimension = models.IntegerField()
    latent_dimension = models.IntegerField()
    training_loss = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Autoencoder: {self.name}"
```

**Syntax Explanation:**
- `nn.Sequential()`: Creates a container that runs layers in sequence, like following recipe steps in order
- `nn.Linear(input_dim, hidden_dim)`: A fully connected layer that transforms input size to hidden size
- `nn.ReLU(True)`: Rectified Linear Unit activation - keeps positive values, zeros out negatives
- `super().__init__()`: Calls the parent class constructor to properly initialize our neural network

### Variational Autoencoders (VAEs)

Now, imagine your compression system doesn't just store exact recipes, but learns to understand the probability distributions of flavors - creating a more flexible system that can generate variations.

```python
# models/vae.py

class VariationalAutoencoder(nn.Module):
    """
    VAE - Like understanding the probability of flavor combinations
    rather than exact recipes, allowing for creative variations
    """
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder network
        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)      # Mean of distribution
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Extract probability distributions of features"""
        hidden = torch.relu(self.encoder_hidden(x))
        mu = self.encoder_mu(hidden)          # Mean of the latent distribution
        logvar = self.encoder_logvar(hidden)  # Log variance for numerical stability
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample from the distribution - the creative part"""
        if self.training:
            std = torch.exp(0.5 * logvar)     # Convert log variance to standard deviation
            eps = torch.randn_like(std)       # Random noise
            return mu + eps * std             # Sample: mean + noise * standard deviation
        else:
            return mu
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)   # Sample from latent space
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

def vae_loss(reconstructed, original, mu, logvar):
    """
    VAE loss combines reconstruction quality with regularization
    Like balancing taste accuracy with creative flexibility
    """
    # Reconstruction loss - how well we recreated the original
    recon_loss = nn.functional.binary_cross_entropy(reconstructed, original, reduction='sum')
    
    # KL divergence - keeps our latent space well-structured for generation
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss
```

**Syntax Explanation:**
- `torch.randn_like(std)`: Creates random numbers with the same shape as `std`
- `torch.exp(0.5 * logvar)`: Exponential function to convert log variance to standard deviation
- `nn.functional.binary_cross_entropy()`: Loss function measuring difference between reconstructed and original
- `.pow(2)`: Element-wise square operation
- `.exp()`: Element-wise exponential operation

---

## 2. Generative Adversarial Networks (GANs) Basics

GANs are like having two chefs in constant competition: one trying to create the most convincing dishes (Generator), and another trying to detect which dishes are authentic versus recreated (Discriminator).

```python
# models/gan.py

class Generator(nn.Module):
    """
    The creative chef - generates new data from random noise
    """
    def __init__(self, noise_dim=100, hidden_dim=128, output_dim=784):
        super(Generator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, noise):
        return self.network(noise)

class Discriminator(nn.Module):
    """
    The critic chef - determines if data is real or generated
    """
    def __init__(self, input_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),      # Slightly different activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()            # Output probability (0-1)
        )
    
    def forward(self, x):
        return self.network(x)

# Django view to handle GAN training
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def train_gan(request):
    """
    Django view to orchestrate the competition between Generator and Discriminator
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 64)
        
        # Initialize our competing networks
        generator = Generator()
        discriminator = Discriminator()
        
        # Optimizers - the learning strategies for each chef
        gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        
        training_history = []
        
        for epoch in range(epochs):
            # Train Discriminator: teach it to distinguish real from fake
            disc_optimizer.zero_grad()
            
            # Real data loss
            real_data = get_real_batch(batch_size)  # Your real data
            real_labels = torch.ones(batch_size, 1)
            real_output = discriminator(real_data)
            real_loss = criterion(real_output, real_labels)
            
            # Fake data loss
            noise = torch.randn(batch_size, 100)
            fake_data = generator(noise)
            fake_labels = torch.zeros(batch_size, 1)
            fake_output = discriminator(fake_data.detach())  # Detach to avoid training generator
            fake_loss = criterion(fake_output, fake_labels)
            
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            disc_optimizer.step()
            
            # Train Generator: teach it to fool the discriminator
            gen_optimizer.zero_grad()
            
            noise = torch.randn(batch_size, 100)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data)  # Don't detach - we want gradients
            gen_labels = torch.ones(batch_size, 1)  # Generator wants discriminator to think it's real
            gen_loss = criterion(fake_output, gen_labels)
            
            gen_loss.backward()
            gen_optimizer.step()
            
            if epoch % 10 == 0:
                training_history.append({
                    'epoch': epoch,
                    'disc_loss': disc_loss.item(),
                    'gen_loss': gen_loss.item()
                })
        
        return JsonResponse({
            'status': 'success',
            'training_history': training_history
        })
```

**Syntax Explanation:**
- `nn.LeakyReLU(0.2)`: Like ReLU but allows small negative values (0.2 * negative_input)
- `nn.Tanh()`: Hyperbolic tangent activation, outputs between -1 and 1
- `optimizer.zero_grad()`: Clears previous gradients before backpropagation
- `.detach()`: Removes tensor from computation graph, preventing gradient flow
- `.item()`: Extracts scalar value from single-element tensor

---

## 3. Image Generation and Manipulation

Let's create a Django application that handles image generation like a master chef's creative station:

```python
# views/image_generation.py

from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

def generate_image(request):
    """
    Like having a station where chefs can create new visual dishes
    """
    if request.method == 'POST':
        # Load your trained model
        generator = Generator()
        generator.load_state_dict(torch.load('path/to/trained/generator.pth'))
        generator.eval()
        
        # Generate new image from random ingredients (noise)
        with torch.no_grad():
            noise = torch.randn(1, 100)  # Random seed
            generated_image = generator(noise)
            
            # Convert tensor to image
            generated_image = generated_image.view(28, 28)  # Assuming MNIST-like data
            generated_image = generated_image.cpu().numpy()
            
            # Create image using matplotlib
            plt.figure(figsize=(5, 5))
            plt.imshow(generated_image, cmap='gray')
            plt.axis('off')
            
            # Save to memory buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            buffer.seek(0)
            
            # Convert to base64 for web display
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return render(request, 'ai_app/generated_image.html', {
                'generated_image': image_base64
            })
    
    return render(request, 'ai_app/generate.html')

def interpolate_images(request):
    """
    Like blending two signature dishes to create something in between
    """
    if request.method == 'POST':
        # Get two latent vectors
        z1 = torch.randn(1, 100)
        z2 = torch.randn(1, 100)
        
        # Create interpolation steps
        steps = 10
        interpolated_images = []
        
        generator = Generator()
        generator.load_state_dict(torch.load('path/to/trained/generator.pth'))
        generator.eval()
        
        for i in range(steps):
            # Linear interpolation
            alpha = i / (steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            with torch.no_grad():
                generated = generator(z_interp)
                image_array = generated.view(28, 28).cpu().numpy()
                
                # Convert to base64
                plt.figure(figsize=(3, 3))
                plt.imshow(image_array, cmap='gray')
                plt.axis('off')
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                interpolated_images.append(image_base64)
        
        return render(request, 'ai_app/interpolation.html', {
            'interpolated_images': interpolated_images
        })
```

**Syntax Explanation:**
- `torch.no_grad()`: Context manager that disables gradient computation for efficiency during inference
- `io.BytesIO()`: Creates an in-memory binary stream for image data
- `base64.b64encode()`: Converts binary data to base64 string for web transmission
- `plt.savefig()`: Saves matplotlib figure to specified format and location

---

## 4. Latent Space Exploration

The latent space is like your flavor profile map - a multi-dimensional space where similar "tastes" are close together:

```python
# utils/latent_exploration.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px

class LatentSpaceExplorer:
    """
    Tool for exploring the hidden dimensions of your generative models
    Like mapping out all possible flavor combinations in your creative space
    """
    
    def __init__(self, model):
        self.model = model
        self.latent_vectors = []
        self.labels = []
    
    def collect_latent_representations(self, dataloader, num_samples=1000):
        """
        Collect latent representations from your trained model
        """
        self.model.eval()
        collected = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                if collected >= num_samples:
                    break
                
                # For VAE, we get mu and logvar
                if hasattr(self.model, 'encode'):
                    mu, logvar = self.model.encode(batch_data)
                    latent = self.model.reparameterize(mu, logvar)
                else:
                    # For regular autoencoder
                    _, latent = self.model(batch_data)
                
                self.latent_vectors.extend(latent.cpu().numpy())
                self.labels.extend(batch_labels.cpu().numpy())
                collected += len(batch_data)
        
        self.latent_vectors = np.array(self.latent_vectors)
        self.labels = np.array(self.labels)
    
    def visualize_2d(self, method='tsne'):
        """
        Create 2D visualization of latent space
        Like creating a map of your flavor combinations
        """
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, random_state=42)
        
        # Reduce dimensionality
        latent_2d = reducer.fit_transform(self.latent_vectors)
        
        # Create interactive plot
        fig = px.scatter(
            x=latent_2d[:, 0], 
            y=latent_2d[:, 1], 
            color=self.labels,
            title=f'Latent Space Visualization ({method.upper()})',
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'}
        )
        
        return fig.to_html()
    
    def find_similar_points(self, target_vector, k=5):
        """
        Find similar points in latent space
        Like finding dishes with similar flavor profiles
        """
        from scipy.spatial.distance import cdist
        
        distances = cdist([target_vector], self.latent_vectors)[0]
        similar_indices = np.argsort(distances)[:k]
        
        return similar_indices, distances[similar_indices]
    
    def generate_interpolation_path(self, start_vector, end_vector, steps=10):
        """
        Create smooth transition between two points in latent space
        """
        interpolation_vectors = []
        for i in range(steps):
            alpha = i / (steps - 1)
            interpolated = (1 - alpha) * start_vector + alpha * end_vector
            interpolation_vectors.append(interpolated)
        
        return np.array(interpolation_vectors)

# Django view for latent space exploration
def explore_latent_space(request):
    """
    Interactive latent space exploration interface
    """
    if request.method == 'GET':
        # Load your trained VAE
        vae = VariationalAutoencoder()
        vae.load_state_dict(torch.load('path/to/trained/vae.pth'))
        
        explorer = LatentSpaceExplorer(vae)
        
        # You would load your actual dataloader here
        # explorer.collect_latent_representations(dataloader)
        
        # Create visualization
        # plot_html = explorer.visualize_2d(method='tsne')
        
        return render(request, 'ai_app/latent_exploration.html', {
            # 'plot_html': plot_html
        })
```

**Syntax Explanation:**
- `hasattr(object, 'attribute')`: Checks if an object has a specific attribute or method
- `np.argsort()`: Returns indices that would sort an array
- `cdist([vector], matrix)`: Computes distances between a point and all points in a matrix
- Context managers (`with torch.no_grad():`): Automatically handle setup and cleanup

---

## Final Quality Project: AI Art Gallery

Now let's create a comprehensive Django application that brings together all our concepts - like opening a revolutionary restaurant that showcases all your generative cooking techniques:

```python
# Final Project Structure:
# ai_gallery/
# ├── models.py          # Database models for storing artworks and models
# ├── views.py           # Views for generation, display, and exploration
# ├── urls.py            # URL routing
# ├── templates/         # HTML templates
# └── static/            # CSS, JS, and generated images

# models.py
class GeneratedArtwork(models.Model):
    GENERATION_METHODS = [
        ('VAE', 'Variational Autoencoder'),
        ('GAN', 'Generative Adversarial Network'),
        ('AUTOENCODER', 'Standard Autoencoder'),
    ]
    
    title = models.CharField(max_length=200)
    generation_method = models.CharField(max_length=20, choices=GENERATION_METHODS)
    image_file = models.ImageField(upload_to='generated_art/')
    latent_vector = models.JSONField()  # Store the latent representation
    creation_timestamp = models.DateTimeField(auto_now_add=True)
    likes = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-creation_timestamp']

# views.py
class AIArtGalleryView(View):
    """
    Main gallery showcasing all generated artworks
    """
    def get(self, request):
        artworks = GeneratedArtwork.objects.all()[:50]  # Latest 50 artworks
        
        context = {
            'artworks': artworks,
            'generation_stats': {
                'total_artworks': GeneratedArtwork.objects.count(),
                'vae_count': GeneratedArtwork.objects.filter(generation_method='VAE').count(),
                'gan_count': GeneratedArtwork.objects.filter(generation_method='GAN').count(),
            }
        }
        return render(request, 'ai_gallery/gallery.html', context)

class CreateArtworkView(View):
    """
    Interactive artwork creation interface
    """
    def get(self, request):
        return render(request, 'ai_gallery/create.html')
    
    def post(self, request):
        method = request.POST.get('method', 'VAE')
        creativity_level = float(request.POST.get('creativity', 0.5))
        
        # Generate artwork based on selected method
        if method == 'VAE':
            artwork_data = self.generate_vae_artwork(creativity_level)
        elif method == 'GAN':
            artwork_data = self.generate_gan_artwork()
        else:
            artwork_data = self.generate_autoencoder_artwork()
        
        # Save artwork to database
        artwork = GeneratedArtwork.objects.create(
            title=f"AI Creation {timezone.now().strftime('%Y%m%d_%H%M%S')}",
            generation_method=method,
            latent_vector=artwork_data['latent_vector'],
        )
        
        # Save image file
        artwork.image_file.save(
            f"artwork_{artwork.id}.png",
            artwork_data['image_file'],
            save=True
        )
        
        return JsonResponse({
            'success': True,
            'artwork_id': artwork.id,
            'image_url': artwork.image_file.url
        })
    
    def generate_vae_artwork(self, creativity_level):
        """Generate artwork using VAE with controllable creativity"""
        vae = VariationalAutoencoder()
        vae.load_state_dict(torch.load('models/trained_vae.pth'))
        vae.eval()
        
        with torch.no_grad():
            # Generate random latent vector
            z = torch.randn(1, 32) * creativity_level
            generated = vae.decoder(z)
            
            # Convert to image
            image_array = generated.view(28, 28).cpu().numpy()
            
            return {
                'image_file': self.array_to_file(image_array),
                'latent_vector': z.cpu().numpy().tolist()
            }

# urls.py
from django.urls import path
from . import views

app_name = 'ai_gallery'
urlpatterns = [
    path('', views.AIArtGalleryView.as_view(), name='gallery'),
    path('create/', views.CreateArtworkView.as_view(), name='create'),
    path('explore/', views.explore_latent_space, name='explore'),
    path('interpolate/<int:artwork1_id>/<int:artwork2_id>/', 
         views.InterpolateArtworkView.as_view(), name='interpolate'),
]
```

---

# Simple Image Generator Project

## Project Overview
You'll create a Django web application that generates simple images using Python's PIL (Pillow) library. Think of this as your digital art station where you can craft various visual recipes - from abstract patterns to geometric designs.

## Project Structure
```
image_generator/
├── manage.py
├── image_generator/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── generator/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── templates/
│       └── generator/
│           ├── index.html
│           └── gallery.html
├── static/
│   └── css/
│       └── style.css
└── media/
    └── generated/
```

## Step 1: Django Setup

**settings.py**
```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'generator',
]

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / "static",
]
```

**Main urls.py**
```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('generator.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

## Step 2: Models

**generator/models.py**
```python
from django.db import models
from django.utils import timezone
import uuid

class GeneratedImage(models.Model):
    PATTERN_CHOICES = [
        ('mandala', 'Mandala Pattern'),
        ('geometric', 'Geometric Shapes'),
        ('gradient', 'Color Gradient'),
        ('noise', 'Random Noise'),
        ('spiral', 'Spiral Pattern'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pattern_type = models.CharField(max_length=20, choices=PATTERN_CHOICES)
    width = models.IntegerField(default=400)
    height = models.IntegerField(default=400)
    primary_color = models.CharField(max_length=7, default='#FF0000')
    secondary_color = models.CharField(max_length=7, default='#0000FF')
    image_file = models.ImageField(upload_to='generated/')
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.pattern_type} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
```

## Step 3: Forms

**generator/forms.py**
```python
from django import forms
from .models import GeneratedImage

class ImageGeneratorForm(forms.ModelForm):
    class Meta:
        model = GeneratedImage
        fields = ['pattern_type', 'width', 'height', 'primary_color', 'secondary_color']
        widgets = {
            'pattern_type': forms.Select(attrs={
                'class': 'form-control',
                'id': 'pattern-select'
            }),
            'width': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '100',
                'max': '800',
                'value': '400'
            }),
            'height': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '100',
                'max': '800',
                'value': '400'
            }),
            'primary_color': forms.TextInput(attrs={
                'type': 'color',
                'class': 'form-control',
                'value': '#FF0000'
            }),
            'secondary_color': forms.TextInput(attrs={
                'type': 'color',
                'class': 'form-control',
                'value': '#0000FF'
            }),
        }
```

## Step 4: Image Generation Logic

**generator/image_generators.py**
```python
from PIL import Image, ImageDraw
import random
import math
import numpy as np

class ImageChef:
    """Your master chef for creating visual delicacies"""
    
    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def create_mandala(width, height, primary_color, secondary_color):
        """Craft an intricate mandala like preparing a delicate pastry"""
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = width // 2, height // 2
        max_radius = min(center_x, center_y) - 20
        
        primary_rgb = ImageChef.hex_to_rgb(primary_color)
        secondary_rgb = ImageChef.hex_to_rgb(secondary_color)
        
        # Create concentric circles like layers of a cake
        for i in range(8):
            radius = max_radius - (i * max_radius // 10)
            color = primary_rgb if i % 2 == 0 else secondary_rgb
            
            # Draw main circle
            draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], outline=color, width=2)
            
            # Add decorative petals
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                x = center_x + radius * 0.8 * math.cos(rad)
                y = center_y + radius * 0.8 * math.sin(rad)
                
                petal_radius = radius // 6
                draw.ellipse([
                    x - petal_radius, y - petal_radius,
                    x + petal_radius, y + petal_radius
                ], fill=color)
        
        return img
    
    @staticmethod
    def create_geometric(width, height, primary_color, secondary_color):
        """Mix geometric shapes like combining ingredients"""
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        primary_rgb = ImageChef.hex_to_rgb(primary_color)
        secondary_rgb = ImageChef.hex_to_rgb(secondary_color)
        
        # Create a grid of shapes
        for i in range(0, width, 60):
            for j in range(0, height, 60):
                shape_type = random.choice(['rectangle', 'ellipse', 'triangle'])
                color = random.choice([primary_rgb, secondary_rgb])
                
                if shape_type == 'rectangle':
                    draw.rectangle([i, j, i+50, j+50], fill=color)
                elif shape_type == 'ellipse':
                    draw.ellipse([i, j, i+50, j+50], fill=color)
                else:  # triangle
                    points = [(i+25, j), (i, j+50), (i+50, j+50)]
                    draw.polygon(points, fill=color)
        
        return img
    
    @staticmethod
    def create_gradient(width, height, primary_color, secondary_color):
        """Blend colors like creating a smooth sauce"""
        img = Image.new('RGB', (width, height))
        
        primary_rgb = ImageChef.hex_to_rgb(primary_color)
        secondary_rgb = ImageChef.hex_to_rgb(secondary_color)
        
        for y in range(height):
            # Calculate blend ratio
            ratio = y / height
            
            # Interpolate between colors
            r = int(primary_rgb[0] * (1 - ratio) + secondary_rgb[0] * ratio)
            g = int(primary_rgb[1] * (1 - ratio) + secondary_rgb[1] * ratio)
            b = int(primary_rgb[2] * (1 - ratio) + secondary_rgb[2] * ratio)
            
            # Draw horizontal line
            for x in range(width):
                img.putpixel((x, y), (r, g, b))
        
        return img
    
    @staticmethod
    def create_noise(width, height, primary_color, secondary_color):
        """Sprinkle random elements like seasoning a dish"""
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        primary_rgb = ImageChef.hex_to_rgb(primary_color)
        secondary_rgb = ImageChef.hex_to_rgb(secondary_color)
        
        # Add random colored pixels
        for _ in range(width * height // 4):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            color = random.choice([primary_rgb, secondary_rgb])
            
            # Create small clusters
            cluster_size = random.randint(1, 5)
            for dx in range(-cluster_size, cluster_size+1):
                for dy in range(-cluster_size, cluster_size+1):
                    if 0 <= x+dx < width and 0 <= y+dy < height:
                        img.putpixel((x+dx, y+dy), color)
        
        return img
    
    @staticmethod
    def create_spiral(width, height, primary_color, secondary_color):
        """Draw spirals like piping decorative cream"""
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = width // 2, height // 2
        primary_rgb = ImageChef.hex_to_rgb(primary_color)
        secondary_rgb = ImageChef.hex_to_rgb(secondary_color)
        
        # Draw multiple spirals
        for spiral in range(3):
            points = []
            for t in range(0, 720, 2):  # 2 full rotations
                angle = math.radians(t + spiral * 120)  # Offset each spiral
                radius = t * 0.15
                
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                
                if 0 <= x < width and 0 <= y < height:
                    points.append((x, y))
            
            # Draw the spiral line
            if len(points) > 1:
                color = primary_rgb if spiral % 2 == 0 else secondary_rgb
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i+1]], fill=color, width=3)
        
        return img
```

## Step 5: Views

**generator/views.py**
```python
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.base import ContentFile
from django.core.paginator import Paginator
from .models import GeneratedImage
from .forms import ImageGeneratorForm
from .image_generators import ImageChef
import io
import uuid

def index(request):
    """Main cooking station where images are crafted"""
    if request.method == 'POST':
        form = ImageGeneratorForm(request.POST)
        if form.is_valid():
            # Get the recipe ingredients
            pattern_type = form.cleaned_data['pattern_type']
            width = form.cleaned_data['width']
            height = form.cleaned_data['height']
            primary_color = form.cleaned_data['primary_color']
            secondary_color = form.cleaned_data['secondary_color']
            
            # Select the appropriate cooking method
            chef_methods = {
                'mandala': ImageChef.create_mandala,
                'geometric': ImageChef.create_geometric,
                'gradient': ImageChef.create_gradient,
                'noise': ImageChef.create_noise,
                'spiral': ImageChef.create_spiral,
            }
            
            # Cook up the image
            img = chef_methods[pattern_type](width, height, primary_color, secondary_color)
            
            # Save the masterpiece
            img_io = io.BytesIO()
            img.save(img_io, format='PNG')
            img_io.seek(0)
            
            # Create database record
            generated_image = form.save(commit=False)
            filename = f"{pattern_type}_{uuid.uuid4().hex[:8]}.png"
            generated_image.image_file.save(
                filename,
                ContentFile(img_io.getvalue()),
                save=True
            )
            
            return redirect('gallery')
    else:
        form = ImageGeneratorForm()
    
    return render(request, 'generator/index.html', {'form': form})

def gallery(request):
    """Display the collection of culinary creations"""
    images = GeneratedImage.objects.all()
    paginator = Paginator(images, 9)  # Show 9 images per page
    
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'generator/gallery.html', {'page_obj': page_obj})

def generate_preview(request):
    """Quick taste test of the image before final preparation"""
    if request.method == 'POST':
        pattern_type = request.POST.get('pattern_type', 'mandala')
        primary_color = request.POST.get('primary_color', '#FF0000')
        secondary_color = request.POST.get('secondary_color', '#0000FF')
        
        # Create small preview (150x150)
        chef_methods = {
            'mandala': ImageChef.create_mandala,
            'geometric': ImageChef.create_geometric,
            'gradient': ImageChef.create_gradient,
            'noise': ImageChef.create_noise,
            'spiral': ImageChef.create_spiral,
        }
        
        img = chef_methods[pattern_type](150, 150, primary_color, secondary_color)
        
        # Convert to base64 for JSON response
        import base64
        img_io = io.BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        img_b64 = base64.b64encode(img_io.getvalue()).decode()
        
        return JsonResponse({
            'success': True,
            'image_data': f'data:image/png;base64,{img_b64}'
        })
    
    return JsonResponse({'success': False})
```

## Step 6: URL Configuration

**generator/urls.py**
```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('gallery/', views.gallery, name='gallery'),
    path('preview/', views.generate_preview, name='generate_preview'),
]
```

## Step 7: Templates

**generator/templates/generator/index.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Generator - Create Your Masterpiece</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .preview-container {
            min-height: 200px;
            border: 2px dashed #dee2e6;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        .pattern-card {
            transition: transform 0.3s ease;
        }
        .pattern-card:hover {
            transform: translateY(-5px);
        }
    </style>
</head>
<body class="bg-light">
    <div class="container mt-5">
        <div class="row">
            <div class="col-md-12 text-center mb-4">
                <h1 class="display-4">Visual Recipe Creator</h1>
                <p class="lead">Mix colors and patterns to create stunning digital art</p>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card shadow-sm">
                    <div class="card-header bg-primary text-white">
                        <h5>Recipe Configuration</h5>
                    </div>
                    <div class="card-body">
                        <form method="post" id="generator-form">
                            {% csrf_token %}
                            
                            <div class="mb-3">
                                <label for="{{ form.pattern_type.id_for_label }}" class="form-label">Pattern Style</label>
                                {{ form.pattern_type }}
                            </div>
                            
                            <div class="row">
                                <div class="col-6">
                                    <div class="mb-3">
                                        <label for="{{ form.width.id_for_label }}" class="form-label">Width</label>
                                        {{ form.width }}
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="mb-3">
                                        <label for="{{ form.height.id_for_label }}" class="form-label">Height</label>
                                        {{ form.height }}
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-6">
                                    <div class="mb-3">
                                        <label for="{{ form.primary_color.id_for_label }}" class="form-label">Primary Color</label>
                                        {{ form.primary_color }}
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="mb-3">
                                        <label for="{{ form.secondary_color.id_for_label }}" class="form-label">Secondary Color</label>
                                        {{ form.secondary_color }}
                                    </div>
                                </div>
                            </div>
                            
                            <button type="button" class="btn btn-outline-primary me-2" onclick="generatePreview()">
                                Preview Recipe
                            </button>
                            <button type="submit" class="btn btn-primary">
                                Create Masterpiece
                            </button>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card shadow-sm">
                    <div class="card-header bg-success text-white">
                        <h5>Preview Station</h5>
                    </div>
                    <div class="card-body">
                        <div class="preview-container" id="preview-container">
                            <div class="text-muted">
                                <i class="fas fa-image fa-3x mb-2"></i>
                                <p>Your preview will appear here</p>
                            </div>
                        </div>
                        <div class="d-grid">
                            <a href="{% url 'gallery' %}" class="btn btn-outline-success">
                                View Gallery
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function generatePreview() {
            const form = document.getElementById('generator-form');
            const formData = new FormData(form);
            
            const previewContainer = document.getElementById('preview-container');
            previewContainer.innerHTML = '<div class="spinner-border text-primary" role="status"></div>';
            
            fetch('/preview/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    previewContainer.innerHTML = `<img src="${data.image_data}" alt="Preview" class="img-fluid rounded">`;
                } else {
                    previewContainer.innerHTML = '<div class="text-danger">Preview failed</div>';
                }
            })
            .catch(error => {
                previewContainer.innerHTML = '<div class="text-danger">Error generating preview</div>';
            });
        }

        // Auto-generate preview when inputs change
        document.addEventListener('change', function(e) {
            if (e.target.form && e.target.form.id === 'generator-form') {
                setTimeout(generatePreview, 300);
            }
        });
    </script>
</body>
</html>
```

**generator/templates/generator/gallery.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gallery - Your Creative Collection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .gallery-item {
            transition: transform 0.3s ease;
        }
        .gallery-item:hover {
            transform: scale(1.05);
        }
        .image-card {
            height: 100%;
        }
    </style>
</head>
<body class="bg-light">
    <div class="container mt-4">
        <div class="row mb-4">
            <div class="col-12">
                <div class="d-flex justify-content-between align-items-center">
                    <h2>Your Culinary Art Gallery</h2>
                    <a href="{% url 'index' %}" class="btn btn-primary">Create New</a>
                </div>
            </div>
        </div>
        
        {% if page_obj %}
        <div class="row g-4">
            {% for image in page_obj %}
            <div class="col-md-4">
                <div class="gallery-item">
                    <div class="card image-card shadow-sm">
                        <img src="{{ image.image_file.url }}" class="card-img-top" alt="{{ image.pattern_type }}" style="height: 250px; object-fit: cover;">
                        <div class="card-body">
                            <h6 class="card-title">{{ image.get_pattern_type_display }}</h6>
                            <p class="card-text">
                                <small class="text-muted">
                                    {{ image.width }}×{{ image.height }} | 
                                    {{ image.created_at|date:"M d, Y H:i" }}
                                </small>
                            </p>
                            <div class="d-flex gap-2">
                                <span class="badge" style="background-color: {{ image.primary_color }};">&nbsp;</span>
                                <span class="badge" style="background-color: {{ image.secondary_color }};">&nbsp;</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Pagination -->
        {% if page_obj.has_other_pages %}
        <nav class="mt-5">
            <ul class="pagination justify-content-center">
                {% if page_obj.has_previous %}
                <li class="page-item">
                    <a class="page-link" href="?page={{ page_obj.previous_page_number }}">Previous</a>
                </li>
                {% endif %}
                
                {% for num in page_obj.paginator.page_range %}
                    {% if page_obj.number == num %}
                    <li class="page-item active">
                        <span class="page-link">{{ num }}</span>
                    </li>
                    {% elif num > page_obj.number|add:'-3' and num < page_obj.number|add:'3' %}
                    <li class="page-item">
                        <a class="page-link" href="?page={{ num }}">{{ num }}</a>
                    </li>
                    {% endif %}
                {% endfor %}
                
                {% if page_obj.has_next %}
                <li class="page-item">
                    <a class="page-link" href="?page={{ page_obj.next_page_number }}">Next</a>
                </li>
                {% endif %}
            </ul>
        </nav>
        {% endif %}
        
        {% else %}
        <div class="row">
            <div class="col-12 text-center">
                <div class="card">
                    <div class="card-body py-5">
                        <h5>No creations yet!</h5>
                        <p class="text-muted">Start creating your first masterpiece</p>
                        <a href="{% url 'index' %}" class="btn btn-primary">Get Cooking</a>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

## Step 8: Database Migration

Run these commands to set up your database:

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

## Step 9: Install Dependencies

```bash
pip install django pillow numpy
```

## Key Learning Concepts Demonstrated

1. **Django MVC Architecture**: Models handle data, Views process logic, Templates render UI
2. **File Handling**: Using Django's FileField to manage generated images
3. **PIL/Pillow Integration**: Creating images programmatically with Python
4. **Form Processing**: Handling user input with Django forms
5. **AJAX Integration**: Real-time preview generation without page refresh
6. **Database Relationships**: Storing metadata about generated images
7. **Static/Media Files**: Proper handling of user-generated content
8. **Pagination**: Managing large collections of generated images
9. **UUID Usage**: Unique identifiers for database records
10. **Mathematical Algorithms**: Converting design concepts into code

This project combines the power of Django's web framework with Python's image processing capabilities, creating a complete full-stack application that generates and manages visual content. The modular design allows easy extension with new pattern types and features.

## Assignment: Personal Style Transfer

**Objective**: Create a Django application that learns your personal "artistic style" and generates new artworks in that style.

**Requirements**:
1. Build a custom VAE that can encode/decode simple drawings or patterns
2. Create a Django interface where users can upload 5-10 sample images of their preferred style
3. Train the VAE on these samples to learn the style patterns
4. Implement a generation endpoint that creates new artworks in the learned style
5. Add a comparison view showing original samples vs. generated variations

**Deliverables**:
- Django app with upload, training, and generation functionality
- Trained VAE model that captures style essence
- Web interface demonstrating before/after style learning
- Brief report explaining how your model captures and reproduces stylistic elements

**Evaluation Criteria**:
- Model successfully learns distinguishing features from uploaded samples
- Generated images show clear stylistic similarity to training samples
- Clean, functional Django interface
- Code demonstrates understanding of VAE principles and latent space manipulation

This assignment differs from our main project by focusing on personalized style learning rather than general image generation, requiring you to work with custom datasets and understand how generative models adapt to specific artistic preferences.

---

## Course Summary

Today you've learned to orchestrate the complex dance between understanding data patterns and generating new creative content. Just as a master chef learns not just to follow recipes but to understand the essence of flavors and create entirely new dishes, you've mastered the art of:

- **Autoencoders**: Compressing data to its essential features and reconstructing it
- **VAEs**: Understanding probability distributions to enable creative generation
- **GANs**: Orchestrating competition between generator and discriminator networks
- **Latent Space**: Navigating the hidden dimensions where creativity lives

Your Django applications now serve as sophisticated creative platforms, capable of generating, exploring, and manipulating AI-created content with the same finesse as a world-class culinary artist.