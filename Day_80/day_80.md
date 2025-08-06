# AI Mastery Course - Day 80: Convolutional Neural Networks (CNNs)

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand and implement convolution and pooling operations from scratch
- Recognize and work with popular CNN architectures (LeNet, AlexNet, VGG)
- Apply transfer learning using pre-trained models for faster training
- Implement data augmentation techniques to improve model robustness
- Create a complete neural network pipeline using Python and Django

---

## Imagine That...

Imagine that you're running the most sophisticated restaurant in the world, where every dish must be identified and classified with absolute precision. Your head chef needs to train a team of specialized sous chefs, each with a unique ability to recognize specific patterns and ingredients in any dish that comes through the door.

Just like how a master chef trains their team to identify the subtle differences between a Bolognese and a Marinara sauce by looking at texture, color, and ingredient distribution, we're going to train our artificial "chefs" (neural networks) to recognize patterns in images with remarkable accuracy.

---

## 1. Convolution and Pooling Operations

### The Art of Pattern Recognition

Think of convolution as teaching your sous chef to examine a dish through a magnifying glass, moving systematically across every inch. The chef uses a special "pattern template" (called a kernel or filter) to identify specific features like edges, textures, or shapes.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define a simple convolution operation
def simple_convolution(image, kernel):
    """
    Performs basic convolution operation
    image: 2D numpy array representing the input image
    kernel: 2D numpy array representing the filter/kernel
    """
    # Get dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate output dimensions
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Initialize output
    output = np.zeros((output_height, output_width))
    
    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest
            roi = image[i:i+kernel_height, j:j+kernel_width]
            # Apply kernel and sum the result
            output[i, j] = np.sum(roi * kernel)
    
    return output

# Example: Edge detection kernel (like a chef detecting crispy edges)
edge_kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

# Sample image (imagine this is a photo of a dish)
sample_image = np.array([[100, 100, 100, 0, 0],
                        [100, 100, 100, 0, 0],
                        [100, 100, 100, 0, 0],
                        [100, 100, 100, 0, 0]])

# Apply convolution
result = simple_convolution(sample_image, edge_kernel)
print("Convolution result:")
print(result)
```

**Syntax Explanation:**
- `np.array()`: Creates a NumPy array, which is like preparing your ingredients in organized containers
- `image.shape`: Gets the dimensions of the image (height, width)
- `np.zeros()`: Creates an array filled with zeros, like preparing empty plates
- The nested loops systematically move the kernel across the image, just like a chef examining every section of a dish

### Pooling: Summarizing the Essential Information

Pooling is like having your chef take notes on only the most important characteristics of each section of the dish, reducing the amount of information while keeping the essential features.

```python
def max_pooling(image, pool_size=2):
    """
    Performs max pooling operation
    image: 2D numpy array
    pool_size: size of the pooling window
    """
    height, width = image.shape
    
    # Calculate output dimensions
    output_height = height // pool_size
    output_width = width // pool_size
    
    # Initialize output
    output = np.zeros((output_height, output_width))
    
    # Perform max pooling
    for i in range(output_height):
        for j in range(output_width):
            # Extract the pooling region
            start_i, start_j = i * pool_size, j * pool_size
            end_i, end_j = start_i + pool_size, start_j + pool_size
            
            # Take the maximum value in this region
            region = image[start_i:end_i, start_j:end_j]
            output[i, j] = np.max(region)
    
    return output

# Example usage
test_image = np.array([[4, 3, 2, 1],
                      [2, 7, 1, 3],
                      [1, 2, 8, 5],
                      [3, 1, 4, 2]])

pooled_result = max_pooling(test_image, pool_size=2)
print("Max pooling result:")
print(pooled_result)
```

**Syntax Explanation:**
- `//` performs integer division (floor division)
- `np.max()`: Finds the maximum value in an array, like identifying the strongest flavor in a section

---

## 2. CNN Architectures: The Master Chef Lineage

### LeNet: The Pioneer Chef

LeNet was one of the first successful CNN architectures, like the grandfather of modern culinary techniques.

```python
def create_lenet():
    """
    Creates a LeNet-5 inspired model
    Perfect for simple image recognition tasks
    """
    model = models.Sequential([
        # First course: Feature extraction
        layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 1)),
        layers.AveragePooling2D((2, 2)),
        
        # Second course: Deeper features
        layers.Conv2D(16, (5, 5), activation='tanh'),
        layers.AveragePooling2D((2, 2)),
        
        # Final preparation: Classification
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')  # 10 dish categories
    ])
    
    return model

# Create and display the model
lenet_model = create_lenet()
lenet_model.summary()
```

### AlexNet: The Revolutionary Chef

```python
def create_alexnet_inspired():
    """
    Creates an AlexNet-inspired model (simplified for demonstration)
    Like a chef who introduced bold new techniques
    """
    model = models.Sequential([
        # First revolutionary layer
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', 
                     input_shape=(224, 224, 3)),
        layers.MaxPooling2D((3, 3), strides=2),
        
        # Second innovation
        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        
        # Third breakthrough
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        
        # Final classification layers
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),  # Prevents overconfidence
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1000, activation='softmax')
    ])
    
    return model

# Note: This model would be quite large for demonstration
print("AlexNet architecture created successfully!")
```

**Syntax Explanation:**
- `models.Sequential()`: Creates a linear stack of layers, like following a recipe step by step
- `layers.Conv2D()`: Creates a 2D convolution layer with specified number of filters and kernel size
- `strides`: How far the filter moves each step (like how carefully a chef examines each section)
- `padding='same'`: Adds padding to maintain input dimensions
- `layers.Dropout()`: Randomly sets some neurons to zero during training to prevent overfitting

### VGG: The Precision Master

```python
def create_vgg_block(filters, num_layers):
    """
    Creates a VGG block - like a standardized cooking technique
    """
    block = []
    for _ in range(num_layers):
        block.extend([
            layers.Conv2D(filters, (3, 3), padding='same', activation='relu'),
        ])
    block.append(layers.MaxPooling2D((2, 2), strides=2))
    return block

def create_vgg16_inspired():
    """
    Creates a VGG16-inspired model
    Like a chef known for consistent, reliable techniques
    """
    model_layers = []
    
    # VGG blocks (like different cooking stages)
    model_layers.extend(create_vgg_block(64, 2))   # Block 1
    model_layers.extend(create_vgg_block(128, 2))  # Block 2
    model_layers.extend(create_vgg_block(256, 3))  # Block 3
    model_layers.extend(create_vgg_block(512, 3))  # Block 4
    model_layers.extend(create_vgg_block(512, 3))  # Block 5
    
    # Classification layers
    model_layers.extend([
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1000, activation='softmax')
    ])
    
    model = models.Sequential(model_layers)
    return model

# Create VGG model
vgg_model = create_vgg16_inspired()
print("VGG16-inspired model created successfully!")
```

---

## 3. Transfer Learning: Learning from Master Chefs

Transfer learning is like having a world-renowned chef teach your kitchen staff. Instead of learning everything from scratch, they build upon proven techniques.

```python
# Using a pre-trained model (like hiring an experienced chef)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

def create_transfer_learning_model(num_classes=10):
    """
    Creates a model using transfer learning
    Like having a master chef adapt their skills to your restaurant's menu
    """
    # Load pre-trained VGG16 model (trained master chef)
    base_model = VGG16(weights='imagenet',  # Pre-trained on ImageNet
                      include_top=False,    # Exclude final classification layer
                      input_shape=(224, 224, 3))
    
    # Freeze the base model layers (respect the master's techniques)
    base_model.trainable = False
    
    # Add custom classification layers (adapt to your menu)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

# Create transfer learning model
transfer_model, base = create_transfer_learning_model(num_classes=5)
transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Transfer learning model ready!")
print(f"Total parameters: {transfer_model.count_params():,}")
print(f"Trainable parameters: {sum([tf.keras.utils.count_params(w) for w in transfer_model.trainable_weights]):,}")
```

**Syntax Explanation:**
- `weights='imagenet'`: Uses weights from a model trained on ImageNet dataset
- `include_top=False`: Excludes the final classification layers
- `base_model.trainable = False`: Freezes the pre-trained layers so they don't change during training
- `GlobalAveragePooling2D()`: Takes the average of each feature map

---

## 4. Data Augmentation: Varying the Ingredients

Data augmentation is like teaching your chefs to recognize dishes even when they're presented differently - rotated, slightly different lighting, or from different angles.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def create_data_augmentation():
    """
    Creates data augmentation pipeline
    Like training chefs to recognize dishes under various conditions
    """
    datagen = ImageDataGenerator(
        rotation_range=20,          # Rotate dishes up to 20 degrees
        width_shift_range=0.2,      # Shift horizontally
        height_shift_range=0.2,     # Shift vertically
        horizontal_flip=True,       # Flip like a mirror image
        zoom_range=0.2,            # Zoom in/out
        brightness_range=[0.8, 1.2], # Vary lighting conditions
        fill_mode='nearest'         # How to fill missing pixels
    )
    return datagen

# Modern approach using tf.keras.utils
def create_modern_augmentation():
    """
    Modern data augmentation using tf.keras layers
    """
    augmentation = models.Sequential([
        layers.RandomRotation(0.1),      # Random rotation
        layers.RandomZoom(0.1),          # Random zoom
        layers.RandomFlip("horizontal"), # Random horizontal flip
        layers.RandomBrightness(0.1),    # Random brightness adjustment
    ])
    return augmentation

# Demonstration function
def demonstrate_augmentation():
    """
    Shows how data augmentation works
    """
    # Create sample data (imagine these are dish photos)
    sample_data = tf.random.normal((32, 224, 224, 3))  # 32 sample images
    
    # Apply augmentation
    augmentation = create_modern_augmentation()
    augmented_data = augmentation(sample_data, training=True)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Augmented data shape: {augmented_data.shape}")
    print("Data augmentation applied successfully!")

demonstrate_augmentation()

# Comprehensive CNN with augmentation
def create_complete_cnn_with_augmentation(input_shape=(224, 224, 3), num_classes=10):
    """
    Creates a complete CNN model with built-in data augmentation
    Like a fully equipped restaurant with versatile chefs
    """
    model = models.Sequential([
        # Data augmentation layers (built into the model)
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),
        
        # Feature extraction layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create the complete model
complete_model = create_complete_cnn_with_augmentation()
complete_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Complete CNN model with augmentation created!")
complete_model.summary()
```

**Syntax Explanation:**
- `ImageDataGenerator()`: Creates a generator that applies random transformations
- `rotation_range`: Maximum degrees to rotate images
- `fill_mode='nearest'`: Strategy for filling in newly created pixels
- `training=True`: Applies augmentation only during training, not during evaluation

---

##Project: Restaurant Dish Classification System

Now let's create a complete image classification system using Django as our web framework - think of it as building the complete restaurant management system.

```python
# models.py - Django model for our restaurant system
from django.db import models
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

class DishClassifier:
    """
    Main classifier class - like the head chef's expertise system
    """
    def __init__(self):
        self.model = None
        self.class_names = ['Pizza', 'Burger', 'Pasta', 'Salad', 'Soup']
        self.load_model()
    
    def load_model(self):
        """Load or create the trained model"""
        try:
            # In a real scenario, you'd load a pre-trained model
            self.model = self.create_demo_model()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def create_demo_model(self):
        """Creates a demonstration model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def preprocess_image(self, image_data):
        """
        Preprocess image for prediction
        Like preparing ingredients before cooking
        """
        # Convert base64 to image if needed
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        
        # Open and resize image
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict_dish(self, image_data):
        """
        Predict the type of dish
        Like having a chef identify a dish instantly
        """
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_data)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            return {
                'dish': self.class_names[predicted_class_idx],
                'confidence': confidence,
                'all_predictions': {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(len(self.class_names))
                }
            }
        except Exception as e:
            return {'error': str(e)}

# Django Model
class DishImage(models.Model):
    """
    Database model to store dish images and predictions
    """
    image = models.ImageField(upload_to='dishes/')
    predicted_dish = models.CharField(max_length=100, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.predicted_dish} ({self.confidence:.2f})"

# views.py - Django views
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json

class DishClassificationView(View):
    """
    Main view for dish classification
    Like the restaurant's order processing system
    """
    
    def __init__(self):
        super().__init__()
        self.classifier = DishClassifier()
    
    def get(self, request):
        """Render the main classification page"""
        return render(request, 'classifier/index.html')
    
    @method_decorator(csrf_exempt)
    def post(self, request):
        """Handle image upload and classification"""
        try:
            # Get uploaded image
            if 'image' in request.FILES:
                image_file = request.FILES['image']
                image_data = image_file.read()
                
                # Classify the dish
                result = self.classifier.predict_dish(image_data)
                
                # Save to database (optional)
                if 'error' not in result:
                    dish_image = DishImage.objects.create(
                        image=image_file,
                        predicted_dish=result['dish'],
                        confidence=result['confidence']
                    )
                
                return JsonResponse(result)
            
            return JsonResponse({'error': 'No image provided'})
            
        except Exception as e:
            return JsonResponse({'error': str(e)})

# Training script for the complete system
def train_restaurant_classifier():
    """
    Complete training pipeline
    Like training your entire kitchen staff
    """
    print("Starting restaurant dish classifier training...")
    
    # Create the model architecture
    model = tf.keras.Sequential([
        # Data augmentation
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomFlip("horizontal"),
        
        # Feature extraction
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Classification
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')  # 5 dish types
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Training configuration
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
        )
    ]
    
    print("Training setup complete!")
    print("In a real scenario, you would now load your dataset and train the model.")
    
    return model

# Run the training setup
if __name__ == "__main__":
    trained_model = train_restaurant_classifier()
```

**Complete Syntax Explanation:**

1. **Django Models**: `models.Model` creates database tables to store our data
2. **ImageField**: Special Django field for handling image uploads
3. **Class-based Views**: `View` provides a clean way to handle HTTP requests
4. **Method Decorators**: `@method_decorator(csrf_exempt)` disables CSRF protection for API endpoints
5. **File Handling**: `request.FILES['image']` accesses uploaded files
6. **JSON Responses**: `JsonResponse()` returns JSON data to the frontend
7. **Callbacks**: TensorFlow callbacks for monitoring and controlling training
8. **Global Average Pooling**: Alternative to flattening that reduces parameters

---

## Project: FoodVision - AI-Powered Recipe Classifier

In this project, you'll create a complete web application that can identify different types of food from images and suggest recipes - like having a master chef who can instantly recognize any dish and tell you how to make it.

### Project Structure
```
foodvision_app/
├── manage.py
├── foodvision/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── classifier/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── migrations/
├── media/
│   └── uploads/
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
│   └── classifier/
├── ml_models/
│   └── food_classifier.h5
└── requirements.txt
```

### Step 1: Django Project Setup

First, create your Django project:

```python
# requirements.txt
Django==4.2.0
tensorflow==2.13.0
Pillow==10.0.0
numpy==1.24.0
opencv-python==4.8.0.74
```

```python
# foodvision/settings.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'your-secret-key-here'
DEBUG = True
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'classifier',
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

ROOT_URLCONF = 'foodvision.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
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

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
```

### Step 2: CNN Model Implementation

```python
# classifier/ml_utils.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import cv2
import os

class FoodClassifierCNN:
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = [
            'pizza', 'burger', 'sushi', 'pasta', 'salad',
            'steak', 'chicken', 'soup', 'sandwich', 'dessert'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build a CNN model with transfer learning using MobileNetV2"""
        # Like having a sous chef who already knows basic cooking techniques
        base_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model - our sous chef's existing skills stay intact
        base_model.trainable = False
        
        # Add our custom layers on top - teaching new recipes
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction - like preparing ingredients"""
        try:
            # Load and resize image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            
            # Convert to array and normalize
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_food(self, image_path):
        """Predict food type from image"""
        if not self.model:
            return None, 0
        
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return None, 0
        
        # Get prediction - like tasting and identifying the dish
        predictions = self.model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]
        
        predicted_class = self.class_names[predicted_class_index]
        
        return predicted_class, float(confidence)
    
    def get_top_predictions(self, image_path, top_k=3):
        """Get top k predictions with confidence scores"""
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return []
        
        predictions = self.model.predict(processed_image)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'class': self.class_names[idx],
                'confidence': float(predictions[idx])
            })
        
        return results
    
    def save_model(self, path):
        """Save the trained model"""
        if self.model:
            self.model.save(path)
    
    def load_model(self, path):
        """Load a pre-trained model"""
        try:
            self.model = keras.models.load_model(path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.build_model()

# Recipe suggestions based on classification
RECIPE_SUGGESTIONS = {
    'pizza': {
        'name': 'Classic Margherita Pizza',
        'ingredients': ['Pizza dough', 'Tomato sauce', 'Mozzarella cheese', 'Fresh basil', 'Olive oil'],
        'instructions': 'Roll out dough, spread sauce, add cheese and basil, bake at 450°F for 12-15 minutes.'
    },
    'burger': {
        'name': 'Gourmet Beef Burger',
        'ingredients': ['Ground beef', 'Burger buns', 'Lettuce', 'Tomato', 'Onion', 'Cheese'],
        'instructions': 'Form patties, grill for 4-5 minutes per side, assemble with toppings.'
    },
    'sushi': {
        'name': 'California Roll',
        'ingredients': ['Sushi rice', 'Nori sheets', 'Crab meat', 'Avocado', 'Cucumber'],
        'instructions': 'Prepare sushi rice, lay nori, add ingredients, roll tightly and slice.'
    },
    'pasta': {
        'name': 'Spaghetti Carbonara',
        'ingredients': ['Spaghetti', 'Eggs', 'Pecorino cheese', 'Pancetta', 'Black pepper'],
        'instructions': 'Cook pasta, fry pancetta, mix eggs and cheese, combine all with pasta water.'
    },
    'salad': {
        'name': 'Caesar Salad',
        'ingredients': ['Romaine lettuce', 'Caesar dressing', 'Parmesan cheese', 'Croutons'],
        'instructions': 'Chop lettuce, toss with dressing, top with cheese and croutons.'
    }
}
```

### Step 3: Django Models

```python
# classifier/models.py
from django.db import models
from django.contrib.auth.models import User

class ImageClassification(models.Model):
    image = models.ImageField(upload_to='uploads/')
    predicted_class = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    def __str__(self):
        return f"{self.predicted_class} ({self.confidence_score:.2%})"
    
    class Meta:
        ordering = ['-created_at']

class Recipe(models.Model):
    name = models.CharField(max_length=200)
    food_category = models.CharField(max_length=100)
    ingredients = models.TextField()
    instructions = models.TextField()
    prep_time = models.IntegerField(help_text="Preparation time in minutes")
    difficulty = models.CharField(
        max_length=20,
        choices=[('Easy', 'Easy'), ('Medium', 'Medium'), ('Hard', 'Hard')]
    )
    
    def __str__(self):
        return self.name
    
    def get_ingredients_list(self):
        return [ingredient.strip() for ingredient in self.ingredients.split(',')]
```

### Step 4: Django Forms

```python
# classifier/forms.py
from django import forms
from .models import ImageClassification

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageClassification
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control-file',
                'accept': 'image/*',
                'id': 'imageInput'
            })
        }
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # Validate file size (max 5MB)
            if image.size > 5 * 1024 * 1024:
                raise forms.ValidationError("Image file too large. Maximum size is 5MB.")
            
            # Validate file type
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            if not any(image.name.lower().endswith(ext) for ext in valid_extensions):
                raise forms.ValidationError("Invalid file type. Please upload a valid image.")
        
        return image
```

### Step 5: Django Views

```python
# classifier/views.py
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import ListView
from django.core.paginator import Paginator
import json
import os
from django.conf import settings

from .models import ImageClassification, Recipe
from .forms import ImageUploadForm
from .ml_utils import FoodClassifierCNN, RECIPE_SUGGESTIONS

# Initialize the CNN model (like having a head chef ready)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'food_classifier.h5')
food_classifier = FoodClassifierCNN(MODEL_PATH)

def home(request):
    """Home page view - the main dining area"""
    recent_classifications = ImageClassification.objects.all()[:6]
    return render(request, 'classifier/home.html', {
        'recent_classifications': recent_classifications
    })

def classify_image(request):
    """Main classification view - where the magic happens"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            classification = form.save(commit=False)
            if request.user.is_authenticated:
                classification.user = request.user
            classification.save()
            
            # Get the image path
            image_path = classification.image.path
            
            # Classify the image - like having our chef identify the dish
            predicted_class, confidence = food_classifier.predict_food(image_path)
            
            if predicted_class:
                # Update the classification record
                classification.predicted_class = predicted_class
                classification.confidence_score = confidence
                classification.save()
                
                # Get top 3 predictions for detailed results
                top_predictions = food_classifier.get_top_predictions(image_path, top_k=3)
                
                # Get recipe suggestion
                recipe_suggestion = RECIPE_SUGGESTIONS.get(predicted_class, None)
                
                context = {
                    'classification': classification,
                    'top_predictions': top_predictions,
                    'recipe_suggestion': recipe_suggestion,
                    'form': ImageUploadForm()  # Reset form
                }
                
                return render(request, 'classifier/results.html', context)
            else:
                messages.error(request, 'Error processing the image. Please try again.')
        else:
            messages.error(request, 'Please upload a valid image file.')
    else:
        form = ImageUploadForm()
    
    return render(request, 'classifier/classify.html', {'form': form})

@csrf_exempt
def classify_ajax(request):
    """AJAX endpoint for real-time classification"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                classification = form.save(commit=False)
                if request.user.is_authenticated:
                    classification.user = request.user
                classification.save()
                
                # Classify image
                predicted_class, confidence = food_classifier.predict_food(classification.image.path)
                
                if predicted_class:
                    classification.predicted_class = predicted_class
                    classification.confidence_score = confidence
                    classification.save()
                    
                    # Get top predictions
                    top_predictions = food_classifier.get_top_predictions(classification.image.path, top_k=3)
                    
                    return JsonResponse({
                        'success': True,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'top_predictions': top_predictions,
                        'image_url': classification.image.url,
                        'recipe': RECIPE_SUGGESTIONS.get(predicted_class, {})
                    })
                else:
                    return JsonResponse({'success': False, 'error': 'Classification failed'})
            else:
                return JsonResponse({'success': False, 'error': 'Invalid form data'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

class ClassificationHistoryView(ListView):
    """View for displaying classification history"""
    model = ImageClassification
    template_name = 'classifier/history.html'
    context_object_name = 'classifications'
    paginate_by = 12
    
    def get_queryset(self):
        if self.request.user.is_authenticated:
            return ImageClassification.objects.filter(user=self.request.user)
        return ImageClassification.objects.none()

def recipe_detail(request, food_category):
    """Display recipe details for a specific food category"""
    recipe_data = RECIPE_SUGGESTIONS.get(food_category)
    if not recipe_data:
        messages.error(request, 'Recipe not found.')
        return redirect('home')
    
    return render(request, 'classifier/recipe_detail.html', {
        'recipe': recipe_data,
        'food_category': food_category
    })

def batch_classify(request):
    """Handle multiple image classification"""
    if request.method == 'POST':
        images = request.FILES.getlist('images')
        results = []
        
        for image in images[:10]:  # Limit to 10 images at once
            # Create temporary classification
            classification = ImageClassification(image=image)
            if request.user.is_authenticated:
                classification.user = request.user
            classification.save()
            
            # Classify
            predicted_class, confidence = food_classifier.predict_food(classification.image.path)
            
            if predicted_class:
                classification.predicted_class = predicted_class
                classification.confidence_score = confidence
                classification.save()
                
                results.append({
                    'image_url': classification.image.url,
                    'predicted_class': predicted_class,
                    'confidence': confidence
                })
        
        return JsonResponse({'success': True, 'results': results})
    
    return render(request, 'classifier/batch_classify.html')
```

### Step 6: URL Configuration

```python
# classifier/urls.py
from django.urls import path
from . import views

app_name = 'classifier'

urlpatterns = [
    path('', views.home, name='home'),
    path('classify/', views.classify_image, name='classify'),
    path('classify-ajax/', views.classify_ajax, name='classify_ajax'),
    path('history/', views.ClassificationHistoryView.as_view(), name='history'),
    path('recipe/<str:food_category>/', views.recipe_detail, name='recipe_detail'),
    path('batch-classify/', views.batch_classify, name='batch_classify'),
]
```

```python
# foodvision/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('classifier.urls')),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

### Step 7: Templates

```html
<!-- templates/classifier/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}FoodVision - AI Recipe Classifier{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 0;
        }
        
        .prediction-card {
            transition: transform 0.3s ease;
            border: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .prediction-card:hover {
            transform: translateY(-5px);
        }
        
        .confidence-bar {
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .recipe-card {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
        }
        
        .upload-area {
            border: 2px dashed #dee2e6;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover, .upload-area.drag-over {
            border-color: #667eea;
            background-color: #f8f9ff;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{% url 'classifier:home' %}">
                <i class="fas fa-utensils"></i> FoodVision
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="{% url 'classifier:home' %}">Home</a>
                <a class="nav-link" href="{% url 'classifier:classify' %}">Classify</a>
                {% if user.is_authenticated %}
                    <a class="nav-link" href="{% url 'classifier:history' %}">History</a>
                {% endif %}
                <a class="nav-link" href="{% url 'classifier:batch_classify' %}">Batch</a>
            </div>
        </div>
    </nav>

    {% if messages %}
        {% for message in messages %}
            <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                {{ message }}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        {% endfor %}
    {% endif %}

    {% block content %}{% endblock %}

    <footer class="bg-dark text-light py-4 mt-5">
        <div class="container text-center">
            <p>&copy; 2024 FoodVision - AI-Powered Recipe Classification</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
```

```html
<!-- templates/classifier/home.html -->
{% extends 'classifier/base.html' %}

{% block content %}
<div class="hero-section">
    <div class="container text-center">
        <h1 class="display-4 mb-4">Welcome to FoodVision</h1>
        <p class="lead">Upload a food image and let our AI chef identify it and suggest recipes!</p>
        <a href="{% url 'classifier:classify' %}" class="btn btn-light btn-lg mt-3">
            <i class="fas fa-camera"></i> Start Classifying
        </a>
    </div>
</div>

<div class="container my-5">
    <div class="row">
        <div class="col-md-4 mb-4">
            <div class="text-center">
                <i class="fas fa-brain fa-3x text-primary mb-3"></i>
                <h3>AI-Powered</h3>
                <p>Advanced CNN technology for accurate food recognition</p>
            </div>
        </div>
        <div class="col-md-4 mb-4">
            <div class="text-center">
                <i class="fas fa-bolt fa-3x text-primary mb-3"></i>
                <h3>Lightning Fast</h3>
                <p>Get results in seconds with our optimized neural network</p>
            </div>
        </div>
        <div class="col-md-4 mb-4">
            <div class="text-center">
                <i class="fas fa-book-open fa-3x text-primary mb-3"></i>
                <h3>Recipe Suggestions</h3>
                <p>Instant recipe recommendations based on identified food</p>
            </div>
        </div>
    </div>
</div>

{% if recent_classifications %}
<div class="container my-5">
    <h2 class="text-center mb-4">Recent Classifications</h2>
    <div class="row">
        {% for classification in recent_classifications %}
        <div class="col-md-4 mb-4">
            <div class="card prediction-card">
                <img src="{{ classification.image.url }}" class="card-img-top" style="height: 200px; object-fit: cover;">
                <div class="card-body">
                    <h5 class="card-title">{{ classification.predicted_class|title }}</h5>
                    <div class="progress mb-2">
                        <div class="progress-bar bg-success" 
                             style="width: {{ classification.confidence_score|floatformat:0 }}%">
                            {{ classification.confidence_score|floatformat:1 }}%
                        </div>
                    </div>
                    <small class="text-muted">{{ classification.created_at|timesince }} ago</small>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</div>
{% endif %}
{% endblock %}
```

```html
<!-- templates/classifier/classify.html -->
{% extends 'classifier/base.html' %}

{% block content %}
<div class="container my-5">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <h2 class="text-center mb-4">Food Image Classifier</h2>
            
            <div class="card">
                <div class="card-body">
                    <form method="post" enctype="multipart/form-data" id="classifyForm">
                        {% csrf_token %}
                        <div class="upload-area" id="uploadArea">
                            <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                            <h4>Drop your food image here</h4>
                            <p class="text-muted">or click to browse</p>
                            {{ form.image }}
                        </div>
                        
                        <div class="text-center mt-4">
                            <button type="submit" class="btn btn-primary btn-lg" id="classifyBtn">
                                <i class="fas fa-search"></i> Classify Food
                            </button>
                        </div>
                    </form>
                    
                    <!-- Real-time preview -->
                    <div id="imagePreview" class="mt-4 text-center" style="display: none;">
                        <img id="previewImg" class="img-fluid rounded" style="max-height: 300px;">
                    </div>
                    
                    <!-- Loading indicator -->
                    <div id="loadingIndicator" class="text-center mt-4" style="display: none;">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Analyzing...</span>
                        </div>
                        <p class="mt-2">Our AI chef is analyzing your image...</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const form = document.getElementById('classifyForm');
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    // Handle drag and drop
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            previewImage(files[0]);
        }
    });
    
    // Handle file selection
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            previewImage(e.target.files[0]);
        }
    });
    
    function previewImage(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
    
    // Handle form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            alert('Please select an image first!');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        
        // Submit form normally (we could also use AJAX here)
        form.submit();
    });
});
</script>
{% endblock %}
```

```html
<!-- templates/classifier/results.html -->
{% extends 'classifier/base.html' %}

{% block content %}
<div class="container my-5">
    <div class="row">
        <div class="col-md-6">
            <div class="card">
                <img src="{{ classification.image.url }}" class="card-img-top" style="max-height: 400px; object-fit: cover;">
                <div class="card-body text-center">
                    <h3 class="text-primary">{{ classification.predicted_class|title }}</h3>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-success" 
                             style="width: {{ classification.confidence_score|floatformat:0 }}%">
                            {{ classification.confidence_score|floatformat:1 }}% confident
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <h3>Detailed Analysis</h3>
            
            {% for prediction in top_predictions %}
            <div class="d-flex justify-content-between align-items-center mb-2">
                <span>{{ prediction.class|title }}</span>
                <div class="flex-grow-1 mx-3">
                    <div class="progress" style="height: 10px;">
                        <div class="progress-bar" 
                             style="width: {{ prediction.confidence|floatformat:0 }}%">
                        </div>
                    </div>
                </div>
                <small>{{ prediction.confidence|floatformat:1 }}%</small>
            </div>
            {% endfor %}

## Assignment: Food Delivery App Dish Verification

**Scenario**: You're building a quality control system for a food delivery app. Restaurants upload photos of their dishes, and your system needs to verify that the dish matches what's advertised on the menu.

**Your Task**:
Create a dish verification system that:

1. **Accepts two images**: The "menu image" (reference) and the "actual dish image" (to verify)

2. **Implements a similarity comparison system** using the following approach:
   ```python
   def create_siamese_network():
       """
       Create a Siamese network for dish comparison
       Returns similarity score between two dish images
       """
       # Your implementation here
       pass
   ```

3. **Creates a Django API endpoint** `/api/verify-dish/` that:
   - Accepts both images via POST request
   - Returns a JSON response with:
     - `similarity_score` (0.0 to 1.0)
     - `is_match` (boolean, True if similarity > 0.8)
     - `confidence_level` (High/Medium/Low)

4. **Implements basic fraud detection** by flagging dishes that:
   - Have similarity score < 0.6 (possible wrong dish)
   - Show signs of being significantly different from the menu image

**Deliverables**:
- Python code for the Siamese network architecture
- Django view and URL configuration
- Sample test cases showing your verification working
- Brief explanation of how your similarity algorithm works

**Bonus Challenge**: Add feature extraction using a pre-trained CNN to compare dishes based on visual features like color distribution, texture, and shape.

This assignment tests your understanding of CNN architectures, transfer learning, and practical application development - different from the main project which focused on single-image classification.

---

## Summary

Today we've journeyed through the world of Convolutional Neural Networks, learning how these powerful "digital chefs" can recognize and classify images with remarkable precision. From basic convolution operations to sophisticated architectures like VGG and AlexNet, we've built a complete understanding of how CNNs work.

Key concepts mastered:
- **Convolution and Pooling**: The fundamental operations that allow networks to detect patterns
- **CNN Architectures**: Historical progression from LeNet to modern designs
- **Transfer Learning**: Leveraging pre-trained models for faster, more effective training
- **Data Augmentation**: Techniques to improve model robustness and generalization
- **Practical Implementation**: Building a complete Django-based image classification system

Remember, like training master chefs, building effective CNNs requires patience, practice, and continuous refinement. The techniques you've learned today form the foundation for advanced computer vision applications in everything from medical imaging to autonomous vehicles.