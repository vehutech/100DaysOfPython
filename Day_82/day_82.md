# AI Mastery Course - Day 82: Advanced CNN Architectures

## Learning Objective
By the end of this lesson, you will master advanced CNN architectures and understand how different network designs solve specific challenges in computer vision, enabling you to choose and implement the right architecture for your deep learning projects.

---

## Introduction:

Imagine that you're running a world-class restaurant where each dish requires different cooking techniques and specialized equipment. Some dishes need quick preparation with minimal ingredients (like a perfect omelet), while others require complex, multi-layered preparation with various cooking stations working together (like a seven-course tasting menu). 

In the world of deep learning, Convolutional Neural Networks (CNNs) work similarly. Just as master chefs have developed specialized techniques and equipment over decades to create different types of cuisine, computer vision researchers have crafted various CNN architectures, each designed to solve specific challenges in image recognition and analysis.

Today, we'll explore four revolutionary "cooking techniques" that have transformed how we approach computer vision tasks.

---

## 1. ResNet and Skip Connections

### The Challenge: The Vanishing Gradient Problem

Just as a message can get distorted when passed through too many people in a game of telephone, information in very deep neural networks can become diluted or lost as it travels from the input to the output layers.

### The Solution: Skip Connections

Think of skip connections like having direct phone lines between different stations in our restaurant. Instead of relying only on messages passed from one station to the next, each station can communicate directly with stations several steps ahead, ensuring critical information never gets lost.

### Code Example: Building a ResNet Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A basic residual block that implements skip connections
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection adjustment if dimensions don't match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Store the input for the skip connection
        residual = x
        
        # Forward pass through the main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add the skip connection (this is the key innovation!)
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

# Example usage
block = ResidualBlock(64, 128, stride=2)
input_tensor = torch.randn(1, 64, 56, 56)  # Batch size 1, 64 channels, 56x56 image
output = block(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
```

**Syntax Explanation:**
- `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`: Creates a 2D convolutional layer
- `nn.BatchNorm2d()`: Normalizes the output to improve training stability
- `self.shortcut()`: Creates the skip connection path that bypasses the main computation
- The `+=` operation adds the original input to the processed output, creating the residual connection

---

## 2. Inception Networks

### The Philosophy: Multiple Cooking Methods Simultaneously

Imagine a master chef who, instead of deciding between grilling, sautéing, or roasting a piece of meat, does all three simultaneously and then combines the results. This parallel approach captures different flavors and textures that no single method could achieve alone.

Inception networks apply this same philosophy to feature extraction, using multiple filter sizes simultaneously to capture features at different scales.

### Code Example: Inception Module

```python
class InceptionModule(nn.Module):
    """
    An Inception module that processes input with multiple parallel paths
    """
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch (captures point-wise features)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution branch (captures local patterns)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),  # Dimension reduction
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 convolution branch (captures broader patterns)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),  # Dimension reduction
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        
        # Max pooling branch (preserves strong features)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Process input through all four branches simultaneously
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        
        # Concatenate all outputs along the channel dimension
        outputs = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)
        return outputs

# Example usage
inception = InceptionModule(192, 64, 96, 128, 16, 32, 32)
input_tensor = torch.randn(1, 192, 28, 28)
output = inception(input_tensor)
print(f"Input channels: {input_tensor.shape[1]}")
print(f"Output channels: {output.shape[1]}")  # Should be 64+128+32+32 = 256
```

**Syntax Explanation:**
- `torch.cat([tensors], dim=1)`: Concatenates tensors along the channel dimension (dim=1)
- Multiple parallel branches process the same input simultaneously
- `padding=1` and `padding=2`: Ensures output spatial dimensions remain the same
- `inplace=True`: Modifies the tensor in-place to save memory

---

## 3. MobileNet and EfficientNet

### The Challenge: Cooking for Mobile Food Trucks

Imagine you need to create gourmet meals, but your equipment is limited to what fits in a small food truck. You need techniques that are both efficient and effective, maximizing flavor while minimizing resource usage.

MobileNet and EfficientNet are designed for this exact scenario in the AI world - delivering high-quality results with minimal computational resources.

### Code Example: MobileNet Depthwise Separable Convolution

```python
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution: the secret ingredient for efficient networks
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution: each input channel is convolved separately
        # Like having each ingredient prepared by a specialist chef
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                     stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Pointwise convolution: combines the separate preparations
        # Like a head chef combining all ingredients into the final dish
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNetBlock(nn.Module):
    """
    A complete MobileNet block with optional stride
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileNetBlock, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, stride)
    
    def forward(self, x):
        return self.conv(x)

# Efficiency comparison
def compare_efficiency():
    # Standard convolution
    standard_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    
    # MobileNet equivalent
    mobile_conv = DepthwiseSeparableConv(128, 256)
    
    # Count parameters
    standard_params = sum(p.numel() for p in standard_conv.parameters())
    mobile_params = sum(p.numel() for p in mobile_conv.parameters())
    
    print(f"Standard convolution parameters: {standard_params:,}")
    print(f"MobileNet convolution parameters: {mobile_params:,}")
    print(f"Parameter reduction: {((standard_params - mobile_params) / standard_params * 100):.1f}%")

compare_efficiency()
```

**Syntax Explanation:**
- `groups=in_channels`: Creates depthwise convolution where each input channel is processed separately
- `p.numel()`: Returns the number of elements in a parameter tensor
- The two-step process (depthwise then pointwise) dramatically reduces computational cost
- `sum(generator)`: Efficiently sums all parameters in the model

---

## 4. Object Detection Basics (YOLO, R-CNN)

### The Philosophy: Finding Ingredients in a Busy Market

Imagine you're a chef walking through a bustling farmers market, trying to identify and locate the best ingredients among hundreds of stalls. You need to not only recognize what each item is but also precisely locate where it is for purchase.

Object detection networks solve this same challenge in images - they must identify what objects are present AND precisely locate where they appear.

### Code Example: YOLO-Style Detection Head

```python
class YOLODetectionHead(nn.Module):
    """
    A simplified YOLO detection head that predicts bounding boxes and classes
    """
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(YOLODetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Each anchor predicts: x, y, width, height, confidence, and class probabilities
        predictions_per_anchor = 5 + num_classes  # 4 bbox coords + 1 confidence + classes
        
        self.detection_layer = nn.Conv2d(
            in_channels, 
            num_anchors * predictions_per_anchor, 
            kernel_size=1
        )
    
    def forward(self, x):
        batch_size, _, grid_h, grid_w = x.shape
        
        # Get raw predictions
        predictions = self.detection_layer(x)
        
        # Reshape to separate anchors and predictions
        predictions = predictions.view(
            batch_size, 
            self.num_anchors, 
            5 + self.num_classes, 
            grid_h, 
            grid_w
        )
        
        # Separate different prediction types
        bbox_coords = predictions[:, :, :4, :, :]      # x, y, w, h
        confidence = predictions[:, :, 4:5, :, :]      # objectness score
        class_probs = predictions[:, :, 5:, :, :]      # class probabilities
        
        # Apply sigmoid to coordinates and confidence (values between 0 and 1)
        bbox_coords = torch.sigmoid(bbox_coords)
        confidence = torch.sigmoid(confidence)
        class_probs = torch.sigmoid(class_probs)
        
        return bbox_coords, confidence, class_probs

# Example of creating a complete detection pipeline
class SimpleObjectDetector(nn.Module):
    """
    A simplified object detector combining feature extraction and detection
    """
    def __init__(self, num_classes=80):  # COCO dataset has 80 classes
        super(SimpleObjectDetector, self).__init__()
        
        # Feature extraction backbone (simplified)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Add more layers here in a real implementation
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Detection head
        self.detection_head = YOLODetectionHead(256, num_classes)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Generate detections
        bbox_coords, confidence, class_probs = self.detection_head(features)
        
        return bbox_coords, confidence, class_probs

# Example usage
detector = SimpleObjectDetector(num_classes=20)  # 20 classes for simplicity
input_image = torch.randn(1, 3, 416, 416)  # Standard YOLO input size

bbox_coords, confidence, class_probs = detector(input_image)
print(f"Bounding box coordinates shape: {bbox_coords.shape}")
print(f"Confidence scores shape: {confidence.shape}")
print(f"Class probabilities shape: {class_probs.shape}")
```

**Syntax Explanation:**
- `.view()`: Reshapes tensors while maintaining the same number of elements
- `torch.sigmoid()`: Applies sigmoid activation to constrain values between 0 and 1
- The detection head outputs multiple predictions per grid cell (anchors)
- Each prediction includes spatial coordinates, confidence, and class probabilities

---

### 8. Project Deployment and Production Considerations

```python
# deployment/docker/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories for models and media
RUN mkdir -p ai_models media

# Expose port
EXPOSE 8000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "main_project.wsgi:application"]
```

```txt
# requirements.txt
Django==4.2.7
djangorestframework==3.14.0
tensorflow==2.13.0
Pillow==10.0.1
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
gunicorn==21.2.0
redis==5.0.1
celery==5.3.4
psycopg2-binary==2.9.9
```

### 9. Final Project Structure

```
advanced_cnn_classifier/
├── manage.py
├── requirements.txt
├── main_project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── classifier_app/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── architectures.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── model_trainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── model_analysis.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── advanced_trainer.py
│   ├── templates/
│   │   └── classifier_app/
│   │       └── dashboard.html
│   └── static/
│       └── classifier_app/
│           ├── css/
│           ├── js/
│           └── images/
├── ai_models/
│   ├── resnet_best.h5
│   ├── inception_best.h5
│   └── mobilenet_best.h5
├── media/
│   └── uploads/
└── deployment/
    ├── docker/
    │   └── Dockerfile
    └── nginx/
        └── nginx.conf
```

### 10. Performance Optimization and Monitoring

```python
# classifier_app/monitoring/performance_monitor.py
import time
import psutil
import tensorflow as tf
from django.core.cache import cache
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'inference_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'model_accuracy': {},
            'error_rates': {}
        }
    
    def monitor_inference(self, model_name):
        """Decorator to monitor inference performance"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.virtual_memory().used
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record successful inference
                    inference_time = time.time() - start_time
                    memory_used = psutil.virtual_memory().used - start_memory
                    
                    self.record_metrics(model_name, {
                        'inference_time': inference_time,
                        'memory_usage': memory_used,
                        'cpu_usage': psutil.cpu_percent(),
                        'status': 'success'
                    })
                    
                    return result
                    
                except Exception as e:
                    # Record failed inference
                    self.record_metrics(model_name, {
                        'status': 'error',
                        'error_type': type(e).__name__
                    })
                    raise
                    
            return wrapper
        return decorator
    
    def record_metrics(self, model_name, metrics):
        """Record performance metrics"""
        timestamp = time.time()
        
        # Store in cache for real-time monitoring
        cache_key = f"metrics_{model_name}_{int(timestamp)}"
        cache.set(cache_key, metrics, timeout=3600)  # Store for 1 hour
        
        # Log important metrics
        if 'inference_time' in metrics:
            logger.info(f"{model_name} inference time: {metrics['inference_time']:.3f}s")
        
        # Update in-memory metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append({
                'timestamp': timestamp,
                'model': model_name,
                'value': value
            })
    
    def get_performance_summary(self, model_name=None, hours=24):
        """Get performance summary for specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        
        summary = {}
        
        for metric_type, records in self.metrics.items():
            filtered_records = [
                r for r in records 
                if r['timestamp'] > cutoff_time and 
                (model_name is None or r['model'] == model_name)
            ]
            
            if filtered_records:
                values = [r['value'] for r in filtered_records if isinstance(r['value'], (int, float))]
                if values:
                    summary[metric_type] = {
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
        
        return summary

# Model optimization utilities
class ModelOptimizer:
    def __init__(self):
        self.optimization_methods = [
            'quantization',
            'pruning',
            'knowledge_distillation'
        ]
    
    def quantize_model(self, model, representative_dataset):
        """Apply post-training quantization"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_data_gen():
            for input_value in representative_dataset.take(100):
                yield [input_value[0]]
        
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        quantized_model = converter.convert()
        return quantized_model
    
    def prune_model(self, model, target_sparsity=0.5):
        """Apply magnitude-based pruning"""
        import tensorflow_model_optimization as tfmot
        
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }
        
        pruned_model = prune_low_magnitude(model, **pruning_params)
        return pruned_model
    
    def knowledge_distillation(self, teacher_model, student_model, train_data, temperature=3.0):
        """Implement knowledge distillation"""
        class DistillationLoss(tf.keras.losses.Loss):
            def __init__(self, temperature=3.0, alpha=0.7):
                super().__init__()
                self.temperature = temperature
                self.alpha = alpha
            
            def call(self, y_true, y_pred):
                teacher_pred, student_pred = y_pred[0], y_pred[1]
                
                # Soft targets from teacher
                soft_targets = tf.nn.softmax(teacher_pred / self.temperature)
                soft_student = tf.nn.softmax(student_pred / self.temperature)
                
                # Distillation loss
                distill_loss = tf.keras.losses.categorical_crossentropy(
                    soft_targets, soft_student
                ) * (self.temperature ** 2)
                
                # Student loss
                student_loss = tf.keras.losses.categorical_crossentropy(
                    y_true, student_pred
                )
                
                return self.alpha * distill_loss + (1 - self.alpha) * student_loss
        
        # Create distillation model
        distillation_model = tf.keras.Model(
            inputs=student_model.input,
            outputs=[teacher_model(student_model.input), student_model.output]
        )
        
        distillation_model.compile(
            optimizer='adam',
            loss=DistillationLoss(temperature=temperature)
        )
        
        return distillation_model

# Advanced caching and model serving
class ModelCache:
    def __init__(self, max_models=3):
        self.cache = {}
        self.max_models = max_models
        self.access_times = {}
    
    def get_model(self, model_name):
        """Get model from cache or load if not present"""
        if model_name in self.cache:
            self.access_times[model_name] = time.time()
            return self.cache[model_name]
        
        # Load model
        model = self._load_model(model_name)
        self._add_to_cache(model_name, model)
        return model
    
    def _load_model(self, model_name):
        """Load model from disk"""
        model_path = f"ai_models/{model_name}_best.h5"
        try:
            return tf.keras.models.load_model(model_path)
        except:
            # Return default model if loading fails
            if model_name == 'resnet':
                return CustomResNet(num_classes=10)
            elif model_name == 'inception':
                return CustomInception(num_classes=10)
            elif model_name == 'mobilenet':
                return CustomMobileNet(num_classes=10)
    
    def _add_to_cache(self, model_name, model):
        """Add model to cache with LRU eviction"""
        if len(self.cache) >= self.max_models:
            # Remove least recently used model
            lru_model = min(self.access_times.keys(), 
                          key=lambda k: self.access_times[k])
            del self.cache[lru_model]
            del self.access_times[lru_model]
        
        self.cache[model_name] = model
        self.access_times[model_name] = time.time()

# Global cache instance
model_cache = ModelCache()
```

### 11. Production Management Commands

```python
# classifier_app/management/commands/train_models.py
from django.core.management.base import BaseCommand
from classifier_app.models.architectures import CustomResNet, CustomInception, CustomMobileNet
from classifier_app.services.model_trainer import AdvancedModelTrainer
import tensorflow as tf
import os

class Command(BaseCommand):
    help = 'Train all CNN models'
    
    def add_arguments(self, parser):
        parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
        parser.add_argument('--models', nargs='+', default=['resnet', 'inception', 'mobilenet'],
                          help='Models to train')
    
    def handle(self, *args, **options):
        self.stdout.write('Starting model training...')
        
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Preprocess data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        # Create validation split
        val_split = int(0.1 * len(x_train))
        x_val, y_val = x_train[:val_split], y_train[:val_split]
        x_train, y_train = x_train[val_split:], y_train[val_split:]
        
        # Initialize trainer
        trainer = AdvancedModelTrainer()
        
        # Train specified models
        for model_name in options['models']:
            self.stdout.write(f'Training {model_name}...')
            
            if model_name == 'resnet':
                model = CustomResNet(num_classes=10)
            elif model_name == 'inception':
                model = CustomInception(num_classes=10)
            elif model_name == 'mobilenet':
                model = CustomMobileNet(num_classes=10)
            else:
                self.stdout.write(f'Unknown model: {model_name}')
                continue
            
            # Train model
            trained_model, history = trainer.train_model(
                model, model_name, 
                (x_train, y_train), (x_val, y_val),
                epochs=options['epochs']
            )
            
            # Save model
            model_path = f'ai_models/{model_name}_best.h5'
            trained_model.save(model_path)
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully trained {model_name}')
            )
        
        self.stdout.write(
            self.style.SUCCESS('All models trained successfully!')
        )
```

```python
# classifier_app/management/commands/evaluate_models.py
from django.core.management.base import BaseCommand
from classifier_app.training.advanced_trainer import ModelEvaluationSuite
import tensorflow as tf

class Command(BaseCommand):
    help = 'Evaluate trained models'
    
    def handle(self, *args, **options):
        self.stdout.write('Loading models and test data...')
        
        # Load test data
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        # Load models
        models = {}
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        for model_name in ['resnet', 'inception', 'mobilenet']:
            try:
                model_path = f'ai_models/{model_name}_best.h5'
                models[model_name] = tf.keras.models.load_model(model_path)
                self.stdout.write(f'Loaded {model_name}')
            except Exception as e:
                self.stdout.write(f'Failed to load {model_name}: {e}')
        
        if not models:
            self.stdout.write('No models found!')
            return
        
        # Evaluate models
        evaluator = ModelEvaluationSuite(models, x_test, y_test, class_names)
        results = evaluator.comprehensive_evaluation()
        report = evaluator.generate_evaluation_report(results)
        
        # Print results
        self.stdout.write('\n=== Model Evaluation Results ===')
        self.stdout.write(f"Best accuracy: {report['summary']['best_accuracy']}")
        self.stdout.write(f"Most confident: {report['summary']['most_confident']}")
        self.stdout.write(f"Most calibrated: {report['summary']['most_calibrated']}")
        
        for model_name, metrics in results.items():
            self.stdout.write(f'\n--- {model_name.upper()} ---')
            self.stdout.write(f"Accuracy: {metrics['test_accuracy']:.4f}")
            self.stdout.write(f"Top-3 Accuracy: {metrics['test_top3_accuracy']:.4f}")
            self.stdout.write(f"Average Confidence: {metrics['avg_confidence']:.4f}")
            
            # Show recommendations
            recommendations = report['recommendations'][model_name]
            for rec in recommendations:
                self.stdout.write(f"  • {rec}")
```

### 12. API Documentation and Testing

```python
# tests/test_models.py
import unittest
import numpy as np
import tensorflow as tf
from classifier_app.models.architectures import CustomResNet, CustomInception, CustomMobileNet

class TestModelArchitectures(unittest.TestCase):
    def setUp(self):
        self.input_shape = (None, 224, 224, 3)
        self.num_classes = 10
        self.test_input = np.random.random((1, 224, 224, 3))
    
    def test_resnet_architecture(self):
        """Test ResNet model creation and forward pass"""
        model = CustomResNet(num_classes=self.num_classes)
        output = model(self.test_input)
        
        self.assertEqual(output.shape, (1, self.num_classes))
        self.assertTrue(np.allclose(np.sum(output.numpy(), axis=1), 1.0, atol=1e-6))
    
    def test_inception_architecture(self):
        """Test Inception model creation and forward pass"""
        model = CustomInception(num_classes=self.num_classes)
        output = model(self.test_input)
        
        self.assertEqual(output.shape, (1, self.num_classes))
        self.assertTrue(np.allclose(np.sum(output.numpy(), axis=1), 1.0, atol=1e-6))
    
    def test_mobilenet_architecture(self):
        """Test MobileNet model creation and forward pass"""
        model = CustomMobileNet(num_classes=self.num_classes)
        output = model(self.test_input)
        
        self.assertEqual(output.shape, (1, self.num_classes))
        self.assertTrue(np.allclose(np.sum(output.numpy(), axis=1), 1.0, atol=1e-6))
    
    def test_model_training_compatibility(self):
        """Test that models can be compiled and trained"""
        for ModelClass in [CustomResNet, CustomInception, CustomMobileNet]:
            model = ModelClass(num_classes=self.num_classes)
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Test with dummy data
            x_dummy = np.random.random((10, 224, 224, 3))
            y_dummy = tf.keras.utils.to_categorical(np.random.randint(0, 10, 10), 10)
            
            # Should not raise any errors
            history = model.fit(x_dummy, y_dummy, epochs=1, verbose=0)
            self.assertIsNotNone(history)

# API endpoint tests
from django.test import TestCase, Client
import json
import base64
from PIL import Image
import io

class TestAPIEndpoints(TestCase):
    def setUp(self):
        self.client = Client()
        
        # Create test image
        img = Image.new('RGB', (224, 224), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        
        self.test_image_b64 = base64.b64encode(buffer.getvalue()).decode()
        self.test_image_data = f"data:image/jpeg;base64,{self.test_image_b64}"
    
    def test_dashboard_view(self):
        """Test dashboard renders correctly"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Advanced CNN Classifier')
    
    def test_classify_image_endpoint(self):
        """Test image classification API"""
        data = {
            'image': self.test_image_data
        }
        
        response = self.client.post(
            '/api/classify/',
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        
        response_data = json.loads(response.content)
        self.assertTrue(response_data['success'])
        self.assertIn('predictions', response_data)
        self.assertIn('model_comparison', response_data)
    
    def test_performance_endpoint(self):
        """Test performance metrics API"""
        response = self.client.get('/api/performance/')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        expected_models = ['resnet', 'inception', 'mobilenet']
        
        for model in expected_models:
            self.assertIn(model, data)
            self.assertIn('accuracy', data[model])
            self.assertIn('model_size_mb', data[model])
            self.assertIn('inference_time_ms', data[model])

if __name__ == '__main__':
    unittest.main()
```

## Project Summary

This advanced image classifier project demonstrates multiple CNN architectures working together in a production-ready Django application. The implementation includes:

**Key Components:**
- **Custom CNN Architectures**: ResNet with skip connections, Inception with multi-scale features, and MobileNet with depthwise separable convolutions
- **Ensemble Learning**: Combines predictions from multiple models for improved accuracy
- **Advanced Training**: Includes data augmentation, learning rate scheduling, and early stopping
- **Performance Monitoring**: Real-time tracking of inference times, memory usage, and accuracy metrics
- **Production Features**: Model caching, quantization, pruning, and containerized deployment

**Technical Highlights:**
- **ResNet Implementation**: Custom residual blocks with skip connections for training deeper networks
- **Inception Module**: Parallel convolution paths with different kernel sizes for multi-scale feature extraction  
- **MobileNet Design**: Depthwise separable convolutions for efficient mobile deployment
- **Django Integration**: RESTful APIs, dynamic model loading, and real-time web interface
- **Optimization Techniques**: Post-training quantization, magnitude-based pruning, and knowledge distillation

The project showcases advanced deep learning concepts while maintaining practical deployment considerations, making it suitable for both educational purposes and real-world applications.# Advanced Image Classifier with Multiple Architectures

## Project Overview
Create a comprehensive image classification system that implements and compares multiple CNN architectures including ResNet, Inception, and MobileNet variants. This project demonstrates advanced deep learning concepts through a unified Django web application.

## Core Implementation

### 1. Django Project Setup

```python
# settings.py
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
    'classifier_app',
    'rest_framework',
]

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# AI Model Configuration
AI_MODELS_PATH = os.path.join(BASE_DIR, 'ai_models')
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
```

### 2. Model Architecture Implementation

```python
# classifier_app/models/architectures.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class ResNetBlock(layers.Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        
        # Main path
        self.conv1 = layers.Conv2D(filters, 3, stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, 1, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Skip connection
        if stride != 1:
            self.skip_conv = layers.Conv2D(filters, 1, stride, padding='same')
            self.skip_bn = layers.BatchNormalization()
        else:
            self.skip_conv = None
            
        self.relu = layers.ReLU()
        
    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Skip connection
        if self.skip_conv:
            skip = self.skip_conv(inputs)
            skip = self.skip_bn(skip)
        else:
            skip = inputs
            
        # Add residual
        x = x + skip
        return self.relu(x)

class CustomResNet(Model):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        
        # Initial layers
        self.conv1 = layers.Conv2D(64, 7, 2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.pool1 = layers.MaxPooling2D(3, 2, padding='same')
        
        # ResNet blocks
        self.block1_1 = ResNetBlock(64)
        self.block1_2 = ResNetBlock(64)
        
        self.block2_1 = ResNetBlock(128, stride=2)
        self.block2_2 = ResNetBlock(128)
        
        self.block3_1 = ResNetBlock(256, stride=2)
        self.block3_2 = ResNetBlock(256)
        
        # Final layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        
        x = self.global_pool(x)
        x = self.dropout(x)
        return self.classifier(x)

class InceptionBlock(layers.Layer):
    def __init__(self, filters_list, **kwargs):
        super().__init__(**kwargs)
        f1, f3_reduce, f3, f5_reduce, f5, pool_proj = filters_list
        
        # 1x1 branch
        self.conv1x1 = layers.Conv2D(f1, 1, activation='relu')
        
        # 3x3 branch
        self.conv3x3_reduce = layers.Conv2D(f3_reduce, 1, activation='relu')
        self.conv3x3 = layers.Conv2D(f3, 3, padding='same', activation='relu')
        
        # 5x5 branch
        self.conv5x5_reduce = layers.Conv2D(f5_reduce, 1, activation='relu')
        self.conv5x5 = layers.Conv2D(f5, 5, padding='same', activation='relu')
        
        # Pool branch
        self.pool = layers.MaxPooling2D(3, 1, padding='same')
        self.pool_proj = layers.Conv2D(pool_proj, 1, activation='relu')
        
    def call(self, inputs):
        # Four parallel branches
        branch1 = self.conv1x1(inputs)
        
        branch2 = self.conv3x3_reduce(inputs)
        branch2 = self.conv3x3(branch2)
        
        branch3 = self.conv5x5_reduce(inputs)
        branch3 = self.conv5x5(branch3)
        
        branch4 = self.pool(inputs)
        branch4 = self.pool_proj(branch4)
        
        # Concatenate all branches
        return tf.concat([branch1, branch2, branch3, branch4], axis=-1)

class CustomInception(Model):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        
        # Initial layers
        self.conv1 = layers.Conv2D(64, 7, 2, padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D(3, 2, padding='same')
        self.conv2 = layers.Conv2D(192, 3, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D(3, 2, padding='same')
        
        # Inception blocks
        self.inception3a = InceptionBlock([64, 96, 128, 16, 32, 32])
        self.inception3b = InceptionBlock([128, 128, 192, 32, 96, 64])
        self.pool3 = layers.MaxPooling2D(3, 2, padding='same')
        
        self.inception4a = InceptionBlock([192, 96, 208, 16, 48, 64])
        self.inception4b = InceptionBlock([160, 112, 224, 24, 64, 64])
        
        # Final layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.4)
        self.classifier = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        
        x = self.global_pool(x)
        x = self.dropout(x)
        return self.classifier(x)

class MobileNetBlock(layers.Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        
        # Depthwise convolution
        self.depthwise = layers.DepthwiseConv2D(3, stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        # Pointwise convolution
        self.pointwise = layers.Conv2D(filters, 1)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        
    def call(self, inputs):
        x = self.depthwise(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return self.relu2(x)

class CustomMobileNet(Model):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        
        # Initial convolution
        self.conv1 = layers.Conv2D(32, 3, 2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        # MobileNet blocks
        self.blocks = [
            MobileNetBlock(64),
            MobileNetBlock(128, stride=2),
            MobileNetBlock(128),
            MobileNetBlock(256, stride=2),
            MobileNetBlock(256),
            MobileNetBlock(512, stride=2),
            MobileNetBlock(512),
            MobileNetBlock(512),
            MobileNetBlock(512),
            MobileNetBlock(512),
            MobileNetBlock(1024, stride=2),
            MobileNetBlock(1024)
        ]
        
        # Final layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.2)
        self.classifier = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.global_pool(x)
        x = self.dropout(x)
        return self.classifier(x)
```

### 3. Model Training and Management

```python
# classifier_app/services/model_trainer.py
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
from django.conf import settings

class AdvancedModelTrainer:
    def __init__(self):
        self.models = {}
        self.histories = {}
        self.model_configs = {
            'resnet': {
                'optimizer': Adam(learning_rate=0.001),
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy', 'top_3_accuracy']
            },
            'inception': {
                'optimizer': Adam(learning_rate=0.0005),
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy', 'top_3_accuracy']
            },
            'mobilenet': {
                'optimizer': Adam(learning_rate=0.001),
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy', 'top_3_accuracy']
            }
        }
        
    def prepare_data(self, train_data, val_data, num_classes):
        """Prepare and augment data for training"""
        # Data augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            rescale=1./255
        )
        
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        return train_datagen, val_datagen
    
    def create_callbacks(self, model_name):
        """Create training callbacks"""
        checkpoint_path = os.path.join(settings.AI_MODELS_PATH, f'{model_name}_best.h5')
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, model, model_name, train_data, val_data, epochs=50):
        """Train a specific model"""
        print(f"Training {model_name} architecture...")
        
        # Compile model
        config = self.model_configs[model_name]
        model.compile(
            optimizer=config['optimizer'],
            loss=config['loss'],
            metrics=config['metrics']
        )
        
        # Create callbacks
        callbacks = self.create_callbacks(model_name)
        
        # Train model
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store results
        self.models[model_name] = model
        self.histories[model_name] = history
        
        return model, history
    
    def evaluate_models(self, test_data):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            test_loss, test_accuracy, test_top3 = model.evaluate(test_data, verbose=0)
            
            # Additional metrics
            predictions = model.predict(test_data)
            
            results[name] = {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy),
                'test_top3_accuracy': float(test_top3),
                'model_size': self.get_model_size(model),
                'inference_time': self.measure_inference_time(model, test_data)
            }
            
        return results
    
    def get_model_size(self, model):
        """Calculate model size in MB"""
        param_count = model.count_params()
        # Assuming float32 parameters
        size_mb = (param_count * 4) / (1024 * 1024)
        return round(size_mb, 2)
    
    def measure_inference_time(self, model, test_data, num_samples=100):
        """Measure average inference time"""
        import time
        
        # Get a batch of test data
        batch = next(iter(test_data.take(1)))
        
        # Warm up
        model.predict(batch[0][:1])
        
        # Measure inference time
        times = []
        for _ in range(num_samples):
            start_time = time.time()
            model.predict(batch[0][:1])
            end_time = time.time()
            times.append(end_time - start_time)
            
        return round(np.mean(times) * 1000, 2)  # Convert to milliseconds
```

### 4. Django Views and API

```python
# classifier_app/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from rest_framework.decorators import api_view
import json
import numpy as np
from PIL import Image
import io
import base64

from .models.architectures import CustomResNet, CustomInception, CustomMobileNet
from .services.model_trainer import AdvancedModelTrainer

class ModelEnsemble:
    def __init__(self):
        self.models = {}
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            self.models['resnet'] = tf.keras.models.load_model('ai_models/resnet_best.h5')
            self.models['inception'] = tf.keras.models.load_model('ai_models/inception_best.h5')
            self.models['mobilenet'] = tf.keras.models.load_model('ai_models/mobilenet_best.h5')
        except:
            print("Models not found, creating new ones...")
            self.models['resnet'] = CustomResNet(num_classes=10)
            self.models['inception'] = CustomInception(num_classes=10)
            self.models['mobilenet'] = CustomMobileNet(num_classes=10)
    
    def preprocess_image(self, image_data):
        """Preprocess image for prediction"""
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB and resize
        image = image.convert('RGB')
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict_ensemble(self, image_array):
        """Make predictions using ensemble of models"""
        predictions = {}
        ensemble_pred = np.zeros((1, len(self.class_names)))
        
        for name, model in self.models.items():
            pred = model.predict(image_array)
            predictions[name] = {
                'probabilities': pred[0].tolist(),
                'predicted_class': self.class_names[np.argmax(pred[0])],
                'confidence': float(np.max(pred[0]))
            }
            ensemble_pred += pred
        
        # Ensemble prediction (average)
        ensemble_pred /= len(self.models)
        predictions['ensemble'] = {
            'probabilities': ensemble_pred[0].tolist(),
            'predicted_class': self.class_names[np.argmax(ensemble_pred[0])],
            'confidence': float(np.max(ensemble_pred[0]))
        }
        
        return predictions

# Global ensemble instance
model_ensemble = ModelEnsemble()

def classifier_dashboard(request):
    """Main dashboard view"""
    context = {
        'models': ['ResNet', 'Inception', 'MobileNet'],
        'classes': model_ensemble.class_names
    }
    return render(request, 'classifier_app/dashboard.html', context)

@csrf_exempt
@api_view(['POST'])
def classify_image(request):
    """API endpoint for image classification"""
    try:
        data = json.loads(request.body)
        image_data = data.get('image')
        
        if not image_data:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        # Preprocess image
        image_array = model_ensemble.preprocess_image(image_data)
        
        # Get predictions from all models
        predictions = model_ensemble.predict_ensemble(image_array)
        
        return JsonResponse({
            'success': True,
            'predictions': predictions,
            'model_comparison': {
                'best_individual': max(predictions.items(), 
                                     key=lambda x: x[1]['confidence'] if x[0] != 'ensemble' else 0)[0],
                'ensemble_confidence': predictions['ensemble']['confidence']
            }
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['GET'])
def model_performance(request):
    """API endpoint to get model performance metrics"""
    # This would typically load from saved metrics
    performance_data = {
        'resnet': {
            'accuracy': 0.892,
            'top3_accuracy': 0.976,
            'model_size_mb': 45.2,
            'inference_time_ms': 23.4,
            'strengths': ['Good generalization', 'Stable training'],
            'weaknesses': ['Large model size', 'Slower inference']
        },
        'inception': {
            'accuracy': 0.887,
            'top3_accuracy': 0.971,
            'model_size_mb': 52.1,
            'inference_time_ms': 28.7,
            'strengths': ['Multi-scale features', 'Good accuracy'],
            'weaknesses': ['Complex architecture', 'Memory intensive']
        },
        'mobilenet': {
            'accuracy': 0.863,
            'top3_accuracy': 0.952,
            'model_size_mb': 12.8,
            'inference_time_ms': 8.2,
            'strengths': ['Lightweight', 'Fast inference'],
            'weaknesses': ['Lower accuracy', 'Limited capacity']
        }
    }
    
    return JsonResponse(performance_data)

@api_view(['POST'])
def train_models(request):
    """API endpoint to trigger model training"""
    try:
        # This would typically be a background task
        trainer = AdvancedModelTrainer()
        
        # Create models
        resnet_model = CustomResNet(num_classes=10)
        inception_model = CustomInception(num_classes=10)
        mobilenet_model = CustomMobileNet(num_classes=10)
        
        # Load CIFAR-10 data (or custom dataset)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Preprocess data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        # Split training data
        val_split = int(0.1 * len(x_train))
        x_val, y_val = x_train[:val_split], y_train[:val_split]
        x_train, y_train = x_train[val_split:], y_train[val_split:]
        
        # Train models (this would be done asynchronously in production)
        results = {}
        
        # Note: In production, use Celery for background tasks
        return JsonResponse({
            'message': 'Training started',
            'status': 'Training models in background...'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
```

### 5. URL Configuration

```python
# classifier_app/urls.py
from django.urls import path
from . import views

app_name = 'classifier_app'

urlpatterns = [
    path('', views.classifier_dashboard, name='dashboard'),
    path('api/classify/', views.classify_image, name='classify'),
    path('api/performance/', views.model_performance, name='performance'),
    path('api/train/', views.train_models, name='train'),
]
```

```python
# main_project/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('classifier_app.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

### 6. Model Utilities and Evaluation

```python
# classifier_app/utils/model_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

class ModelAnalyzer:
    def __init__(self, models, class_names):
        self.models = models
        self.class_names = class_names
        
    def generate_confusion_matrices(self, test_data, test_labels):
        """Generate confusion matrices for all models"""
        matrices = {}
        
        for name, model in self.models.items():
            predictions = model.predict(test_data)
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(test_labels, axis=1)
            
            cm = confusion_matrix(true_classes, pred_classes)
            matrices[name] = cm
            
        return matrices
    
    def analyze_model_behaviors(self, test_data, test_labels):
        """Analyze where models agree/disagree"""
        all_predictions = {}
        
        for name, model in self.models.items():
            predictions = model.predict(test_data)
            all_predictions[name] = np.argmax(predictions, axis=1)
        
        # Find samples where all models agree
        agreement_mask = np.ones(len(test_data), dtype=bool)
        for i in range(len(test_data)):
            preds = [all_predictions[name][i] for name in self.models.keys()]
            if len(set(preds)) > 1:
                agreement_mask[i] = False
        
        agreement_rate = np.mean(agreement_mask)
        
        return {
            'agreement_rate': agreement_rate,
            'disagreement_samples': np.where(~agreement_mask)[0],
            'predictions': all_predictions
        }
    
    def feature_importance_analysis(self, model, test_image):
        """Analyze feature importance using gradient-based methods"""
        # Simplified Grad-CAM implementation
        last_conv_layer = None
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Conv layer
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(test_image)
            class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]
        
        # Calculate gradients
        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        return heatmap.numpy()

class EnsembleOptimizer:
    def __init__(self, models, validation_data, validation_labels):
        self.models = models
        self.val_data = validation_data
        self.val_labels = validation_labels
        
    def optimize_ensemble_weights(self):
        """Find optimal weights for ensemble combination"""
        from scipy.optimize import minimize
        
        # Get predictions from all models
        model_predictions = []
        for model in self.models.values():
            preds = model.predict(self.val_data)
            model_predictions.append(preds)
        
        model_predictions = np.array(model_predictions)
        
        def ensemble_loss(weights):
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
            
            ensemble_pred = np.sum(
                weights[:, np.newaxis, np.newaxis] * model_predictions, 
                axis=0
            )
            
            # Cross-entropy loss
            epsilon = 1e-15
            ensemble_pred = np.clip(ensemble_pred, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(self.val_labels * np.log(ensemble_pred), axis=1))
            
            return loss
        
        # Initial weights (equal)
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # Constraints: weights sum to 1, all positive
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(ensemble_loss, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return dict(zip(self.models.keys(), result.x))

# Advanced data augmentation strategies
class AdvancedAugmentation:
    def __init__(self):
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0
        
    def mixup(self, x, y, alpha=0.2):
        """MixUp augmentation"""
        batch_size = tf.shape(x)[0]
        lambda_val = tf.random.uniform([batch_size, 1, 1, 1], 0, alpha)
        
        # Shuffle the batch
        indices = tf.random.shuffle(tf.range(batch_size))
        x_shuffled = tf.gather(x, indices)
        y_shuffled = tf.gather(y, indices)
        
        # Mix inputs and labels
        x_mixed = lambda_val * x + (1 - lambda_val) * x_shuffled
        y_mixed = lambda_val[:, :, 0, 0:1] * y + (1 - lambda_val[:, :, 0, 0:1]) * y_shuffled
        
        return x_mixed, y_mixed
    
    def cutmix(self, x, y, alpha=1.0):
        """CutMix augmentation"""
        batch_size = tf.shape(x)[0]
        image_size = tf.shape(x)[1]
        
        # Sample lambda from beta distribution
        lambda_val = tf.random.uniform([batch_size], 0, alpha)
        
        # Random crop coordinates
        cut_ratio = tf.sqrt(1 - lambda_val)
        cut_w = tf.cast(cut_ratio * tf.cast(image_size, tf.float32), tf.int32)
        cut_h = cut_w
        
        cx = tf.random.uniform([batch_size], 0, image_size, dtype=tf.int32)
        cy = tf.random.uniform([batch_size], 0, image_size, dtype=tf.int32)
        
        # Create masks and apply CutMix
        # Simplified implementation - in practice, would use proper masking
        indices = tf.random.shuffle(tf.range(batch_size))
        x_shuffled = tf.gather(x, indices)
        y_shuffled = tf.gather(y, indices)
        
        return x_shuffled, y_shuffled  # Simplified for demonstration
```

### 7. Advanced Training Pipeline

```python
# classifier_app/training/advanced_trainer.py
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import wandb  # For experiment tracking (optional)

class AdvancedTrainingPipeline:
    def __init__(self, models, config):
        self.models = models
        self.config = config
        self.augmentation = AdvancedAugmentation()
        
    def create_advanced_callbacks(self, model_name):
        """Create sophisticated callbacks for training"""
        callbacks = []
        
        # Learning rate scheduling
        def lr_schedule(epoch, lr):
            if epoch < 10:
                return lr
            elif epoch < 20:
                return lr * 0.1
            else:
                return lr * 0.01
        
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule))
        
        # Custom validation callback
        class ValidationCallback(Callback):
            def __init__(self, validation_data):
                self.validation_data = validation_data
                self.best_val_acc = 0
                
            def on_epoch_end(self, epoch, logs=None):
                val_loss, val_acc = self.model.evaluate(
                    self.validation_data, verbose=0
                )
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    print(f"New best validation accuracy: {val_acc:.4f}")
                    
                # Log to wandb if available
                try:
                    wandb.log({
                        'epoch': epoch,
                        'val_accuracy': val_acc,
                        'val_loss': val_loss
                    })
                except:
                    pass
        
        # Add other callbacks
        callbacks.extend([
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7
            )
        ])
        
        return callbacks
    
    def train_with_advanced_techniques(self, model, model_name, train_data, val_data):
        """Train model with advanced techniques"""
        
        # Compile with advanced optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.config['learning_rate'],
            weight_decay=0.01
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Create callbacks
        callbacks = self.create_advanced_callbacks(model_name)
        
        # Apply data augmentation during training
        def augmented_generator(data_gen):
            for x_batch, y_batch in data_gen:
                if np.random.random() > 0.5:
                    x_batch, y_batch = self.augmentation.mixup(x_batch, y_batch)
                yield x_batch, y_batch
        
        # Train model
        history = model.fit(
            train_data,
            epochs=self.config['epochs'],
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history

class ModelEvaluationSuite:
    def __init__(self, models, test_data, test_labels, class_names):
        self.models = models
        self.test_data = test_data
        self.test_labels = test_labels
        self.class_names = class_names
        
    def comprehensive_evaluation(self):
        """Perform comprehensive evaluation of all models"""
        results = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Basic metrics
            test_loss, test_acc, test_top3 = model.evaluate(
                self.test_data, self.test_labels, verbose=0
            )
            
            # Detailed predictions
            predictions = model.predict(self.test_data)
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(self.test_labels, axis=1)
            
            # Classification report
            class_report = classification_report(
                true_classes, pred_classes,
                target_names=self.class_names,
                output_dict=True
            )
            
            # Per-class accuracy
            per_class_acc = {}
            for i, class_name in enumerate(self.class_names):
                class_mask = true_classes == i
                if np.sum(class_mask) > 0:
                    per_class_acc[class_name] = np.mean(
                        pred_classes[class_mask] == true_classes[class_mask]
                    )
            
            # Confidence analysis
            max_confidences = np.max(predictions, axis=1)
            avg_confidence = np.mean(max_confidences)
            confidence_correct = np.mean(max_confidences[pred_classes == true_classes])
            confidence_incorrect = np.mean(max_confidences[pred_classes != true_classes])
            
            results[name] = {
                'test_accuracy': test_acc,
                'test_top3_accuracy': test_top3,
                'test_loss': test_loss,
                'classification_report': class_report,
                'per_class_accuracy': per_class_acc,
                'avg_confidence': avg_confidence,
                'confidence_when_correct': confidence_correct,
                'confidence_when_incorrect': confidence_incorrect,
                'prediction_distribution': np.bincount(pred_classes)
            }
            
        return results
    
    def generate_evaluation_report(self, results):
        """Generate comprehensive evaluation report"""
        report = {
            'summary': {
                'best_accuracy': max(results.keys(), key=lambda k: results[k]['test_accuracy']),
                'most_confident': max(results.keys(), key=lambda k: results[k]['avg_confidence']),
                'most_calibrated': min(results.keys(), key=lambda k: abs(
                    results[k]['confidence_when_correct'] - results[k]['confidence_when_incorrect']
                ))
            },
            'detailed_comparison': results,
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results):
        """Generate usage recommendations based on results"""
        recommendations = {}
        
        for name, metrics in results.items():
            rec = []
            
            if metrics['test_accuracy'] > 0.9:
                rec.append("High accuracy - suitable for production")
            
            if metrics['avg_confidence'] > 0.8:
                rec.append("High confidence predictions")
                
            if metrics['confidence_when_correct'] - metrics['confidence_when_incorrect'] > 0.2:
                rec.append("Well-calibrated confidence scores")
            
            # Analyze per-class performance
            weak_classes = [
                cls for cls, acc in metrics['per_class_accuracy'].items() 
                if acc < 0.7
            ]
            if weak_classes:
                rec.append(f"Struggles with: {', '.join(weak_classes)}")
            
            recommendations[name] = rec
            
        return recommendations
```

### 8. Frontend Implementation

```html
<!-- classifier_app/templates/classifier_app/dashboard.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced CNN Classifier</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            color: #333;
            margin: 0;
            font-size: 2.5em;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .upload-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .drag-drop-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .drag-drop-area:hover {
            border-color: #764ba2;
            background: rgba(102, 126, 234, 0.05);
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .model-result {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .model-result h3 {
            margin: 0 0 15px 0;
            color: #333;
        }
        
        .confidence-bar {
            background: #e9ecef;
            border-radius: 10px;
            height: 10px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .performance-section {
            margin-top: 40px;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 30px;
        }
        
        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Advanced CNN Classifier</h1>
            <p>Compare ResNet, Inception, and MobileNet architectures</p>
        </div>
        
        <div class="upload-section">
            <div class="drag-drop-area" id="dragDropArea">
                <h3>Drop an image here or click to upload</h3>
                <p>Supported formats: JPG, PNG, GIF</p>
                <input type="file" id="imageInput" accept="image/*" style="display: none;">
            </div>
        </div>
        
        <div class="loading" id="loadingDiv">
            <div class="spinner"></div>
            <p>Analyzing image with multiple CNN architectures...</p>
        </div>
        
        <div id="resultsContainer" style="display: none;">
            <div class="results-grid" id="resultsGrid">
                <!-- Results will be populated here -->
            </div>
        </div>
        
        <div class="performance-section">
            <h2>Model Performance Comparison</h2>
            <div style="height: 400px;">
                <canvas id="performanceChart"></canvas>
            </div>
            <button class="btn" onclick="loadPerformanceData()">Load Performance Metrics</button>
        </div>
    </div>

    <script>
        let performanceChart = null;
        
        // File upload handling
        const dragDropArea = document.getElementById('dragDropArea');
        const imageInput = document.getElementById('imageInput');
        
        dragDropArea.addEventListener('click', () => imageInput.click());
        dragDropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            dragDropArea.style.borderColor = '#764ba2';
            dragDropArea.style.background = 'rgba(102, 126, 234, 0.1)';
        });
        
        dragDropArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dragDropArea.style.borderColor = '#667eea';
            dragDropArea.style.background = 'transparent';
        });
        
        dragDropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            dragDropArea.style.borderColor = '#667eea';
            dragDropArea.style.background = 'transparent';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleImageUpload(files[0]);
            }
        });
        
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleImageUpload(e.target.files[0]);
            }
        });
        
        function handleImageUpload(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                classifyImage(e.target.result);
            };
            reader.readAsDataURL(file);
        }
        
        async function classifyImage(imageData) {
            document.getElementById('loadingDiv').style.display = 'block';
            document.getElementById('resultsContainer').style.display = 'none';
            
            try {
                const response = await fetch('/api/classify/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: imageData
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data.predictions, data.model_comparison);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error classifying image: ' + error.message);
            } finally {
                document.getElementById('loadingDiv').style.display = 'none';
            }
        }
        
        function displayResults(predictions, comparison) {
            const resultsGrid = document.getElementById('resultsGrid');
            resultsGrid.innerHTML = '';
            
            const modelOrder = ['resnet', 'inception', 'mobilenet', 'ensemble'];
            const modelNames = {
                'resnet': 'ResNet',
                'inception': 'Inception',
                'mobilenet': 'MobileNet',
                'ensemble': 'Ensemble'
            };
            
            modelOrder.forEach(modelKey => {
                if (predictions[modelKey]) {
                    const prediction = predictions[modelKey];
                    const resultDiv = document.createElement('div');
                    resultDiv.className = 'model-result';
                    
                    if (modelKey === 'ensemble') {
                        resultDiv.style.borderLeft = '4px solid #28a745';
                    }
                    
                    resultDiv.innerHTML = `
                        <h3>${modelNames[modelKey]} ${modelKey === 'ensemble' ? '🏆' : ''}</h3>
                        <p><strong>Prediction:</strong> ${prediction.predicted_class}</p>
                        <p><strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(2)}%</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${prediction.confidence * 100}%"></div>
                        </div>
                        <div class="top-predictions">
                            <h4>Top 3 Predictions:</h4>
                            ${getTopPredictions(prediction.probabilities)}
                        </div>
                    `;
                    
                    resultsGrid.appendChild(resultDiv);
                }
            });
            
            // Add comparison summary
            const summaryDiv = document.createElement('div');
            summaryDiv.className = 'model-result';
            summaryDiv.style.borderLeft = '4px solid #ffc107';
            summaryDiv.innerHTML = `
                <h3>📊 Analysis Summary</h3>
                <p><strong>Best Individual Model:</strong> ${comparison.best_individual}</p>
                <p><strong>Ensemble Confidence:</strong> ${(comparison.ensemble_confidence * 100).toFixed(2)}%</p>
                <p><strong>Recommendation:</strong> ${comparison.ensemble_confidence > 0.9 ? 'High confidence prediction' : 'Consider ensemble result for better accuracy'}</p>
            `;
            resultsGrid.appendChild(summaryDiv);
            
            document.getElementById('resultsContainer').style.display = 'block';
        }
        
        function getTopPredictions(probabilities) {
            const classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
            const predictions = probabilities.map((prob, index) => ({
                class: classes[index],
                probability: prob
            }));
            
            predictions.sort((a, b) => b.probability - a.probability);
            
            return predictions.slice(0, 3).map(pred => 
                `<div style="margin: 5px 0;">
                    ${pred.class}: ${(pred.probability * 100).toFixed(1)}%
                </div>`
            ).join('');
        }
        
        async function loadPerformanceData() {
            try {
                const response = await fetch('/api/performance/');
                const data = await response.json();
                
                createPerformanceChart(data);
            } catch (error) {
                console.error('Error loading performance data:', error);
            }
        }
        
        function createPerformanceChart(data) {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            
            if (performanceChart) {
                performanceChart.destroy();
            }
            
            const models = Object.keys(data);
            const accuracyData = models.map(model => data[model].accuracy * 100);
            const sizeData = models.map(model => data[model].model_size_mb);
            const speedData = models.map(model => 1000 / data[model].inference_time_ms); // Inferences per second
            
            performanceChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Accuracy (%)', 'Speed (fps)', 'Efficiency (1/MB)', 'Top-3 Accuracy (%)', 'Overall Score'],
                    datasets: [
                        {
                            label: 'ResNet',
                            data: [
                                data.resnet.accuracy * 100,
                                1000 / data.resnet.inference_time_ms,
                                100 / data.resnet.model_size_mb,
                                data.resnet.top3_accuracy * 100,
                                calculateOverallScore(data.resnet)
                            ],
                            backgroundColor: 'rgba(102, 126, 234, 0.2)',
                            borderColor: 'rgba(102, 126, 234, 1)',
                            pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                        },
                        {
                            label: 'Inception',
                            data: [
                                data.inception.accuracy * 100,
                                1000 / data.inception.inference_time_ms,
                                100 / data.inception.model_size_mb,
                                data.inception.top3_accuracy * 100,
                                calculateOverallScore(data.inception)
                            ],
                            backgroundColor: 'rgba(118, 75, 162, 0.2)',
                            borderColor: 'rgba(118, 75, 162, 1)',
                            pointBackgroundColor: 'rgba(118, 75, 162, 1)',
                        },
                        {
                            label: 'MobileNet',
                            data: [
                                data.mobilenet.accuracy * 100,
                                1000 / data.mobilenet.inference_time_ms,
                                100 / data.mobilenet.model_size_mb,
                                data.mobilenet.top3_accuracy * 100,
                                calculateOverallScore(data.mobilenet)
                            ],
                            backgroundColor: 'rgba(40, 167, 69, 0.2)',
                            borderColor: 'rgba(40, 167, 69, 1)',
                            pointBackgroundColor: 'rgba(40, 167, 69, 1)',
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        title: {
                            display: true,
                            text: 'Model Performance Comparison'
                        }
                    }
                }
            });
        }
        
        function calculateOverallScore(modelData) {
            // Weighted score considering accuracy (40%), speed (30%), efficiency (30%)
            const accuracyScore = modelData.accuracy * 100;
            const speedScore = Math.min((1000 / modelData.inference_time_ms) * 2, 100); // Normalize speed
            const efficiencyScore = Math.min((100 / modelData.model_size_mb) * 10, 100); // Normalize efficiency
            
            return (accuracyScore * 0.4 + speedScore * 0.3 + efficiencyScore * 0.3);
        }
        
        // Load performance data on page load
        window.addEventListener('load', () => {
            loadPerformanceData();
        });
    </script>
</body>
</html>

## Assignment: Architecture Performance Analyzer

Create a Python program that implements and compares the computational efficiency of different CNN architectures. Your program should:

1. **Implement three different conv blocks**: Standard convolution, ResNet block with skip connection, and MobileNet depthwise separable convolution
2. **Create a timing function** that measures forward pass time for each architecture
3. **Calculate parameter counts** for each architecture
4. **Generate a comparison report** showing:
   - Parameter count for each architecture
   - Forward pass time (averaged over 100 runs)
   - Memory usage estimation
   - Accuracy vs efficiency trade-offs discussion

**Requirements:**
- Use the code examples provided as starting points
- Test with input tensors of size (1, 3, 224, 224) - standard ImageNet size
- Include error handling for different input sizes
- Create visualizations comparing the three architectures
- Write a brief analysis explaining when you would choose each architecture

**Deliverable:** Submit a Jupyter notebook with your implementation, timing results, and analysis. Include at least one graph showing the parameter count vs computational time trade-offs.

This assignment will deepen your understanding of how architectural choices impact both performance and efficiency - crucial knowledge for deploying models in real-world scenarios where computational resources are often limited.

---

## Summary

Today, we've explored four revolutionary approaches to CNN architecture design:

1. **ResNet's skip connections** solve the vanishing gradient problem, like having direct communication lines in our restaurant
2. **Inception networks** use parallel processing paths, like a chef using multiple cooking methods simultaneously  
3. **MobileNet** achieves efficiency through depthwise separable convolutions, perfect for resource-constrained environments
4. **Object detection networks** like YOLO locate and classify objects simultaneously, like identifying ingredients in a busy market

Each architecture represents a different solution to specific challenges in computer vision, and understanding their strengths helps you choose the right tool for your particular task. Just as master chefs select different techniques based on their ingredients and desired outcomes, you now have the knowledge to select the appropriate CNN architecture for your deep learning projects.