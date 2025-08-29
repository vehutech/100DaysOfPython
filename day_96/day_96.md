# AI Mastery Course - Day 96: AI System Architecture

## Learning Objective
By the end of this lesson, you will understand how to design and implement scalable AI system architectures using Python and Django, enabling you to build robust, distributed AI applications that can handle real-world production demands.

---

## Introduction

Imagine that you're running the busiest restaurant in the city. Your establishment has multiple cooking stations, each specialized for different types of dishes - appetizers, main courses, desserts, and beverages. Each station operates independently but coordinates seamlessly to deliver complete meals to hundreds of customers simultaneously. Just like this well-orchestrated culinary operation, modern AI systems require carefully designed architectures that can scale, distribute workloads, and deliver intelligent responses efficiently across multiple platforms and environments.

---

## 1. Scalable ML System Design

### The Foundation of Your Culinary Empire

Just as a master restaurateur plans their kitchen layout for maximum efficiency, we need to design our ML systems with scalability in mind from the ground up.

#### Key Principles:
- **Separation of Concerns**: Like having dedicated prep cooks, line cooks, and plating specialists
- **Horizontal Scaling**: Adding more cooking stations when demand increases
- **Load Distribution**: Spreading orders across multiple preparation areas
- **Data Pipeline Management**: Ensuring ingredients (data) flow smoothly through the entire process

#### Code Example: Django-based ML Model Server

```python
# models.py - Our recipe book for AI models
from django.db import models
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MLModel(models.Model):
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=20)
    model_file = models.FileField(upload_to='ml_models/')
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)
    
    class Meta:
        unique_together = ['name', 'version']

    def load_model(self):
        """Load the pickled model - like retrieving a recipe from storage"""
        with open(self.model_file.path, 'rb') as f:
            return pickle.load(f)

# views.py - Our head chef coordinating all operations
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loaded models like a sous chef manages recipes"""
    def __init__(self):
        self.loaded_models = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def get_model(self, model_name):
        if model_name not in self.loaded_models:
            try:
                ml_model = MLModel.objects.get(name=model_name, is_active=True)
                self.loaded_models[model_name] = ml_model.load_model()
                logger.info(f"Model {model_name} loaded successfully")
            except MLModel.DoesNotExist:
                logger.error(f"Model {model_name} not found")
                return None
        return self.loaded_models[model_name]

# Global model manager - our kitchen coordinator
model_manager = ModelManager()

@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    """Main prediction endpoint - like our order processing system"""
    try:
        # Parse incoming order (request data)
        data = json.loads(request.body)
        model_name = data.get('model_name')
        features = np.array(data.get('features')).reshape(1, -1)
        
        # Get the right specialist (model) for this dish (prediction)
        model = model_manager.get_model(model_name)
        if model is None:
            return JsonResponse({
                'error': 'Model not available'
            }, status=404)
        
        # Prepare the dish (make prediction)
        prediction = model.predict(features)
        probability = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        
        # Plate and serve (return response)
        response = {
            'prediction': prediction.tolist(),
            'model_name': model_name,
            'timestamp': timezone.now().isoformat()
        }
        
        if probability is not None:
            response['probability'] = probability.tolist()
            
        return JsonResponse(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JsonResponse({
            'error': 'Internal server error'
        }, status=500)
```

**Syntax Explanation:**
- `@csrf_exempt`: Disables CSRF protection for API endpoints (like having a separate entrance for delivery orders)
- `@require_http_methods(["POST"])`: Restricts endpoint to POST requests only
- `ThreadPoolExecutor(max_workers=4)`: Creates a pool of worker threads for parallel processing
- `reshape(1, -1)`: Transforms input data into the format expected by scikit-learn models
- `hasattr(model, 'predict_proba')`: Checks if the model supports probability predictions

---

## 2. Microservices for AI Applications

### Creating Specialized Kitchen Stations

Just as a professional establishment has specialized stations for different culinary tasks, microservices architecture breaks down AI applications into focused, independent services.

#### Code Example: Authentication Microservice

```python
# auth_service/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
import jwt
from datetime import datetime, timedelta
from django.conf import settings

class APIUser(AbstractUser):
    """Extended user model for API access"""
    api_key = models.CharField(max_length=255, unique=True, null=True, blank=True)
    rate_limit = models.IntegerField(default=1000)  # requests per hour
    is_premium = models.BooleanField(default=False)
    
    def generate_api_key(self):
        """Generate a new API key - like issuing a staff badge"""
        import secrets
        self.api_key = f"ak_{secrets.token_urlsafe(32)}"
        self.save()
        return self.api_key
    
    def create_access_token(self):
        """Create JWT token - like a time-stamped order ticket"""
        payload = {
            'user_id': self.id,
            'username': self.username,
            'is_premium': self.is_premium,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, settings.SECRET_KEY, algorithm='HS256')

# auth_service/middleware.py
from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin
import jwt
from django.conf import settings
from .models import APIUser
import time
from django.core.cache import cache

class APIAuthMiddleware(MiddlewareMixin):
    """Authentication middleware - like a maître d' checking reservations"""
    
    def process_request(self, request):
        # Skip authentication for certain paths
        skip_paths = ['/health/', '/docs/']
        if any(request.path.startswith(path) for path in skip_paths):
            return None
            
        # Extract API key or token
        api_key = request.headers.get('X-API-Key')
        auth_header = request.headers.get('Authorization')
        
        if api_key:
            return self._authenticate_api_key(request, api_key)
        elif auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            return self._authenticate_token(request, token)
        else:
            return JsonResponse({
                'error': 'Authentication required'
            }, status=401)
    
    def _authenticate_api_key(self, request, api_key):
        """Authenticate using API key"""
        try:
            user = APIUser.objects.get(api_key=api_key, is_active=True)
            
            # Rate limiting - like controlling order frequency per table
            cache_key = f"rate_limit_{user.id}"
            current_requests = cache.get(cache_key, 0)
            
            if current_requests >= user.rate_limit:
                return JsonResponse({
                    'error': 'Rate limit exceeded'
                }, status=429)
            
            # Increment request count
            cache.set(cache_key, current_requests + 1, 3600)  # 1 hour
            request.user = user
            
        except APIUser.DoesNotExist:
            return JsonResponse({
                'error': 'Invalid API key'
            }, status=401)
    
    def _authenticate_token(self, request, token):
        """Authenticate using JWT token"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
            user = APIUser.objects.get(id=payload['user_id'])
            request.user = user
            
        except jwt.ExpiredSignatureError:
            return JsonResponse({
                'error': 'Token expired'
            }, status=401)
        except (jwt.DecodeError, APIUser.DoesNotExist):
            return JsonResponse({
                'error': 'Invalid token'
            }, status=401)
```

**Syntax Explanation:**
- `AbstractUser`: Django's built-in base class for custom user models
- `secrets.token_urlsafe(32)`: Generates a cryptographically secure random string
- `jwt.encode()` and `jwt.decode()`: Create and verify JSON Web Tokens
- `MiddlewareMixin`: Base class for Django middleware components
- `cache.get()` and `cache.set()`: Redis/Memcached operations for rate limiting

---

## 3. Real-time Inference Systems

### The Express Lane Kitchen

Some orders need to be prepared and served immediately - like a bustling coffee bar during morning rush hour. Real-time inference systems provide instant AI responses.

#### Code Example: WebSocket-based Real-time Predictions

```python
# consumers.py - Our real-time order processing system
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
import asyncio
import numpy as np
from .models import MLModel
import logging

logger = logging.getLogger(__name__)

class PredictionConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time predictions"""
    
    async def connect(self):
        """Accept WebSocket connection - like seating a customer"""
        self.user = self.scope["user"]
        if not self.user.is_authenticated:
            await self.close()
            return
            
        # Join a prediction group based on user type
        self.group_name = f"predictions_{self.user.id}"
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Send welcome message with available models
        available_models = await self.get_available_models()
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'available_models': available_models,
            'message': 'Connected to real-time prediction service'
        }))
    
    async def disconnect(self, close_code):
        """Handle disconnection - like clearing a table"""
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        """Handle incoming prediction requests"""
        try:
            data = json.loads(text_data)
            request_type = data.get('type', 'predict')
            
            if request_type == 'predict':
                await self.handle_prediction(data)
            elif request_type == 'batch_predict':
                await self.handle_batch_prediction(data)
            else:
                await self.send_error('Unknown request type')
                
        except json.JSONDecodeError:
            await self.send_error('Invalid JSON format')
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            await self.send_error('Internal server error')
    
    async def handle_prediction(self, data):
        """Process single prediction request"""
        model_name = data.get('model_name')
        features = data.get('features')
        request_id = data.get('request_id')
        
        if not all([model_name, features, request_id]):
            await self.send_error('Missing required fields', request_id)
            return
        
        # Load model and make prediction
        try:
            model = await self.load_model(model_name)
            features_array = np.array(features).reshape(1, -1)
            
            # Simulate async prediction (in real scenario, might be GPU computation)
            await asyncio.sleep(0.1)  # Small delay for realism
            
            prediction = await database_sync_to_async(model.predict)(features_array)
            
            # Send result back to client
            await self.send(text_data=json.dumps({
                'type': 'prediction_result',
                'request_id': request_id,
                'model_name': model_name,
                'prediction': prediction.tolist(),
                'processing_time': '0.1s',
                'status': 'success'
            }))
            
        except Exception as e:
            await self.send_error(f'Prediction failed: {str(e)}', request_id)
    
    async def handle_batch_prediction(self, data):
        """Process multiple predictions in parallel"""
        batch_data = data.get('batch')
        batch_id = data.get('batch_id')
        
        if not batch_data or not batch_id:
            await self.send_error('Invalid batch data')
            return
        
        # Process predictions concurrently
        tasks = []
        for item in batch_data:
            task = self.process_single_item(item)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Send batch results
        await self.send(text_data=json.dumps({
            'type': 'batch_result',
            'batch_id': batch_id,
            'results': results,
            'total_processed': len(results)
        }))
    
    @database_sync_to_async
    def load_model(self, model_name):
        """Load ML model asynchronously"""
        model_record = MLModel.objects.get(name=model_name, is_active=True)
        return model_record.load_model()
    
    @database_sync_to_async
    def get_available_models(self):
        """Get list of available models"""
        models = MLModel.objects.filter(is_active=True).values_list('name', flat=True)
        return list(models)
    
    async def send_error(self, message, request_id=None):
        """Send error message to client"""
        error_data = {
            'type': 'error',
            'message': message,
            'timestamp': asyncio.get_event_loop().time()
        }
        if request_id:
            error_data['request_id'] = request_id
            
        await self.send(text_data=json.dumps(error_data))

# routing.py - URL routing for WebSocket connections
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/predictions/$', consumers.PredictionConsumer.as_asgi()),
]
```

**Syntax Explanation:**
- `AsyncWebsocketConsumer`: Django Channels class for handling WebSocket connections
- `database_sync_to_async`: Decorator that makes synchronous database operations work in async context
- `asyncio.gather()`: Runs multiple async operations concurrently
- `self.channel_layer.group_add()`: Adds WebSocket connection to a group for broadcasting
- `await asyncio.sleep(0.1)`: Non-blocking delay in async function

---

## 4. Edge AI and Mobile Deployment

### The Food Truck Operation

Sometimes you need to bring your culinary expertise directly to where customers are - like operating a gourmet food truck that serves high-quality meals in remote locations with limited infrastructure.

#### Code Example: Model Optimization for Edge Deployment

```python
# model_optimizer.py - Preparing recipes for mobile kitchens
import tensorflow as tf
import tensorflow_lite as tflite
import numpy as np
from django.core.management.base import BaseCommand
from django.conf import settings
import os
import pickle

class ModelOptimizer:
    """Optimize models for edge deployment"""
    
    def __init__(self):
        self.optimization_strategies = {
            'quantization': self.apply_quantization,
            'pruning': self.apply_pruning,
            'compression': self.apply_compression
        }
    
    def optimize_tensorflow_model(self, model_path, output_path, strategy='quantization'):
        """Convert TensorFlow model to TensorFlow Lite"""
        # Load the original model - like getting the master recipe
        model = tf.keras.models.load_model(model_path)
        
        # Create TensorFlow Lite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply optimization strategy
        if strategy == 'quantization':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._get_representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # Convert model - like adapting recipe for smaller kitchen
        tflite_model = converter.convert()
        
        # Save optimized model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Verify model performance
        original_size = os.path.getsize(model_path)
        optimized_size = len(tflite_model)
        compression_ratio = (1 - optimized_size / original_size) * 100
        
        print(f"Model optimized: {compression_ratio:.1f}% size reduction")
        return output_path
    
    def _get_representative_dataset(self):
        """Generate representative data for quantization"""
        # This would typically use actual training data samples
        for _ in range(100):
            # Generate random data matching your model's input shape
            yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]
    
    def create_edge_inference_api(self, model_path):
        """Create lightweight inference API for edge devices"""
        return EdgeInferenceAPI(model_path)

class EdgeInferenceAPI:
    """Lightweight inference API for edge deployment"""
    
    def __init__(self, model_path):
        # Load TensorFlow Lite model - like setting up a mobile kitchen
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def predict(self, input_data):
        """Make prediction with optimized model"""
        # Preprocess input - like prep work in a food truck
        if isinstance(input_data, list):
            input_data = np.array(input_data, dtype=np.float32)
        
        # Ensure correct input shape
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            input_data
        )
        
        # Run inference - like cooking the order
        self.interpreter.invoke()
        
        # Get prediction - like plating the dish
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        return output_data
    
    def get_model_info(self):
        """Get model metadata"""
        return {
            'input_shape': self.input_details[0]['shape'].tolist(),
            'output_shape': self.output_details[0]['shape'].tolist(),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_dtype': str(self.output_details[0]['dtype'])
        }

# Django view for edge model serving
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
from PIL import Image
import io

# Global edge API instance - like having a ready mobile kitchen
edge_api = None

def initialize_edge_api():
    """Initialize the edge API on startup"""
    global edge_api
    model_path = os.path.join(settings.MEDIA_ROOT, 'edge_models', 'optimized_model.tflite')
    if os.path.exists(model_path):
        edge_api = EdgeInferenceAPI(model_path)

@csrf_exempt
def edge_predict(request):
    """Lightweight prediction endpoint for edge deployment"""
    global edge_api
    
    if edge_api is None:
        return JsonResponse({
            'error': 'Edge model not initialized'
        }, status=503)
    
    try:
        data = json.loads(request.body)
        
        # Handle different input types
        if 'image_base64' in data:
            # Decode base64 image
            image_data = base64.b64decode(data['image_base64'])
            image = Image.open(io.BytesIO(image_data))
            # Convert to numpy array and preprocess
            input_data = np.array(image.resize((224, 224))) / 255.0
        else:
            input_data = np.array(data['features'])
        
        # Make prediction
        prediction = edge_api.predict(input_data)
        
        return JsonResponse({
            'prediction': prediction.tolist(),
            'model_info': edge_api.get_model_info(),
            'inference_time': 'sub-100ms'  # Typical edge inference time
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'Prediction failed: {str(e)}'
        }, status=500)

# Management command for model optimization
class Command(BaseCommand):
    """Django management command to optimize models for edge deployment"""
    help = 'Optimize ML models for edge deployment'
    
    def add_arguments(self, parser):
        parser.add_argument('--model-path', type=str, required=True)
        parser.add_argument('--output-path', type=str, required=True)
        parser.add_argument('--strategy', type=str, default='quantization')
    
    def handle(self, *args, **options):
        optimizer = ModelOptimizer()
        
        output_path = optimizer.optimize_tensorflow_model(
            options['model_path'],
            options['output_path'],
            options['strategy']
        )
        
        self.stdout.write(
            self.style.SUCCESS(f'Model optimized and saved to {output_path}')
        )
```

**Syntax Explanation:**
- `tf.lite.TFLiteConverter`: Converts TensorFlow models to TensorFlow Lite format
- `converter.optimizations = [tf.lite.Optimize.DEFAULT]`: Applies default optimizations
- `converter.representative_dataset`: Provides sample data for quantization
- `self.interpreter.allocate_tensors()`: Allocates memory for the TensorFlow Lite model
- `np.expand_dims(input_data, axis=0)`: Adds batch dimension to input data
- `base64.b64decode()`: Decodes base64-encoded image data

---

## Final Quality Project: Distributed AI Restaurant Management System

Create a comprehensive AI system that manages multiple aspects of a virtual restaurant chain:

### Project Requirements:

1. **Order Prediction Service**: Predict peak hours and popular items
2. **Real-time Inventory Tracking**: WebSocket-based inventory updates
3. **Customer Sentiment Analysis**: Analyze reviews in real-time
4. **Mobile App Integration**: Optimized models for mobile ordering
5. **Multi-location Coordination**: Microservices architecture

### Implementation Steps:

```python
# Project structure:
restaurant_ai/
├── core/                 # Main Django project
├── auth_service/         # Authentication microservice
├── prediction_service/   # ML predictions
├── inventory_service/    # Real-time inventory
├── sentiment_service/    # Text analysis
├── mobile_api/          # Edge deployment
└── orchestrator/        # Service coordination

# Key files to implement:
# 1. docker-compose.yml for microservices
# 2. Kubernetes deployment configs
# 3. Model training pipeline
# 4. API Gateway with rate limiting
# 5. Monitoring and logging system
```

This project will integrate all concepts learned, creating a production-ready AI system that can handle real-world scale and complexity.

---

# Day 96: Scalable AI Architecture - Complete Implementation

## Project: Intelligent Recipe Recommendation System

You'll build a complete scalable AI architecture that handles multiple cooking styles, dietary preferences, and real-time recommendations - just like how a master kitchen operates with different stations working in harmony.

### Core Architecture Components

```python
# project_structure/
├── ai_kitchen/
│   ├── models/
│   │   ├── recipe_classifier.py
│   │   ├── nutrition_analyzer.py
│   │   └── preference_engine.py
│   ├── services/
│   │   ├── inference_service.py
│   │   ├── model_manager.py
│   │   └── cache_service.py
│   ├── api/
│   │   ├── views.py
│   │   ├── serializers.py
│   │   └── urls.py
│   └── deployment/
│       ├── docker/
│       ├── kubernetes/
│       └── monitoring/
```

### 1. Model Components (The Cooking Stations)

```python
# ai_kitchen/models/recipe_classifier.py
import tensorflow as tf
import numpy as np
from django.conf import settings
import joblib
from typing import Dict, List, Tuple
import logging

class RecipeClassifier:
    """
    Classifies recipes by cuisine type and cooking method
    Like having specialized stations for different cooking styles
    """
    
    def __init__(self):
        self.cuisine_model = None
        self.cooking_method_model = None
        self.vectorizer = None
        self.label_encoders = {}
        self.logger = logging.getLogger(__name__)
    
    def load_models(self):
        """Load pre-trained models - like setting up your stations"""
        try:
            model_path = settings.ML_MODELS_PATH
            
            # Load the main classification model
            self.cuisine_model = tf.keras.models.load_model(
                f'{model_path}/cuisine_classifier.h5'
            )
            
            # Load text vectorizer
            self.vectorizer = joblib.load(
                f'{model_path}/recipe_vectorizer.pkl'
            )
            
            # Load label encoders
            self.label_encoders = joblib.load(
                f'{model_path}/label_encoders.pkl'
            )
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise
    
    def preprocess_recipe(self, ingredients: List[str], 
                         instructions: str) -> np.ndarray:
        """
        Prepare recipe data for classification
        Like prepping ingredients before cooking
        """
        # Combine ingredients and instructions
        recipe_text = " ".join(ingredients) + " " + instructions
        
        # Clean and normalize text
        recipe_text = recipe_text.lower().strip()
        
        # Vectorize the text
        if self.vectorizer is None:
            raise ValueError("Vectorizer not loaded")
            
        features = self.vectorizer.transform([recipe_text])
        return features.toarray()
    
    def classify_recipe(self, ingredients: List[str], 
                       instructions: str) -> Dict[str, any]:
        """
        Classify recipe and return predictions with confidence
        """
        try:
            # Preprocess the recipe
            features = self.preprocess_recipe(ingredients, instructions)
            
            # Get predictions
            cuisine_pred = self.cuisine_model.predict(features)
            cuisine_proba = tf.nn.softmax(cuisine_pred).numpy()
            
            # Decode predictions
            cuisine_idx = np.argmax(cuisine_proba[0])
            cuisine_confidence = float(cuisine_proba[0][cuisine_idx])
            
            cuisine_name = self.label_encoders['cuisine'].inverse_transform([cuisine_idx])[0]
            
            return {
                'cuisine_type': cuisine_name,
                'confidence': cuisine_confidence,
                'all_probabilities': {
                    self.label_encoders['cuisine'].inverse_transform([i])[0]: float(prob)
                    for i, prob in enumerate(cuisine_proba[0])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            return {'error': str(e)}

# ai_kitchen/models/nutrition_analyzer.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List
import joblib

class NutritionAnalyzer:
    """
    Analyzes nutritional content of recipes
    Like having a nutrition expert in your kitchen
    """
    
    def __init__(self):
        self.nutrition_model = None
        self.ingredient_encoder = None
        self.nutrition_scaler = None
        
    def load_models(self):
        """Load nutrition analysis models"""
        model_path = settings.ML_MODELS_PATH
        
        self.nutrition_model = joblib.load(
            f'{model_path}/nutrition_regressor.pkl'
        )
        self.ingredient_encoder = joblib.load(
            f'{model_path}/ingredient_encoder.pkl'
        )
        self.nutrition_scaler = joblib.load(
            f'{model_path}/nutrition_scaler.pkl'
        )
    
    def analyze_nutrition(self, ingredients: List[Dict]) -> Dict[str, float]:
        """
        Calculate nutritional values from ingredients
        Each ingredient dict should have: {'name': str, 'amount': float, 'unit': str}
        """
        try:
            # Process ingredients into feature vector
            features = self._encode_ingredients(ingredients)
            
            # Predict nutrition values
            nutrition_raw = self.nutrition_model.predict([features])
            nutrition_scaled = self.nutrition_scaler.inverse_transform(nutrition_raw)
            
            nutrition_dict = {
                'calories': float(nutrition_scaled[0][0]),
                'protein_g': float(nutrition_scaled[0][1]),
                'carbs_g': float(nutrition_scaled[0][2]),
                'fat_g': float(nutrition_scaled[0][3]),
                'fiber_g': float(nutrition_scaled[0][4]),
                'sugar_g': float(nutrition_scaled[0][5]),
                'sodium_mg': float(nutrition_scaled[0][6])
            }
            
            return nutrition_dict
            
        except Exception as e:
            return {'error': f"Nutrition analysis failed: {str(e)}"}
    
    def _encode_ingredients(self, ingredients: List[Dict]) -> np.ndarray:
        """Convert ingredient list to feature vector"""
        # Create feature vector based on ingredient categories
        feature_vector = np.zeros(len(self.ingredient_encoder.categories_[0]))
        
        for ingredient in ingredients:
            try:
                # Get ingredient category index
                ingredient_name = ingredient['name'].lower()
                if ingredient_name in self.ingredient_encoder.categories_[0]:
                    idx = list(self.ingredient_encoder.categories_[0]).index(ingredient_name)
                    feature_vector[idx] += ingredient.get('amount', 1.0)
            except:
                continue  # Skip unknown ingredients
                
        return feature_vector
```

### 2. Service Layer (The Kitchen Management)

```python
# ai_kitchen/services/inference_service.py
import asyncio
import redis
import json
from typing import Dict, List, Optional
from django.conf import settings
from ..models.recipe_classifier import RecipeClassifier
from ..models.nutrition_analyzer import NutritionAnalyzer
from .cache_service import CacheService
import time

class InferenceService:
    """
    Manages AI inference requests with caching and load balancing
    Like the head of kitchen coordinating all stations
    """
    
    def __init__(self):
        self.recipe_classifier = RecipeClassifier()
        self.nutrition_analyzer = NutritionAnalyzer()
        self.cache_service = CacheService()
        self._models_loaded = False
        
    async def initialize(self):
        """Initialize all models and services"""
        if not self._models_loaded:
            # Load models in parallel
            await asyncio.gather(
                self._load_classifier(),
                self._load_nutrition_analyzer()
            )
            self._models_loaded = True
    
    async def _load_classifier(self):
        """Load recipe classifier model"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.recipe_classifier.load_models)
    
    async def _load_nutrition_analyzer(self):
        """Load nutrition analyzer model"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.nutrition_analyzer.load_models)
    
    async def analyze_recipe(self, recipe_data: Dict) -> Dict:
        """
        Complete recipe analysis with caching
        """
        # Create cache key
        cache_key = self._generate_cache_key(recipe_data)
        
        # Check cache first
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            return cached_result
        
        # Ensure models are loaded
        await self.initialize()
        
        # Extract recipe components
        ingredients = recipe_data.get('ingredients', [])
        instructions = recipe_data.get('instructions', '')
        ingredient_details = recipe_data.get('ingredient_details', [])
        
        start_time = time.time()
        
        # Run analysis in parallel
        classification_task = asyncio.create_task(
            self._classify_recipe_async(ingredients, instructions)
        )
        nutrition_task = asyncio.create_task(
            self._analyze_nutrition_async(ingredient_details)
        )
        
        # Wait for both analyses to complete
        classification_result, nutrition_result = await asyncio.gather(
            classification_task, nutrition_task
        )
        
        # Combine results
        final_result = {
            'classification': classification_result,
            'nutrition': nutrition_result,
            'processing_time_ms': round((time.time() - start_time) * 1000, 2),
            'timestamp': time.time()
        }
        
        # Cache the result for 1 hour
        await self.cache_service.set(cache_key, final_result, ttl=3600)
        
        return final_result
    
    async def _classify_recipe_async(self, ingredients: List[str], 
                                   instructions: str) -> Dict:
        """Async wrapper for recipe classification"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.recipe_classifier.classify_recipe, 
            ingredients, 
            instructions
        )
    
    async def _analyze_nutrition_async(self, ingredient_details: List[Dict]) -> Dict:
        """Async wrapper for nutrition analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.nutrition_analyzer.analyze_nutrition, 
            ingredient_details
        )
    
    def _generate_cache_key(self, recipe_data: Dict) -> str:
        """Generate unique cache key for recipe data"""
        import hashlib
        recipe_str = json.dumps(recipe_data, sort_keys=True)
        return f"recipe_analysis:{hashlib.md5(recipe_str.encode()).hexdigest()}"

# ai_kitchen/services/cache_service.py
import redis
import json
import asyncio
from typing import Optional, Dict, Any
from django.conf import settings

class CacheService:
    """
    Redis-based caching service for AI inference results
    Like a prep station that stores commonly used ingredients
    """
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
    
    async def get(self, key: str) -> Optional[Dict]:
        """Get cached result"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.redis_client.get, key
            )
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Dict, ttl: int = 3600):
        """Set cache with TTL"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self.redis_client.setex, 
                key, 
                ttl, 
                json.dumps(value)
            )
        except Exception as e:
            print(f"Cache set error: {e}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache keys matching pattern"""
        try:
            loop = asyncio.get_event_loop()
            keys = await loop.run_in_executor(
                None, self.redis_client.keys, pattern
            )
            if keys:
                await loop.run_in_executor(
                    None, self.redis_client.delete, *keys
                )
        except Exception as e:
            print(f"Cache invalidation error: {e}")
```

### 3. Django API Layer (The Service Counter)

```python
# ai_kitchen/api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator
from django.views.decorators.vary import vary_on_headers
import asyncio
from ..services.inference_service import InferenceService
from .serializers import RecipeAnalysisSerializer
import logging

logger = logging.getLogger(__name__)

class RecipeAnalysisView(APIView):
    """
    Main endpoint for recipe analysis
    Like the main order window where customers place requests
    """
    
    def __init__(self):
        super().__init__()
        self.inference_service = InferenceService()
    
    async def analyze_recipe_async(self, recipe_data):
        """Async recipe analysis"""
        return await self.inference_service.analyze_recipe(recipe_data)
    
    def post(self, request):
        """
        Analyze a recipe for cuisine type and nutrition
        """
        try:
            # Validate input data
            serializer = RecipeAnalysisSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    {'errors': serializer.errors}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            recipe_data = serializer.validated_data
            
            # Run async analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.analyze_recipe_async(recipe_data)
                )
            finally:
                loop.close()
            
            return Response(result, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Recipe analysis failed: {str(e)}")
            return Response(
                {'error': 'Analysis failed', 'details': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class BatchAnalysisView(APIView):
    """
    Batch processing endpoint for multiple recipes
    Like handling catering orders efficiently
    """
    
    def __init__(self):
        super().__init__()
        self.inference_service = InferenceService()
    
    async def process_batch_async(self, recipes_data):
        """Process multiple recipes concurrently"""
        tasks = []
        for recipe_data in recipes_data:
            task = self.inference_service.analyze_recipe(recipe_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    def post(self, request):
        """Process multiple recipes in batch"""
        try:
            recipes = request.data.get('recipes', [])
            
            if not recipes or len(recipes) > 50:  # Limit batch size
                return Response(
                    {'error': 'Invalid batch size (1-50 recipes)'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Process batch
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                results = loop.run_until_complete(
                    self.process_batch_async(recipes)
                )
            finally:
                loop.close()
            
            return Response({
                'results': results,
                'processed_count': len(results)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {str(e)}")
            return Response(
                {'error': 'Batch analysis failed'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

# ai_kitchen/api/serializers.py
from rest_framework import serializers

class IngredientDetailSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=200)
    amount = serializers.FloatField(default=1.0)
    unit = serializers.CharField(max_length=50, default='cup')

class RecipeAnalysisSerializer(serializers.Serializer):
    """
    Serializer for recipe analysis requests
    """
    ingredients = serializers.ListField(
        child=serializers.CharField(max_length=200),
        min_length=1,
        max_length=50
    )
    instructions = serializers.CharField(max_length=5000)
    ingredient_details = serializers.ListField(
        child=IngredientDetailSerializer(),
        required=False,
        default=list
    )
    recipe_name = serializers.CharField(max_length=200, required=False)
    servings = serializers.IntegerField(default=4, min_value=1, max_value=20)
```

### 4. Deployment Configuration (The Kitchen Setup)

```python
# ai_kitchen/deployment/docker/Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p /app/ml_models

# Set environment variables
ENV PYTHONPATH=/app
ENV DJANGO_SETTINGS_MODULE=ai_kitchen.settings.production

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "gevent", "ai_kitchen.wsgi:application"]
```

```yaml
# ai_kitchen/deployment/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-kitchen-api
  labels:
    app: ai-kitchen-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-kitchen-api
  template:
    metadata:
      labels:
        app: ai-kitchen-api
    spec:
      containers:
      - name: ai-kitchen-api
        image: ai-kitchen:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: DB_HOST
          value: "postgres-service"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready/
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ai-kitchen-service
spec:
  selector:
    app: ai-kitchen-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 5. Monitoring and Scaling (Kitchen Performance Tracking)

```python
# ai_kitchen/monitoring/metrics.py
import time
import psutil
import prometheus_client
from django.core.management.base import BaseCommand
from prometheus_client import Counter, Histogram, Gauge
import threading

# Define metrics
REQUEST_COUNT = Counter(
    'recipe_analysis_requests_total', 
    'Total recipe analysis requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'recipe_analysis_duration_seconds',
    'Recipe analysis request duration'
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_type']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections_total',
    'Number of active connections'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage'
)

class MetricsCollector:
    """
    Collects and exports system and application metrics
    Like monitoring the kitchen's performance and efficiency
    """
    
    def __init__(self):
        self.running = False
        self.thread = None
    
    def start_collection(self):
        """Start metrics collection in background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._collect_system_metrics)
            self.thread.daemon = True
            self.thread.start()
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while self.running:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                SYSTEM_MEMORY_USAGE.set(memory.percent)
                
                time.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                print(f"Metrics collection error: {e}")
                time.sleep(60)  # Wait longer on error

# Middleware for request metrics
class MetricsMiddleware:
    """Django middleware to collect request metrics"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        start_time = time.time()
        
        response = self.get_response(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.path,
            status=response.status_code
        ).inc()
        
        return response
```

### 6. Load Testing and Performance Validation

```python
# ai_kitchen/tests/load_test.py
import asyncio
import aiohttp
import time
import json
from typing import List, Dict

class LoadTester:
    """
    Load testing for the AI system
    Like stress-testing your kitchen during rush hour
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
    
    async def create_session(self):
        """Create HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def single_request(self, recipe_data: Dict) -> Dict:
        """Make single API request"""
        try:
            start_time = time.time()
            
            async with self.session.post(
                f'{self.base_url}/api/analyze/',
                json=recipe_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                result = await response.json()
                duration = time.time() - start_time
                
                return {
                    'success': response.status == 200,
                    'duration': duration,
                    'status_code': response.status,
                    'response_size': len(json.dumps(result))
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    async def run_load_test(self, recipe_data: Dict, 
                           concurrent_users: int = 10,
                           requests_per_user: int = 5) -> Dict:
        """
        Run load test with specified parameters
        """
        await self.create_session()
        
        print(f"Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        # Create all tasks
        tasks = []
        for user in range(concurrent_users):
            for request in range(requests_per_user):
                task = self.single_request(recipe_data)
                tasks.append(task)
        
        # Execute all requests
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.time() - start_time
        
        await self.close_session()
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get('success')]
        failed_requests = [r for r in results if not (isinstance(r, dict) and r.get('success'))]
        
        durations = [r['duration'] for r in successful_requests]
        
        stats = {
            'total_requests': len(results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(results) * 100,
            'total_duration': total_duration,
            'requests_per_second': len(results) / total_duration,
            'avg_response_time': sum(durations) / len(durations) if durations else 0,
            'min_response_time': min(durations) if durations else 0,
            'max_response_time': max(durations) if durations else 0
        }
        
        return stats

# Usage example
async def main():
    """Run load test example"""
    load_tester = LoadTester('http://localhost:8000')
    
    test_recipe = {
        'ingredients': ['chicken breast', 'olive oil', 'garlic', 'lemon'],
        'instructions': 'Season chicken, heat oil, cook chicken until done, add garlic and lemon.',
        'ingredient_details': [
            {'name': 'chicken breast', 'amount': 1.0, 'unit': 'pound'},
            {'name': 'olive oil', 'amount': 2.0, 'unit': 'tablespoon'},
            {'name': 'garlic', 'amount': 3.0, 'unit': 'clove'},
            {'name': 'lemon', 'amount': 0.5, 'unit': 'piece'}
        ]
    }
    
    # Test with different load levels
    for users in [5, 10, 25, 50]:
        print(f"\n=== Testing with {users} concurrent users ===")
        stats = await load_tester.run_load_test(
            test_recipe, 
            concurrent_users=users, 
            requests_per_user=3
        )
        
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Requests/second: {stats['requests_per_second']:.2f}")
        print(f"Avg response time: {stats['avg_response_time']:.3f}s")
        print(f"Max response time: {stats['max_response_time']:.3f}s")

if __name__ == '__main__':
    asyncio.run(main())
```

### 7. Complete Settings Configuration

```python
# ai_kitchen/settings/production.py
import os
from .base import *

# Production settings
DEBUG = False
ALLOWED_HOSTS = ['*']

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME', 'ai_kitchen'),
        'USER': os.environ.get('DB_USER', 'postgres'),
        'PASSWORD': os.environ.get('DB_PASSWORD', ''),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
        'OPTIONS': {
            'MAX_CONNS': 20,
        }
    }
}

# Redis configuration
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))

# ML Models path
ML_MODELS_PATH = os.environ.get('ML_MODELS_PATH', '/app/ml_models')

# Caching
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': f'redis://{REDIS_HOST}:{REDIS_PORT}/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '/app/logs/ai_kitchen.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'ai_kitchen': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# Security
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
```

**: Multi-layer caching with Redis
- **Load Balancing**: Kubernetes-based auto-scaling
- **Monitoring**: Comprehensive metrics collection
- **Error Handling**: Graceful degradation and recovery

### 8. Auto-Scaling Configuration

```python
# ai_kitchen/deployment/kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-kitchen-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-kitchen-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### 9. Model Management and Versioning

```python
# ai_kitchen/services/model_manager.py
import asyncio
import json
import os
import shutil
from typing import Dict, List, Optional
from datetime import datetime
import requests
from django.conf import settings
import logging

class ModelManager:
    """
    Manages AI model versions and updates
    Like managing recipe variations and improvements
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_registry = {}
        self.active_models = {}
        
    async def initialize(self):
        """Initialize model manager"""
        await self.load_model_registry()
        await self.load_active_models()
    
    async def load_model_registry(self):
        """Load model registry from configuration"""
        registry_path = os.path.join(settings.ML_MODELS_PATH, 'registry.json')
        
        try:
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
            else:
                # Create default registry
                self.model_registry = {
                    'cuisine_classifier': {
                        'current_version': 'v1.0.0',
                        'versions': {
                            'v1.0.0': {
                                'path': 'cuisine_classifier.h5',
                                'performance_metrics': {
                                    'accuracy': 0.85,
                                    'f1_score': 0.83
                                },
                                'created_at': '2025-01-15T10:00:00Z'
                            }
                        }
                    },
                    'nutrition_regressor': {
                        'current_version': 'v1.0.0', 
                        'versions': {
                            'v1.0.0': {
                                'path': 'nutrition_regressor.pkl',
                                'performance_metrics': {
                                    'mse': 0.15,
                                    'r2_score': 0.78
                                },
                                'created_at': '2025-01-15T10:00:00Z'
                            }
                        }
                    }
                }
                await self.save_model_registry()
                
        except Exception as e:
            self.logger.error(f"Failed to load model registry: {e}")
            raise
    
    async def load_active_models(self):
        """Load currently active model versions"""
        for model_name, model_info in self.model_registry.items():
            current_version = model_info['current_version']
            self.active_models[model_name] = {
                'version': current_version,
                'path': model_info['versions'][current_version]['path'],
                'loaded_at': datetime.now().isoformat()
            }
    
    async def deploy_new_model(self, model_name: str, model_file: str, 
                             version: str, performance_metrics: Dict) -> bool:
        """
        Deploy a new model version with A/B testing capability
        """
        try:
            # Validate model file exists
            model_path = os.path.join(settings.ML_MODELS_PATH, model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Add to registry
            if model_name not in self.model_registry:
                self.model_registry[model_name] = {
                    'current_version': version,
                    'versions': {}
                }
            
            self.model_registry[model_name]['versions'][version] = {
                'path': model_file,
                'performance_metrics': performance_metrics,
                'created_at': datetime.now().isoformat(),
                'status': 'deployed'
            }
            
            # Save registry
            await self.save_model_registry()
            
            self.logger.info(f"Successfully deployed {model_name} version {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            return False
    
    async def rollback_model(self, model_name: str, target_version: str) -> bool:
        """Rollback to previous model version"""
        try:
            if (model_name in self.model_registry and 
                target_version in self.model_registry[model_name]['versions']):
                
                # Update current version
                self.model_registry[model_name]['current_version'] = target_version
                self.active_models[model_name] = {
                    'version': target_version,
                    'path': self.model_registry[model_name]['versions'][target_version]['path'],
                    'loaded_at': datetime.now().isoformat()
                }
                
                await self.save_model_registry()
                
                self.logger.info(f"Rolled back {model_name} to version {target_version}")
                return True
            else:
                raise ValueError(f"Invalid model or version: {model_name}/{target_version}")
                
        except Exception as e:
            self.logger.error(f"Model rollback failed: {e}")
            return False
    
    async def save_model_registry(self):
        """Save model registry to disk"""
        registry_path = os.path.join(settings.ML_MODELS_PATH, 'registry.json')
        
        try:
            os.makedirs(settings.ML_MODELS_PATH, exist_ok=True)
            with open(registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model"""
        return self.active_models.get(model_name)
    
    def list_all_models(self) -> Dict:
        """List all models and their versions"""
        return {
            'active_models': self.active_models,
            'registry': self.model_registry
        }
```

### 10. Circuit Breaker Pattern Implementation

```python
# ai_kitchen/services/circuit_breaker.py
import time
import asyncio
from enum import Enum
from typing import Callable, Any, Optional
import logging

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """
    Circuit breaker pattern for handling service failures
    Like having backup cooking methods when equipment fails
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: int = 60,
                 expected_exception: Exception = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.logger = logging.getLogger(__name__)
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset if we were half open
            if self.state == CircuitState.HALF_OPEN:
                self._reset()
                
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
    
    def _record_failure(self):
        """Record a failure and update circuit state"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if (self.failure_count >= self.failure_threshold or 
            self.state == CircuitState.HALF_OPEN):
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
    
    def _reset(self):
        """Reset circuit breaker to closed state"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.logger.info("Circuit breaker RESET to CLOSED state")
    
    @property
    def status(self) -> Dict:
        """Get current circuit breaker status"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'failure_threshold': self.failure_threshold,
            'timeout': self.timeout
        }

# Integration with inference service
class ResilientInferenceService(InferenceService):
    """
    Enhanced inference service with circuit breaker protection
    """
    
    def __init__(self):
        super().__init__()
        self.classifier_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=30
        )
        self.nutrition_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=30
        )
    
    async def analyze_recipe(self, recipe_data: Dict) -> Dict:
        """Recipe analysis with circuit breaker protection"""
        try:
            # Use circuit breakers for each service
            classification_result = await self.classifier_breaker.call(
                self._classify_recipe_with_fallback,
                recipe_data.get('ingredients', []),
                recipe_data.get('instructions', '')
            )
            
            nutrition_result = await self.nutrition_breaker.call(
                self._analyze_nutrition_with_fallback,
                recipe_data.get('ingredient_details', [])
            )
            
            return {
                'classification': classification_result,
                'nutrition': nutrition_result,
                'circuit_breaker_status': {
                    'classifier': self.classifier_breaker.status,
                    'nutrition': self.nutrition_breaker.status
                }
            }
            
        except Exception as e:
            self.logger.error(f"Resilient analysis failed: {e}")
            return {
                'error': 'Service temporarily unavailable',
                'fallback_used': True
            }
    
    async def _classify_recipe_with_fallback(self, ingredients: List[str], 
                                           instructions: str) -> Dict:
        """Classification with fallback logic"""
        try:
            return await self._classify_recipe_async(ingredients, instructions)
        except Exception as e:
            # Fallback to rule-based classification
            return self._rule_based_classification(ingredients, instructions)
    
    async def _analyze_nutrition_with_fallback(self, ingredient_details: List[Dict]) -> Dict:
        """Nutrition analysis with fallback logic"""
        try:
            return await self._analyze_nutrition_async(ingredient_details)
        except Exception as e:
            # Fallback to basic nutrition estimation
            return self._basic_nutrition_estimation(ingredient_details)
    
    def _rule_based_classification(self, ingredients: List[str], 
                                 instructions: str) -> Dict:
        """Simple rule-based fallback classification"""
        # Basic keyword matching for cuisine types
        asian_keywords = ['soy', 'ginger', 'sesame', 'rice', 'noodles']
        italian_keywords = ['pasta', 'tomato', 'basil', 'parmesan', 'olive']
        mexican_keywords = ['chili', 'cumin', 'lime', 'cilantro', 'beans']
        
        text = ' '.join(ingredients + [instructions]).lower()
        
        scores = {
            'asian': sum(1 for kw in asian_keywords if kw in text),
            'italian': sum(1 for kw in italian_keywords if kw in text),
            'mexican': sum(1 for kw in mexican_keywords if kw in text)
        }
        
        best_cuisine = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_cuisine] / max(1, sum(scores.values()))
        
        return {
            'cuisine_type': best_cuisine if confidence > 0 else 'unknown',
            'confidence': confidence,
            'fallback_method': 'rule_based'
        }
    
    def _basic_nutrition_estimation(self, ingredient_details: List[Dict]) -> Dict:
        """Basic nutrition estimation fallback"""
        # Simple approximations based on common ingredients
        nutrition_db = {
            'chicken': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
            'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3},
            'oil': {'calories': 884, 'protein': 0, 'carbs': 0, 'fat': 100},
            'vegetables': {'calories': 25, 'protein': 1, 'carbs': 5, 'fat': 0.1}
        }
        
        total_nutrition = {'calories': 0, 'protein_g': 0, 'carbs_g': 0, 'fat_g': 0}
        
        for ingredient in ingredient_details:
            ingredient_name = ingredient.get('name', '').lower()
            amount = ingredient.get('amount', 1.0)
            
            # Simple matching
            nutrition = nutrition_db.get('vegetables')  # default
            for key in nutrition_db:
                if key in ingredient_name:
                    nutrition = nutrition_db[key]
                    break
            
            # Scale by amount (rough approximation)
            scale_factor = amount / 100  # per 100g approximation
            for nutrient, value in nutrition.items():
                if nutrient == 'calories':
                    total_nutrition['calories'] += value * scale_factor
                else:
                    total_nutrition[f'{nutrient}_g'] += value * scale_factor
        
        total_nutrition['fallback_method'] = 'basic_estimation'
        return total_nutrition
```

### 11. Complete Integration Test Suite

```python
# ai_kitchen/tests/integration_test.py
import pytest
import asyncio
import json
from django.test import TestCase, TransactionTestCase
from django.test.client import Client
from unittest.mock import patch, MagicMock
from ..services.inference_service import InferenceService
from ..services.model_manager import ModelManager

class IntegrationTestSuite(TransactionTestCase):
    """
    Complete integration tests for the AI architecture
    Like testing the entire kitchen workflow end-to-end
    """
    
    def setUp(self):
        """Set up test environment"""
        self.client = Client()
        self.test_recipe = {
            'ingredients': ['chicken breast', 'olive oil', 'garlic', 'lemon'],
            'instructions': 'Season chicken, cook in oil with garlic and lemon.',
            'ingredient_details': [
                {'name': 'chicken breast', 'amount': 1.0, 'unit': 'pound'},
                {'name': 'olive oil', 'amount': 2.0, 'unit': 'tablespoon'}
            ]
        }
    
    @patch('ai_kitchen.models.recipe_classifier.RecipeClassifier.load_models')
    @patch('ai_kitchen.models.nutrition_analyzer.NutritionAnalyzer.load_models')
    def test_complete_recipe_analysis_flow(self, mock_nutrition_load, mock_classifier_load):
        """Test complete recipe analysis workflow"""
        
        # Mock model loading
        mock_classifier_load.return_value = None
        mock_nutrition_load.return_value = None
        
        # Mock model predictions
        with patch('ai_kitchen.models.recipe_classifier.RecipeClassifier.classify_recipe') as mock_classify:
            mock_classify.return_value = {
                'cuisine_type': 'mediterranean',
                'confidence': 0.85,
                'all_probabilities': {'mediterranean': 0.85, 'italian': 0.15}
            }
            
            with patch('ai_kitchen.models.nutrition_analyzer.NutritionAnalyzer.analyze_nutrition') as mock_nutrition:
                mock_nutrition.return_value = {
                    'calories': 250.0,
                    'protein_g': 35.0,
                    'carbs_g': 2.0,
                    'fat_g': 12.0
                }
                
                # Make API request
                response = self.client.post(
                    '/api/analyze/',
                    json.dumps(self.test_recipe),
                    content_type='application/json'
                )
                
                # Verify response
                self.assertEqual(response.status_code, 200)
                data = response.json()
                
                self.assertIn('classification', data)
                self.assertIn('nutrition', data)
                self.assertEqual(data['classification']['cuisine_type'], 'mediterranean')
                self.assertEqual(data['nutrition']['calories'], 250.0)
    
    def test_batch_processing(self):
        """Test batch recipe processing"""
        batch_data = {
            'recipes': [self.test_recipe] * 3
        }
        
        with patch('ai_kitchen.services.inference_service.InferenceService.analyze_recipe') as mock_analyze:
            mock_analyze.return_value = {
                'classification': {'cuisine_type': 'test', 'confidence': 0.8},
                'nutrition': {'calories': 200}
            }
            
            response = self.client.post(
                '/api/batch-analyze/',
                json.dumps(batch_data),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data['processed_count'], 3)
            self.assertIn('results', data)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self):
        """Test circuit breaker under failure conditions"""
        from ..services.circuit_breaker import CircuitBreaker
        
        # Create circuit breaker
        breaker = CircuitBreaker(failure_threshold=2, timeout=1)
        
        # Function that always fails
        async def failing_function():
            raise Exception("Service error")
        
        # Test failure accumulation
        for i in range(3):
            try:
                await breaker.call(failing_function)
            except Exception:
                pass
        
        # Circuit should be open now
        self.assertEqual(breaker.state.value, 'open')
        
        # Subsequent calls should fail fast
        with self.assertRaises(Exception):
            await breaker.call(failing_function)
    
    @pytest.mark.asyncio
    async def test_model_manager_deployment(self):
        """Test model deployment and versioning"""
        manager = ModelManager()
        await manager.initialize()
        
        # Test model deployment
        success = await manager.deploy_new_model(
            'test_model',
            'test_model.pkl',
            'v2.0.0',
            {'accuracy': 0.90}
        )
        
        self.assertTrue(success)
        
        # Test rollback
        rollback_success = await manager.rollback_model('test_model', 'v1.0.0')
        self.assertTrue(rollback_success)
    
    def test_load_balancing_configuration(self):
        """Test that load balancing configuration is valid"""
        # This would typically test Kubernetes configurations
        # For now, we'll test the Django settings
        from django.conf import settings
        
        # Verify essential settings are present
        self.assertTrue(hasattr(settings, 'ML_MODELS_PATH'))
        self.assertTrue(hasattr(settings, 'REDIS_HOST'))
        
    @patch('redis.Redis')
    def test_caching_functionality(self, mock_redis):
        """Test Redis caching integration"""
        from ..services.cache_service import CacheService
        
        # Mock Redis client
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        cache_service = CacheService()
        
        # Test cache operations
        asyncio.run(cache_service.set('test_key', {'data': 'test'}, 3600))
        mock_redis_client.setex.assert_called_once()
        
    def test_metrics_collection(self):
        """Test metrics collection and export"""
        from ..monitoring.metrics import MetricsCollector, REQUEST_COUNT
        
        # Test metrics collector initialization
        collector = MetricsCollector()
        collector.start_collection()
        
        # Verify metrics are being collected
        initial_count = REQUEST_COUNT._value._value
        
        # Simulate request
        REQUEST_COUNT.labels(method='POST', endpoint='/api/analyze/', status=200).inc()
        
        # Verify count increased
        new_count = REQUEST_COUNT._value._value
        self.assertGreater(new_count, initial_count)
        
        collector.stop_collection()
    
    def test_error_handling_and_graceful_degradation(self):
        """Test system behavior under various error conditions"""
        
        # Test invalid input handling
        invalid_recipe = {'ingredients': []}  # Missing required fields
        
        response = self.client.post(
            '/api/analyze/',
            json.dumps(invalid_recipe),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('errors', response.json())
    
    def tearDown(self):
        """Clean up after tests"""
        # Clean up any test artifacts
        pass

# Performance benchmarking
class PerformanceBenchmark:
    """
    Benchmark the AI system performance
    """
    
    @staticmethod
    async def benchmark_single_inference():
        """Benchmark single inference time"""
        service = InferenceService()
        await service.initialize()
        
        test_recipe = {
            'ingredients': ['chicken', 'rice', 'vegetables'],
            'instructions': 'Cook together',
            'ingredient_details': []
        }
        
        import time
        start_time = time.time()
        
        result = await service.analyze_recipe(test_recipe)
        
        inference_time = time.time() - start_time
        
        print(f"Single inference time: {inference_time:.3f}s")
        return inference_time
    
    @staticmethod
    async def benchmark_concurrent_load():
        """Benchmark concurrent request handling"""
        service = InferenceService()
        await service.initialize()
        
        test_recipe = {
            'ingredients': ['chicken', 'rice'],
            'instructions': 'Cook',
            'ingredient_details': []
        }
        
        # Create concurrent tasks
        tasks = []
        for _ in range(50):
            task = service.analyze_recipe(test_recipe)
            tasks.append(task)
        
        import time
        start_time = time.time()
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        successful_results = [r for r in results if 'error' not in r]
        
        print(f"Concurrent load test:")
        print(f"  Total requests: 50")
        print(f"  Successful: {len(successful_results)}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Requests/second: {50/total_time:.2f}")
        
        return {
            'total_time': total_time,
            'success_rate': len(successful_results) / 50,
            'rps': 50 / total_time
        }

# Run benchmarks
if __name__ == '__main__':
    async def run_benchmarks():
        print("Running performance benchmarks...")
        
        single_time = await PerformanceBenchmark.benchmark_single_inference()
        concurrent_stats = await PerformanceBenchmark.benchmark_concurrent_load()
        
        print(f"\n=== Performance Summary ===")
        print(f"Single inference: {single_time:.3f}s")
        print(f"Concurrent RPS: {concurrent_stats['rps']:.2f}")
        print(f"Success rate: {concurrent_stats['success_rate']*100:.1f}%")
    
    asyncio.run(run_benchmarks())
```

This complete scalable AI architecture demonstrates:

**Key Architecture Principles:**
- **Microservices Design**: Separate services for different AI models
- **Async Processing**: Non-blocking operations for better performance  
- **Caching Strategy**: Multi-layer caching with Redis
- **Load Balancing**: Kubernetes-based auto-scaling
- **Monitoring**: Comprehensive metrics collection
- **Error Handling**: Graceful degradation and recovery
- **Circuit Breakers**: Fault tolerance and resilience
- **Model Management**: Versioning and deployment automation
- **Performance Testing**: Load testing and benchmarking

The system scales like a professional kitchen operation - each component (model, service, cache) works independently but coordinates seamlessly to handle varying loads while maintaining quality and performance.

## Assignment: Implement Model A/B Testing Infrastructure

**Task**: Build a Django-based system that allows for A/B testing of different ML models in production.

**Requirements**:
1. Create a `ModelExperiment` model that tracks different model versions
2. Implement traffic splitting (70% Model A, 30% Model B)
3. Track prediction accuracy and response times
4. Create an admin interface to view experiment results
5. Implement automatic model switching based on performance metrics

**Deliverables**:
- Django models for experiment tracking
- API endpoint that randomly routes requests to different models
- Dashboard showing A/B test results
- Documentation explaining your traffic splitting strategy

**Evaluation Criteria**:
- Code quality and organization
- Proper error handling
- Statistical significance in A/B testing
- User interface design
- Performance optimization

This assignment tests your understanding of system architecture, model management, and production ML practices while being distinctly different from the comprehensive restaurant management project.