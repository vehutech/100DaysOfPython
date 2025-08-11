# AI Mastery Course - Day 85: Model Optimization & Deployment

## Learning Objectives
By the end of this lesson, you will be able to:
- Optimize trained models through quantization and pruning techniques
- Deploy models for mobile applications using TensorFlow Lite
- Set up model serving infrastructure with TensorFlow Serving
- Implement performance monitoring systems for production ML models

---

## Introduction

Imagine that you've spent weeks perfecting a complex recipe in your culinary laboratory. Your dish tastes incredible, but now you need to serve it to hundreds of customers efficiently. You can't just scale up the same complicated process - you need to streamline it, package it properly, and ensure consistent quality at scale. This is exactly what happens when we take our trained machine learning models from development to production.

Just as a master chef optimizes recipes for different serving contexts - perhaps creating a simplified version for a food truck or a pre-packaged version for retail - we need to optimize our models for different deployment scenarios while maintaining their essential capabilities.

---

## 1. Model Quantization and Pruning

### The Art of Refinement

Think of model quantization like reducing a complex sauce recipe. Instead of using 32 different spices in precise measurements, you identify the 8 most impactful ingredients that deliver 95% of the flavor. The result is faster to prepare, uses fewer resources, but maintains the essence of the original.

### Quantization Implementation

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Load your pre-trained model
model = keras.models.load_model('my_trained_model.h5')

# Post-training quantization - the "simplification" process
def quantize_model(model, representative_dataset):
    """
    Convert a model to use 8-bit integers instead of 32-bit floats
    Like converting precise measurements to practical portions
    """
    # Create a TensorFlow Lite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable quantization optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative data for calibration
    converter.representative_dataset = representative_dataset
    
    # Ensure all operations use integers (full quantization)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert the model
    quantized_model = converter.convert()
    
    return quantized_model

# Example representative dataset function
def representative_dataset_generator():
    """
    Provides sample data to calibrate the quantization
    Like taste-testing during the simplification process
    """
    for _ in range(100):
        # Generate sample data matching your model's input shape
        sample_data = np.random.random((1, 224, 224, 3)).astype(np.float32)
        yield [sample_data]

# Apply quantization
quantized_model = quantize_model(model, representative_dataset_generator)

# Save the optimized model
with open('optimized_model.tflite', 'wb') as f:
    f.write(quantized_model)

print(f"Original model size: {len(tf.io.read_file('my_trained_model.h5'))} bytes")
print(f"Quantized model size: {len(quantized_model)} bytes")
```

### Pruning Implementation

```python
import tensorflow_model_optimization as tfmot

def create_pruned_model(base_model, target_sparsity=0.5):
    """
    Remove less important connections, like removing garnishes
    that don't significantly impact the main flavor
    """
    # Define pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=1000
        )
    }
    
    # Apply pruning to the model
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        base_model, **pruning_params
    )
    
    return pruned_model

# Create and compile pruned model
pruned_model = create_pruned_model(model)
pruned_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune the pruned model
# This is like adjusting seasoning after removing some ingredients
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruned_model.fit(train_data, epochs=5, callbacks=callbacks)

# Remove pruning wrapper and export final model
final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
```

**Syntax Explanation:**
- `tf.lite.Optimize.DEFAULT`: Applies standard optimization techniques
- `representative_dataset`: Provides sample data for calibration during quantization
- `target_spec.supported_ops`: Specifies which operations the target hardware supports
- `PolynomialDecay`: Gradually increases sparsity (removes connections) over time
- `prune_low_magnitude`: Removes weights with the smallest absolute values

---

## 2. TensorFlow Lite and Mobile Deployment

### Portable Culinary Creations

Converting your model for mobile is like creating a travel-friendly version of your signature dish. It needs to maintain the core experience while being practical for on-the-go consumption.

```python
import tensorflow as tf

class MobileFriendlyPreprocessor:
    """
    Handles data preparation for mobile inference
    Like pre-cutting ingredients for quick assembly
    """
    def __init__(self, input_size=(224, 224)):
        self.input_size = input_size
    
    def preprocess_image(self, image_path):
        """Prepare image data for mobile model inference"""
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        
        # Resize to model input size
        image = tf.image.resize(image, self.input_size)
        
        # Normalize pixel values (recipe standardization)
        image = tf.cast(image, tf.float32) / 255.0
        
        # Add batch dimension
        image = tf.expand_dims(image, 0)
        
        return image

class TFLiteInferenceEngine:
    """
    Manages model loading and inference on mobile devices
    Like a portable cooking station with essential tools
    """
    def __init__(self, model_path):
        # Load the TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model input shape: {self.input_details[0]['shape']}")
        print(f"Model output shape: {self.output_details[0]['shape']}")
    
    def predict(self, input_data):
        """Run inference on preprocessed data"""
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            input_data.astype(np.float32)
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        return output_data

# Usage example
preprocessor = MobileFriendlyPreprocessor()
mobile_model = TFLiteInferenceEngine('optimized_model.tflite')

# Process an image and get predictions
image_data = preprocessor.preprocess_image('test_image.jpg')
predictions = mobile_model.predict(image_data)

print(f"Prediction confidence: {np.max(predictions):.3f}")
print(f"Predicted class: {np.argmax(predictions)}")
```

**Syntax Explanation:**
- `tf.lite.Interpreter`: Loads and manages TensorFlow Lite models
- `allocate_tensors()`: Allocates memory for model tensors
- `get_input_details()`: Retrieves information about model inputs
- `set_tensor()`: Provides input data to the model
- `invoke()`: Executes the model inference

---

## 3. Model Serving with TensorFlow Serving

### Professional Service Infrastructure

Setting up TensorFlow Serving is like establishing a professional restaurant service system. You need consistent quality, efficient order processing, and the ability to handle multiple customers simultaneously.

```python
import requests
import json
import numpy as np
from typing import Dict, Any

class ModelServingClient:
    """
    Client for interacting with TensorFlow Serving
    Like a waiter taking orders to the kitchen
    """
    def __init__(self, server_url: str, model_name: str, version: int = 1):
        self.server_url = server_url
        self.model_name = model_name
        self.version = version
        self.predict_url = f"{server_url}/v1/models/{model_name}:predict"
    
    def prepare_request_data(self, input_array: np.ndarray) -> Dict[str, Any]:
        """Format data for TensorFlow Serving API"""
        return {
            "signature_name": "serving_default",
            "instances": input_array.tolist()
        }
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Send prediction request to serving infrastructure"""
        request_data = self.prepare_request_data(input_data)
        
        try:
            response = requests.post(
                self.predict_url,
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Prediction request failed: {e}")
            return None
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Retrieve information about the served model"""
        metadata_url = f"{self.server_url}/v1/models/{self.model_name}/metadata"
        
        try:
            response = requests.get(metadata_url, timeout=5)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Failed to get model metadata: {e}")
            return None

# Docker command to start TensorFlow Serving (run in terminal)
"""
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/your/saved_model,target=/models/your_model \
  -e MODEL_NAME=your_model \
  -t tensorflow/serving
"""

# Usage example
serving_client = ModelServingClient(
    server_url="http://localhost:8501",
    model_name="image_classifier"
)

# Check if the service is ready
metadata = serving_client.get_model_metadata()
if metadata:
    print("Model serving is ready!")
    print(f"Model metadata: {json.dumps(metadata, indent=2)}")

# Make a prediction
test_data = np.random.random((1, 224, 224, 3))  # Example image data
result = serving_client.predict(test_data)

if result:
    predictions = result['predictions'][0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.3f}")
```

### Creating a Scalable Service

```python
from flask import Flask, request, jsonify
import tensorflow as tf
import logging

class ProductionMLService:
    """
    Production-ready ML service with proper error handling
    Like a well-managed restaurant with quality controls
    """
    def __init__(self, model_path: str):
        self.model = tf.saved_model.load(model_path)
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for production monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def health_check(self) -> Dict[str, str]:
        """Verify service is functioning properly"""
        try:
            # Test inference with dummy data
            dummy_input = tf.constant([[0.5, 0.5, 0.5]])
            _ = self.model(dummy_input)
            return {"status": "healthy", "message": "Model is ready"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}

# Flask application setup
app = Flask(__name__)
ml_service = ProductionMLService('/path/to/your/saved_model')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for load balancers"""
    return jsonify(ml_service.health_check())

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Parse input data
        input_data = request.json['instances']
        input_tensor = tf.constant(input_data)
        
        # Make prediction
        predictions = ml_service.model(input_tensor)
        
        # Return results
        return jsonify({
            'predictions': predictions.numpy().tolist(),
            'status': 'success'
        })
    
    except Exception as e:
        ml_service.logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**Syntax Explanation:**
- `tf.saved_model.load()`: Loads a SavedModel format for serving
- `requests.post()`: Makes HTTP POST requests to the serving API
- `response.raise_for_status()`: Raises an exception for HTTP error codes
- `Flask(__name__)`: Creates a Flask web application instance
- `@app.route()`: Decorates functions as HTTP endpoints

---

## 4. Performance Monitoring in Production

### Quality Assurance System

Production monitoring is like having food critics and customer feedback systems constantly evaluating your restaurant. You need real-time insights into how well your service is performing.

```python
import time
import psutil
import numpy as np
from collections import defaultdict
from datetime import datetime
import json

class ModelPerformanceMonitor:
    """
    Comprehensive monitoring system for ML models in production
    Like a restaurant manager tracking service quality
    """
    def __init__(self):
        self.metrics = defaultdict(list)
        self.prediction_cache = []
        self.start_time = time.time()
    
    def record_inference_metrics(self, inference_time: float, 
                                input_size: int, prediction_confidence: float):
        """Track key performance indicators"""
        current_time = datetime.now().isoformat()
        
        # Record timing metrics
        self.metrics['inference_times'].append({
            'timestamp': current_time,
            'duration_ms': inference_time * 1000,
            'input_size': input_size
        })
        
        # Record prediction quality metrics
        self.metrics['prediction_confidence'].append({
            'timestamp': current_time,
            'confidence': prediction_confidence,
            'is_high_confidence': prediction_confidence > 0.8
        })
        
        # System resource monitoring
        self.record_system_metrics()
    
    def record_system_metrics(self):
        """Monitor system resource usage"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        self.metrics['system_resources'].append({
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3)
        })
    
    def detect_model_drift(self, recent_predictions: list, 
                          baseline_distribution: dict) -> dict:
        """
        Detect if model behavior is changing over time
        Like noticing if your signature dish tastes different
        """
        recent_dist = self._calculate_prediction_distribution(recent_predictions)
        
        # Calculate distribution drift using KL divergence
        drift_score = self._kl_divergence(baseline_distribution, recent_dist)
        
        return {
            'drift_detected': drift_score > 0.1,  # Threshold for concern
            'drift_score': drift_score,
            'recent_distribution': recent_dist,
            'recommendation': self._get_drift_recommendation(drift_score)
        }
    
    def _calculate_prediction_distribution(self, predictions: list) -> dict:
        """Calculate distribution of prediction classes"""
        if not predictions:
            return {}
        
        classes, counts = np.unique(predictions, return_counts=True)
        total = len(predictions)
        
        return {int(cls): count/total for cls, count in zip(classes, counts)}
    
    def _kl_divergence(self, p_dist: dict, q_dist: dict) -> float:
        """Calculate KL divergence between two distributions"""
        if not p_dist or not q_dist:
            return 0.0
        
        kl_div = 0.0
        for key in p_dist:
            if key in q_dist and q_dist[key] > 0:
                kl_div += p_dist[key] * np.log(p_dist[key] / q_dist[key])
        
        return kl_div
    
    def _get_drift_recommendation(self, drift_score: float) -> str:
        """Provide actionable recommendations based on drift severity"""
        if drift_score < 0.05:
            return "Model performance is stable"
        elif drift_score < 0.1:
            return "Monitor closely - slight drift detected"
        else:
            return "Consider model retraining - significant drift detected"
    
    def generate_performance_report(self) -> dict:
        """Create comprehensive performance summary"""
        if not self.metrics['inference_times']:
            return {"error": "No performance data available"}
        
        # Calculate summary statistics
        inference_times = [m['duration_ms'] for m in self.metrics['inference_times']]
        confidences = [m['confidence'] for m in self.metrics['prediction_confidence']]
        
        return {
            'summary': {
                'total_predictions': len(inference_times),
                'avg_inference_time_ms': np.mean(inference_times),
                'p95_inference_time_ms': np.percentile(inference_times, 95),
                'avg_confidence': np.mean(confidences),
                'high_confidence_rate': np.mean([c > 0.8 for c in confidences])
            },
            'recent_performance': {
                'last_hour_predictions': len([
                    m for m in self.metrics['inference_times']
                    if (datetime.now() - datetime.fromisoformat(m['timestamp'])).seconds < 3600
                ]),
                'system_health': 'good' if psutil.cpu_percent() < 80 else 'concerning'
            },
            'timestamp': datetime.now().isoformat()
        }

# Usage in production service
class MonitoredMLService:
    """ML service with integrated monitoring"""
    def __init__(self, model_path: str):
        self.model = tf.saved_model.load(model_path)
        self.monitor = ModelPerformanceMonitor()
        self.baseline_distribution = {}  # Set from historical data
    
    def predict_with_monitoring(self, input_data):
        """Make prediction while collecting performance metrics"""
        start_time = time.time()
        
        # Make prediction
        predictions = self.model(input_data)
        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
        confidence = tf.reduce_max(predictions).numpy()
        
        # Record performance metrics
        inference_time = time.time() - start_time
        self.monitor.record_inference_metrics(
            inference_time=inference_time,
            input_size=input_data.shape[0],
            prediction_confidence=float(confidence)
        )
        
        return {
            'prediction': int(predicted_class),
            'confidence': float(confidence),
            'inference_time_ms': inference_time * 1000
        }

# Example usage
monitored_service = MonitoredMLService('/path/to/model')

# Make monitored predictions
for i in range(100):
    test_input = tf.random.normal((1, 10))  # Example input
    result = monitored_service.predict_with_monitoring(test_input)
    print(f"Prediction: {result['prediction']}, "
          f"Confidence: {result['confidence']:.3f}")

# Generate performance report
report = monitored_service.monitor.generate_performance_report()
print(json.dumps(report, indent=2))
```

**Syntax Explanation:**
- `defaultdict(list)`: Creates a dictionary with default empty lists
- `psutil.cpu_percent()`: Gets current CPU usage percentage  
- `datetime.now().isoformat()`: Creates ISO format timestamp strings
- `np.percentile(data, 95)`: Calculates the 95th percentile value
- `tf.argmax()`: Finds the index of the maximum value along an axis

---

# Day 85: Deployed ML Model API - Final Project

## Project: Restaurant Revenue Predictor API

You're the head chef launching a digital kitchen management system. Your restaurant chain needs to predict daily revenue based on various factors like weather, day of the week, and seasonal trends. You'll create a complete ML-powered API that other restaurant locations can use to forecast their earnings.

## Project Structure
```
restaurant_ml_api/
├── manage.py
├── requirements.txt
├── restaurant_predictor/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── ml_api/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── serializers.py
│   ├── ml_model.py
│   └── migrations/
└── trained_models/
    └── revenue_predictor.pkl
```

## Step 1: Create the ML Model Training Script

**File: `ml_api/ml_model.py`**
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime, timedelta

class RevenuePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.weather_encoder = LabelEncoder()
        self.day_encoder = LabelEncoder()
        self.is_trained = False
        
    def generate_training_data(self, n_samples=1000):
        """Generate synthetic restaurant revenue data"""
        np.random.seed(42)
        
        # Generate features
        dates = pd.date_range(start='2023-01-01', periods=n_samples)
        data = {
            'date': dates,
            'day_of_week': dates.day_name(),
            'month': dates.month,
            'is_weekend': dates.weekday >= 5,
            'weather': np.random.choice(['sunny', 'rainy', 'cloudy', 'snowy'], n_samples, p=[0.4, 0.2, 0.3, 0.1]),
            'temperature': np.random.normal(20, 10, n_samples),
            'special_event': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'marketing_spend': np.random.uniform(100, 1000, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create revenue based on realistic patterns
        base_revenue = 2000
        weekend_boost = df['is_weekend'] * 800
        weather_impact = df['weather'].map({'sunny': 300, 'cloudy': 0, 'rainy': -200, 'snowy': -400})
        temp_impact = np.where(df['temperature'].between(15, 25), 200, -100)
        event_boost = df['special_event'] * 600
        marketing_impact = df['marketing_spend'] * 0.3
        seasonal_impact = np.sin(df['month'] / 12 * 2 * np.pi) * 400
        
        df['revenue'] = (base_revenue + weekend_boost + weather_impact + 
                        temp_impact + event_boost + marketing_impact + 
                        seasonal_impact + np.random.normal(0, 200, n_samples))
        
        # Ensure no negative revenue
        df['revenue'] = np.maximum(df['revenue'], 500)
        
        return df
    
    def train(self):
        """Train the revenue prediction model"""
        print("🍳 Preparing the recipe data for our prediction model...")
        
        # Generate training data
        df = self.generate_training_data()
        
        # Prepare features
        df['weather_encoded'] = self.weather_encoder.fit_transform(df['weather'])
        df['day_encoded'] = self.day_encoder.fit_transform(df['day_of_week'])
        
        # Select features for training
        feature_columns = ['month', 'is_weekend', 'weather_encoded', 'day_encoded', 
                          'temperature', 'special_event', 'marketing_spend']
        
        X = df[feature_columns]
        y = df['revenue']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        print("🔥 Training the model with our restaurant data...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"📊 Training Results:")
        print(f"   Train MAE: ${train_mae:.2f}")
        print(f"   Test MAE: ${test_mae:.2f}")
        print(f"   Train R²: {train_r2:.3f}")
        print(f"   Test R²: {test_r2:.3f}")
        
        self.is_trained = True
        
        # Save encoders for later use
        self.weather_classes = self.weather_encoder.classes_
        self.day_classes = self.day_encoder.classes_
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def predict(self, features):
        """Make revenue prediction"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        feature_array = np.array([
            features['month'],
            features['is_weekend'],
            features['weather_encoded'],
            features['day_encoded'],
            features['temperature'],
            features['special_event'],
            features['marketing_spend']
        ]).reshape(1, -1)
        
        # Scale and predict
        feature_scaled = self.scaler.transform(feature_array)
        prediction = self.model.predict(feature_scaled)[0]
        
        return max(prediction, 500)  # Ensure minimum revenue
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'weather_encoder': self.weather_encoder,
            'day_encoder': self.day_encoder,
            'weather_classes': self.weather_classes,
            'day_classes': self.day_classes,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"🏪 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.weather_encoder = model_data['weather_encoder']
        self.day_encoder = model_data['day_encoder']
        self.weather_classes = model_data['weather_classes']
        self.day_classes = model_data['day_classes']
        self.is_trained = model_data['is_trained']
        
        print("🍽️ Model loaded successfully!")

# Training script
def train_and_save_model():
    predictor = RevenuePredictor()
    results = predictor.train()
    predictor.save_model('trained_models/revenue_predictor.pkl')
    return results

if __name__ == "__main__":
    train_and_save_model()
```

## Step 2: Django Settings Configuration

**File: `restaurant_predictor/settings.py`**
```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'your-secret-key-here-change-in-production'
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'ml_api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'restaurant_predictor.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
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

WSGI_APPLICATION = 'restaurant_predictor.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [],
    'DEFAULT_PERMISSION_CLASSES': [],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour'
    }
}

CORS_ALLOW_ALL_ORIGINS = True  # Only for development

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ML Model Settings
ML_MODEL_PATH = os.path.join(BASE_DIR, 'trained_models', 'revenue_predictor.pkl')
```

## Step 3: Django Models for Prediction History

**File: `ml_api/models.py`**
```python
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import json

class PredictionRequest(models.Model):
    WEATHER_CHOICES = [
        ('sunny', 'Sunny'),
        ('rainy', 'Rainy'),
        ('cloudy', 'Cloudy'),
        ('snowy', 'Snowy'),
    ]
    
    DAY_CHOICES = [
        ('Monday', 'Monday'),
        ('Tuesday', 'Tuesday'),
        ('Wednesday', 'Wednesday'),
        ('Thursday', 'Thursday'),
        ('Friday', 'Friday'),
        ('Saturday', 'Saturday'),
        ('Sunday', 'Sunday'),
    ]
    
    # Input features
    restaurant_name = models.CharField(max_length=200, default="Unknown Restaurant")
    prediction_date = models.DateField()
    day_of_week = models.CharField(max_length=10, choices=DAY_CHOICES)
    month = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(12)])
    weather = models.CharField(max_length=10, choices=WEATHER_CHOICES)
    temperature = models.FloatField()
    special_event = models.BooleanField(default=False)
    marketing_spend = models.FloatField(validators=[MinValueValidator(0)])
    
    # Output
    predicted_revenue = models.FloatField()
    confidence_interval = models.JSONField(default=dict)  # Store prediction confidence
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50, default="v1.0")
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.restaurant_name} - {self.prediction_date} - ${self.predicted_revenue:.2f}"
    
    @property
    def is_weekend(self):
        return self.day_of_week in ['Saturday', 'Sunday']
    
    def to_feature_dict(self):
        """Convert model instance to ML model input format"""
        return {
            'month': self.month,
            'is_weekend': self.is_weekend,
            'weather': self.weather,
            'day_of_week': self.day_of_week,
            'temperature': self.temperature,
            'special_event': int(self.special_event),
            'marketing_spend': self.marketing_spend,
        }

class ModelPerformance(models.Model):
    """Track model performance metrics"""
    model_version = models.CharField(max_length=50)
    mae = models.FloatField()  # Mean Absolute Error
    r2_score = models.FloatField()  # R-squared score
    training_date = models.DateTimeField(auto_now_add=True)
    sample_count = models.IntegerField()
    
    class Meta:
        ordering = ['-training_date']
        
    def __str__(self):
        return f"Model {self.model_version} - MAE: ${self.mae:.2f}, R²: {self.r2_score:.3f}"
```

## Step 4: API Serializers

**File: `ml_api/serializers.py`**
```python
from rest_framework import serializers
from .models import PredictionRequest, ModelPerformance
from datetime import date

class PredictionRequestSerializer(serializers.ModelSerializer):
    # Add validation for prediction date
    prediction_date = serializers.DateField()
    
    class Meta:
        model = PredictionRequest
        fields = [
            'restaurant_name', 'prediction_date', 'day_of_week', 
            'month', 'weather', 'temperature', 'special_event', 
            'marketing_spend'
        ]
    
    def validate_prediction_date(self, value):
        if value < date.today():
            raise serializers.ValidationError("Cannot predict revenue for past dates")
        return value
    
    def validate_temperature(self, value):
        if value < -50 or value > 60:
            raise serializers.ValidationError("Temperature must be between -50°C and 60°C")
        return value
    
    def validate_marketing_spend(self, value):
        if value < 0:
            raise serializers.ValidationError("Marketing spend cannot be negative")
        if value > 10000:
            raise serializers.ValidationError("Marketing spend seems unusually high")
        return value

class PredictionResponseSerializer(serializers.ModelSerializer):
    confidence_score = serializers.SerializerMethodField()
    recommendation = serializers.SerializerMethodField()
    
    class Meta:
        model = PredictionRequest
        fields = [
            'id', 'restaurant_name', 'prediction_date', 'day_of_week',
            'month', 'weather', 'temperature', 'special_event',
            'marketing_spend', 'predicted_revenue', 'confidence_score',
            'recommendation', 'created_at', 'model_version'
        ]
        read_only_fields = ['id', 'predicted_revenue', 'created_at', 'model_version']
    
    def get_confidence_score(self, obj):
        """Calculate confidence based on feature values"""
        confidence = 0.85  # Base confidence
        
        # Adjust based on weather (sunny weather = more predictable)
        if obj.weather == 'sunny':
            confidence += 0.1
        elif obj.weather == 'rainy':
            confidence -= 0.05
        
        # Weekend predictions are generally more reliable
        if obj.is_weekend:
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    def get_recommendation(self, obj):
        """Provide business recommendations based on prediction"""
        revenue = obj.predicted_revenue
        recommendations = []
        
        if revenue > 3000:
            recommendations.append("High revenue expected! Consider expanding staff for the day.")
        elif revenue < 1500:
            recommendations.append("Lower revenue predicted. Consider promotional activities.")
        
        if obj.weather == 'rainy':
            recommendations.append("Rainy weather expected. Promote delivery services.")
        
        if obj.is_weekend and obj.special_event:
            recommendations.append("Weekend + special event combo! Prepare for high demand.")
        
        return recommendations

class ModelPerformanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelPerformance
        fields = '__all__'
```

## Step 5: API Views

**File: `ml_api/views.py`**
```python
from rest_framework import status
from rest_framework.decorators import api_view, throttle_classes
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from rest_framework.views import APIView
from django.conf import settings
from django.shortcuts import get_object_or_404
from .models import PredictionRequest, ModelPerformance
from .serializers import (
    PredictionRequestSerializer, 
    PredictionResponseSerializer,
    ModelPerformanceSerializer
)
from .ml_model import RevenuePredictor
import os
from datetime import datetime
import traceback

# Global model instance
predictor = None

def load_model():
    """Load the ML model on startup"""
    global predictor
    if predictor is None:
        try:
            predictor = RevenuePredictor()
            if os.path.exists(settings.ML_MODEL_PATH):
                predictor.load_model(settings.ML_MODEL_PATH)
                print("🍽️ Revenue prediction model loaded successfully!")
            else:
                print("⚠️ Model file not found. Training new model...")
                predictor.train()
                predictor.save_model(settings.ML_MODEL_PATH)
                print("✅ New model trained and saved!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            predictor = None
    return predictor

class PredictRevenueView(APIView):
    """Main API endpoint for revenue predictions"""
    
    def post(self, request):
        """Make a revenue prediction"""
        try:
            # Load model if not already loaded
            model = load_model()
            if model is None:
                return Response(
                    {"error": "ML model is not available. Please try again later."},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            
            # Validate input data
            serializer = PredictionRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    {"error": "Invalid input data", "details": serializer.errors},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Prepare features for prediction
            validated_data = serializer.validated_data
            features = {
                'month': validated_data['month'],
                'is_weekend': validated_data['day_of_week'] in ['Saturday', 'Sunday'],
                'weather_encoded': list(model.weather_classes).index(validated_data['weather']),
                'day_encoded': list(model.day_classes).index(validated_data['day_of_week']),
                'temperature': validated_data['temperature'],
                'special_event': int(validated_data['special_event']),
                'marketing_spend': validated_data['marketing_spend'],
            }
            
            # Make prediction
            predicted_revenue = model.predict(features)
            
            # Save prediction to database
            prediction_obj = PredictionRequest.objects.create(
                **validated_data,
                predicted_revenue=predicted_revenue,
                model_version="v1.0"
            )
            
            # Return response
            response_serializer = PredictionResponseSerializer(prediction_obj)
            return Response({
                "success": True,
                "message": "Revenue prediction completed successfully",
                "prediction": response_serializer.data
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {"error": f"Prediction failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@api_view(['GET'])
@throttle_classes([AnonRateThrottle])
def get_prediction_history(request):
    """Get prediction history for a restaurant"""
    restaurant_name = request.query_params.get('restaurant_name')
    limit = min(int(request.query_params.get('limit', 10)), 100)  # Max 100 records
    
    queryset = PredictionRequest.objects.all()
    
    if restaurant_name:
        queryset = queryset.filter(restaurant_name__icontains=restaurant_name)
    
    predictions = queryset[:limit]
    serializer = PredictionResponseSerializer(predictions, many=True)
    
    return Response({
        "success": True,
        "count": predictions.count(),
        "predictions": serializer.data
    })

@api_view(['GET'])
def get_model_stats(request):
    """Get model performance statistics"""
    try:
        latest_performance = ModelPerformance.objects.first()
        total_predictions = PredictionRequest.objects.count()
        
        # Calculate recent prediction statistics
        recent_predictions = PredictionRequest.objects.order_by('-created_at')[:100]
        avg_predicted_revenue = sum(p.predicted_revenue for p in recent_predictions) / len(recent_predictions) if recent_predictions else 0
        
        stats = {
            "model_info": {
                "version": "v1.0",
                "type": "Random Forest Regressor",
                "features": ["month", "is_weekend", "weather", "day_of_week", "temperature", "special_event", "marketing_spend"]
            },
            "performance": ModelPerformanceSerializer(latest_performance).data if latest_performance else None,
            "usage_stats": {
                "total_predictions": total_predictions,
                "average_predicted_revenue": round(avg_predicted_revenue, 2),
                "model_loaded": predictor is not None and predictor.is_trained
            }
        }
        
        return Response({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        return Response(
            {"error": f"Failed to retrieve stats: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def retrain_model(request):
    """Retrain the model (admin endpoint)"""
    try:
        global predictor
        predictor = RevenuePredictor()
        results = predictor.train()
        predictor.save_model(settings.ML_MODEL_PATH)
        
        # Save performance metrics
        ModelPerformance.objects.create(
            model_version="v1.0",
            mae=results['test_mae'],
            r2_score=results['test_r2'],
            sample_count=1000  # From our synthetic data
        )
        
        return Response({
            "success": True,
            "message": "Model retrained successfully",
            "performance": results
        })
        
    except Exception as e:
        return Response(
            {"error": f"Retraining failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def health_check(request):
    """API health check endpoint"""
    return Response({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None and predictor.is_trained if predictor else False,
        "version": "1.0.0"
    })
```

## Step 6: URL Configuration

**File: `ml_api/urls.py`**
```python
from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.PredictRevenueView.as_view(), name='predict_revenue'),
    path('history/', views.get_prediction_history, name='prediction_history'),
    path('stats/', views.get_model_stats, name='model_stats'),
    path('retrain/', views.retrain_model, name='retrain_model'),
    path('health/', views.health_check, name='health_check'),
]
```

**File: `restaurant_predictor/urls.py`**
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('ml_api.urls')),
]
```

## Step 7: Requirements File

**File: `requirements.txt`**
```
Django==4.2.7
djangorestframework==3.14.0
django-cors-headers==4.3.1
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.24.3
joblib==1.3.2
gunicorn==21.2.0
python-dotenv==1.0.0
```

## Step 8: Model Training and Management Commands

**File: `ml_api/management/__init__.py`** (empty file)

**File: `ml_api/management/commands/__init__.py`** (empty file)

**File: `ml_api/management/commands/train_model.py`**
```python
from django.core.management.base import BaseCommand
from django.conf import settings
from ml_api.ml_model import RevenuePredictor
from ml_api.models import ModelPerformance

class Command(BaseCommand):
    help = 'Train and save the ML model'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--samples',
            type=int,
            default=1000,
            help='Number of training samples to generate',
        )
    
    def handle(self, *args, **options):
        self.stdout.write('🍳 Starting model training...')
        
        try:
            predictor = RevenuePredictor()
            
            # Override the default sample size if provided
            original_method = predictor.generate_training_data
            def generate_with_custom_samples(n_samples=options['samples']):
                return original_method(n_samples)
            predictor.generate_training_data = generate_with_custom_samples
            
            # Train the model
            results = predictor.train()
            
            # Save the model
            predictor.save_model(settings.ML_MODEL_PATH)
            
            # Save performance metrics
            ModelPerformance.objects.create(
                model_version="v1.0",
                mae=results['test_mae'],
                r2_score=results['test_r2'],
                sample_count=options['samples']
            )
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'✅ Model trained successfully!\n'
                    f'   📊 Test MAE: ${results["test_mae"]:.2f}\n'
                    f'   📊 Test R²: {results["test_r2"]:.3f}\n'
                    f'   📁 Saved to: {settings.ML_MODEL_PATH}'
                )
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Training failed: {e}')
            )
```

## Step 9: Deployment Setup

**File: `deploy.py`**
```python
import os
import subprocess
import sys
from pathlib import Path

def setup_project():
    """Complete project setup and deployment preparation"""
    print("🏪 Setting up Restaurant Revenue Predictor API...")
    
    # Install requirements
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Create database and run migrations
    print("🗄️ Setting up database...")
    subprocess.run([sys.executable, "manage.py", "makemigrations"])
    subprocess.run([sys.executable, "manage.py", "migrate"])
    
    # Train the initial model
    print("🤖 Training initial ML model...")
    subprocess.run([sys.executable, "manage.py", "train_model", "--samples", "2000"])
    
    print("✅ Setup complete! Your API is ready to serve predictions.")
    print("\n🚀 To start the server, run:")
    print("   python manage.py runserver 0.0.0.0:8000")
    print("\n📡 API endpoints will be available at:")
    print("   POST /api/v1/predict/        - Make predictions")
    print("   GET  /api/v1/history/        - View prediction history")
    print("   GET  /api/v1/stats/          - View model statistics")
    print("   GET  /api/v1/health/         - Health check")

if __name__ == "__main__":
    setup_project()
```

## Step 10: API Testing Script

**File: `test_api.py`**
```python
import requests
import json
from datetime import date, timedelta

BASE_URL = "http://localhost:8000/api/v1"

def test_api():
    """Test the complete API functionality"""
    print("🧪 Testing Restaurant Revenue Predictor API...")
    
    # Test 1: Health Check
    print("\n1. Testing health check...")
    response = requests.get(f"{BASE_URL}/health/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test 2: Make a prediction
    print("\n2. Testing revenue prediction...")
    prediction_data = {
        "restaurant_name": "Chef Mario's Bistro",
        "prediction_date": str(date.today() + timedelta(days=7)),
        "day_of_week": "Saturday",
        "month": 12,
        "weather": "sunny",
        "temperature": 22.5,
        "special_event": True,
        "marketing_spend": 750.0
    }
    
    response = requests.post(f"{BASE_URL}/predict/", json=prediction_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 201:
        result = response.json()
        print(f"   Predicted Revenue: ${result['prediction']['predicted_revenue']:.2f}")
        print(f"   Confidence: {result['prediction']['confidence_score']:.2%}")
        print(f"   Recommendations: {result['prediction']['recommendation']}")
    else:
        print(f"   Error: {response.json()}")
    
    # Test 3: Get prediction history
    print("\n3. Testing prediction history...")
    response = requests.get(f"{BASE_URL}/history/?restaurant_name=Chef Mario's&limit=5")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        history = response.json()
        print(f"   Found {history['count']} predictions")
    
    # Test 4: Get model statistics
    print("\n4. Testing model statistics...")
    response = requests.get(f"{BASE_URL}/stats/")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        stats = response.json()['stats']
        print(f"   Model Version: {stats['model_info']['version']}")
        print(f"   Total Predictions: {stats['usage_stats']['total_predictions']}")
        print(f"   Model Loaded: {stats['usage_stats']['model_loaded']}")
    
    # Test 5: Batch predictions for different scenarios
    print("\n5. Testing various prediction scenarios...")
    scenarios = [
        {
            "name": "Rainy Tuesday",
            "data": {
                "restaurant_name": "Downtown Diner",
                "prediction_date": str(date.today() + timedelta(days=1)),
                "day_of_week": "Tuesday",
                "month": 11,
                "weather": "rainy",
                "temperature": 15.0,
                "special_event": False,
                "marketing_spend": 200.0
            }
        },
        {
            "name": "Sunny Weekend with Event",
            "data": {
                "restaurant_name": "Rooftop Grill",
                "prediction_date": str(date.today() + timedelta(days=3)),
                "day_of_week": "Sunday",
                "month": 12,
                "weather": "sunny",
                "temperature": 25.0,
                "special_event": True,
                "marketing_spend": 1200.0
            }
        },
        {
            "name": "Cold Snowy Day",
            "data": {
                "restaurant_name": "Cozy Cafe",
                "prediction_date": str(date.today() + timedelta(days=5)),
                "day_of_week": "Wednesday",
                "month": 1,
                "weather": "snowy",
                "temperature": -5.0,
                "special_event": False,
                "marketing_spend": 300.0
            }
        }
    ]
    
    for scenario in scenarios:
        response = requests.post(f"{BASE_URL}/predict/", json=scenario["data"])
        if response.status_code == 201:
            result = response.json()
            revenue = result['prediction']['predicted_revenue']
            print(f"   {scenario['name']}: ${revenue:.2f}")
        else:
            print(f"   {scenario['name']}: Error - {response.json()}")
    
    print("\n✅ API testing completed!")

if __name__ == "__main__":
    test_api()
```

## Step 11: Production Deployment Configuration

**File: `production_settings.py`**
```python
from .settings import *
import os
from dotenv import load_dotenv

load_dotenv()

# Production settings
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com', 'api.your-restaurant-chain.com']

# Security settings
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'change-me-in-production')
SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# Database for production (PostgreSQL recommended)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'restaurant_predictor'),
        'USER': os.getenv('DB_USER', 'postgres'),
        'PASSWORD': os.getenv('DB_PASSWORD', ''),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

# Caching with Redis
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.getenv('REDIS_URL', 'redis://localhost:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Enhanced API throttling for production
REST_FRAMEWORK['DEFAULT_THROTTLE_RATES'] = {
    'anon': '50/hour',
    'user': '200/hour'
}

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'api.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'ml_api': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# Static files for production
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATIC_URL = '/static/'
```

**File: `gunicorn.conf.py`**
```python
# Gunicorn configuration for production deployment
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "restaurant_predictor_api"

# Server mechanics
preload_app = True
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
```

**File: `Dockerfile`**
```dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE restaurant_predictor.production_settings

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
COPY requirements-prod.txt /app/
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy project
COPY . /app/

# Create directory for model files
RUN mkdir -p /app/trained_models

# Collect static files
RUN python manage.py collectstatic --noinput

# Create non-root user
RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# Run gunicorn
CMD ["gunicorn", "--config", "gunicorn.conf.py", "restaurant_predictor.wsgi:application"]
```

**File: `requirements-prod.txt`**
```
-r requirements.txt
psycopg2-binary==2.9.9
django-redis==5.4.0
python-dotenv==1.0.0
whitenoise==6.6.0
sentry-sdk==1.38.0
```

**File: `docker-compose.yml`**
```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: restaurant_predictor
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  web:
    build: .
    command: gunicorn --config gunicorn.conf.py restaurant_predictor.wsgi:application
    volumes:
      - .:/app
      - ./trained_models:/app/trained_models
    ports:
      - "8000:8000"
    environment:
      - DJANGO_SETTINGS_MODULE=restaurant_predictor.production_settings
      - DB_HOST=db
      - DB_NAME=restaurant_predictor
      - DB_USER=postgres
      - DB_PASSWORD=your_db_password
      - REDIS_URL=redis://redis:6379/1
    depends_on:
      - db
      - redis
    restart: unless-stopped

volumes:
  postgres_data:
```

## Step 12: Complete Project Setup Script

**File: `run_project.py`**
```python
#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"   ✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed: {e.stderr}")
        return None

def setup_and_run():
    """Complete setup and run the API server"""
    print("🏪 Welcome to Restaurant Revenue Predictor API Setup!")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install requirements
    run_command("pip install -r requirements.txt", "Installing dependencies")
    
    # Django setup
    run_command("python manage.py makemigrations", "Creating database migrations")
    run_command("python manage.py migrate", "Applying database migrations")
    
    # Train model
    print("\n🤖 Training ML model...")
    run_command("python manage.py train_model --samples 2000", "Training revenue prediction model")
    
    # Create superuser (optional)
    print("\n👤 Django admin setup (optional - press Ctrl+C to skip)")
    try:
        subprocess.run("python manage.py createsuperuser", shell=True)
    except KeyboardInterrupt:
        print("\n   Skipping admin user creation")
    
    # Start development server
    print("\n🚀 Starting development server...")
    print("=" * 60)
    print("📡 API will be available at: http://localhost:8000/api/v1/")
    print("📊 API Endpoints:")
    print("   POST /api/v1/predict/     - Make revenue predictions")
    print("   GET  /api/v1/history/     - View prediction history")
    print("   GET  /api/v1/stats/       - Model performance stats")
    print("   GET  /api/v1/health/      - Health check")
    print("   POST /api/v1/retrain/     - Retrain model")
    print("\n🧪 Test the API by running: python test_api.py")
    print("⏹️  Stop the server with Ctrl+C")
    print("=" * 60)
    
    try:
        subprocess.run("python manage.py runserver 0.0.0.0:8000", shell=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Thanks for using Restaurant Revenue Predictor API!")

if __name__ == "__main__":
    setup_and_run()
```

## Step 13: API Documentation

**File: `API_DOCUMENTATION.md`**
```markdown
# 🍽️ Restaurant Revenue Predictor API Documentation

## Overview
A production-ready ML-powered API that predicts restaurant daily revenue based on weather, date, and operational factors.

## Base URL
- Development: `http://localhost:8000/api/v1/`
- Production: `https://your-domain.com/api/v1/`

## Authentication
Currently open API. In production, implement API key authentication.

## Endpoints

### 1. Make Prediction
**POST** `/predict/`

Predict restaurant revenue for a specific date and conditions.

**Request Body:**
```json
{
    "restaurant_name": "Chef Mario's Bistro",
    "prediction_date": "2024-12-25",
    "day_of_week": "Saturday",
    "month": 12,
    "weather": "sunny",
    "temperature": 22.5,
    "special_event": true,
    "marketing_spend": 750.0
}
```

**Response (201 Created):**
```json
{
    "success": true,
    "message": "Revenue prediction completed successfully",
    "prediction": {
        "id": 1,
        "restaurant_name": "Chef Mario's Bistro",
        "prediction_date": "2024-12-25",
        "predicted_revenue": 3420.50,
        "confidence_score": 0.92,
        "recommendation": [
            "High revenue expected! Consider expanding staff for the day.",
            "Weekend + special event combo! Prepare for high demand."
        ],
        "model_version": "v1.0"
    }
}
```

### 2. Get Prediction History
**GET** `/history/`

Retrieve past predictions with optional filtering.

**Query Parameters:**
- `restaurant_name` (optional): Filter by restaurant name
- `limit` (optional): Number of records (max 100, default 10)

**Response (200 OK):**
```json
{
    "success": true,
    "count": 5,
    "predictions": [...]
}
```

### 3. Model Statistics
**GET** `/stats/`

Get model performance and usage statistics.

**Response (200 OK):**
```json
{
    "success": true,
    "stats": {
        "model_info": {
            "version": "v1.0",
            "type": "Random Forest Regressor",
            "features": ["month", "is_weekend", "weather", ...]
        },
        "performance": {
            "mae": 234.56,
            "r2_score": 0.847
        },
        "usage_stats": {
            "total_predictions": 1250,
            "average_predicted_revenue": 2450.30
        }
    }
}
```

### 4. Health Check
**GET** `/health/`

Check API and model status.

**Response (200 OK):**
```json
{
    "status": "healthy",
    "timestamp": "2024-08-08T10:30:00Z",
    "model_loaded": true,
    "version": "1.0.0"
}
```

### 5. Retrain Model
**POST** `/retrain/`

Retrain the ML model (admin endpoint).

**Response (200 OK):**
```json
{
    "success": true,
    "message": "Model retrained successfully",
    "performance": {
        "train_mae": 198.45,
        "test_mae": 234.56,
        "train_r2": 0.923,
        "test_r2": 0.847
    }
}
```

## Error Responses

**400 Bad Request:**
```json
{
    "error": "Invalid input data",
    "details": {
        "temperature": ["Temperature must be between -50°C and 60°C"]
    }
}
```

**503 Service Unavailable:**
```json
{
    "error": "ML model is not available. Please try again later."
}
```

## Rate Limiting
- Development: 100 requests/hour
- Production: 50 requests/hour (anonymous), 200 requests/hour (authenticated)

## Data Types
- **weather**: "sunny", "rainy", "cloudy", "snowy"
- **day_of_week**: "Monday", "Tuesday", ..., "Sunday"
- **month**: Integer 1-12
- **temperature**: Float (°C)
- **marketing_spend**: Float (currency units)
- **special_event**: Boolean

## Example Usage (Python)
```python
import requests

# Make a prediction
data = {
    "restaurant_name": "Downtown Bistro",
    "prediction_date": "2024-12-31",
    "day_of_week": "Sunday",
    "month": 12,
    "weather": "sunny",
    "temperature": 20.0,
    "special_event": True,
    "marketing_spend": 1000.0
}

response = requests.post("http://localhost:8000/api/v1/predict/", json=data)
result = response.json()
print(f"Predicted Revenue: ${result['prediction']['predicted_revenue']:.2f}")
```
```

## Final Project Summary

Your **Restaurant Revenue Predictor API** is now a complete, production-ready ML system that demonstrates:

### 🧠 **Machine Learning Integration**
- **Random Forest model** trained on realistic restaurant data
- **Feature engineering** with weather, seasonality, and business factors  
- **Model persistence** using joblib for consistent predictions
- **Performance tracking** with MAE and R² metrics

### 🍽️ **Django API Architecture** 
- **RESTful endpoints** for predictions, history, and statistics
- **Data validation** with Django serializers
- **Database models** for prediction logging and performance tracking
- **Custom management commands** for model training

### 🚀 **Production Deployment**
- **Docker containerization** with multi-stage builds
- **Database integration** (SQLite for dev, PostgreSQL for production)
- **Caching layer** with Redis for improved performance
- **Rate limiting and security** configurations
- **Health monitoring** and logging systems

### 🧪 **Quality Assurance**
- **Comprehensive testing script** covering all endpoints
- **Error handling** for edge cases and failures  
- **API documentation** with examples and usage patterns
- **Setup automation** for easy deployment

The project simulates a real-world scenario where restaurant chains need data-driven revenue forecasting. Just like a head chef planning ingredients and staffing based on expected crowd sizes, your API helps restaurants optimize their operations using weather patterns, seasonal trends, and marketing investments.

Run `python run_project.py` to get started, then test with `python test_api.py` to see your ML model serving predictions through a robust Django API!

## Assignment: Model Performance Analyzer

Create a comprehensive model performance analyzer that monitors a deployed machine learning model and provides actionable insights about its production behavior.

### Requirements:

1. **Build a monitoring system** that tracks:
   - Inference latency and throughput
   - Prediction confidence distributions  
   - Resource utilization (CPU, memory)
   - Model drift detection over time

2. **Implement alerting logic** that:
   - Flags when inference times exceed acceptable thresholds
   - Detects significant changes in prediction patterns
   - Monitors system resource exhaustion
   - Provides clear recommendations for each alert type

3. **Create a dashboard interface** using Python (Flask/Django) that:
   - Displays real-time performance metrics
   - Shows historical trends with visualizations
   - Presents drift analysis results
   - Allows administrators to set monitoring thresholds

4. **Include optimization recommendations** that:
   - Suggest when to retrain based on drift severity
   - Recommend infrastructure scaling based on load patterns
   - Identify potential model quantization opportunities
   - Provide cost-benefit analysis for optimization strategies

Your solution should handle at least 1000 prediction requests, maintain 7 days of historical data, and provide meaningful insights that would help a team maintain a production ML system effectively.

Submit your code along with a brief report explaining your monitoring strategy and key insights discovered during testing.