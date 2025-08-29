# AI Mastery Course - Day 95: Advanced Tools & Frameworks

## Learning Objective
By the end of this lesson, you will understand how to select, implement, and orchestrate advanced AI tools and frameworks, compare deep learning libraries effectively, leverage transformer models, track experiments professionally, and containerize ML models for scalable deployment.

---

## Introduction

Imagine that you've spent months perfecting recipes in your home kitchen, and now you're ready to equip a professional restaurant. Just as a master culinary artist needs to understand the nuances between different ovens, knife sets, and ingredient sourcing systems, an AI practitioner must master the sophisticated tools that separate hobbyist experiments from production-ready systems.

Today, we're not just learning individual tools – we're understanding how they work together in harmony, like a well-orchestrated kitchen where every appliance, tracking system, and preparation method serves a specific purpose in creating exceptional results.

---

## 1. PyTorch vs TensorFlow Deep Dive

### Understanding the Landscape

Think of PyTorch and TensorFlow as two different cooking philosophies. One emphasizes intuitive, dynamic preparation (PyTorch), while the other focuses on optimized, structured workflows (TensorFlow). Both can create world-class dishes, but understanding when to use each is crucial.

### PyTorch: The Dynamic Approach

PyTorch operates like cooking by intuition – you can taste and adjust as you go.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple neural network
class FlexibleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FlexibleNet, self).__init__()
        # nn.Linear creates a fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()  # Activation function
        
    def forward(self, x):
        # Define forward pass - this can change dynamically
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model instance
model = FlexibleNet(input_size=10, hidden_size=20, output_size=1)

# Sample data
X = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randn(100, 1)   # Target values

# Define loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop - notice the dynamic nature
for epoch in range(100):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

**Syntax Explanation:**
- `nn.Module`: Base class for all neural network modules
- `nn.Linear(in_features, out_features)`: Creates a linear transformation layer
- `loss.backward()`: Computes gradients using automatic differentiation
- `optimizer.step()`: Updates model parameters based on computed gradients

### TensorFlow: The Structured Approach

TensorFlow is like having a well-documented recipe system – everything is planned and optimized.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# TensorFlow's structured approach
class StructuredNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(StructuredNet, self).__init__()
        # Define layers upfront
        self.dense1 = layers.Dense(hidden_size, activation='relu', input_shape=(input_size,))
        self.dense2 = layers.Dense(output_size)
        
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# Create and compile model
model = StructuredNet(input_size=10, hidden_size=20, output_size=1)
model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error
    metrics=['mae']  # Mean Absolute Error for monitoring
)

# Generate sample data
X_train = np.random.randn(100, 10)
y_train = np.random.randn(100, 1)

# Training with built-in methods
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,  # Use 20% for validation
    verbose=0  # Silent training
)

print(f"Final loss: {history.history['loss'][-1]:.4f}")
```

**Syntax Explanation:**
- `tf.keras.Model`: High-level API for building models
- `layers.Dense(units, activation)`: Fully connected layer with specified activation
- `model.compile()`: Configures the model for training
- `model.fit()`: Trains the model with automatic batching and validation

### When to Choose Which?

```python
# Decision framework
def choose_framework(project_requirements):
    """
    Function to guide framework selection
    """
    pytorch_score = 0
    tensorflow_score = 0
    
    factors = {
        'research_flexibility': pytorch_score + 2,
        'production_deployment': tensorflow_score + 2,
        'dynamic_architectures': pytorch_score + 3,
        'mobile_deployment': tensorflow_score + 3,
        'debugging_ease': pytorch_score + 2,
        'ecosystem_maturity': tensorflow_score + 1
    }
    
    return "PyTorch" if pytorch_score > tensorflow_score else "TensorFlow"
```

---

## 2. Hugging Face Transformers Library

### The Universal Toolkit

Hugging Face is like having access to a library of time-tested recipes from master cooks worldwide. Instead of creating everything from scratch, you can build upon proven techniques.

```python
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, Trainer, TrainingArguments
)
import torch
from datasets import Dataset

# Quick start with pipelines - the "instant meal" approach
def quick_sentiment_analysis():
    # Create a sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis")
    
    texts = [
        "I love working with AI models!",
        "This documentation is confusing.",
        "The results exceeded our expectations."
    ]
    
    results = classifier(texts)
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']}, Confidence: {result['score']:.3f}\n")

quick_sentiment_analysis()

# Advanced custom model fine-tuning
class CustomTextClassifier:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
    
    def prepare_data(self, texts, labels):
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,     # Cut off long texts
            padding=True,        # Pad short texts
            max_length=512,      # Maximum sequence length
            return_tensors="pt"  # Return PyTorch tensors
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)
        })
        
        return dataset
    
    def train(self, train_dataset, eval_dataset=None):
        # Training arguments - like setting oven temperature and timing
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch" if eval_dataset else "no"
        )
        
        # Create trainer - your sous chef
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Start training
        trainer.train()
        return trainer

# Example usage
texts = ["Great product!", "Terrible experience!", "It's okay.", "Amazing quality!"]
labels = [1, 0, 0, 1]  # 1 for positive, 0 for negative

classifier = CustomTextClassifier()
train_dataset = classifier.prepare_data(texts, labels)
```

**Syntax Explanation:**
- `AutoTokenizer.from_pretrained()`: Loads pre-trained tokenizer matching the model
- `return_tensors="pt"`: Returns PyTorch tensors instead of lists
- `TrainingArguments`: Configuration class for training hyperparameters
- `Trainer`: High-level training interface that handles the training loop

---

## 3. Weights & Biases for Experiment Tracking

### Professional Recipe Documentation

Imagine keeping detailed notes of every cooking experiment – ingredients, techniques, timings, and results. W&B is your digital lab notebook for AI experiments.

```python
import wandb
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Initialize experiment tracking
def initialize_experiment(project_name, config):
    wandb.init(
        project=project_name,
        config=config,
        tags=["tutorial", "pytorch", "classification"]
    )
    return wandb.config

# Enhanced training with comprehensive tracking
class TrackedMLExperiment:
    def __init__(self, config):
        self.config = config
        self.model = self.create_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def create_model(self):
        return nn.Sequential(
            nn.Linear(self.config.input_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),  # Prevents overfitting
            nn.Linear(self.config.hidden_size, self.config.num_classes)
        )
    
    def train_epoch(self, train_loader):
        self.model.train()  # Set model to training mode
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Log batch metrics to W&B
            if batch_idx % 10 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_accuracy": 100. * correct / total
                })
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient computation
            for data, target in val_loader:
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return val_loss / len(val_loader), 100. * correct / total
    
    def run_experiment(self, train_loader, val_loader, epochs):
        best_val_accuracy = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Log epoch metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                wandb.save('best_model.pth')  # Upload to W&B
            
            print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Log final summary
        wandb.log({"best_val_accuracy": best_val_accuracy})

# Example experiment configuration
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "hidden_size": 128,
    "dropout_rate": 0.3,
    "input_size": 20,
    "num_classes": 2
}

# Run experiment (commented out to avoid actual W&B calls)
# config = initialize_experiment("ai-mastery-tutorial", config)
# experiment = TrackedMLExperiment(config)
```

**Syntax Explanation:**
- `wandb.init()`: Initializes experiment tracking session
- `wandb.log()`: Logs metrics to the W&B dashboard
- `torch.no_grad()`: Context manager that disables gradient computation for efficiency
- `wandb.save()`: Uploads files to W&B for versioning

---

## 4. Docker for ML Model Deployment

### Packaging Your Creation

Docker is like creating a complete meal kit – everything needed to reproduce your dish, regardless of the kitchen it's prepared in.

```dockerfile
# Dockerfile for ML model deployment
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]
```

```python
# app.py - Flask API for model serving
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import pickle

app = Flask(__name__)

# Model definition (same as training)
class DeploymentModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_classes=2):
        super(DeploymentModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1)  # Get probabilities
        )
    
    def forward(self, x):
        return self.network(x)

# Load trained model
model = DeploymentModel()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(features)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = output.numpy()[0]
            prediction = int(np.argmax(probabilities))
        
        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities.tolist(),
            "confidence": float(max(probabilities))
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Batch prediction endpoint
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.json
        features_list = np.array(data['features_list'])
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(features_list)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = outputs.numpy()
            predictions = np.argmax(probabilities, axis=1)
        
        return jsonify({
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
```

```bash
# docker-compose.yml for complete deployment
version: '3.8'
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/best_model.pth
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```python
# requirements.txt content
"""
torch==1.12.1
flask==2.3.3
numpy==1.24.3
scikit-learn==1.3.0
requests==2.31.0
gunicorn==21.2.0
"""

# Docker management utilities
import subprocess
import json

class DockerMLManager:
    def __init__(self, image_name, container_name):
        self.image_name = image_name
        self.container_name = container_name
    
    def build_image(self):
        """Build Docker image"""
        cmd = f"docker build -t {self.image_name} ."
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    
    def run_container(self, port=8000):
        """Run container"""
        cmd = f"docker run -d --name {self.container_name} -p {port}:{port} {self.image_name}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    
    def check_health(self):
        """Check container health"""
        import requests
        try:
            response = requests.get("http://localhost:8000/health")
            return response.status_code == 200
        except:
            return False
```

**Syntax Explanation:**
- `FROM python:3.9-slim`: Base image with Python pre-installed
- `WORKDIR /app`: Sets the working directory inside the container
- `COPY requirements.txt .`: Copies file to current directory in container
- `RUN pip install --no-cache-dir`: Installs Python packages without caching
- `CMD ["python", "app.py"]`: Default command when container starts
- `@app.route()`: Flask decorator to define API endpoints

---

## Final Project: Intelligent Model Comparison System

### Project Overview

Create a comprehensive system that compares PyTorch and TensorFlow models, tracks experiments with W&B, and deploys the best model using Docker. This project integrates all four concepts we've covered.

```python
# main_project.py
import torch
import torch.nn as nn
import tensorflow as tf
from transformers import pipeline
import wandb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import json

class ModelComparisonSystem:
    def __init__(self, project_name="model-comparison"):
        self.project_name = project_name
        self.results = {}
        
    def prepare_data(self):
        """Prepare iris dataset for comparison"""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_pytorch_model(self, input_size=4, hidden_size=10, num_classes=3):
        """Create PyTorch model"""
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def train_pytorch_model(self, X_train, y_train, X_test, y_test):
        """Train PyTorch model with tracking"""
        wandb.init(project=self.project_name, name="pytorch-run", reinit=True)
        
        model = self.create_pytorch_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        start_time = time.time()
        
        for epoch in range(100):
            # Training
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = (predicted == y_test_tensor).float().mean()
            
            if epoch % 10 == 0:
                wandb.log({
                    "epoch": epoch,
                    "loss": loss.item(),
                    "accuracy": accuracy.item()
                })
        
        training_time = time.time() - start_time
        
        # Final evaluation
        with torch.no_grad():
            final_outputs = model(X_test_tensor)
            _, final_predicted = torch.max(final_outputs.data, 1)
            final_accuracy = (final_predicted == y_test_tensor).float().mean()
        
        wandb.log({
            "final_accuracy": final_accuracy.item(),
            "training_time": training_time
        })
        wandb.finish()
        
        return {
            "framework": "PyTorch",
            "accuracy": final_accuracy.item(),
            "training_time": training_time,
            "model": model
        }
    
    def create_tensorflow_model(self, input_size=4, hidden_size=10, num_classes=3):
        """Create TensorFlow model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_size,)),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_tensorflow_model(self, X_train, y_train, X_test, y_test):
        """Train TensorFlow model with tracking"""
        wandb.init(project=self.project_name, name="tensorflow-run", reinit=True)
        
        model = self.create_tensorflow_model()
        
        start_time = time.time()
        
        # Custom callback for W&B logging
        class WandbCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                wandb.log({
                    "epoch": epoch,
                    "loss": logs.get('loss'),
                    "accuracy": logs.get('accuracy'),
                    "val_loss": logs.get('val_loss'),
                    "val_accuracy": logs.get('val_accuracy')
                })
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[WandbCallback()]
        )
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_loss, final_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        wandb.log({
            "final_accuracy": final_accuracy,
            "training_time": training_time
        })
        wandb.finish()
        
        return {
            "framework": "TensorFlow",
            "accuracy": final_accuracy,
            "training_time": training_time,
            "model": model
        }
    
    def add_transformer_analysis(self, text_samples=None):
        """Add transformer-based text analysis"""
        if text_samples is None:
            text_samples = [
                "This model performs exceptionally well.",
                "The results are disappointing and concerning.",
                "Average performance with room for improvement."
            ]
        
        # Use Hugging Face pipeline for sentiment analysis
        sentiment_analyzer = pipeline("sentiment-analysis")
        
        results = []
        for text in text_samples:
            result = sentiment_analyzer(text)[0]
            results.append({
                "text": text,
                "sentiment": result['label'],
                "confidence": result['score']
            })
        
        return results
    
    def run_comparison(self):
        """Run complete comparison"""
        print("🚀 Starting Model Comparison System...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Train PyTorch model
        print("🔥 Training PyTorch model...")
        pytorch_results = self.train_pytorch_model(X_train, y_train, X_test, y_test)
        
        # Train TensorFlow model
        print("🧠 Training TensorFlow model...")
        tensorflow_results = self.train_tensorflow_model(X_train, y_train, X_test, y_test)
        
        # Add transformer analysis
        print("🤖 Running transformer analysis...")
        transformer_results = self.add_transformer_analysis()
        
        # Compare results
        comparison = {
            "pytorch": pytorch_results,
            "tensorflow": tensorflow_results,
            "transformer_analysis": transformer_results,
            "winner": "PyTorch" if pytorch_results['accuracy'] > tensorflow_results['accuracy'] else "TensorFlow"
        }
        
        # Save results
        with open('comparison_results.json', 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        return comparison

# Run the complete system
if __name__ == "__main__":
    system = ModelComparisonSystem()
    results = system.run_comparison()
    
    print("\n📊 COMPARISON RESULTS:")
    print(f"PyTorch Accuracy: {results['pytorch']['accuracy']:.4f}")
    print(f"TensorFlow Accuracy: {results['tensorflow']['accuracy']:.4f}")
    print(f"Winner: {results['winner']}")
```

---

# Professional-Grade ML System - Day 95 Final Project

## Project Overview
You'll create a complete ML system that combines sentiment analysis with real-time processing capabilities. Think of this as setting up a fully equipped restaurant kitchen where every station works in harmony - from prep work (data processing) to final plating (model deployment).

## Project Structure
```
ml_system/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── model_trainer.py
│   ├── api_server.py
│   └── utils.py
├── models/
├── data/
├── requirements.txt
├── config.py
└── run.py
```

## Step 1: Environment Setup & Configuration

**config.py**
```python
import os
from pathlib import Path

# Like organizing spice racks - everything has its place
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Hugging Face model configuration
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TOKENIZER_NAME = MODEL_NAME

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Weights & Biases configuration
WANDB_PROJECT = "professional-ml-system"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "your-username")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///sentiment_data.db")

# Create directories
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
```

**requirements.txt**
```txt
torch==2.1.0
transformers==4.35.0
datasets==2.14.5
wandb==0.15.12
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.4.2
sqlalchemy==2.0.23
pandas==2.1.2
numpy==1.24.3
scikit-learn==1.3.1
python-multipart==0.0.6
aiofiles==23.2.1
```

## Step 2: Data Processing Module

**src/data_processor.py**
```python
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import wandb
from config import *

Base = declarative_base()

class SentimentData(Base):
    """Database model - like a recipe card that stores all ingredients"""
    __tablename__ = 'sentiment_data'
    
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    predicted_sentiment = Column(String(20))
    confidence_score = Column(Float)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class DataProcessor:
    """Master preparation station - handles all data cooking"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(self.engine)
        SessionLocal = sessionmaker(bind=self.engine)
        self.db_session = SessionLocal()
        
    def prepare_training_data(self, csv_path: str) -> DatasetDict:
        """
        Like mise en place - preparing all ingredients before cooking
        Expected CSV format: text, label (0=negative, 1=neutral, 2=positive)
        """
        wandb.init(project=WANDB_PROJECT, job_type="data_preparation")
        
        df = pd.read_csv(csv_path)
        
        # Log data statistics
        wandb.log({
            "total_samples": len(df),
            "label_distribution": df['label'].value_counts().to_dict()
        })
        
        # Split data like organizing ingredients by cooking order
        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Tokenize like chopping vegetables - consistent sizes
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        columns_to_return = ['input_ids', 'attention_mask', 'label']
        train_dataset.set_format(type="torch", columns=columns_to_return)
        val_dataset.set_format(type="torch", columns=columns_to_return)
        test_dataset.set_format(type="torch", columns=columns_to_return)
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        wandb.finish()
        return dataset_dict
    
    def store_prediction(self, text: str, sentiment: str, confidence: float, processing_time: float):
        """Store results like logging each dish served"""
        record = SentimentData(
            text=text,
            predicted_sentiment=sentiment,
            confidence_score=confidence,
            processing_time=processing_time
        )
        self.db_session.add(record)
        self.db_session.commit()
```

## Step 3: Model Training Module

**src/model_trainer.py**
```python
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer
)
from datasets import DatasetDict
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from config import *

class ModelTrainer:
    """Head culinary instructor - trains the main cooking skills"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_metrics(self, eval_pred):
        """Quality control - taste testing each batch"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, dataset: DatasetDict, model_name: str = MODEL_NAME):
        """
        Main cooking process - slow and steady wins the race
        Like developing perfect seasoning through patience
        """
        wandb.init(project=WANDB_PROJECT, job_type="training")
        
        # Load pre-trained model - like starting with quality base ingredients
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,  # negative, neutral, positive
            problem_type="single_label_classification"
        )
        
        # Training arguments - recipe instructions
        training_args = TrainingArguments(
            output_dir=str(MODEL_DIR / "checkpoints"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(MODEL_DIR / "logs"),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="wandb"
        )
        
        # Initialize trainer - your sous-chef assistant
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )
        
        # Start training - the actual cooking process
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        # Evaluate on test set - final quality check
        test_results = trainer.evaluate(dataset['test'])
        
        # Log final metrics
        wandb.log({
            "training_time_minutes": training_time / 60,
            "test_accuracy": test_results.get("eval_accuracy"),
            "test_f1": test_results.get("eval_f1")
        })
        
        # Save the final model - preserving the perfected recipe
        final_model_path = MODEL_DIR / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        wandb.finish()
        return trainer, test_results
    
    def load_trained_model(self, model_path: str = None):
        """Load pre-trained model - like accessing your signature dishes"""
        if model_path is None:
            model_path = str(MODEL_DIR / "final_model")
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, text: str) -> tuple:
        """
        Make prediction - like preparing a dish to order
        Returns (sentiment_label, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_trained_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Convert to readable format
        predicted_class = torch.argmax(predictions, dim=-1).cpu().numpy()[0]
        confidence = predictions.max().cpu().numpy()
        
        # Map to sentiment labels
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment_label = sentiment_map[predicted_class]
        
        return sentiment_label, float(confidence)
```

## Step 4: API Server Module

**src/api_server.py**
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any
import time
import pandas as pd
import io
import asyncio
from datetime import datetime
import wandb

from .model_trainer import ModelTrainer
from .data_processor import DataProcessor
from config import *

# Pydantic models - like standardized recipe cards
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    processing_time: float
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    average_processing_time: float

# Initialize the main service - opening the restaurant
app = FastAPI(
    title="Professional ML Sentiment Analysis System",
    description="A production-ready sentiment analysis API with monitoring",
    version="1.0.0"
)

# Global instances - your permanent kitchen staff
model_trainer = ModelTrainer()
data_processor = DataProcessor()

@app.on_event("startup")
async def startup_event():
    """Prep work before opening - like preparing stations"""
    try:
        model_trainer.load_trained_model()
        wandb.init(project=WANDB_PROJECT, job_type="api_serving")
        print("🍳 Kitchen is ready! All stations operational.")
    except Exception as e:
        print(f"⚠️ Startup warning: {e}")
        print("Model will be loaded on first prediction request")

@app.on_event("shutdown")
async def shutdown_event():
    """Closing time cleanup"""
    wandb.finish()
    data_processor.db_session.close()

@app.get("/")
async def root():
    """Welcome message - like greeting customers"""
    return {
        "message": "Professional ML System API",
        "status": "operational",
        "endpoints": ["/predict", "/batch_predict", "/health", "/stats"]
    }

@app.get("/health")
async def health_check():
    """System health check - like checking all equipment"""
    try:
        # Quick model test
        test_prediction = model_trainer.predict("This is a test")
        return {
            "status": "healthy",
            "model_loaded": True,
            "database_connected": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(
    input_data: TextInput,
    background_tasks: BackgroundTasks
):
    """
    Single prediction endpoint - like preparing one special dish
    """
    try:
        start_time = time.time()
        
        # Make prediction
        sentiment, confidence = model_trainer.predict(input_data.text)
        
        processing_time = time.time() - start_time
        
        # Log metrics in background
        background_tasks.add_task(
            log_prediction_metrics,
            input_data.text,
            sentiment,
            confidence,
            processing_time
        )
        
        return PredictionResponse(
            sentiment=sentiment,
            confidence=confidence,
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict_sentiment(file: UploadFile = File(...)):
    """
    Batch prediction - like catering a large event
    Expects CSV with 'text' column
    """
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if 'text' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="CSV must contain 'text' column"
            )
        
        predictions = []
        processing_times = []
        
        # Process each text - like preparing multiple orders
        for _, row in df.iterrows():
            start_time = time.time()
            sentiment, confidence = model_trainer.predict(row['text'])
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            predictions.append(PredictionResponse(
                sentiment=sentiment,
                confidence=confidence,
                processing_time=processing_time,
                timestamp=datetime.utcnow().isoformat()
            ))
            
            # Store in database
            data_processor.store_prediction(
                row['text'], sentiment, confidence, processing_time
            )
        
        # Log batch metrics
        wandb.log({
            "batch_size": len(predictions),
            "average_processing_time": sum(processing_times) / len(processing_times),
            "batch_timestamp": datetime.utcnow().isoformat()
        })
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            average_processing_time=sum(processing_times) / len(processing_times)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    """Performance analytics - like daily sales reports"""
    try:
        # Query recent predictions
        recent_predictions = data_processor.db_session.query(
            data_processor.SentimentData
        ).order_by(
            data_processor.SentimentData.created_at.desc()
        ).limit(100).all()
        
        if not recent_predictions:
            return {"message": "No predictions found"}
        
        # Calculate statistics
        sentiments = [p.predicted_sentiment for p in recent_predictions]
        confidences = [p.confidence_score for p in recent_predictions]
        processing_times = [p.processing_time for p in recent_predictions]
        
        from collections import Counter
        sentiment_counts = Counter(sentiments)
        
        return {
            "total_predictions": len(recent_predictions),
            "sentiment_distribution": dict(sentiment_counts),
            "average_confidence": sum(confidences) / len(confidences),
            "average_processing_time": sum(processing_times) / len(processing_times),
            "last_prediction": recent_predictions[0].created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def log_prediction_metrics(text: str, sentiment: str, confidence: float, processing_time: float):
    """Background task for logging - like recording each order"""
    # Store in database
    data_processor.store_prediction(text, sentiment, confidence, processing_time)
    
    # Log to Weights & Biases
    wandb.log({
        "prediction_confidence": confidence,
        "processing_time": processing_time,
        "sentiment": sentiment,
        "text_length": len(text)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
```

## Step 5: Docker Configuration

**docker/Dockerfile**
```dockerfile
# Multi-stage build - like organizing kitchen prep and service areas
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies - essential kitchen tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first - like gathering all ingredients
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user - kitchen safety protocols
RUN useradd --create-home --shell /bin/bash mluser
RUN chown -R mluser:mluser /app
USER mluser

# Expose port
EXPOSE 8000

# Health check - regular kitchen inspections
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker/docker-compose.yml**
```yaml
version: '3.8'

services:
  # Main application service - the head kitchen
  ml-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://mluser:mlpassword@postgres:5432/ml_system
      - WANDB_ENTITY=your-username
    volumes:
      - ../models:/app/models
      - ../data:/app/data
    depends_on:
      - postgres
    restart: unless-stopped
    networks:
      - ml-network

  # Database service - the pantry storage
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=ml_system
      - POSTGRES_USER=mluser
      - POSTGRES_PASSWORD=mlpassword
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - ml-network

  # Monitoring service - quality control station
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped
    networks:
      - ml-network

volumes:
  postgres_data:

networks:
  ml-network:
    driver: bridge
```

## Step 6: Main Runner Script

**run.py**
```python
#!/usr/bin/env python3
"""
Main orchestrator - like the executive chef coordinating all operations
"""

import argparse
import asyncio
from pathlib import Path

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.api_server import app
import uvicorn

def train_pipeline(data_path: str):
    """Full training pipeline - like developing a new signature dish"""
    print("🔥 Starting training pipeline...")
    
    # Initialize components
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Prepare data
    print("📊 Preparing training data...")
    dataset = processor.prepare_training_data(data_path)
    
    # Train model
    print("🏋️ Training model...")
    trained_model, results = trainer.train_model(dataset)
    
    print(f"✅ Training complete! Test F1: {results.get('eval_f1', 'N/A'):.4f}")
    return results

def serve_api():
    """Start the API server - opening for business"""
    print("🚀 Starting ML API server...")
    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )

def main():
    parser = argparse.ArgumentParser(description="Professional ML System")
    parser.add_argument("command", choices=["train", "serve"], 
                       help="Command to execute")
    parser.add_argument("--data", type=str, help="Path to training data CSV")
    
    args = parser.parse_args()
    
    if args.command == "train":
        if not args.data:
            print("❌ --data argument required for training")
            return
        train_pipeline(args.data)
    elif args.command == "serve":
        serve_api()

if __name__ == "__main__":
    main()
```

## Step 7: Usage & Deployment

**Training the Model:**
```bash
# Prepare your data CSV with columns: text, label (0,1,2)
python run.py train --data data/sentiment_training.csv
```

**Running the API:**
```bash
# Local development
python run.py serve

# With Docker
docker-compose -f docker/docker-compose.yml up --build
```

**Testing the System:**
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This system works amazingly well!"}
)
print(response.json())

# Health check
health = requests.get("http://localhost:8000/health")
print(health.json())

# Statistics
stats = requests.get("http://localhost:8000/stats")
print(stats.json())
```

## Key Professional Features Implemented:

1. **Experiment Tracking**: Weights & Biases integration for monitoring training runs and API usage
2. **Model Versioning**: Proper model saving/loading with Hugging Face Transformers
3. **Database Integration**: SQLAlchemy for storing predictions and analytics
4. **API Production Ready**: FastAPI with proper error handling, validation, and documentation
5. **Containerization**: Docker setup for consistent deployment across environments
6. **Monitoring**: Health checks, metrics logging, and performance analytics
7. **Scalability**: Async processing, batch predictions, and background tasks
8. **Code Organization**: Modular structure following software engineering best practices

This system demonstrates enterprise-level ML engineering practices while maintaining the organic flow of a well-organized culinary operation - each component works in harmony to deliver consistent, high-quality results.

## Assignment: Experiment Tracking Dashboard

### Task Description

Create a comprehensive experiment tracking system that automatically runs multiple ML experiments with different hyperparameters, logs everything to Weights & Biases, and generates a summary report comparing all runs.

### Requirements

1. **Implement AutoML Experiment Runner**: Create a class that automatically tests different combinations of:
   - Learning rates: [0.001, 0.01, 0.1]
   - Hidden layer sizes: [32, 64, 128]
   - Dropout rates: [0.1, 0.3, 0.5]

2. **Advanced W&B Integration**: 
   - Use W&B Sweeps for hyperparameter optimization
   - Log model architecture visualizations
   - Track system metrics (GPU usage, memory)
   - Create custom charts and tables

3. **Multi-Framework Support**:
   - Run identical experiments in both PyTorch and TensorFlow
   - Compare performance and training characteristics
   - Generate framework recommendation based on results

4. **Automated Reporting**:
   - Generate markdown report with best configurations
   - Create visualizations comparing different runs
   - Include statistical significance testing

### Starter Code Framework

```python
class AutoMLExperimentRunner:
    def __init__(self, project_name="automl-experiments"):
        self.project_name = project_name
        self.experiment_configs = self.generate_configurations()
    
    def generate_configurations(self):
        """Generate all hyperparameter combinations"""
        # Your implementation here
        pass
    
    def run_pytorch_sweep(self):
        """Run PyTorch experiments with W&B sweeps"""
        # Your implementation here
        pass
    
    def run_tensorflow_sweep(self):
        """Run TensorFlow experiments with W&B sweeps"""
        # Your implementation here
        pass
    
    def analyze_results(self):
        """Analyze and compare all experimental results"""
        # Your implementation here
        pass
    
    def generate_report(self):
        """Generate comprehensive markdown report"""
        # Your implementation here
        pass

# Your task: Complete this implementation
```

### Submission Guidelines

- Complete implementation with all methods filled in
- Include proper error handling and logging
- Add comprehensive docstrings
- Create sample visualizations showing your results
- Write a brief explanation of your findings (which configurations work best and why)

### Evaluation Criteria

- **Functionality** (40%): All experiments run successfully
- **W&B Integration** (25%): Proper use of advanced W&B features
- **Code Quality** (20%): Clean, well-documented code
- **Analysis** (15%): Insightful comparison and recommendations

---

## Conclusion

Today's journey took us through the professional toolkit that transforms experimental AI projects into production-ready systems. Like a master chef who understands not just individual ingredients but how they combine to create extraordinary experiences, you now have the knowledge to orchestrate PyTorch's flexibility with TensorFlow's structure, leverage the collective wisdom embedded in Hugging Face transformers, maintain professional experimental standards with W&B tracking, and package everything into deployable containers with Docker.

The synergy between these tools creates possibilities far greater than any single component. Your next AI project