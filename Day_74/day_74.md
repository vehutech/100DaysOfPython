# AI Mastery Course - Day 74: Model Evaluation & Validation with Python

## Learning Objective
By the end of this lesson, you will be able to implement comprehensive model evaluation and validation techniques in Django applications, including train/validation/test splits, cross-validation, performance metrics calculation, and ROC curve analysis to ensure your AI models perform reliably in production.

---

Imagine you're the head chef at a prestigious restaurant, and you've just created a revolutionary new recipe for the perfect pasta dish. But here's the catch - you can't just serve it to customers immediately! 

Just like a responsible chef would:
- **Test the recipe** with a small batch first (train/validation split)
- **Have different chefs try it** multiple times to ensure consistency (cross-validation)
- **Measure the results** - taste, presentation, cooking time (performance metrics)
- **Chart the success rate** across different customer preferences (ROC curves)

In the AI kitchen, model evaluation is your quality control system. You're not just cooking up predictions; you're ensuring every "dish" (prediction) that leaves your kitchen meets the highest standards!

---

## Lesson 1: Train/Validation/Test Splits - The Three-Station Kitchen

### The Chef's Approach
Think of your data like ingredients in a professional kitchen with three stations:
- **Prep Station (Training Data)**: Where you learn the recipe
- **Tasting Station (Validation Data)**: Where you adjust seasoning and technique  
- **Customer Station (Test Data)**: The final judgment - untouched until service

### Django Implementation

```python
# models.py
from django.db import models
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

class DataSplitter(models.Model):
    """Model to handle data splitting operations"""
    dataset_name = models.CharField(max_length=100)
    train_size = models.FloatField(default=0.7)
    validation_size = models.FloatField(default=0.15)
    test_size = models.FloatField(default=0.15)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def split_data(self, X, y):
        """
        Split data into train, validation, and test sets
        Like organizing ingredients across three kitchen stations
        """
        # First split: separate test data (the customer station)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=42,
            stratify=y  # Ensures balanced distribution like balanced menu
        )
        
        # Second split: separate train and validation (prep and tasting stations)
        val_size_adjusted = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_temp
        )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val, 
            'X_test': X_test, 'y_test': y_test
        }

# views.py
from django.shortcuts import render
from django.http import JsonResponse
from .models import DataSplitter
import pandas as pd

def split_dataset_view(request):
    """View to handle data splitting requests"""
    if request.method == 'POST':
        # Load your dataset (example with sample data)
        data = pd.read_csv('your_dataset.csv')  # Replace with your data source
        X = data.drop('target', axis=1)  # Features
        y = data['target']  # Target variable
        
        # Create splitter instance
        splitter = DataSplitter.objects.create(
            dataset_name="Customer Preference Dataset",
            train_size=0.7,
            validation_size=0.15,
            test_size=0.15
        )
        
        # Perform the split
        splits = splitter.split_data(X, y)
        
        # Save splits for later use
        for split_name, split_data in splits.items():
            joblib.dump(split_data, f'data/{split_name}.pkl')
        
        return JsonResponse({
            'message': 'Data split successfully!',
            'train_samples': len(splits['X_train']),
            'validation_samples': len(splits['X_val']),
            'test_samples': len(splits['X_test'])
        })
    
    return render(request, 'split_data.html')
```

**Syntax Explanation:**
- `train_test_split()`: Scikit-learn function that randomly divides data
- `stratify=y`: Ensures each split maintains the same proportion of each class (like ensuring each station gets a balanced mix of ingredients)
- `random_state=42`: Sets a seed for reproducible results (like following the same recipe steps every time)

---

## Lesson 2: Cross-Validation Techniques - The Multiple Chef Test

### The Chef's Approach
Imagine you want to test your recipe's consistency. You'd have 5 different experienced chefs each make the dish, rotating who does what. That's exactly what cross-validation does - it tests your model multiple times with different data combinations!

### Django Implementation

```python
# models.py (continued)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class CrossValidator(models.Model):
    """Model to handle cross-validation operations"""
    model_name = models.CharField(max_length=100)
    cv_folds = models.IntegerField(default=5)
    cv_scores = models.JSONField(default=list)  # Store cross-validation scores
    mean_score = models.FloatField(null=True, blank=True)
    std_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def perform_cross_validation(self, X, y, model=None):
        """
        Perform k-fold cross-validation
        Like having multiple chefs test the same recipe
        """
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create stratified k-fold (ensures balanced "chef assignments")
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        
        # Store results
        self.cv_scores = cv_scores.tolist()
        self.mean_score = np.mean(cv_scores)
        self.std_score = np.std(cv_scores)
        self.save()
        
        return {
            'individual_scores': cv_scores,
            'mean_accuracy': self.mean_score,
            'std_deviation': self.std_score,
            'confidence_interval': (
                self.mean_score - 2*self.std_score, 
                self.mean_score + 2*self.std_score
            )
        }

# views.py (continued)
def cross_validate_model(request):
    """View to perform cross-validation"""
    if request.method == 'POST':
        # Load training data
        X_train = joblib.load('data/X_train.pkl')
        y_train = joblib.load('data/y_train.pkl')
        
        # Create cross-validator
        cv_validator = CrossValidator.objects.create(
            model_name="Random Forest Classifier",
            cv_folds=5
        )
        
        # Perform cross-validation
        results = cv_validator.perform_cross_validation(X_train, y_train)
        
        return JsonResponse({
            'message': 'Cross-validation completed!',
            'results': results,
            'interpretation': f"Your model performs consistently with "
                           f"{results['mean_accuracy']:.3f} ± {results['std_deviation']:.3f} accuracy"
        })
    
    return render(request, 'cross_validate.html')
```

**Syntax Explanation:**
- `StratifiedKFold`: Ensures each fold maintains class distribution (like ensuring each chef gets the same ingredient proportions)
- `cross_val_score()`: Automatically performs k-fold cross-validation
- `JSONField`: Django field type that stores JSON data (lists, dictionaries) in the database

---

## Lesson 3: Performance Metrics - The Taste Test Scorecard

### The Chef's Approach
When judging a dish, you don't just say "good" or "bad." You evaluate:
- **Accuracy**: Overall deliciousness (how often you get it right)
- **Precision**: When you say it's perfect, how often is it actually perfect?
- **Recall**: Of all the perfect dishes possible, how many did you identify?
- **F1-Score**: The balanced judgment between precision and recall

### Django Implementation

```python
# models.py (continued)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

class PerformanceEvaluator(models.Model):
    """Model to calculate and store performance metrics"""
    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score_value = models.FloatField(null=True, blank=True)
    confusion_matrix_data = models.JSONField(default=dict)
    classification_report_data = models.JSONField(default=dict)
    evaluation_date = models.DateTimeField(auto_now_add=True)
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive performance metrics
        Like creating a detailed taste test scorecard
        """
        # Basic metrics (the four pillars of evaluation)
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred, average='weighted')
        self.recall = recall_score(y_true, y_pred, average='weighted')
        self.f1_score_value = f1_score(y_true, y_pred, average='weighted')
        
        # Detailed analysis
        self.confusion_matrix_data = confusion_matrix(y_true, y_pred).tolist()
        
        # Classification report (detailed breakdown per class)
        report = classification_report(y_true, y_pred, output_dict=True)
        self.classification_report_data = report
        
        self.save()
        
        return self.get_metrics_summary()
    
    def get_metrics_summary(self):
        """Return a chef-friendly interpretation of metrics"""
        return {
            'accuracy': {
                'value': self.accuracy,
                'interpretation': f"Your model gets it right {self.accuracy:.1%} of the time - like a chef with {self.accuracy:.1%} customer satisfaction!"
            },
            'precision': {
                'value': self.precision,
                'interpretation': f"When your model says 'this is class A', it's correct {self.precision:.1%} of the time - high precision means no false promises!"
            },
            'recall': {
                'value': self.recall,
                'interpretation': f"Your model catches {self.recall:.1%} of all actual instances - high recall means nothing slips through the cracks!"
            },
            'f1_score': {
                'value': self.f1_score_value,
                'interpretation': f"F1-score of {self.f1_score_value:.3f} represents the perfect balance between precision and recall - like balancing flavor and presentation!"
            }
        }

# views.py (continued)
def evaluate_model_performance(request):
    """View to evaluate model performance"""
    if request.method == 'POST':
        # Load validation data and make predictions
        X_val = joblib.load('data/X_val.pkl')
        y_val = joblib.load('data/y_val.pkl')
        
        # Load your trained model (assuming it's saved)
        model = joblib.load('models/trained_model.pkl')  # You'd save this after training
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Create evaluator and calculate metrics
        evaluator = PerformanceEvaluator.objects.create(
            model_name="Random Forest - Validation Set"
        )
        
        metrics_summary = evaluator.calculate_metrics(y_val, y_pred)
        
        return JsonResponse({
            'message': 'Model evaluation completed!',
            'metrics': metrics_summary,
            'confusion_matrix': evaluator.confusion_matrix_data
        })
    
    return render(request, 'evaluate_performance.html')
```

**Syntax Explanation:**
- `average='weighted'`: Calculates metrics for each class and averages them weighted by class frequency
- `output_dict=True`: Returns classification report as a dictionary instead of a string
- `.tolist()`: Converts NumPy arrays to Python lists (JSON-serializable)

---

## Lesson 4: ROC Curves and AUC - The Master Chef's Rating System

### The Chef's Approach
Imagine you're judging a cooking competition. You need to plot how well each chef performs at different "difficulty thresholds." ROC curves show you exactly that - how your model performs across all possible decision thresholds, like rating dishes from "definitely amazing" to "definitely terrible."

### Django Implementation

```python
# models.py (continued)
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class ROCAnalyzer(models.Model):
    """Model to handle ROC curve analysis"""
    model_name = models.CharField(max_length=100)
    auc_score = models.FloatField(null=True, blank=True)
    roc_curve_image = models.TextField(blank=True)  # Base64 encoded image
    fpr_data = models.JSONField(default=list)
    tpr_data = models.JSONField(default=list)
    thresholds_data = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def calculate_roc_auc(self, y_true, y_pred_proba):
        """
        Calculate ROC curve and AUC score
        Like creating a master performance chart for a chef
        """
        # Calculate ROC curve points
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # Calculate AUC (Area Under the Curve)
        self.auc_score = auc(fpr, tpr)
        
        # Store curve data
        self.fpr_data = fpr.tolist()
        self.tpr_data = tpr.tolist()
        self.thresholds_data = thresholds.tolist()
        
        # Generate ROC curve plot
        self.roc_curve_image = self._generate_roc_plot(fpr, tpr)
        
        self.save()
        
        return self.get_roc_interpretation()
    
    def _generate_roc_plot(self, fpr, tpr):
        """Generate ROC curve visualization"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {self.auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Guess (AUC = 0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Bad dishes called good)')
        plt.ylabel('True Positive Rate (Good dishes called good)')
        plt.title('ROC Curve - Model Performance Across All Thresholds')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def get_roc_interpretation(self):
        """Provide chef-friendly ROC interpretation"""
        if self.auc_score >= 0.9:
            performance = "Exceptional! Your model is like a Michelin-star chef - nearly perfect discrimination!"
        elif self.auc_score >= 0.8:
            performance = "Excellent! Your model performs like an experienced head chef - reliable and accurate!"
        elif self.auc_score >= 0.7:
            performance = "Good! Your model is like a skilled line cook - does well but has room for improvement!"
        elif self.auc_score >= 0.6:
            performance = "Fair performance - like a culinary student, showing promise but needs more training!"
        else:
            performance = "Poor performance - time to go back to culinary school and rethink the recipe!"
        
        return {
            'auc_score': self.auc_score,
            'performance_level': performance,
            'interpretation': f"AUC of {self.auc_score:.3f} means your model can distinguish between classes {self.auc_score:.1%} better than random guessing",
            'roc_curve_image': self.roc_curve_image
        }

# views.py (continued)
def analyze_roc_curve(request):
    """View to perform ROC analysis"""
    if request.method == 'POST':
        # Load validation data
        X_val = joblib.load('data/X_val.pkl')
        y_val = joblib.load('data/y_val.pkl')
        
        # Load trained model
        model = joblib.load('models/trained_model.pkl')
        
        # Get prediction probabilities (needed for ROC)
        y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of positive class
        
        # Create ROC analyzer
        roc_analyzer = ROCAnalyzer.objects.create(
            model_name="Random Forest - ROC Analysis"
        )
        
        # Perform ROC analysis
        roc_results = roc_analyzer.calculate_roc_auc(y_val, y_pred_proba)
        
        return JsonResponse({
            'message': 'ROC analysis completed!',
            'results': roc_results
        })
    
    return render(request, 'roc_analysis.html')
```

**Syntax Explanation:**
- `predict_proba()`: Returns prediction probabilities instead of hard classifications
- `[:, 1]`: Selects the probability of the positive class (column 1)
- `base64.b64encode()`: Converts binary image data to text format for storage/transmission
- `BytesIO()`: Creates an in-memory buffer for temporary file operations

---

# Model Evaluation Framework - Django Project

## The Master Chef's Quality Control Kitchen

Just as a master chef needs a systematic way to evaluate and validate every dish before it reaches the customer, we need a comprehensive framework to evaluate our machine learning models. Think of this framework as your kitchen's quality control station - where every model gets thoroughly tested, measured, and validated before deployment.

## Project Overview

We'll build a Django-based model evaluation framework that can handle multiple ML models, perform various evaluation techniques, and provide comprehensive performance analytics. This is like creating a state-of-the-art testing kitchen where every recipe (model) goes through rigorous quality checks.

## Project Structure

```
model_evaluation_framework/
├── manage.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── evaluator/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   ├── utils.py
│   └── ml_evaluator.py
├── templates/
│   └── evaluator/
│       ├── base.html
│       ├── dashboard.html
│       ├── upload_model.html
│       └── evaluation_results.html
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
└── requirements.txt
```

## Core Implementation

### 1. Django Models (models.py)

```python
from django.db import models
from django.contrib.auth.models import User
import uuid
import json

class MLModel(models.Model):
    """Represents a machine learning model - like a recipe in our kitchen"""
    
    MODEL_TYPES = [
        ('classification', 'Classification'),
        ('regression', 'Regression'),
        ('clustering', 'Clustering'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    description = models.TextField(blank=True)
    model_file = models.FileField(upload_to='models/')
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"

class Dataset(models.Model):
    """Represents datasets used for evaluation - like ingredients for testing"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    data_file = models.FileField(upload_to='datasets/')
    target_column = models.CharField(max_length=100)
    feature_columns = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class EvaluationRun(models.Model):
    """Represents an evaluation session - like a complete quality test"""
    
    VALIDATION_METHODS = [
        ('train_test_split', 'Train-Test Split'),
        ('cross_validation', 'Cross Validation'),
        ('bootstrap', 'Bootstrap Validation'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    validation_method = models.CharField(max_length=20, choices=VALIDATION_METHODS)
    test_size = models.FloatField(default=0.2)
    cv_folds = models.IntegerField(default=5)
    random_state = models.IntegerField(default=42)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, default='pending')
    
    def __str__(self):
        return f"Evaluation: {self.model.name} on {self.dataset.name}"

class EvaluationMetrics(models.Model):
    """Stores evaluation results - like taste test scores"""
    
    evaluation_run = models.OneToOneField(EvaluationRun, on_delete=models.CASCADE)
    
    # Classification Metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    roc_auc = models.FloatField(null=True, blank=True)
    
    # Regression Metrics
    mse = models.FloatField(null=True, blank=True)
    rmse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    r2_score = models.FloatField(null=True, blank=True)
    
    # Detailed Results
    confusion_matrix = models.JSONField(null=True, blank=True)
    classification_report = models.JSONField(null=True, blank=True)
    feature_importance = models.JSONField(null=True, blank=True)
    roc_curve_data = models.JSONField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Metrics for {self.evaluation_run}"
```

### 2. ML Evaluation Engine (ml_evaluator.py)

```python
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class ModelEvaluator:
    """
    The Master Chef's Quality Control System
    This class handles all model evaluation processes like a head chef
    overseeing quality control in a professional kitchen.
    """
    
    def __init__(self, model_path, dataset_path, target_column):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.model = None
        self.data = None
        self.X = None
        self.y = None
        self.results = {}
        
    def load_model(self):
        """Load the trained model - like bringing the dish to the tasting station"""
        try:
            self.model = joblib.load(self.model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def load_data(self):
        """Load and prepare the dataset - like preparing ingredients for testing"""
        try:
            # Handle different file formats
            if self.dataset_path.endswith('.csv'):
                self.data = pd.read_csv(self.dataset_path)
            elif self.dataset_path.endswith('.xlsx'):
                self.data = pd.read_excel(self.dataset_path)
            
            # Prepare features and target
            self.y = self.data[self.target_column]
            self.X = self.data.drop(columns=[self.target_column])
            
            # Handle categorical variables
            for col in self.X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col])
            
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def train_test_evaluation(self, test_size=0.2, random_state=42):
        """
        Perform train-test split evaluation
        Like dividing ingredients into cooking and tasting portions
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state,
            stratify=self.y if self._is_classification() else None
        )
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        if self._is_classification():
            return self._calculate_classification_metrics(y_test, y_pred, X_test)
        else:
            return self._calculate_regression_metrics(y_test, y_pred)
    
    def cross_validation_evaluation(self, cv_folds=5, random_state=42):
        """
        Perform cross-validation evaluation
        Like testing the recipe multiple times with different ingredient batches
        """
        if self._is_classification():
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            # Calculate various metrics
            accuracy_scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring='accuracy')
            precision_scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring='precision_weighted')
            recall_scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring='recall_weighted')
            f1_scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring='f1_weighted')
            
            return {
                'accuracy': {
                    'mean': np.mean(accuracy_scores),
                    'std': np.std(accuracy_scores),
                    'scores': accuracy_scores.tolist()
                },
                'precision': {
                    'mean': np.mean(precision_scores),
                    'std': np.std(precision_scores),
                    'scores': precision_scores.tolist()
                },
                'recall': {
                    'mean': np.mean(recall_scores),
                    'std': np.std(recall_scores),
                    'scores': recall_scores.tolist()
                },
                'f1_score': {
                    'mean': np.mean(f1_scores),
                    'std': np.std(f1_scores),
                    'scores': f1_scores.tolist()
                }
            }
        else:
            # Regression cross-validation
            mse_scores = cross_val_score(self.model, self.X, self.y, cv=cv_folds, scoring='neg_mean_squared_error')
            mae_scores = cross_val_score(self.model, self.X, self.y, cv=cv_folds, scoring='neg_mean_absolute_error')
            r2_scores = cross_val_score(self.model, self.X, self.y, cv=cv_folds, scoring='r2')
            
            return {
                'mse': {
                    'mean': -np.mean(mse_scores),
                    'std': np.std(mse_scores),
                    'scores': (-mse_scores).tolist()
                },
                'mae': {
                    'mean': -np.mean(mae_scores),
                    'std': np.std(mae_scores),
                    'scores': (-mae_scores).tolist()
                },
                'r2_score': {
                    'mean': np.mean(r2_scores),
                    'std': np.std(r2_scores),
                    'scores': r2_scores.tolist()
                }
            }
    
    def _calculate_classification_metrics(self, y_true, y_pred, X_test):
        """Calculate comprehensive classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # ROC Curve for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                y_proba = self.model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_true, y_proba)
                metrics['roc_auc'] = auc(fpr, tpr)
                metrics['roc_curve_data'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            except:
                pass
        
        # Feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.X.columns.tolist()
            importance_data = list(zip(feature_names, self.model.feature_importances_))
            metrics['feature_importance'] = sorted(importance_data, key=lambda x: x[1], reverse=True)
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred):
        """Calculate comprehensive regression metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
    
    def _is_classification(self):
        """Determine if the problem is classification or regression"""
        return hasattr(self.model, 'predict_proba') or len(np.unique(self.y)) < 10
    
    def generate_evaluation_report(self, validation_method='train_test_split', **kwargs):
        """
        Generate comprehensive evaluation report
        Like creating a complete quality assessment document
        """
        if not self.load_model() or not self.load_data():
            return None
        
        if validation_method == 'train_test_split':
            results = self.train_test_evaluation(**kwargs)
        elif validation_method == 'cross_validation':
            results = self.cross_validation_evaluation(**kwargs)
        else:
            raise ValueError("Invalid validation method")
        
        return results
```

### 3. Django Views (views.py)

```python
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.core.files.storage import default_storage
from django.conf import settings
import os
import json
from datetime import datetime

from .models import MLModel, Dataset, EvaluationRun, EvaluationMetrics
from .forms import ModelUploadForm, DatasetUploadForm, EvaluationForm
from .ml_evaluator import ModelEvaluator

@login_required
def dashboard(request):
    """Main dashboard - like the head chef's overview of all quality tests"""
    context = {
        'models': MLModel.objects.filter(created_by=request.user, is_active=True),
        'datasets': Dataset.objects.all(),
        'recent_evaluations': EvaluationRun.objects.filter(
            model__created_by=request.user
        ).order_by('-created_at')[:10],
        'total_models': MLModel.objects.filter(created_by=request.user).count(),
        'total_evaluations': EvaluationRun.objects.filter(
            model__created_by=request.user
        ).count()
    }
    return render(request, 'evaluator/dashboard.html', context)

@login_required
def upload_model(request):
    """Upload a new model - like adding a new recipe to test"""
    if request.method == 'POST':
        form = ModelUploadForm(request.POST, request.FILES)
        if form.is_valid():
            model = form.save(commit=False)
            model.created_by = request.user
            model.save()
            messages.success(request, 'Model uploaded successfully!')
            return redirect('evaluator:dashboard')
    else:
        form = ModelUploadForm()
    
    return render(request, 'evaluator/upload_model.html', {'form': form})

@login_required
def upload_dataset(request):
    """Upload a new dataset - like adding new ingredients to test with"""
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, 'Dataset uploaded successfully!')
            return redirect('evaluator:dashboard')
    else:
        form = DatasetUploadForm()
    
    return render(request, 'evaluator/upload_dataset.html', {'form': form})

@login_required
def run_evaluation(request, model_id):
    """Start a new evaluation run - like beginning a comprehensive quality test"""
    model = get_object_or_404(MLModel, id=model_id, created_by=request.user)
    
    if request.method == 'POST':
        form = EvaluationForm(request.POST)
        if form.is_valid():
            evaluation_run = form.save(commit=False)
            evaluation_run.model = model
            evaluation_run.save()
            
            # Run evaluation in background (in production, use Celery)
            try:
                _execute_evaluation(evaluation_run)
                messages.success(request, 'Evaluation completed successfully!')
                return redirect('evaluator:evaluation_results', evaluation_run.id)
            except Exception as e:
                messages.error(request, f'Evaluation failed: {str(e)}')
                evaluation_run.status = 'failed'
                evaluation_run.save()
    else:
        form = EvaluationForm()
    
    context = {
        'form': form,
        'model': model,
        'datasets': Dataset.objects.all()
    }
    return render(request, 'evaluator/run_evaluation.html', context)

def _execute_evaluation(evaluation_run):
    """
    Execute the actual evaluation process
    Like conducting the actual taste test and quality assessment
    """
    evaluation_run.status = 'running'
    evaluation_run.save()
    
    # Get file paths
    model_path = evaluation_run.model.model_file.path
    dataset_path = evaluation_run.dataset.data_file.path
    target_column = evaluation_run.dataset.target_column
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path, dataset_path, target_column)
    
    # Set evaluation parameters
    eval_params = {
        'test_size': evaluation_run.test_size,
        'random_state': evaluation_run.random_state
    }
    
    if evaluation_run.validation_method == 'cross_validation':
        eval_params = {
            'cv_folds': evaluation_run.cv_folds,
            'random_state': evaluation_run.random_state
        }
    
    # Run evaluation
    results = evaluator.generate_evaluation_report(
        validation_method=evaluation_run.validation_method,
        **eval_params
    )
    
    if results:
        # Save metrics
        metrics = EvaluationMetrics(evaluation_run=evaluation_run)
        
        # Save appropriate metrics based on model type
        if 'accuracy' in results:
            if isinstance(results['accuracy'], dict):  # Cross-validation results
                metrics.accuracy = results['accuracy']['mean']
                metrics.precision = results['precision']['mean']
                metrics.recall = results['recall']['mean']
                metrics.f1_score = results['f1_score']['mean']
            else:  # Single evaluation results
                metrics.accuracy = results.get('accuracy')
                metrics.precision = results.get('precision')
                metrics.recall = results.get('recall')
                metrics.f1_score = results.get('f1_score')
                metrics.roc_auc = results.get('roc_auc')
                metrics.confusion_matrix = results.get('confusion_matrix')
                metrics.classification_report = results.get('classification_report')
                metrics.roc_curve_data = results.get('roc_curve_data')
        
        if 'mse' in results:
            if isinstance(results['mse'], dict):  # Cross-validation results
                metrics.mse = results['mse']['mean']
                metrics.mae = results['mae']['mean']
                metrics.r2_score = results['r2_score']['mean']
            else:  # Single evaluation results
                metrics.mse = results.get('mse')
                metrics.rmse = results.get('rmse')
                metrics.mae = results.get('mae')
                metrics.r2_score = results.get('r2_score')
        
        metrics.feature_importance = results.get('feature_importance')
        metrics.save()
        
        evaluation_run.status = 'completed'
        evaluation_run.completed_at = datetime.now()
    else:
        evaluation_run.status = 'failed'
    
    evaluation_run.save()

@login_required
def evaluation_results(request, evaluation_id):
    """Display evaluation results - like presenting the quality test report"""
    evaluation_run = get_object_or_404(EvaluationRun, id=evaluation_id)
    
    # Check if user has permission to view this evaluation
    if evaluation_run.model.created_by != request.user:
        messages.error(request, 'You do not have permission to view this evaluation.')
        return redirect('evaluator:dashboard')
    
    try:
        metrics = evaluation_run.evaluationmetrics
    except EvaluationMetrics.DoesNotExist:
        metrics = None
    
    context = {
        'evaluation_run': evaluation_run,
        'metrics': metrics,
        'model': evaluation_run.model,
        'dataset': evaluation_run.dataset
    }
    
    return render(request, 'evaluator/evaluation_results.html', context)

@login_required
def api_evaluation_status(request, evaluation_id):
    """API endpoint to check evaluation status"""
    evaluation_run = get_object_or_404(EvaluationRun, id=evaluation_id)
    
    if evaluation_run.model.created_by != request.user:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    return JsonResponse({
        'status': evaluation_run.status,
        'completed_at': evaluation_run.completed_at.isoformat() if evaluation_run.completed_at else None
    })
```

### 4. Django Forms (forms.py)

```python
from django import forms
from .models import MLModel, Dataset, EvaluationRun

class ModelUploadForm(forms.ModelForm):
    class Meta:
        model = MLModel
        fields = ['name', 'model_type', 'description', 'model_file']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter model name'}),
            'model_type': forms.Select(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Describe your model...'}),
            'model_file': forms.FileInput(attrs={'class': 'form-control', 'accept': '.pkl,.joblib,.pickle'})
        }

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'data_file', 'target_column']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter dataset name'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Describe your dataset...'}),
            'data_file': forms.FileInput(attrs={'class': 'form-control', 'accept': '.csv,.xlsx'}),
            'target_column': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter target column name'})
        }

class EvaluationForm(forms.ModelForm):
    class Meta:
        model = EvaluationRun
        fields = ['dataset', 'validation_method', 'test_size', 'cv_folds', 'random_state']
        widgets = {
            'dataset': forms.Select(attrs={'class': 'form-control'}),
            'validation_method': forms.Select(attrs={'class': 'form-control'}),
            'test_size': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'min': '0.1', 'max': '0.5'}),
            'cv_folds': forms.NumberInput(attrs={'class': 'form-control', 'min': '2', 'max': '10'}),
            'random_state': forms.NumberInput(attrs={'class': 'form-control'})
        }
```

### 5. Templates

#### base.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Model Evaluation Framework{% endblock %}

#### evaluation_results.html
```html
{% extends 'evaluator/base.html' %}

{% block content %}
<div class="row">
    <div class="col-12">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <div>
                <h1><i class="fas fa-clipboard-check me-2"></i>Quality Test Results</h1>
                <p class="text-muted">Complete evaluation report for {{ model.name }}</p>
            </div>
            <a href="{% url 'evaluator:dashboard' %}" class="btn btn-outline-secondary">
                <i class="fas fa-arrow-left me-1"></i>Back to Kitchen
            </a>
        </div>
    </div>
</div>

<!-- Evaluation Overview -->
<div class="row mb-4">
    <div class="col-12">
        <div class="metric-card">
            <h5><i class="fas fa-info-circle me-2"></i>Test Overview</h5>
            <div class="row">
                <div class="col-md-3">
                    <strong>Recipe (Model):</strong><br>
                    <span class="text-primary">{{ model.name }}</span>
                </div>
                <div class="col-md-3">
                    <strong>Ingredients (Dataset):</strong><br>
                    <span class="text-success">{{ dataset.name }}</span>
                </div>
                <div class="col-md-3">
                    <strong>Test Method:</strong><br>
                    <span class="text-info">{{ evaluation_run.get_validation_method_display }}</span>
                </div>
                <div class="col-md-3">
                    <strong>Status:</strong><br>
                    <span class="badge bg-{% if evaluation_run.status == 'completed' %}success{% elif evaluation_run.status == 'failed' %}danger{% else %}warning{% endif %}">
                        {{ evaluation_run.status|title }}
                    </span>
                </div>
            </div>
        </div>
    </div>
</div>

{% if metrics %}
    <!-- Classification Metrics -->
    {% if metrics.accuracy %}
    <div class="row mb-4">
        <div class="col-12">
            <div class="metric-card">
                <h5><i class="fas fa-bullseye me-2"></i>Classification Performance - Taste Test Scores</h5>
                <div class="row">
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-primary">{{ metrics.accuracy|floatformat:4 }}</h3>
                            <p class="mb-0">Accuracy<br><small class="text-muted">Overall correctness</small></p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-success">{{ metrics.precision|floatformat:4 }}</h3>
                            <p class="mb-0">Precision<br><small class="text-muted">True positive rate</small></p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-warning">{{ metrics.recall|floatformat:4 }}</h3>
                            <p class="mb-0">Recall<br><small class="text-muted">Sensitivity</small></p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-info">{{ metrics.f1_score|floatformat:4 }}</h3>
                            <p class="mb-0">F1-Score<br><small class="text-muted">Balanced measure</small></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- ROC Curve -->
    {% if metrics.roc_auc %}
    <div class="row mb-4">
        <div class="col-md-6">
            <div class="metric-card">
                <h5><i class="fas fa-chart-area me-2"></i>ROC Analysis</h5>
                <div class="text-center mb-3">
                    <h3 class="text-primary">{{ metrics.roc_auc|floatformat:4 }}</h3>
                    <p class="mb-0">Area Under Curve (AUC)</p>
                </div>
                <canvas id="rocCurve" width="400" height="300"></canvas>
            </div>
        </div>
        <div class="col-md-6">
            <div class="metric-card">
                <h5><i class="fas fa-table me-2"></i>Confusion Matrix</h5>
                <div id="confusionMatrix"></div>
            </div>
        </div>
    </div>
    {% endif %}
    {% endif %}

    <!-- Regression Metrics -->
    {% if metrics.mse %}
    <div class="row mb-4">
        <div class="col-12">
            <div class="metric-card">
                <h5><i class="fas fa-ruler me-2"></i>Regression Performance - Measurement Accuracy</h5>
                <div class="row">
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-danger">{{ metrics.mse|floatformat:4 }}</h3>
                            <p class="mb-0">MSE<br><small class="text-muted">Mean Squared Error</small></p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-warning">{{ metrics.rmse|floatformat:4 }}</h3>
                            <p class="mb-0">RMSE<br><small class="text-muted">Root Mean Squared Error</small></p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-info">{{ metrics.mae|floatformat:4 }}</h3>
                            <p class="mb-0">MAE<br><small class="text-muted">Mean Absolute Error</small></p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-success">{{ metrics.r2_score|floatformat:4 }}</h3>
                            <p class="mb-0">R² Score<br><small class="text-muted">Coefficient of Determination</small></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    {% endif %}

    <!-- Feature Importance -->
    {% if metrics.feature_importance %}
    <div class="row mb-4">
        <div class="col-12">
            <div class="metric-card">
                <h5><i class="fas fa-weight-hanging me-2"></i>Key Ingredients - Feature Importance</h5>
                <canvas id="featureImportance" width="800" height="400"></canvas>
            </div>
        </div>
    </div>
    {% endif %}

    <!-- Detailed Classification Report -->
    {% if metrics.classification_report %}
    <div class="row mb-4">
        <div class="col-12">
            <div class="metric-card">
                <h5><i class="fas fa-file-alt me-2"></i>Detailed Quality Report</h5>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Class</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                                <th>Support</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for class_name, metrics_data in metrics.classification_report.items %}
                                {% if class_name != 'accuracy' and class_name != 'macro avg' and class_name != 'weighted avg' %}
                                <tr>
                                    <td><strong>{{ class_name }}</strong></td>
                                    <td>{{ metrics_data.precision|floatformat:4 }}</td>
                                    <td>{{ metrics_data.recall|floatformat:4 }}</td>
                                    <td>{{ metrics_data.f1-score|floatformat:4 }}</td>
                                    <td>{{ metrics_data.support }}</td>
                                </tr>
                                {% endif %}
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    {% endif %}

{% else %}
    <div class="row">
        <div class="col-12">
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                No evaluation results available yet. The quality test may still be running or may have failed.
            </div>
        </div>
    </div>
{% endif %}
{% endblock %}

{% block extra_js %}
<script>
    // ROC Curve Chart
    {% if metrics.roc_curve_data %}
    const rocCtx = document.getElementById('rocCurve').getContext('2d');
    new Chart(rocCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'ROC Curve (AUC = {{ metrics.roc_auc|floatformat:4 }})',
                data: {{ metrics.roc_curve_data.fpr|safe }}.map((fpr, index) => ({
                    x: fpr,
                    y: {{ metrics.roc_curve_data.tpr|safe }}[index]
                })),
                borderColor: 'rgb(102, 126, 234)',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                fill: true
            }, {
                label: 'Random Classifier',
                data: [{x: 0, y: 0}, {x: 1, y: 1}],
                borderColor: 'rgb(255, 99, 132)',
                borderDash: [5, 5],
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'False Positive Rate'
                    },
                    min: 0,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: 'True Positive Rate'
                    },
                    min: 0,
                    max: 1
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'ROC Curve - Recipe Performance Analysis'
                }
            }
        }
    });
    {% endif %}

    // Feature Importance Chart
    {% if metrics.feature_importance %}
    const featureCtx = document.getElementById('featureImportance').getContext('2d');
    const featureData = {{ metrics.feature_importance|safe }};
    
    new Chart(featureCtx, {
        type: 'bar',
        data: {
            labels: featureData.slice(0, 10).map(item => item[0]),
            datasets: [{
                label: 'Importance Score',
                data: featureData.slice(0, 10).map(item => item[1]),
                backgroundColor: 'rgba(102, 126, 234, 0.8)',
                borderColor: 'rgb(102, 126, 234)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Importance Score'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Top 10 Most Important Ingredients (Features)'
                }
            }
        }
    });
    {% endif %}

    // Confusion Matrix Display
    {% if metrics.confusion_matrix %}
    const confusionData = {{ metrics.confusion_matrix|safe }};
    const confusionHtml = `
        <table class="table table-bordered text-center">
            <thead>
                <tr>
                    <th rowspan="2">Actual</th>
                    <th colspan="${confusionData[0].length}">Predicted</th>
                </tr>
                <tr>
                    ${confusionData[0].map((_, index) => `<th>Class ${index}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
                ${confusionData.map((row, rowIndex) => `
                    <tr>
                        <th>Class ${rowIndex}</th>
                        ${row.map(value => `<td class="p-3 ${rowIndex === row.indexOf(value) && value === Math.max(...row) ? 'bg-success text-white' : ''}">${value}</td>`).join('')}
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    document.getElementById('confusionMatrix').innerHTML = confusionHtml;
    {% endif %}
</script>
{% endblock %}
```

### 6. URL Configuration

#### evaluator/urls.py
```python
from django.urls import path
from . import views

app_name = 'evaluator'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload-model/', views.upload_model, name='upload_model'),
    path('upload-dataset/', views.upload_dataset, name='upload_dataset'),
    path('evaluate/<uuid:model_id>/', views.run_evaluation, name='run_evaluation'),
    path('results/<uuid:evaluation_id>/', views.evaluation_results, name='evaluation_results'),
    path('api/status/<uuid:evaluation_id>/', views.api_evaluation_status, name='api_evaluation_status'),
]
```

#### config/urls.py
```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('evaluator.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

### 7. Settings Configuration

#### config/settings.py
```python
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
    'evaluator',
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

ROOT_URLCONF = 'config.urls'

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

# Media files (uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Static files
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024  # 100MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024  # 100MB

LOGIN_URL = '/admin/login/'
```

### 8. Requirements and Setup

#### requirements.txt
```
Django==4.2.7
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.24.3
joblib==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
openpyxl==3.1.2
pillow==10.1.0
```

#### Setup Commands
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run development server
python manage.py runserver
```

## Usage Instructions

### 1. Upload a Model
Navigate to the dashboard and click "Add New Recipe (Model)". Upload a trained scikit-learn model saved with joblib.

### 2. Upload a Dataset
Click "Add New Ingredients (Dataset)" and upload your test dataset in CSV or Excel format.

### 3. Run Evaluation
Select a model from your collection and click "Test Recipe". Choose your dataset and evaluation method, then start the evaluation.

### 4. View Results
Once complete, view comprehensive results including performance metrics, visualizations, and detailed reports.

## Key Features Implemented

1. **Comprehensive Model Support**: Handles both classification and regression models
2. **Multiple Validation Methods**: Train-test split and cross-validation
3. **Rich Metrics**: All major performance indicators with visualizations
4. **User-Friendly Interface**: Kitchen/chef analogy throughout
5. **File Management**: Secure upload and storage of models and datasets
6. **Real-time Status**: Track evaluation progress
7. **Detailed Reporting**: Professional-grade evaluation reports
8. **Responsive Design**: Works on all devices

This framework serves as your complete "quality control kitchen" for machine learning models, providing professional-grade evaluation capabilities with an intuitive, chef-inspired interface.</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .chef-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 0;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg chef-header">
        <div class="container">
            <a class="navbar-brand" href="{% url 'evaluator:dashboard' %}">
                <i class="fas fa-chart-line me-2"></i>
                Chef's Quality Control Kitchen
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link text-white" href="{% url 'evaluator:dashboard' %}">
                    <i class="fas fa-tachometer-alt me-1"></i>Dashboard
                </a>
            </div>
        </div>
    </nav>

    <main class="container my-4">
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}

        {% block content %}{% endblock %}
    </main>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
```

#### dashboard.html
```html
{% extends 'evaluator/base.html' %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h1><i class="fas fa-utensils me-2"></i>Master Chef's Quality Control Dashboard</h1>
        <p class="text-muted">Welcome to your model evaluation kitchen - where every recipe gets thoroughly tested!</p>
    </div>
</div>

<!-- Stats Cards -->
<div class="row mb-4">
    <div class="col-md-6">
        <div class="metric-card">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h3 class="text-primary">{{ total_models }}</h3>
                    <p class="mb-0">Active Recipes (Models)</p>
                </div>
                <i class="fas fa-robot fa-2x text-primary"></i>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="metric-card">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h3 class="text-success">{{ total_evaluations }}</h3>
                    <p class="mb-0">Quality Tests Completed</p>
                </div>
                <i class="fas fa-check-circle fa-2x text-success"></i>
            </div>
        </div>
    </div>
</div>

<!-- Action Buttons -->
<div class="row mb-4">
    <div class="col-12">
        <div class="metric-card">
            <h5><i class="fas fa-plus-circle me-2"></i>Quick Actions</h5>
            <div class="d-flex gap-2 flex-wrap">
                <a href="{% url 'evaluator:upload_model' %}" class="btn btn-primary">
                    <i class="fas fa-upload me-1"></i>Add New Recipe (Model)
                </a>
                <a href="{% url 'evaluator:upload_dataset' %}" class="btn btn-success">
                    <i class="fas fa-database me-1"></i>Add New Ingredients (Dataset)
                </a>
            </div>
        </div>
    </div>
</div>

<!-- Models Section -->
<div class="row">
    <div class="col-md-6">
        <div class="metric-card">
            <h5><i class="fas fa-robot me-2"></i>Your Recipe Collection</h5>
            {% if models %}
                <div class="list-group list-group-flush">
                    {% for model in models %}
                        <div class="list-group-item d-flex justify-content-between align-items-center">
                            <div>
                                <h6 class="mb-1">{{ model.name }}</h6>
                                <small class="text-muted">{{ model.get_model_type_display }}</small>
                            </div>
                            <div>
                                <a href="{% url 'evaluator:run_evaluation' model.id %}" class="btn btn-sm btn-outline-primary">
                                    <i class="fas fa-flask me-1"></i>Test Recipe
                                </a>
                            </div>
                        </div>
                    {% endfor %}
                </div>
            {% else %}
                <p class="text-muted">No recipes in your kitchen yet. Upload your first model to get started!</p>
            {% endif %}
        </div>
    </div>

    <div class="col-md-6">
        <div class="metric-card">
            <h5><i class="fas fa-history me-2"></i>Recent Quality Tests</h5>
            {% if recent_evaluations %}
                <div class="list-group list-group-flush">
                    {% for eval in recent_evaluations %}
                        <div class="list-group-item">
                            <div class="d-flex w-100 justify-content-between">
                                <h6 class="mb-1">{{ eval.model.name }}</h6>
                                <small class="text-muted">{{ eval.created_at|timesince }} ago</small>
                            </div>
                            <p class="mb-1">{{ eval.dataset.name }} - {{ eval.get_validation_method_display }}</p>
                            <div class="d-flex justify-content-between align-items-center">
                                <span class="badge bg-{% if eval.status == 'completed' %}success{% elif eval.status == 'failed' %}danger{% else %}warning{% endif %}">
                                    {{ eval.status|title }}
                                </span>
                                {% if eval.status == 'completed' %}
                                    <a href="{% url 'evaluator:evaluation_results' eval.id %}" class="btn btn-sm btn-outline-info">
                                        <i class="fas fa-eye me-1"></i>View Results
                                    </a>
                                {% endif %}
                            </div>
                        </div>
                    {% endfor %}
                </div>
            {% else %}
                <p class="text-muted">No quality tests completed yet. Start testing your models!</p>
            {% endif %}
        </div>
    </div>
</div>
{% endblock %}

## Assignment: The Master Chef's Challenge

### Your Mission
You are the head chef at "AI Bistro," and you need to evaluate three different "recipe models" (classification algorithms) to determine which one best predicts customer satisfaction with your new fusion menu.

**Dataset**: Use the Wine Quality dataset (available from UCI ML Repository) where you'll predict whether a wine is "high quality" (rating ≥ 7) or not.

**Your Task**:
1. **Data Preparation**: Split the wine dataset using your Django DataSplitter (70% train, 15% validation, 15% test)

2. **Model Training**: Train three different "chef algorithms":
   - Random Forest (the experienced head chef)
   - Logistic Regression (the efficient line cook)  
   - Support Vector Machine (the perfectionist pastry chef)

3. **Cross-Validation**: Use your CrossValidator to test each model's consistency with 5-fold cross-validation

4. **Performance Evaluation**: Calculate accuracy, precision, recall, and F1-score for each model on the validation set

5. **ROC Analysis**: Generate ROC curves and calculate AUC scores for all three models

6. **Final Recommendation**: Write a "Chef's Report" explaining which model you'd deploy in production and why, using the kitchen analogy throughout.

**Deliverables**:
- Django views that handle each evaluation step
- A comparison table showing all metrics
- ROC curve visualizations for all three models
- A 300-word "Chef's Report" with your final recommendation

**Bonus Challenge**: Create a Django template that displays all results in a beautiful dashboard, as if you're presenting to the restaurant owner!

---

## Final Project: Restaurant Quality Prediction System

Create a complete Django application that helps restaurant owners predict customer satisfaction scores based on various factors (service speed, food quality, ambiance, price, etc.). Your system should:

1. **Data Management**: Handle multiple restaurant datasets with proper train/validation/test splits
2. **Model Pipeline**: Implement automated cross-validation for model selection
3. **Comprehensive Evaluation**: Calculate and display all performance metrics with visual representations
4. **ROC Dashboard**: Interactive ROC curve comparisons between different models
5. **Prediction Interface**: Allow new restaurant data input with confidence intervals
6. **Reporting System**: Generate automated evaluation reports for restaurant managers

This project integrates all concepts learned, creating a production-ready system that any restaurant owner could use to improve their business through data-driven insights!

**Remember**: In the kitchen of machine learning, evaluation isn't just about getting the right answer - it's about understanding how consistently and reliably your model performs across different scenarios. Just like a master chef who tastes, adjusts, and perfects their dishes, a skilled data scientist continuously evaluates and improves their models!