# AI Mastery Course: Day 91 - MLOps & Production Systems

## Learning Objectives
By the end of this lesson, you will be able to:
- Implement automated ML pipeline workflows using Python and Django
- Set up model versioning and registry systems for production environments
- Design and execute A/B testing frameworks for machine learning models
- Build monitoring systems to detect model drift and performance degradation

---

## Introduction

Imagine that you're the head chef of a world-renowned restaurant that serves thousands of customers daily. Your signature dishes (ML models) need to be prepared consistently, updated with seasonal ingredients (new data), and monitored for quality. Just as a master chef doesn't just cook once and forget about it, a machine learning engineer must ensure their models continue performing excellently in production, adapting to changing conditions while maintaining the highest standards.

In today's digital restaurant, we'll learn how to set up the kitchen operations that keep your AI dishes fresh, consistent, and delightful for your customers.

---

## 1. ML Pipeline Automation

Think of your ML pipeline as the prep work that happens before service begins. Just as a chef prepares ingredients, marinates proteins, and sets up stations for smooth service, we need automated systems that prepare, process, and deploy our models seamlessly.

### Setting Up Django for ML Pipeline Management

```python
# models.py - Our recipe database
from django.db import models
from django.contrib.auth.models import User
import uuid

class MLPipeline(models.Model):
    """Like a recipe card that tracks our cooking process"""
    PIPELINE_STATUS = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=PIPELINE_STATUS, default='queued')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    config = models.JSONField(default=dict)  # Our recipe ingredients and instructions
    
    def __str__(self):
        return f"{self.name} - {self.status}"

class PipelineStep(models.Model):
    """Individual cooking steps in our recipe"""
    pipeline = models.ForeignKey(MLPipeline, on_delete=models.CASCADE, related_name='steps')
    step_name = models.CharField(max_length=100)
    order = models.IntegerField()
    status = models.CharField(max_length=20, choices=MLPipeline.PIPELINE_STATUS, default='queued')
    logs = models.TextField(blank=True)
    execution_time = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['order']
```

### Pipeline Automation Service

```python
# services/pipeline_service.py
import time
import logging
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
from django.conf import settings
import os

logger = logging.getLogger(__name__)

class MLPipelineService:
    """Our head chef orchestrating the entire cooking process"""
    
    def __init__(self, pipeline_instance):
        self.pipeline = pipeline_instance
        self.model = None
        
    def execute_pipeline(self):
        """Execute the full recipe from start to finish"""
        try:
            self.pipeline.status = 'running'
            self.pipeline.save()
            
            # Execute each step like following a recipe
            for step in self.pipeline.steps.all():
                self._execute_step(step)
                
            self.pipeline.status = 'completed'
            self.pipeline.save()
            logger.info(f"Pipeline {self.pipeline.name} completed successfully")
            
        except Exception as e:
            self.pipeline.status = 'failed'
            self.pipeline.save()
            logger.error(f"Pipeline {self.pipeline.name} failed: {str(e)}")
            raise
    
    def _execute_step(self, step):
        """Execute individual cooking step with timing"""
        start_time = time.time()
        step.status = 'running'
        step.save()
        
        try:
            # Map step names to methods (like having different cooking techniques)
            step_methods = {
                'data_preparation': self._prepare_ingredients,
                'model_training': self._cook_the_model,
                'model_evaluation': self._taste_test,
                'model_saving': self._preserve_the_dish
            }
            
            if step.step_name in step_methods:
                result = step_methods[step.step_name]()
                step.logs = f"Step completed successfully: {result}"
            else:
                raise ValueError(f"Unknown step: {step.step_name}")
                
            step.status = 'completed'
            step.execution_time = time.time() - start_time
            step.save()
            
        except Exception as e:
            step.status = 'failed'
            step.logs = f"Error: {str(e)}"
            step.execution_time = time.time() - start_time
            step.save()
            raise
    
    def _prepare_ingredients(self):
        """Data preparation - like chopping vegetables and marinating meat"""
        config = self.pipeline.config
        data_path = config.get('data_path', 'sample_data.csv')
        
        # In a real scenario, this would load from your data source
        # For demo, we'll create sample data
        import numpy as np
        np.random.seed(42)
        
        # Creating sample data (like preparing ingredients)
        n_samples = 1000
        X = np.random.randn(n_samples, 4)  # 4 features
        y = (X[:, 0] + X[:, 1] > X[:, 2] + X[:, 3]).astype(int)  # Binary target
        
        # Split like organizing ingredients into different prep containers
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Store in our mise en place (everything in its place)
        self.training_data = (X_train, y_train)
        self.test_data = (X_test, y_test)
        
        return f"Prepared {len(X_train)} training samples and {len(X_test)} test samples"
    
    def _cook_the_model(self):
        """Model training - the actual cooking process"""
        X_train, y_train = self.training_data
        config = self.pipeline.config.get('model_config', {})
        
        # Initialize our cooking method (Random Forest)
        self.model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 10),
            random_state=42
        )
        
        # Cook the model (training)
        self.model.fit(X_train, y_train)
        
        return f"Model trained with {len(X_train)} samples"
    
    def _taste_test(self):
        """Model evaluation - quality control like a chef tasting the dish"""
        X_test, y_test = self.test_data
        predictions = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        
        # Store evaluation results (like notes from the taste test)
        evaluation_results = {
            'accuracy': float(accuracy),
            'classification_report': classification_report(y_test, predictions, output_dict=True)
        }
        
        # Update pipeline config with results
        self.pipeline.config.setdefault('evaluation', {}).update(evaluation_results)
        self.pipeline.save()
        
        return f"Model accuracy: {accuracy:.4f}"
    
    def _preserve_the_dish(self):
        """Model saving - like properly storing a signature dish recipe"""
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models', str(self.pipeline.id))
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'model.joblib')
        joblib.dump(self.model, model_path)
        
        # Update pipeline with model location
        self.pipeline.config['model_path'] = model_path
        self.pipeline.save()
        
        return f"Model saved to {model_path}"
```

### Django Views for Pipeline Management

```python
# views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import ListView
import json
from .models import MLPipeline, PipelineStep
from .services.pipeline_service import MLPipelineService
import threading

class PipelineListView(ListView):
    """Our restaurant's order management system"""
    model = MLPipeline
    template_name = 'pipelines/list.html'
    context_object_name = 'pipelines'
    paginate_by = 20

@login_required
def create_pipeline(request):
    """Create a new cooking order"""
    if request.method == 'POST':
        data = json.loads(request.body)
        
        # Create the main pipeline (order ticket)
        pipeline = MLPipeline.objects.create(
            name=data['name'],
            created_by=request.user,
            config=data.get('config', {})
        )
        
        # Create pipeline steps (cooking steps)
        steps_config = [
            {'name': 'data_preparation', 'order': 1},
            {'name': 'model_training', 'order': 2},
            {'name': 'model_evaluation', 'order': 3},
            {'name': 'model_saving', 'order': 4},
        ]
        
        for step_config in steps_config:
            PipelineStep.objects.create(
                pipeline=pipeline,
                step_name=step_config['name'],
                order=step_config['order']
            )
        
        return JsonResponse({'pipeline_id': str(pipeline.id), 'status': 'created'})
    
    return render(request, 'pipelines/create.html')

@csrf_exempt
def execute_pipeline(request, pipeline_id):
    """Start cooking - execute the pipeline"""
    if request.method == 'POST':
        pipeline = get_object_or_404(MLPipeline, id=pipeline_id)
        
        # Start cooking in background (like sending order to kitchen)
        def run_pipeline():
            service = MLPipelineService(pipeline)
            service.execute_pipeline()
        
        thread = threading.Thread(target=run_pipeline)
        thread.start()
        
        return JsonResponse({'status': 'Pipeline execution started'})
    
    return JsonResponse({'error': 'Invalid method'}, status=405)
```

---

## 2. Model Versioning and Registry

Just as a master chef maintains different versions of their signature recipes - perhaps a summer version with fresh herbs, a winter version with heartier ingredients, and experimental versions they're perfecting - we need to track and manage different versions of our models.

### Model Registry System

```python
# models.py (additional models)
class ModelRegistry(models.Model):
    """Our recipe book with different versions of each signature dish"""
    MODEL_STAGES = [
        ('development', 'Development'),
        ('staging', 'Staging'),
        ('production', 'Production'),
        ('archived', 'Archived'),
    ]
    
    name = models.CharField(max_length=200)  # Dish name
    version = models.CharField(max_length=50)  # Version like "v1.2.3" or "summer_2024"
    stage = models.CharField(max_length=20, choices=MODEL_STAGES, default='development')
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Model metadata (ingredients list and nutritional info)
    model_path = models.CharField(max_length=500)
    metrics = models.JSONField(default=dict)  # Performance metrics
    metadata = models.JSONField(default=dict)  # Additional info
    
    # Relationship to pipeline that created this version
    source_pipeline = models.ForeignKey(MLPipeline, on_delete=models.SET_NULL, null=True)
    
    class Meta:
        unique_together = ['name', 'version']
    
    def __str__(self):
        return f"{self.name} v{self.version} ({self.stage})"

class ModelComparison(models.Model):
    """Compare different versions like a chef comparing taste tests"""
    name = models.CharField(max_length=200)
    baseline_model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE, related_name='baseline_comparisons')
    challenger_model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE, related_name='challenger_comparisons')
    comparison_results = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
```

### Model Registry Service

```python
# services/model_registry_service.py
import os
import joblib
import json
from typing import Dict, Any, Optional
from django.conf import settings
from django.core.files.storage import default_storage
import shutil

class ModelRegistryService:
    """Our sommelier managing the wine cellar of model versions"""
    
    @staticmethod
    def register_model(pipeline_instance, version: str, stage: str = 'development') -> ModelRegistry:
        """Register a new model version like cataloging a new vintage"""
        
        # Extract model info from completed pipeline
        model_path = pipeline_instance.config.get('model_path')
        metrics = pipeline_instance.config.get('evaluation', {})
        
        if not model_path or not os.path.exists(model_path):
            raise ValueError("Model file not found in pipeline results")
        
        # Create registry entry
        model_entry = ModelRegistry.objects.create(
            name=pipeline_instance.name,
            version=version,
            stage=stage,
            created_by=pipeline_instance.created_by,
            model_path=model_path,
            metrics=metrics,
            source_pipeline=pipeline_instance,
            metadata={
                'training_date': pipeline_instance.updated_at.isoformat(),
                'pipeline_config': pipeline_instance.config
            }
        )
        
        return model_entry
    
    @staticmethod
    def promote_model(model_id: str, new_stage: str) -> ModelRegistry:
        """Promote a model to the next stage (like promoting a sous chef)"""
        model = ModelRegistry.objects.get(id=model_id)
        
        # If promoting to production, demote current production model
        if new_stage == 'production':
            ModelRegistry.objects.filter(
                name=model.name, 
                stage='production'
            ).update(stage='archived')
        
        model.stage = new_stage
        model.save()
        
        return model
    
    @staticmethod
    def load_model(model_name: str, version: Optional[str] = None, stage: Optional[str] = None):
        """Load a model like retrieving a recipe from the cookbook"""
        query = ModelRegistry.objects.filter(name=model_name)
        
        if version:
            query = query.filter(version=version)
        elif stage:
            query = query.filter(stage=stage)
        else:
            # Default to production, fall back to latest
            prod_model = query.filter(stage='production').first()
            if prod_model:
                query = ModelRegistry.objects.filter(id=prod_model.id)
            else:
                query = query.order_by('-created_at')
        
        model_entry = query.first()
        if not model_entry:
            raise ValueError(f"No model found for {model_name}")
        
        # Load the actual model
        model = joblib.load(model_entry.model_path)
        
        return model, model_entry
    
    @staticmethod
    def compare_models(baseline_id: str, challenger_id: str) -> Dict[str, Any]:
        """Compare two models like a blind taste test"""
        baseline = ModelRegistry.objects.get(id=baseline_id)
        challenger = ModelRegistry.objects.get(id=challenger_id)
        
        baseline_metrics = baseline.metrics
        challenger_metrics = challenger.metrics
        
        # Simple comparison logic
        comparison = {
            'baseline': {
                'model': f"{baseline.name} v{baseline.version}",
                'accuracy': baseline_metrics.get('accuracy', 0)
            },
            'challenger': {
                'model': f"{challenger.name} v{challenger.version}",
                'accuracy': challenger_metrics.get('accuracy', 0)
            },
            'improvement': challenger_metrics.get('accuracy', 0) - baseline_metrics.get('accuracy', 0),
            'recommendation': 'promote' if challenger_metrics.get('accuracy', 0) > baseline_metrics.get('accuracy', 0) else 'keep_baseline'
        }
        
        # Save comparison results
        ModelComparison.objects.create(
            name=f"{baseline.name}_comparison_{baseline.version}_vs_{challenger.version}",
            baseline_model=baseline,
            challenger_model=challenger,
            comparison_results=comparison,
            created_by=baseline.created_by  # Simplified for demo
        )
        
        return comparison
```

---

## 3. A/B Testing for ML Models

Imagine you're introducing a new signature sauce alongside your classic recipe. You wouldn't replace the beloved original immediately - instead, you'd serve both versions to different tables, carefully noting which one gets better reactions, more compliments, and higher satisfaction scores.

### A/B Testing Framework

```python
# models.py (additional)
class ABTest(models.Model):
    """Our taste test experiment setup"""
    TEST_STATUS = [
        ('draft', 'Draft'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('paused', 'Paused'),
    ]
    
    name = models.CharField(max_length=200)
    description = models.TextField()
    
    # The two recipes we're testing
    control_model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE, related_name='control_tests')
    treatment_model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE, related_name='treatment_tests')
    
    # Test configuration
    traffic_split = models.FloatField(default=0.5)  # 50/50 split
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    status = models.CharField(max_length=20, choices=TEST_STATUS, default='draft')
    
    # Success metrics we're tracking
    success_metrics = models.JSONField(default=list)
    
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

class ABTestResult(models.Model):
    """Individual customer feedback (prediction result)"""
    test = models.ForeignKey(ABTest, on_delete=models.CASCADE, related_name='results')
    model_used = models.CharField(max_length=20, choices=[('control', 'Control'), ('treatment', 'Treatment')])
    
    # Request details
    input_data = models.JSONField()  # What we served
    prediction = models.JSONField()  # What the model predicted
    actual_outcome = models.JSONField(null=True)  # What actually happened (if available)
    
    # Performance metrics
    response_time = models.FloatField()  # How fast was service
    confidence_score = models.FloatField(null=True)  # Model confidence
    
    timestamp = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(max_length=100, null=True)  # To group related requests
```

### A/B Testing Service

```python
# services/ab_testing_service.py
import random
import time
import hashlib
from typing import Dict, Any, Tuple
from datetime import datetime
from django.utils import timezone
from .model_registry_service import ModelRegistryService

class ABTestingService:
    """Our maître d' deciding which menu version to serve each customer"""
    
    def __init__(self, test_id: str):
        self.test = ABTest.objects.get(id=test_id)
        self.control_model = None
        self.treatment_model = None
        self._load_models()
    
    def _load_models(self):
        """Load both versions of our recipes"""
        self.control_model, _ = ModelRegistryService.load_model(
            self.test.control_model.name, 
            version=self.test.control_model.version
        )
        self.treatment_model, _ = ModelRegistryService.load_model(
            self.test.treatment_model.name,
            version=self.test.treatment_model.version
        )
    
    def get_model_assignment(self, user_id: str) -> str:
        """Decide which version to serve this customer (consistent assignment)"""
        if self.test.status != 'running':
            return 'control'  # Default to control if test isn't running
        
        # Consistent assignment based on user_id (same customer gets same version)
        hash_value = int(hashlib.md5(f"{self.test.id}_{user_id}".encode()).hexdigest(), 16)
        assignment_value = (hash_value % 100) / 100.0
        
        return 'treatment' if assignment_value < self.test.traffic_split else 'control'
    
    def make_prediction(self, user_id: str, input_data: Dict[str, Any]) -> Tuple[Dict, str]:
        """Serve the appropriate dish version and record the interaction"""
        assignment = self.get_model_assignment(user_id)
        start_time = time.time()
        
        # Choose the right recipe
        if assignment == 'treatment':
            model = self.treatment_model
            model_name = f"{self.test.treatment_model.name} v{self.test.treatment_model.version}"
        else:
            model = self.control_model
            model_name = f"{self.test.control_model.name} v{self.test.control_model.version}"
        
        # Make prediction (serve the dish)
        try:
            # Convert input_data to format expected by model
            import numpy as np
            features = np.array([list(input_data.values())])  # Simplified for demo
            
            prediction = model.predict(features)[0]
            confidence = None
            
            # Try to get prediction probability if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                confidence = max(probabilities)
        
        except Exception as e:
            # Fallback to control if treatment fails
            if assignment == 'treatment':
                features = np.array([list(input_data.values())])
                prediction = self.control_model.predict(features)[0]
                assignment = 'control'  # Log as control since we fell back
                confidence = None
            else:
                raise e
        
        response_time = time.time() - start_time
        
        # Record the interaction
        result = ABTestResult.objects.create(
            test=self.test,
            model_used=assignment,
            input_data=input_data,
            prediction={'value': int(prediction), 'model': model_name},
            response_time=response_time,
            confidence_score=confidence,
            session_id=user_id  # Simplified for demo
        )
        
        return {
            'prediction': int(prediction),
            'confidence': confidence,
            'model_version': assignment,
            'response_time': response_time
        }, assignment
    
    def get_test_results(self) -> Dict[str, Any]:
        """Analyze how our taste test is going"""
        results = self.test.results.all()
        
        control_results = results.filter(model_used='control')
        treatment_results = results.filter(model_used='treatment')
        
        analysis = {
            'test_name': self.test.name,
            'status': self.test.status,
            'total_requests': results.count(),
            'control': {
                'count': control_results.count(),
                'avg_response_time': control_results.aggregate(avg_time=models.Avg('response_time'))['avg_time'] or 0,
                'avg_confidence': control_results.filter(confidence_score__isnull=False).aggregate(avg_conf=models.Avg('confidence_score'))['avg_conf']
            },
            'treatment': {
                'count': treatment_results.count(),
                'avg_response_time': treatment_results.aggregate(avg_time=models.Avg('response_time'))['avg_time'] or 0,
                'avg_confidence': treatment_results.filter(confidence_score__isnull=False).aggregate(avg_conf=models.Avg('confidence_score'))['avg_conf']
            }
        }
        
        # Calculate traffic split
        if analysis['total_requests'] > 0:
            analysis['actual_split'] = {
                'control': analysis['control']['count'] / analysis['total_requests'],
                'treatment': analysis['treatment']['count'] / analysis['total_requests']
            }
        
        return analysis
```

---

## 4. Monitoring and Drift Detection

Just as an experienced chef constantly tastes dishes throughout service, checks ingredient quality, and notices when something seems different about the usual preparation, we need systems that continuously monitor our models for changes in performance or data patterns.

### Monitoring System

```python
# models.py (additional)
class ModelMonitor(models.Model):
    """Our quality control inspector watching over the service"""
    model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE)
    
    # What we're watching for
    drift_threshold = models.FloatField(default=0.1)  # Acceptable drift level
    performance_threshold = models.FloatField(default=0.05)  # Min performance drop before alert
    
    # Monitoring configuration
    monitoring_frequency = models.CharField(max_length=50, default='hourly')
    alert_channels = models.JSONField(default=list)  # Email, Slack, etc.
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

class DriftDetection(models.Model):
    """Records of when we detected something unusual in our ingredients"""
    DRIFT_TYPES = [
        ('data_drift', 'Data Drift'),
        ('concept_drift', 'Concept Drift'),
        ('performance_drift', 'Performance Drift'),
    ]
    
    monitor = models.ForeignKey(ModelMonitor, on_delete=models.CASCADE, related_name='detections')
    drift_type = models.CharField(max_length=20, choices=DRIFT_TYPES)
    drift_score = models.FloatField()  # How significant the drift is
    detected_at = models.DateTimeField(auto_now_add=True)
    
    # Details about what drifted
    affected_features = models.JSONField(default=list)
    drift_details = models.JSONField(default=dict)
    
    # Actions taken
    alert_sent = models.BooleanField(default=False)
    action_taken = models.TextField(blank=True)

class ModelPerformanceLog(models.Model):
    """Daily performance reports like end-of-service kitchen reports"""
    model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE)
    date = models.DateField()
    
    # Performance metrics
    total_predictions = models.IntegerField()
    avg_response_time = models.FloatField()
    error_rate = models.FloatField()
    
    # If we have ground truth data
    accuracy = models.FloatField(null=True)
    precision = models.FloatField(null=True)
    recall = models.FloatField(null=True)
    
    # Resource usage
    cpu_usage = models.FloatField(null=True)
    memory_usage = models.FloatField(null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['model', 'date']
```

### Drift Detection Service

```python
# services/drift_detection_service.py
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)

class DriftDetectionService:
    """Our head chef's quality control system"""
    
    def __init__(self, model_monitor: ModelMonitor):
        self.monitor = model_monitor
        self.model = model_monitor.model
        
    def check_for_drift(self, recent_data: np.ndarray, baseline_data: np.ndarray) -> Dict[str, Any]:
        """Check if our ingredients have changed significantly"""
        drift_results = {
            'data_drift_detected': False,
            'drift_score': 0.0,
            'affected_features': [],
            'drift_details': {}
        }
        
        # Statistical test for each feature (like testing each ingredient)
        n_features = recent_data.shape[1] if len(recent_data.shape) > 1 else 1
        p_values = []
        
        for feature_idx in range(n_features):
            if len(recent_data.shape) > 1:
                recent_feature = recent_data[:, feature_idx]
                baseline_feature = baseline_data[:, feature_idx]
            else:
                recent_feature = recent_data
                baseline_feature = baseline_data
            
            # Kolmogorov-Smirnov test for distribution drift
            statistic, p_value = stats.ks_2samp(baseline_feature, recent_feature)
            p_values.append(p_value)
            
            # If p-value is low, distributions are significantly different
            if p_value < 0.05:  # Traditional significance level
                drift_results['affected_features'].append(feature_idx)
                drift_results['drift_details'][f'feature_{feature_idx}'] = {
                    'p_value': p_value,
                    'statistic': statistic,
                    'baseline_mean': float(np.mean(baseline_feature)),
                    'recent_mean': float(np.mean(recent_feature)),
                    'baseline_std': float(np.std(baseline_feature)),
                    'recent_std': float(np.std(recent_feature))
                }
        
        # Overall drift score (average of negative log p-values)
        drift_results['drift_score'] = float(np.mean([-np.log10(p + 1e-10) for p in p_values]))
        
        # Determine if drift is significant
        drift_results['data_drift_detected'] = (
            drift_results['drift_score'] > -np.log10(self.monitor.drift_threshold) or
            len(drift_results['affected_features']) > n_features * 0.3  # More than 30% of features drifted
        )
        
        if drift_results['data_drift_detected']:
            self._record_drift_detection('data_drift', drift_results)
            logger.warning(f"Data drift detected for model {self.model.name}")
        
        return drift_results
    
    def check_performance_drift(self, recent_performance: Dict[str, float], baseline_performance: Dict[str, float]) -> Dict[str, Any]:
        """Check if our dish quality has degraded"""
        performance_drift = {
            'performance_drift_detected': False,
            'performance_changes': {},
            'overall_degradation': 0.0
        }
        
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        degradations = []
        
        for metric in key_metrics:
            if metric in recent_performance and metric in baseline_performance:
                recent_val = recent_performance[metric]
                baseline_val = baseline_performance[metric]
                change = recent_val - baseline_val
                degradations.append(change)
                
                performance_drift['performance_changes'][metric] = {
                    'baseline': baseline_val,
                    'recent': recent_val,
                    'change': change,
                    'relative_change': change / baseline_val if baseline_val != 0 else 0
                }
        
        # Overall degradation (negative means worse performance)
        performance_drift['overall_degradation'] = float(np.mean(degradations)) if degradations else 0.0
        
        # Check if degradation exceeds threshold
        performance_drift['performance_drift_detected'] = (
            performance_drift['overall_degradation'] < -self.monitor.performance_threshold
        )
        
        if performance_drift['performance_drift_detected']:
            self._record_drift_detection('performance_drift', performance_drift)
            logger.warning(f"Performance drift detected for model {self.model.name}")
        
        return performance_drift
    
    def _record_drift_detection(self, drift_type: str, drift_data: Dict[str, Any]):
        """Record drift detection like writing up an incident report"""
        DriftDetection.objects.create(
            monitor=self.monitor,
            drift_type=drift_type,
            drift_score=drift_data.get('drift_score', drift_data.get('overall_degradation', 0)),
            affected_features=drift_data.get('affected_features', []),
            drift_details=drift_data
        )
        
        # Send alerts if configured
        self._send_alerts(drift_type, drift_data)
    
    def _send_alerts(self, drift_type: str, drift_data: Dict[str, Any]):
        """Alert the head chef when something's wrong"""
        # In a real implementation, this would send emails, Slack messages, etc.
        alert_message = f"""
        🚨 DRIFT ALERT: {drift_type.replace('_', ' ').title()}
        Model: {self.model.name} v{self.model.version}
        Drift Score: {drift_data.get('drift_score', drift_data.get('overall_degradation', 0)):.4f}
        Time: {timezone.now()}
        
        Details: {drift_data}
        """
        
        logger.warning(alert_message)
        # TODO: Implement actual alerting (email, Slack, PagerDuty, etc.)

class ModelMonitoringService:
    """Our restaurant manager overseeing daily operations"""
    
    @staticmethod
    def daily_performance_check():
        """Run daily checks on all monitored models"""
        active_monitors = ModelMonitor.objects.filter(is_active=True)
        
        for monitor in active_monitors:
            try:
                ModelMonitoringService._check_model_performance(monitor)
            except Exception as e:
                logger.error(f"Error monitoring model {monitor.model.name}: {str(e)}")
    
    @staticmethod
    def _check_model_performance(monitor: ModelMonitor):
        """Check individual model performance like reviewing daily sales"""
        model = monitor.model
        today = timezone.now().date()
        
        # Get recent predictions (in real scenario, from prediction logs)
        # For demo, we'll simulate some performance data
        total_predictions = np.random.randint(100, 1000)
        avg_response_time = np.random.uniform(0.1, 0.5)
        error_rate = np.random.uniform(0.01, 0.1)
        
        # Simulate performance metrics
        accuracy = max(0.7, np.random.normal(0.85, 0.05))
        precision = max(0.6, np.random.normal(0.80, 0.05))
        recall = max(0.6, np.random.normal(0.82, 0.05))
        
        # Record daily performance
        performance_log, created = ModelPerformanceLog.objects.get_or_create(
            model=model,
            date=today,
            defaults={
                'total_predictions': total_predictions,
                'avg_response_time': avg_response_time,
                'error_rate': error_rate,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
        )
        
        # Check for performance drift by comparing to baseline
        baseline_period = today - timedelta(days=30)
        baseline_logs = ModelPerformanceLog.objects.filter(
            model=model,
            date__gte=baseline_period,
            date__lt=today - timedelta(days=7)  # Exclude recent week
        )
        
        if baseline_logs.exists():
            baseline_performance = {
                'accuracy': baseline_logs.aggregate(avg_acc=models.Avg('accuracy'))['avg_acc'],
                'precision': baseline_logs.aggregate(avg_prec=models.Avg('precision'))['avg_prec'],
                'recall': baseline_logs.aggregate(avg_recall=models.Avg('recall'))['avg_recall']
            }
            
            recent_performance = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
            
            # Check for drift
            drift_detector = DriftDetectionService(monitor)
            drift_detector.check_performance_drift(recent_performance, baseline_performance)
```

### Django Management Command for Monitoring

```python
# management/commands/monitor_models.py
from django.core.management.base import BaseCommand
from myapp.services.drift_detection_service import ModelMonitoringService

class Command(BaseCommand):
    """Daily monitoring routine - like opening checklist for the restaurant"""
    help = 'Run daily model monitoring checks'
    
    def handle(self, *args, **options):
        self.stdout.write('Starting daily model monitoring...')
        
        try:
            ModelMonitoringService.daily_performance_check()
            self.stdout.write(
                self.style.SUCCESS('Daily monitoring completed successfully')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error during monitoring: {str(e)}')
            )
```

---

## Code Syntax Explanations

Let me break down the key Python and Django syntax patterns used in our MLOps system:

### Django Model Patterns
```python
# UUIDField for unique identifiers
id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
```
**Explanation**: Creates a unique identifier that's harder to guess than sequential IDs, perfect for model versioning where security matters.

### JSONField Usage
```python
config = models.JSONField(default=dict)
metrics = models.JSONField(default=dict)
```
**Explanation**: Stores flexible data structures directly in the database. Like having a flexible recipe card that can accommodate different ingredients and instructions.

### Foreign Key Relationships
```python
source_pipeline = models.ForeignKey(MLPipeline, on_delete=models.SET_NULL, null=True)
```
**Explanation**: `SET_NULL` means if the pipeline is deleted, we keep the model registry entry but clear the relationship - like keeping the dish on the menu even if we lose the original recipe notes.

### Threading for Background Tasks
```python
def run_pipeline():
    service = MLPipelineService(pipeline)
    service.execute_pipeline()

thread = threading.Thread(target=run_pipeline)
thread.start()
```
**Explanation**: Runs the pipeline in the background so the web request returns immediately. Like sending an order to the kitchen without making the customer wait at the counter.

### List Comprehensions with Statistical Operations
```python
drift_score = float(np.mean([-np.log10(p + 1e-10) for p in p_values]))
```
**Explanation**: Transforms p-values into a drift score. The `1e-10` prevents taking log of zero, and the negative log transforms small p-values (significant drift) into large scores.

### Dictionary.get() with Defaults
```python
n_estimators=config.get('n_estimators', 100)
```
**Explanation**: Safe way to get configuration values with fallbacks, like having standard portions when a recipe doesn't specify exact amounts.

---

# Complete MLOps Pipeline Project

## Project Overview
You'll create a comprehensive MLOps pipeline that manages the entire machine learning lifecycle - from data ingestion to model deployment and monitoring. Think of this as setting up a master chef's complete kitchen operation where ingredients flow in, get processed through standardized recipes, quality is continuously monitored, and the final dishes are served consistently to customers.

## Project Structure
```
mlops_pipeline/
├── data/
│   ├── raw/
│   ├── processed/
│   └── validation/
├── models/
│   ├── training/
│   ├── registry/
│   └── artifacts/
├── pipeline/
│   ├── data_processing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── deployment.py
├── monitoring/
│   ├── drift_detection.py
│   ├── performance_metrics.py
│   └── alerts.py
├── api/
│   ├── app.py
│   └── models.py
├── config/
│   └── pipeline_config.yaml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── main.py
```

## 1. Data Pipeline Component

### data_processing.py
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import yaml
from datetime import datetime
import os
import joblib

class DataProcessor:
    """
    Like a prep chef who ensures all ingredients are properly cleaned,
    cut, and organized before the main cooking begins.
    """
    
    def __init__(self, config_path="config/pipeline_config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def ingest_data(self, data_path):
        """Raw ingredient delivery - bringing in fresh data"""
        try:
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                data = pd.read_json(data_path)
            else:
                raise ValueError("Unsupported file format")
            
            self.logger.info(f"Data ingested: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        
        except Exception as e:
            self.logger.error(f"Data ingestion failed: {str(e)}")
            raise
    
    def validate_data_quality(self, data):
        """Quality inspection - like checking ingredients for freshness"""
        quality_report = {
            'total_rows': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Check for data quality issues
        if quality_report['duplicate_rows'] > 0:
            self.logger.warning(f"Found {quality_report['duplicate_rows']} duplicate rows")
        
        missing_threshold = self.config['data_quality']['missing_threshold']
        high_missing_cols = [col for col, missing in quality_report['missing_values'].items() 
                           if missing > len(data) * missing_threshold]
        
        if high_missing_cols:
            self.logger.warning(f"High missing values in columns: {high_missing_cols}")
        
        return quality_report, data
    
    def preprocess_features(self, data, target_column):
        """Recipe preparation - transforming raw ingredients into cooking-ready form"""
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = self.label_encoder.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=self.config['data_split']['test_size'],
            random_state=self.config['data_split']['random_state'],
            stratify=y if len(y.unique()) < 20 else None
        )
        
        # Save preprocessing artifacts
        os.makedirs('models/artifacts', exist_ok=True)
        joblib.dump(self.scaler, 'models/artifacts/scaler.pkl')
        joblib.dump(self.label_encoder, 'models/artifacts/label_encoder.pkl')
        
        self.logger.info(f"Preprocessing complete. Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Store prepared ingredients in organized containers"""
        os.makedirs('data/processed', exist_ok=True)
        
        X_train.to_csv('data/processed/X_train.csv', index=False)
        X_test.to_csv('data/processed/X_test.csv', index=False)
        pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
        pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)
        
        self.logger.info("Processed data saved successfully")
```

## 2. Model Training & Registry Component

### model_training.py
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib

class ModelTrainer:
    """
    Master chef who experiments with different recipes,
    keeps track of what works, and maintains a cookbook of successful dishes.
    """
    
    def __init__(self, config):
        self.config = config
        self.client = MlflowClient()
        
        # Set up MLflow
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'sqlite:///mlflow.db'))
        mlflow.set_experiment(config.get('experiment_name', 'mlops_pipeline'))
    
    def train_multiple_models(self, X_train, y_train, X_test, y_test):
        """Test kitchen - trying different recipes to find the best one"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        model_results = {}
        
        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
                
                # Log parameters and metrics
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(model, f"{model_name}_model")
                
                # Store results
                model_results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'run_id': mlflow.active_run().info.run_id
                }
                
                print(f"{model_name} trained - F1 Score: {metrics['test_f1']:.4f}")
        
        return model_results
    
    def calculate_metrics(self, y_train, y_pred_train, y_test, y_pred_test):
        """Quality control - measuring how well each recipe performs"""
        return {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_precision': precision_score(y_train, y_pred_train, average='weighted'),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
            'train_recall': recall_score(y_train, y_pred_train, average='weighted'),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted'),
            'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted')
        }
    
    def register_best_model(self, model_results):
        """Recipe book - officially recording the best recipe for future use"""
        # Find best model based on test F1 score
        best_model_name = max(model_results.keys(), 
                            key=lambda k: model_results[k]['metrics']['test_f1'])
        best_model_info = model_results[best_model_name]
        
        # Register model in MLflow Model Registry
        model_uri = f"runs:/{best_model_info['run_id']}/{best_model_name}_model"
        
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name="production_model"
        )
        
        # Transition to staging
        self.client.transition_model_version_stage(
            name="production_model",
            version=model_details.version,
            stage="Staging"
        )
        
        # Save model locally
        model_path = f"models/registry/model_v{model_details.version}.pkl"
        joblib.dump(best_model_info['model'], model_path)
        
        # Save model metadata
        metadata = {
            'model_name': best_model_name,
            'version': model_details.version,
            'metrics': best_model_info['metrics'],
            'timestamp': datetime.now().isoformat(),
            'run_id': best_model_info['run_id'],
            'model_path': model_path
        }
        
        with open(f"models/registry/model_v{model_details.version}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Best model ({best_model_name}) registered as version {model_details.version}")
        
        return model_details, metadata
```

## 3. Deployment & Serving Component

### api/app.py
```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import os
from monitoring.drift_detection import DriftDetector
from monitoring.performance_metrics import PerformanceTracker

app = Flask(__name__)

class ModelServer:
    """
    Restaurant service - taking orders and serving dishes consistently
    """
    
    def __init__(self):
        self.load_model()
        self.load_preprocessors()
        self.drift_detector = DriftDetector()
        self.performance_tracker = PerformanceTracker()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Get the signature dish ready for service"""
        try:
            # Load latest model version
            model_files = [f for f in os.listdir('models/registry') if f.endswith('.pkl')]
            latest_model = max(model_files, key=lambda x: int(x.split('_v')[1].split('.')[0]))
            
            self.model = joblib.load(f'models/registry/{latest_model}')
            
            # Load metadata
            metadata_file = latest_model.replace('.pkl', '_metadata.json')
            with open(f'models/registry/{metadata_file}', 'r') as f:
                self.model_metadata = json.load(f)
            
            self.logger.info(f"Model loaded: {latest_model}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise
    
    def load_preprocessors(self):
        """Prepare the prep station with standard tools"""
        self.scaler = joblib.load('models/artifacts/scaler.pkl')
        self.label_encoder = joblib.load('models/artifacts/label_encoder.pkl')
    
    def preprocess_input(self, data):
        """Standard prep work - ensuring consistency"""
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Apply same preprocessing as training
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            data[col] = self.label_encoder.transform(data[col].astype(str))
        
        # Fill missing values
        data = data.fillna(data.median())
        
        # Scale features
        data_scaled = self.scaler.transform(data)
        
        return data_scaled
    
    def predict(self, input_data):
        """Main service - delivering the final product"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)
            prediction_proba = self.model.predict_proba(processed_data)
            
            # Log prediction for monitoring
            self.log_prediction(input_data, prediction, prediction_proba)
            
            return {
                'prediction': prediction.tolist(),
                'probability': prediction_proba.tolist(),
                'model_version': self.model_metadata['version'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return {'error': str(e)}
    
    def log_prediction(self, input_data, prediction, prediction_proba):
        """Kitchen log - keeping track of what was served"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_data': input_data if isinstance(input_data, dict) else input_data.to_dict('records'),
            'prediction': prediction.tolist(),
            'max_probability': np.max(prediction_proba),
            'model_version': self.model_metadata['version']
        }
        
        # Save to log file
        os.makedirs('logs', exist_ok=True)
        log_file = f"logs/predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

# Initialize model server
model_server = ModelServer()

@app.route('/predict', methods=['POST'])
def predict():
    """Order taking - receiving customer requests"""
    try:
        data = request.get_json()
        result = model_server.predict(data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Kitchen status check"""
    return jsonify({
        'status': 'healthy',
        'model_version': model_server.model_metadata['version'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Menu information"""
    return jsonify(model_server.model_metadata)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

## 4. Monitoring & Drift Detection

### monitoring/drift_detection.py
```python
import pandas as pd
import numpy as np
from scipy import stats
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DriftDetector:
    """
    Quality inspector - constantly checking if ingredients and 
    cooking conditions are still meeting standards
    """
    
    def __init__(self, reference_data_path='data/processed/X_train.csv'):
        self.reference_data = pd.read_csv(reference_data_path)
        self.drift_threshold = 0.05  # p-value threshold
        self.feature_drift_scores = {}
        
    def detect_data_drift(self, new_data):
        """Ingredient inspection - checking if new ingredients match standards"""
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_drift': False,
            'drifted_features': [],
            'feature_scores': {}
        }
        
        # Ensure same columns
        common_columns = set(self.reference_data.columns) & set(new_data.columns)
        
        for column in common_columns:
            # Kolmogorov-Smirnov test for distribution comparison
            ks_statistic, p_value = stats.ks_2samp(
                self.reference_data[column].dropna(),
                new_data[column].dropna()
            )
            
            drift_results['feature_scores'][column] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'drift_detected': p_value < self.drift_threshold
            }
            
            if p_value < self.drift_threshold:
                drift_results['drifted_features'].append(column)
                drift_results['overall_drift'] = True
        
        # Calculate overall drift score
        drift_results['drift_score'] = np.mean([
            score['ks_statistic'] for score in drift_results['feature_scores'].values()
        ])
        
        return drift_results
    
    def detect_prediction_drift(self, predictions_log_path):
        """Recipe consistency check - ensuring output quality remains stable"""
        # Load recent predictions
        recent_predictions = self.load_recent_predictions(predictions_log_path)
        
        if len(recent_predictions) < 100:  # Need minimum samples
            return {'warning': 'Insufficient data for drift detection'}
        
        # Convert to DataFrame
        df = pd.DataFrame(recent_predictions)
        
        # Check prediction distribution over time
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_pred_dist = df.groupby('hour')['prediction'].apply(lambda x: x.value_counts(normalize=True))
        
        # Simple drift detection based on prediction variance
        prediction_variance = df['max_probability'].var()
        
        drift_detected = prediction_variance > 0.1  # Threshold for variance
        
        return {
            'timestamp': datetime.now().isoformat(),
            'prediction_drift_detected': drift_detected,
            'prediction_variance': float(prediction_variance),
            'total_predictions': len(recent_predictions)
        }
    
    def load_recent_predictions(self, log_path, hours=24):
        """Load recent service logs"""
        try:
            with open(log_path, 'r') as f:
                predictions = [json.loads(line) for line in f]
            
            # Filter recent predictions
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_predictions = [
                p for p in predictions 
                if datetime.fromisoformat(p['timestamp']) > cutoff_time
            ]
            
            return recent_predictions
        
        except FileNotFoundError:
            return []
    
    def generate_drift_report(self, drift_results):
        """Daily quality report"""
        report = f"""
=== DRIFT DETECTION REPORT ===
Timestamp: {drift_results['timestamp']}
Overall Drift Detected: {drift_results['overall_drift']}
Drift Score: {drift_results.get('drift_score', 0):.4f}

Drifted Features: {', '.join(drift_results['drifted_features']) if drift_results['drifted_features'] else 'None'}

Feature Details:
"""
        for feature, scores in drift_results['feature_scores'].items():
            status = "DRIFT" if scores['drift_detected'] else "OK"
            report += f"  {feature}: {status} (p-value: {scores['p_value']:.4f})\n"
        
        return report
```

## 5. Complete Pipeline Orchestrator

### main.py
```python
import yaml
import logging
import os
from datetime import datetime
import schedule
import time
import threading

from pipeline.data_processing import DataProcessor
from pipeline.model_training import ModelTrainer
from monitoring.drift_detection import DriftDetector
from monitoring.performance_metrics import PerformanceTracker

class MLOpsPipeline:
    """
    Executive chef - orchestrating the entire operation,
    ensuring everything runs smoothly and on schedule
    """
    
    def __init__(self, config_path="config/pipeline_config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.setup_logging()
        self.data_processor = DataProcessor(config_path)
        self.model_trainer = ModelTrainer(self.config)
        self.drift_detector = DriftDetector()
        self.performance_tracker = PerformanceTracker()
        
    def setup_logging(self):
        """Set up the kitchen communication system"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_full_pipeline(self, data_path, target_column):
        """Complete service preparation - from ingredient delivery to ready-to-serve dishes"""
        self.logger.info("Starting full MLOps pipeline execution")
        
        try:
            # Step 1: Data Processing
            self.logger.info("Step 1: Processing ingredients")
            raw_data = self.data_processor.ingest_data(data_path)
            
            quality_report, validated_data = self.data_processor.validate_data_quality(raw_data)
            self.save_quality_report(quality_report)
            
            X_train, X_test, y_train, y_test = self.data_processor.preprocess_features(
                validated_data, target_column
            )
            self.data_processor.save_processed_data(X_train, X_test, y_train, y_test)
            
            # Step 2: Model Training
            self.logger.info("Step 2: Testing recipes")
            model_results = self.model_trainer.train_multiple_models(
                X_train, y_train, X_test, y_test
            )
            
            # Step 3: Model Registration
            self.logger.info("Step 3: Recording the best recipe")
            model_details, metadata = self.model_trainer.register_best_model(model_results)
            
            # Step 4: Performance Baseline
            self.performance_tracker.set_baseline_metrics(metadata['metrics'])
            
            self.logger.info("Pipeline execution completed successfully")
            return {
                'status': 'success',
                'model_version': model_details.version,
                'best_model_f1': metadata['metrics']['test_f1']
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_monitoring_cycle(self):
        """Daily quality inspection"""
        self.logger.info("Running monitoring cycle")
        
        try:
            # Check for data drift
            if os.path.exists('data/new_data.csv'):  # New data arrived
                new_data = self.data_processor.ingest_data('data/new_data.csv')
                drift_results = self.drift_detector.detect_data_drift(new_data)
                
                if drift_results['overall_drift']:
                    self.logger.warning("Data drift detected! Consider retraining model.")
                    self.send_alert('data_drift', drift_results)
                
                # Save drift report
                report = self.drift_detector.generate_drift_report(drift_results)
                with open(f"logs/drift_report_{datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
                    f.write(report)
            
            # Check prediction drift
            log_file = f"logs/predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
            if os.path.exists(log_file):
                pred_drift_results = self.drift_detector.detect_prediction_drift(log_file)
                
                if pred_drift_results.get('prediction_drift_detected'):
                    self.logger.warning("Prediction drift detected!")
                    self.send_alert('prediction_drift', pred_drift_results)
        
        except Exception as e:
            self.logger.error(f"Monitoring cycle failed: {str(e)}")
    
    def send_alert(self, alert_type, details):
        """Emergency communication system"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'severity': 'high' if alert_type in ['data_drift', 'prediction_drift'] else 'medium',
            'details': details,
            'action_required': 'Review model performance and consider retraining'
        }
        
        # Save alert
        os.makedirs('alerts', exist_ok=True)
        alert_file = f"alerts/alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=2)
        
        self.logger.critical(f"ALERT: {alert_type} - {alert['action_required']}")
    
    def save_quality_report(self, quality_report):
        """Archive quality inspection results"""
        os.makedirs('reports', exist_ok=True)
        report_file = f"reports/quality_report_{datetime.now().strftime('%Y%m%d')}.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump(quality_report, f, indent=2)
    
    def schedule_pipeline(self):
        """Set up the daily operation schedule"""
        # Schedule monitoring every 4 hours
        schedule.every(4).hours.do(self.run_monitoring_cycle)
        
        # Schedule weekly full pipeline run (if new data available)
        schedule.every().sunday.at("02:00").do(self.check_and_retrain)
        
        self.logger.info("Pipeline scheduling activated")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def check_and_retrain(self):
        """Weekly assessment - decide if we need to update our recipes"""
        if os.path.exists('data/new_data.csv'):
            self.logger.info("New data detected. Initiating retraining...")
            result = self.run_full_pipeline('data/new_data.csv', self.config['target_column'])
            
            if result['status'] == 'success':
                # Archive old data and replace
                os.rename('data/new_data.csv', f"data/processed/data_{datetime.now().strftime('%Y%m%d')}.csv")

def main():
    """Restaurant opening - start all operations"""
    pipeline = MLOpsPipeline()
    
    # Run initial pipeline if training data exists
    if os.path.exists('data/raw/training_data.csv'):
        print("Running initial pipeline setup...")
        result = pipeline.run_full_pipeline(
            'data/raw/training_data.csv',
            pipeline.config['target_column']
        )
        print(f"Pipeline setup result: {result}")
    
    # Start scheduling in background
    scheduler_thread = threading.Thread(target=pipeline.schedule_pipeline)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    print("MLOps Pipeline is running!")
    print("- API server: python api/app.py")
    print("- Monitoring: Check logs/ directory")
    print("- Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down pipeline...")

if __name__ == "__main__":
    main()
```

## 6. Configuration & Docker Setup

### config/pipeline_config.yaml
```yaml
# MLOps Pipeline Configuration
experiment_name: "production_mlops_pipeline"
mlflow_uri: "sqlite:///mlflow.db"

# Data configuration
target_column: "target"
data_split:
  test_size: 0.2
  random_state: 42

# Data quality thresholds
data_quality:
  missing_threshold: 0.3
  drift_threshold: 0.05

# Model training
model_selection:
  metric: "f1_score"
  cross_validation_folds: 5

# Monitoring
monitoring:
  drift_detection_frequency: 4  # hours
  performance_check_frequency: 24  # hours
  alert_thresholds:
    accuracy_drop: 0.05
    drift_score: 0.1

# Deployment
deployment:
  model_serve_port: 5000
  health_check_interval: 300  # seconds
  max_prediction_batch_size: 1000
```

### docker/Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed models/registry models/artifacts reports alerts

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "main.py"]

### docker/docker-compose.yml
```yaml
version: '3.8'

services:
  mlops-pipeline:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "5000:5000"
    volumes:
      - ../data:/app/data
      - ../logs:/app/logs
      - ../models:/app/models
      - ../reports:/app/reports
      - ../alerts:/app/alerts
    environment:
      - PYTHONPATH=/app
      - FLASK_ENV=production
    restart: unless-stopped
    
  mlflow-server:
    image: python:3.9-slim
    ports:
      - "5001:5001"
    volumes:
      - ../mlruns:/mlruns
      - ../mlflow.db:/mlflow.db
    command: >
      sh -c "pip install mlflow &&
             mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlruns"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
```

## 7. Performance Monitoring Component

### monitoring/performance_metrics.py
```python
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

class PerformanceTracker:
    """
    Restaurant critic - keeping detailed records of service quality
    and customer satisfaction over time
    """
    
    def __init__(self, db_path='performance_metrics.db'):
        self.db_path = db_path
        self.setup_database()
        self.logger = logging.getLogger(__name__)
        
    def setup_database(self):
        """Set up the review record books"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                model_version TEXT,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                prediction_count INTEGER,
                avg_confidence REAL,
                metric_type TEXT
            )
        ''')
        
        # Create predictions table for detailed tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                model_version TEXT,
                prediction INTEGER,
                confidence REAL,
                actual_label INTEGER,
                feedback_received BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction_with_feedback(self, prediction, confidence, actual_label=None, model_version="1.0"):
        """Record each dish served and customer feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO prediction_logs 
            (timestamp, model_version, prediction, confidence, actual_label, feedback_received)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            model_version,
            prediction,
            confidence,
            actual_label,
            actual_label is not None
        ))
        
        conn.commit()
        conn.close()
    
    def calculate_live_metrics(self, time_window_hours=24):
        """Daily performance assessment - how well did we serve today?"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent predictions with feedback
        cutoff_time = (datetime.now() - timedelta(hours=time_window_hours)).isoformat()
        
        query = '''
            SELECT prediction, actual_label, confidence, model_version
            FROM prediction_logs 
            WHERE timestamp > ? AND feedback_received = TRUE
        '''
        
        df = pd.read_sql(query, conn, params=[cutoff_time])
        conn.close()
        
        if len(df) == 0:
            return None
        
        # Calculate metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'time_window_hours': time_window_hours,
            'total_predictions': len(df),
            'model_version': df['model_version'].iloc[-1],
            'accuracy': accuracy_score(df['actual_label'], df['prediction']),
            'precision': precision_score(df['actual_label'], df['prediction'], average='weighted'),
            'recall': recall_score(df['actual_label'], df['prediction'], average='weighted'),
            'f1_score': f1_score(df['actual_label'], df['prediction'], average='weighted'),
            'avg_confidence': df['confidence'].mean(),
            'confidence_std': df['confidence'].std()
        }
        
        # Store metrics
        self.store_metrics(metrics)
        
        return metrics
    
    def store_metrics(self, metrics):
        """Archive performance records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (timestamp, model_version, accuracy, precision_score, recall, f1_score, 
             prediction_count, avg_confidence, metric_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['timestamp'],
            metrics['model_version'],
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['total_predictions'],
            metrics['avg_confidence'],
            'live_feedback'
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Metrics stored: F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}")
    
    def set_baseline_metrics(self, baseline_metrics):
        """Establish the gold standard - our best recipe performance"""
        baseline_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_version': 'baseline',
            'accuracy': baseline_metrics['test_accuracy'],
            'precision': baseline_metrics['test_precision'],
            'recall': baseline_metrics['test_recall'],
            'f1_score': baseline_metrics['test_f1'],
            'prediction_count': 0,
            'avg_confidence': 0.0,
            'metric_type': 'baseline'
        }
        
        self.store_metrics(baseline_entry)
        self.logger.info("Baseline metrics established")
    
    def detect_performance_degradation(self, threshold=0.05):
        """Quality control alert - detect if service standards are dropping"""
        conn = sqlite3.connect(self.db_path)
        
        # Get baseline metrics
        baseline_query = '''
            SELECT * FROM performance_metrics 
            WHERE metric_type = 'baseline' 
            ORDER BY timestamp DESC 
            LIMIT 1
        '''
        baseline = pd.read_sql(baseline_query, conn)
        
        # Get recent live metrics
        recent_query = '''
            SELECT * FROM performance_metrics 
            WHERE metric_type = 'live_feedback' 
            ORDER BY timestamp DESC 
            LIMIT 5
        '''
        recent = pd.read_sql(recent_query, conn)
        conn.close()
        
        if len(baseline) == 0 or len(recent) == 0:
            return {'status': 'insufficient_data'}
        
        # Compare metrics
        baseline_f1 = baseline['f1_score'].iloc[0]
        recent_avg_f1 = recent['f1_score'].mean()
        
        performance_drop = baseline_f1 - recent_avg_f1
        degradation_detected = performance_drop > threshold
        
        return {
            'degradation_detected': degradation_detected,
            'baseline_f1': float(baseline_f1),
            'recent_avg_f1': float(recent_avg_f1),
            'performance_drop': float(performance_drop),
            'threshold': threshold,
            'recommendation': 'Consider model retraining' if degradation_detected else 'Performance stable'
        }
    
    def generate_performance_report(self, days=7):
        """Weekly performance summary report"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = '''
            SELECT * FROM performance_metrics 
            WHERE timestamp > ? AND metric_type = 'live_feedback'
            ORDER BY timestamp
        '''
        
        df = pd.read_sql(query, conn, params=[cutoff_date])
        conn.close()
        
        if len(df) == 0:
            return "No performance data available for the specified period."
        
        # Calculate summary statistics
        report = f"""
=== PERFORMANCE REPORT ({days} days) ===
Period: {cutoff_date[:10]} to {datetime.now().strftime('%Y-%m-%d')}

Summary Statistics:
- Total Evaluations: {len(df)}
- Average F1 Score: {df['f1_score'].mean():.4f} (±{df['f1_score'].std():.4f})
- Average Accuracy: {df['accuracy'].mean():.4f} (±{df['accuracy'].std():.4f})
- Average Confidence: {df['avg_confidence'].mean():.4f}

Performance Trends:
- Best F1 Score: {df['f1_score'].max():.4f}
- Worst F1 Score: {df['f1_score'].min():.4f}
- Latest F1 Score: {df['f1_score'].iloc[-1]:.4f}

Model Versions Used: {', '.join(df['model_version'].unique())}

Recommendations:
"""
        
        # Add recommendations based on trends
        if len(df) >= 3:
            recent_trend = np.polyfit(range(len(df[-5:])), df['f1_score'].tail(5), 1)[0]
            if recent_trend < -0.01:
                report += "⚠️  Declining performance trend detected - consider model refresh\n"
            elif recent_trend > 0.01:
                report += "✅ Improving performance trend - current approach working well\n"
            else:
                report += "📊 Stable performance - continue monitoring\n"
        
        return report
    
    def create_performance_dashboard(self, output_path='reports/performance_dashboard.png'):
        """Visual performance review - create charts for management"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent data
        query = '''
            SELECT * FROM performance_metrics 
            WHERE metric_type = 'live_feedback'
            ORDER BY timestamp
        '''
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if len(df) < 2:
            self.logger.warning("Insufficient data for dashboard creation")
            return
        
        # Create dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MLOps Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # F1 Score over time
        axes[0, 0].plot(df['timestamp'], df['f1_score'], marker='o', linewidth=2)
        axes[0, 0].set_title('F1 Score Over Time')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy vs Confidence
        axes[0, 1].scatter(df['avg_confidence'], df['accuracy'], alpha=0.6, s=60)
        axes[0, 1].set_title('Accuracy vs Average Confidence')
        axes[0, 1].set_xlabel('Average Confidence')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Prediction volume
        axes[1, 0].bar(range(len(df)), df['prediction_count'], alpha=0.7)
        axes[1, 0].set_title('Prediction Volume Over Time')
        axes[1, 0].set_ylabel('Number of Predictions')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Metrics comparison
        metrics_to_plot = ['accuracy', 'precision_score', 'recall', 'f1_score']
        for i, metric in enumerate(metrics_to_plot):
            axes[1, 1].plot(df['timestamp'], df[metric], marker='o', label=metric.replace('_', ' ').title())
        
        axes[1, 1].set_title('All Metrics Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Format x-axis for all time-based plots
        for ax in [axes[0, 0], axes[1, 1]]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance dashboard saved to {output_path}")
```

## 8. Requirements File

### requirements.txt
```txt
# Core ML libraries
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
scipy==1.11.1

# MLOps and tracking
mlflow==2.5.0
joblib==1.3.1

# Web framework and API
flask==2.3.2
gunicorn==21.0.1

# Data processing
PyYAML==6.0.1
openpyxl==3.1.2

# Monitoring and visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# Scheduling and utilities
schedule==1.2.0
python-dateutil==2.8.2

# Database
sqlite3

# System utilities
psutil==5.9.5
requests==2.31.0

# Development and testing
pytest==7.4.0
pytest-cov==4.1.0
black==23.7.0
flake8==6.0.0

# Docker and containerization
docker==6.1.3

# Additional utilities
click==8.1.6
python-dotenv==1.0.0
```

## 9. Sample Data Generator (for testing)

### generate_sample_data.py
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(n_samples=10000, n_features=10):
    """
    Create sample data for testing our complete operation
    """
    np.random.seed(42)
    
    # Generate feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Generate synthetic data
    data = {}
    for feature in feature_names:
        if feature in ['feature_0', 'feature_5']:  # Categorical features
            data[feature] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
        else:  # Numerical features
            data[feature] = np.random.normal(0, 1, n_samples)
    
    # Create target variable with some correlation to features
    target_weights = np.random.uniform(-1, 1, n_features)
    
    # Convert categorical to numerical for target calculation
    temp_data = pd.DataFrame(data)
    for col in ['feature_0', 'feature_5']:
        temp_data[col] = pd.Categorical(temp_data[col]).codes
    
    # Generate target
    target_score = np.sum([temp_data[f'feature_{i}'] * target_weights[i] 
                          for i in range(n_features)], axis=0)
    target = (target_score > np.median(target_score)).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['target'] = target
    
    return df

def create_sample_datasets():
    """Prepare all ingredients for our test kitchen"""
    os.makedirs('data/raw', exist_ok=True)
    
    # Training data
    train_data = generate_sample_data(8000, 10)
    train_data.to_csv('data/raw/training_data.csv', index=False)
    print(f"Training data created: {train_data.shape}")
    
    # New data for drift testing (slightly different distribution)
    np.random.seed(123)  # Different seed for drift
    new_data = generate_sample_data(2000, 10)
    # Add some drift
    new_data['feature_1'] = new_data['feature_1'] + 0.5  # Shift distribution
    new_data['feature_3'] = new_data['feature_3'] * 1.2  # Scale change
    new_data.to_csv('data/new_data.csv', index=False)
    print(f"New data created: {new_data.shape}")
    
    # Validation data for testing API
    val_data = generate_sample_data(100, 10)
    val_data.drop('target', axis=1).to_csv('data/validation_input.csv', index=False)
    val_data[['target']].to_csv('data/validation_labels.csv', index=False)
    print(f"Validation data created: {val_data.shape}")

if __name__ == "__main__":
    create_sample_datasets()
    print("Sample datasets generated successfully!")
```

## 10. API Testing Script

### test_api.py
```python
import requests
import pandas as pd
import json
import time
import numpy as np

def test_complete_pipeline():
    """
    Complete service test - from order taking to dish delivery
    """
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Complete MLOps Pipeline")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1️⃣ Testing kitchen status...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ System healthy - Model version: {health_data['model_version']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # Test 2: Model info
    print("\n2️⃣ Checking menu information...")
    response = requests.get(f"{base_url}/model_info")
    if response.status_code == 200:
        model_info = response.json()
        print(f"✅ Model info retrieved:")
        print(f"   - Model: {model_info['model_name']}")
        print(f"   - F1 Score: {model_info['metrics']['test_f1']:.4f}")
        print(f"   - Version: {model_info['version']}")
    
    # Test 3: Single prediction
    print("\n3️⃣ Testing single order...")
    sample_input = {
        'feature_0': 'A',
        'feature_1': 0.5,
        'feature_2': -0.3,
        'feature_3': 1.2,
        'feature_4': 0.8,
        'feature_5': 'B',
        'feature_6': -0.5,
        'feature_7': 0.0,
        'feature_8': 0.7,
        'feature_9': -1.1
    }
    
    response = requests.post(f"{base_url}/predict", json=sample_input)
    if response.status_code == 200:
        prediction = response.json()
        print(f"✅ Single prediction successful:")
        print(f"   - Prediction: {prediction['prediction'][0]}")
        print(f"   - Confidence: {max(prediction['probability'][0]):.4f}")
    else:
        print(f"❌ Single prediction failed: {response.status_code}")
    
    # Test 4: Batch predictions
    print("\n4️⃣ Testing batch orders...")
    if 'data/validation_input.csv' in [f for f in os.listdir('data') if f.endswith('.csv')]:
        val_data = pd.read_csv('data/validation_input.csv').head(10)
        batch_input = val_data.to_dict('records')
        
        start_time = time.time()
        response = requests.post(f"{base_url}/predict", json=batch_input)
        end_time = time.time()
        
        if response.status_code == 200:
            predictions = response.json()
            print(f"✅ Batch prediction successful:")
            print(f"   - Processed: {len(predictions['prediction'])} samples")
            print(f"   - Time taken: {end_time - start_time:.2f} seconds")
            print(f"   - Avg confidence: {np.mean([max(prob) for prob in predictions['probability']]):.4f}")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
    
    # Test 5: Load testing
    print("\n5️⃣ Testing service under pressure...")
    successful_requests = 0
    total_requests = 50
    
    for i in range(total_requests):
        try:
            response = requests.post(f"{base_url}/predict", 
                                   json=sample_input, 
                                   timeout=5)
            if response.status_code == 200:
                successful_requests += 1
        except:
            pass
    
    success_rate = (successful_requests / total_requests) * 100
    print(f"✅ Load test completed:")
    print(f"   - Success rate: {success_rate:.1f}% ({successful_requests}/{total_requests})")
    
    if success_rate >= 95:
        print("🎉 Pipeline is production-ready!")
    elif success_rate >= 80:
        print("⚠️  Pipeline needs optimization")
    else:
        print("❌ Pipeline requires significant improvements")

def simulate_production_usage():
    """Simulate real restaurant service with varying customer flow"""
    base_url = "http://localhost:5000"
    
    print("\n🏪 Simulating production usage...")
    
    # Load validation data
    val_data = pd.read_csv('data/validation_input.csv')
    val_labels = pd.read_csv('data/validation_labels.csv')
    
    # Simulate different times of day with varying load
    scenarios = [
        ("Morning Rush", 20, 0.1),      # 20 requests, 0.1s interval
        ("Lunch Peak", 50, 0.05),       # 50 requests, 0.05s interval  
        ("Afternoon Lull", 10, 0.5),    # 10 requests, 0.5s interval
        ("Dinner Rush", 40, 0.08),      # 40 requests, 0.08s interval
    ]
    
    for scenario_name, num_requests, interval in scenarios:
        print(f"\n📊 {scenario_name}: {num_requests} requests...")
        
        successful = 0
        total_latency = 0
        
        for i in range(num_requests):
            # Random sample from validation data
            sample_idx = np.random.randint(0, len(val_data))
            sample_input = val_data.iloc[sample_idx].to_dict()
            
            start_time = time.time()
            try:
                response = requests.post(f"{base_url}/predict", 
                                       json=sample_input,
                                       timeout=10)
                end_time = time.time()
                
                if response.status_code == 200:
                    successful += 1
                    total_latency += (end_time - start_time)
                
            except Exception as e:
                pass
            
            time.sleep(interval)
        
        avg_latency = total_latency / successful if successful > 0 else 0
        print(f"   - Success: {successful}/{num_requests} ({successful/num_requests*100:.1f}%)")
        print(f"   - Avg Latency: {avg_latency*1000:.1f}ms")

if __name__ == "__main__":
    import os
    test_complete_pipeline()
    simulate_production_usage()
```

## Usage Instructions

### 1. Setup and Installation
```bash
# Clone and setup
git clone <your-repo>
cd mlops_pipeline

# Install dependencies  
pip install -r requirements.txt

# Generate sample data
python generate_sample_data.py

# Initialize pipeline
python main.py
```

### 2. Run API Server (in separate terminal)
```bash
python api/app.py
```

### 3. Test the Pipeline
```bash
python test_api.py
```

### 4. Docker Deployment
```bash
cd docker
docker-compose up -d
```

### 5. Monitor Performance
- MLflow UI: http://localhost:5001
- API Health: http://localhost:5000/health  
- Grafana Dashboard: http://localhost:3000
- Check logs: `tail -f logs/pipeline_*.log`

This complete MLOps pipeline demonstrates production-ready machine learning operations with automated training, monitoring, drift detection, and deployment - like running a world-class restaurant where quality is consistently maintained and improvements are continuously made based on customer feedback.

## Assignment: Model Performance Dashboard

**Task**: Create a Django web interface that displays real-time model performance metrics and drift detection alerts.

**Requirements**:

1. **Dashboard View** (`views.py`):
   - Create a view that shows all active models and their current status
   - Display recent drift detections with severity levels
   - Show performance trends over the last 30 days

2. **Template** (`dashboard.html`):
   - Use charts (Chart.js) to visualize performance trends
   - Create alert cards for recent drift detections
   - Add filtering by model name and time period

3. **AJAX Updates** (`JavaScript`):
   - Implement auto-refresh every 30 seconds for live monitoring
   - Add click handlers to drill down into specific models

4. **Model Extension**:
   - Add a `ModelAlert` model to track different types of alerts
   - Include severity levels: 'low', 'medium', 'high', 'critical'

**Deliverables**:
- Django view function with proper error handling
- HTML template with responsive design
- JavaScript for interactivity
- At least 3 different chart types showing different metrics

**Success Criteria**:
- Dashboard loads without errors
- Charts display sample data correctly
- Auto-refresh functionality works
- Responsive design works on mobile devices

This assignment tests your ability to integrate all the MLOps concepts into a practical monitoring interface - like creating a digital kitchen display that shows the status of all dishes being prepared, with alerts when something needs attention.

---

## Summary

Today we've built a comprehensive MLOps system that handles the full lifecycle of machine learning models in production. Just as a world-class restaurant operates with precision, consistency, and continuous quality monitoring, our system ensures that ML models perform reliably at scale.

**Key Concepts Mastered**:

1. **Pipeline Automation**: Automated workflows that handle model training, evaluation, and deployment
2. **Model Versioning**: Systematic tracking and management of different model versions
3. **A/B Testing**: Scientific comparison of model versions in production
4. **Drift Detection**: Automated monitoring for data and performance changes

The beauty of this system lies in its orchestration - each component works independently but contributes to a seamless operation where models are continuously improved, monitored, and maintained without manual intervention.

Remember: MLOps isn't just about technology - it's about creating sustainable, reliable systems that allow data science teams to focus on innovation while ensuring production models serve customers with excellence, day after day.