# AI Mastery Course: Day 76 Ensemble Methods

## Learning Objective
By the end of this lesson, you will master ensemble methods in machine learning and implement them using Django as your web framework. You'll understand how to combine multiple models like a master chef combines ingredients to create extraordinary dishes, and build a Django application that serves ensemble predictions.

---

## Imagine That...

Imagine you're the head chef of the world's most prestigious restaurant. You have four talented sous chefs, each with their own specialty - one excels at appetizers, another at main courses, one at desserts, and the last at fusion cuisine. Instead of relying on just one chef for the entire meal, you orchestrate all four to work together, each contributing their expertise to create a dining experience that's far superior to what any single chef could achieve alone.

This is exactly how ensemble methods work in machine learning. Just as you combine the strengths of multiple chefs to create an extraordinary meal, ensemble methods combine multiple machine learning models to create predictions that are more accurate and robust than any individual model could produce.

---

## Lesson 1: Bagging and Boosting - The Foundation of Team Cooking

Think of **bagging** like having multiple chefs prepare the same dish independently, each using slightly different ingredients from your pantry. At the end, you taste all versions and take the average flavor profile - this reduces the chance of any single chef's mistake ruining the dish.

**Boosting** is like having chefs work in sequence, where each chef learns from the previous chef's mistakes and focuses on improving the areas where the previous chef struggled.

### Code Implementation with Django

```python
# models.py - Django Models for Ensemble Learning
from django.db import models
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

class EnsembleModel(models.Model):
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=50)  # 'bagging' or 'boosting'
    accuracy = models.FloatField(null=True, blank=True)
    model_file = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def save_model(self, model):
        """Save the trained model to file"""
        if not os.path.exists('models'):
            os.makedirs('models')
        model_path = f'models/{self.name}_{self.model_type}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        self.model_file = model_path
        self.save()
    
    def load_model(self):
        """Load the trained model from file"""
        with open(self.model_file, 'rb') as f:
            return pickle.load(f)

# views.py - Django Views
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import EnsembleModel

def bagging_demo(request):
    """Demonstrate bagging like multiple chefs cooking independently"""
    
    # Generate sample data (like ingredients in our kitchen)
    np.random.seed(42)
    X = np.random.randn(1000, 4)  # 4 features (ingredients)
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.1 > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create bagging classifier (multiple chefs working independently)
    bagging_chef = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=10,  # 10 different chefs
        random_state=42
    )
    
    # Train our ensemble of chefs
    bagging_chef.fit(X_train, y_train)
    
    # Make predictions (get the consensus from all chefs)
    predictions = bagging_chef.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Save the model
    model_instance = EnsembleModel.objects.create(
        name="Restaurant_Bagging_Team",
        model_type="bagging",
        accuracy=accuracy
    )
    model_instance.save_model(bagging_chef)
    
    context = {
        'accuracy': round(accuracy * 100, 2),
        'model_name': "Bagging Ensemble",
        'description': "Like having 10 chefs cook the same dish independently and averaging their results"
    }
    
    return render(request, 'ensemble/bagging_results.html', context)

def boosting_demo(request):
    """Demonstrate boosting like chefs learning from each other's mistakes"""
    
    # Same kitchen setup (data)
    np.random.seed(42)
    X = np.random.randn(1000, 4)
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.1 > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create boosting classifier (chefs learning from previous mistakes)
    boosting_chef = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,  # 50 learning iterations
        learning_rate=1.0,
        random_state=42
    )
    
    # Train our learning sequence of chefs
    boosting_chef.fit(X_train, y_train)
    
    # Make predictions
    predictions = boosting_chef.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Save the model
    model_instance = EnsembleModel.objects.create(
        name="Restaurant_Boosting_Team",
        model_type="boosting",
        accuracy=accuracy
    )
    model_instance.save_model(boosting_chef)
    
    context = {
        'accuracy': round(accuracy * 100, 2),
        'model_name': "Boosting Ensemble",
        'description': "Like chefs working in sequence, each learning from the previous chef's mistakes"
    }
    
    return render(request, 'ensemble/boosting_results.html', context)
```

### Syntax Explanation:
- `BaggingClassifier(base_estimator=DecisionTreeClassifier())`: Creates multiple versions of the base model with different subsets of data
- `n_estimators=10`: Like having 10 different chefs
- `AdaBoostClassifier()`: Implements adaptive boosting where each model learns from previous mistakes
- `learning_rate=1.0`: Controls how much each new chef learns from previous mistakes

---

## Lesson 2: Random Forest Deep Dive - The Dream Kitchen Team

Random Forest is like having a kitchen full of specialized chefs, each trained on different recipes (random subsets of ingredients) and different cooking techniques (random subsets of methods). Each chef votes on the final dish, and the majority wins.

### Code Implementation

```python
# views.py (continued)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def random_forest_kitchen(request):
    """Deep dive into Random Forest - our dream kitchen team"""
    
    # Create a more complex dataset (gourmet ingredients)
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create our Random Forest kitchen team
    forest_kitchen = RandomForestClassifier(
        n_estimators=100,      # 100 specialized chefs
        max_depth=10,          # Each chef can use up to 10 cooking steps
        max_features='sqrt',   # Each chef uses sqrt(20) ≈ 4 ingredients randomly
        bootstrap=True,        # Each chef gets a random sample of dishes to practice on
        random_state=42,
        n_jobs=-1             # All chefs work simultaneously (parallel processing)
    )
    
    # Train our forest of chefs
    forest_kitchen.fit(X_train, y_train)
    
    # Get predictions and confidence scores
    predictions = forest_kitchen.predict(X_test)
    probabilities = forest_kitchen.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Feature importance (which ingredients matter most)
    feature_importance = forest_kitchen.feature_importances_
    
    # Save the model
    model_instance = EnsembleModel.objects.create(
        name="Random_Forest_Kitchen",
        model_type="random_forest",
        accuracy=accuracy
    )
    model_instance.save_model(forest_kitchen)
    
    context = {
        'accuracy': round(accuracy * 100, 2),
        'n_estimators': 100,
        'feature_importance': feature_importance.tolist()[:10],  # Top 10 features
        'description': "100 specialized chefs, each expert in different ingredient combinations"
    }
    
    return render(request, 'ensemble/random_forest_results.html', context)
```

### Syntax Explanation:
- `max_features='sqrt'`: Each tree uses √(total features) randomly selected features
- `bootstrap=True`: Each tree trains on a random sample (with replacement) of the training data
- `n_jobs=-1`: Uses all available CPU cores for parallel processing
- `feature_importances_`: Shows which features (ingredients) are most important for predictions

---

## Lesson 3: Gradient Boosting - The Master Chef's Sequential Training

Gradient Boosting is like a master chef training apprentices in sequence. Each new apprentice focuses specifically on the dishes that previous apprentices struggled with, gradually building a team that can handle any culinary challenge.

### Code Implementation

```python
# views.py (continued)
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

def gradient_boosting_academy(request):
    """Gradient Boosting - Master Chef's Sequential Training Academy"""
    
    # Prepare our training kitchen (complex dataset)
    X, y = make_classification(
        n_samples=3000,
        n_features=25,
        n_informative=20,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Traditional Gradient Boosting (Classic training method)
    classic_academy = GradientBoostingClassifier(
        n_estimators=200,      # 200 training sessions
        learning_rate=0.1,     # How much each apprentice learns from mistakes
        max_depth=3,           # Complexity of each apprentice's skills
        random_state=42
    )
    
    # XGBoost (Modern, efficient training)
    xgb_academy = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric='logloss'
    )
    
    # LightGBM (Ultra-fast training)
    lgb_academy = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=-1
    )
    
    # Train all three academies
    models = {
        'Classic GB': classic_academy,
        'XGBoost': xgb_academy,
        'LightGBM': lgb_academy
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = {
            'accuracy': round(accuracy * 100, 2),
            'model': model
        }
        
        # Save each model
        model_instance = EnsembleModel.objects.create(
            name=f"Gradient_Boosting_{name}",
            model_type="gradient_boosting",
            accuracy=accuracy
        )
        model_instance.save_model(model)
    
    context = {
        'results': results,
        'description': "Three different master chef training academies, each with their own approach"
    }
    
    return render(request, 'ensemble/gradient_boosting_results.html', context)
```

### Syntax Explanation:
- `learning_rate=0.1`: Controls how much each new model learns from previous mistakes (smaller = more conservative learning)
- `n_estimators=200`: Number of sequential models (apprentices) to train
- `eval_metric='logloss'`: Metric used to evaluate model performance during training
- `verbose=-1`: Suppresses training output for cleaner logs

---

## Lesson 4: Voting and Stacking Classifiers - The Ultimate Kitchen Council

**Voting** is like having your best chefs each prepare their signature dish, then having a panel of food critics vote on which approach to use for the final meal.

**Stacking** is like having a master chef who tastes all the other chefs' dishes and then creates a final, refined version based on what they learned from each dish.

### Code Implementation

```python
# views.py (continued)
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def kitchen_council(request):
    """Voting and Stacking - The Ultimate Kitchen Council"""
    
    # Prepare our gourmet dataset
    X, y = make_classification(
        n_samples=2500,
        n_features=15,
        n_informative=12,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Our panel of expert chefs (base models)
    chef_forest = RandomForestClassifier(n_estimators=50, random_state=42)
    chef_svm = SVC(probability=True, random_state=42)  # probability=True for soft voting
    chef_knn = KNeighborsClassifier(n_neighbors=5)
    chef_xgb = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
    
    # Voting Classifier - Democratic Kitchen Council
    voting_council = VotingClassifier(
        estimators=[
            ('forest_chef', chef_forest),
            ('svm_chef', chef_svm),
            ('knn_chef', chef_knn),
            ('xgb_chef', chef_xgb)
        ],
        voting='soft'  # Use probability-based voting (more nuanced than hard voting)
    )
    
    # Stacking Classifier - Master Chef Coordinator
    stacking_master = StackingClassifier(
        estimators=[
            ('forest_chef', chef_forest),
            ('svm_chef', chef_svm),
            ('knn_chef', chef_knn),
            ('xgb_chef', chef_xgb)
        ],
        final_estimator=LogisticRegression(),  # Master chef who makes final decision
        cv=5  # 5-fold cross-validation for training the meta-learner
    )
    
    # Train both councils
    voting_council.fit(X_train, y_train)
    stacking_master.fit(X_train, y_train)
    
    # Get predictions from both approaches
    voting_predictions = voting_council.predict(X_test)
    stacking_predictions = stacking_master.predict(X_test)
    
    voting_accuracy = accuracy_score(y_test, voting_predictions)
    stacking_accuracy = accuracy_score(y_test, stacking_predictions)
    
    # Save both models
    voting_model = EnsembleModel.objects.create(
        name="Kitchen_Voting_Council",
        model_type="voting",
        accuracy=voting_accuracy
    )
    voting_model.save_model(voting_council)
    
    stacking_model = EnsembleModel.objects.create(
        name="Kitchen_Stacking_Master",
        model_type="stacking", 
        accuracy=stacking_accuracy
    )
    stacking_model.save_model(stacking_master)
    
    context = {
        'voting_accuracy': round(voting_accuracy * 100, 2),
        'stacking_accuracy': round(stacking_accuracy * 100, 2),
        'voting_description': "4 expert chefs vote democratically on each dish",
        'stacking_description': "Master chef learns from all 4 experts and makes refined decisions"
    }
    
    return render(request, 'ensemble/council_results.html', context)

# API endpoint for making predictions
@csrf_exempt
def predict_with_ensemble(request):
    """API endpoint to make predictions using trained ensemble models"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            model_id = data.get('model_id')
            features = data.get('features')  # List of feature values
            
            # Load the requested model
            model_instance = EnsembleModel.objects.get(id=model_id)
            model = model_instance.load_model()
            
            # Make prediction
            prediction = model.predict([features])[0]
            probability = None
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba([features])[0].tolist()
            
            return JsonResponse({
                'prediction': int(prediction),
                'probability': probability,
                'model_name': model_instance.name,
                'model_type': model_instance.model_type
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)
```

### Syntax Explanation:
- `voting='soft'`: Uses predicted probabilities for voting (more nuanced than simple majority)
- `estimators=[('name', model), ...]`: List of base models with names
- `final_estimator=LogisticRegression()`: Meta-learner that combines base model predictions
- `cv=5`: Cross-validation folds for training the stacking meta-learner
- `probability=True`: Enables probability predictions for SVM

---

## Django Templates

```html
<!-- templates/ensemble/base.html -->
<!DOCTYPE html>
<html>
<head>
    <title>AI Mastery - Ensemble Kitchen</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="#">🍳 Ensemble Kitchen</a>
            <div class="navbar-nav">
                <a class="nav-link" href="{% url 'bagging_demo' %}">Bagging</a>
                <a class="nav-link" href="{% url 'random_forest_kitchen' %}">Random Forest</a>
                <a class="nav-link" href="{% url 'gradient_boosting_academy' %}">Gradient Boosting</a>
                <a class="nav-link" href="{% url 'kitchen_council' %}">Voting & Stacking</a>
            </div>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% block content %}
        {% endblock %}
    </div>
</body>
</html>

<!-- templates/ensemble/council_results.html -->
{% extends 'ensemble/base.html' %}

{% block content %}
<div class="row">
    <div class="col-md-12">
        <h2>🏆 The Ultimate Kitchen Council Results</h2>
        <p class="lead">Comparing democratic voting vs master chef coordination</p>
    </div>
</div>

<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h4>🗳️ Voting Council</h4>
            </div>
            <div class="card-body">
                <h3 class="text-primary">{{ voting_accuracy }}% Accuracy</h3>
                <p>{{ voting_description }}</p>
                <div class="progress mb-3">
                    <div class="progress-bar" style="width: {{ voting_accuracy }}%"></div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-6">
        <div class="card">
            <div class="card-header bg-success text-white">
                <h4>🎯 Stacking Master</h4>
            </div>
            <div class="card-body">
                <h3 class="text-success">{{ stacking_accuracy }}% Accuracy</h3>
                <p>{{ stacking_description }}</p>
                <div class="progress mb-3">
                    <div class="progress-bar bg-success" style="width: {{ stacking_accuracy }}%"></div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

---

## Final Quality Project: Restaurant Recommendation Ensemble System

### Project Overview
Build a complete Django web application that uses ensemble methods to recommend restaurants based on user preferences, reviews, and dining history.

### Project Structure

```python
# models.py - Complete project models
class Restaurant(models.Model):
    name = models.CharField(max_length=200)
    cuisine_type = models.CharField(max_length=100)
    price_range = models.IntegerField(choices=[(1, '$'), (2, '$$'), (3, '$$$'), (4, '$$$$')])
    avg_rating = models.FloatField()
    location = models.CharField(max_length=200)
    features = models.JSONField()  # Store cuisine features, ambiance scores, etc.

class UserProfile(models.Model):
    user = models.OneToOneField('auth.User', on_delete=models.CASCADE)
    preferred_cuisines = models.JSONField(default=list)
    price_preference = models.IntegerField(default=2)
    dining_history = models.JSONField(default=list)

class RecommendationModel(models.Model):
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=50)
    accuracy = models.FloatField()
    model_file = models.CharField(max_length=200)
    is_active = models.BooleanField(default=True)

# views.py - Main recommendation system
class RestaurantRecommendationSystem:
    def __init__(self):
        self.ensemble_model = None
        self.feature_columns = ['price_range', 'avg_rating', 'cuisine_encoded', 
                               'location_encoded', 'ambiance_score', 'service_score']
    
    def prepare_training_data(self):
        """Prepare training data from user dining history"""
        # Get all user profiles with dining history
        profiles = UserProfile.objects.exclude(dining_history=[])
        
        X, y = [], []
        for profile in profiles:
            for restaurant_id, rating in profile.dining_history:
                try:
                    restaurant = Restaurant.objects.get(id=restaurant_id)
                    features = self.extract_features(restaurant, profile)
                    X.append(features)
                    y.append(1 if rating >= 4 else 0)  # Binary: liked/disliked
                except Restaurant.DoesNotExist:
                    continue
        
        return np.array(X), np.array(y)
    
    def extract_features(self, restaurant, user_profile):
        """Extract features for the ensemble model"""
        features = [
            restaurant.price_range,
            restaurant.avg_rating,
            self.encode_cuisine(restaurant.cuisine_type),
            self.encode_location(restaurant.location),
            restaurant.features.get('ambiance_score', 3.0),
            restaurant.features.get('service_score', 3.0)
        ]
        return features
    
    def train_ensemble(self):
        """Train our restaurant recommendation ensemble"""
        X, y = self.prepare_training_data()
        
        if len(X) < 100:  # Need sufficient data
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create our ensemble of recommendation chefs
        base_models = [
            ('taste_expert', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('price_expert', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('location_expert', xgb.XGBClassifier(n_estimators=100, random_state=42)),
            ('review_expert', SVC(probability=True, random_state=42))
        ]
        
        # Use stacking with a sophisticated meta-learner
        self.ensemble_model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        )
        
        # Train the ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate performance
        predictions = self.ensemble_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Save the model
        model_instance = RecommendationModel.objects.create(
            name="Restaurant_Ensemble_v1",
            model_type="stacking_ensemble",
            accuracy=accuracy
        )
        
        # Save to file
        model_path = f'models/restaurant_ensemble_{model_instance.id}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        
        model_instance.model_file = model_path
        model_instance.save()
        
        return True
    
    def recommend_restaurants(self, user_profile, num_recommendations=5):
        """Get restaurant recommendations for a user"""
        if not self.ensemble_model:
            # Load the latest active model
            try:
                latest_model = RecommendationModel.objects.filter(is_active=True).latest('id')
                with open(latest_model.model_file, 'rb') as f:
                    self.ensemble_model = pickle.load(f)
            except:
                return []
        
        # Get all restaurants user hasn't tried
        tried_restaurants = [item[0] for item in user_profile.dining_history]
        available_restaurants = Restaurant.objects.exclude(id__in=tried_restaurants)
        
        recommendations = []
        for restaurant in available_restaurants:
            features = self.extract_features(restaurant, user_profile)
            probability = self.ensemble_model.predict_proba([features])[0][1]  # Probability of liking
            
            recommendations.append({
                'restaurant': restaurant,
                'likelihood_score': probability,
                'reasons': self.explain_recommendation(restaurant, user_profile)
            })
        
        # Sort by likelihood and return top recommendations
        recommendations.sort(key=lambda x: x['likelihood_score'], reverse=True)
        return recommendations[:num_recommendations]

# Main Django view
def get_recommendations(request):
    """Main view for getting restaurant recommendations"""
    if not request.user.is_authenticated:
        return redirect('login')
    
    user_profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    # Initialize recommendation system
    rec_system = RestaurantRecommendationSystem()
    
    # Get recommendations
    recommendations = rec_system.recommend_restaurants(user_profile)
    
    context = {
        'recommendations': recommendations,
        'user_profile': user_profile
    }
    
    return render(request, 'restaurant/recommendations.html', context)
```

### Project Features:
1. **Multi-Model Ensemble**: Uses 4 different algorithms specialized in different aspects
2. **Real-time Predictions**: API endpoints for getting recommendations
3. **Explainable AI**: Provides reasons for each recommendation
4. **User Learning**: System improves as users provide feedback
5. **Django Integration**: Full web interface with user authentication

---

# Ensemble Model Competition Project

## Project Overview
Build a comprehensive ensemble model that combines multiple machine learning algorithms to compete in a predictive modeling challenge. This project simulates a real-world data science competition where you'll create a robust prediction system using advanced ensemble techniques.

## Project Specifications

### Dataset
We'll use the **Wine Quality Dataset** - a classic machine learning competition dataset with 1,599 samples and 11 features predicting wine quality scores (0-10).

### Core Requirements
1. Implement at least 4 different base models
2. Create 3 different ensemble methods
3. Optimize hyperparameters for each component
4. Achieve cross-validation score > 0.75
5. Generate competition-ready predictions

## Implementation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
def load_competition_data():
    """Load wine quality dataset for competition"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Base Models - Our Chef's Core Ingredients
class BaseModelChef:
    """Creates individual models like preparing different cooking techniques"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
    
    def prepare_ingredients(self, X_train, X_test):
        """Scale features like preparing ingredients consistently"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def train_base_models(self, X_train, y_train):
        """Train multiple base models like mastering different cooking methods"""
        
        # Model 1: Random Forest - The reliable slow-cooking method
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='r2')
        rf_grid.fit(X_train, y_train)
        self.models['random_forest'] = rf_grid.best_estimator_
        
        # Model 2: XGBoost - The precision technique
        xgb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6]
        }
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='r2')
        xgb_grid.fit(X_train, y_train)
        self.models['xgboost'] = xgb_grid.best_estimator_
        
        # Model 3: LightGBM - The efficient modern method
        lgb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50]
        }
        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=5, scoring='r2')
        lgb_grid.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_grid.best_estimator_
        
        # Model 4: Neural Network - The complex fusion technique
        mlp_params = {
            'hidden_layer_sizes': [(100,), (100, 50)],
            'learning_rate_init': [0.001, 0.01]
        }
        mlp = MLPRegressor(random_state=42, max_iter=500)
        mlp_grid = GridSearchCV(mlp, mlp_params, cv=5, scoring='r2')
        mlp_grid.fit(X_train, y_train)
        self.models['neural_network'] = mlp_grid.best_estimator_
        
        return self.models

# Ensemble Methods - Our Master Chef's Combination Techniques
class EnsembleChef:
    """Combines models like a master chef combining flavors"""
    
    def __init__(self, base_models):
        self.base_models = base_models
        self.ensembles = {}
    
    def create_voting_ensemble(self):
        """Voting ensemble - Like getting opinions from multiple expert chefs"""
        voting_regressor = VotingRegressor([
            ('rf', self.base_models['random_forest']),
            ('xgb', self.base_models['xgboost']),
            ('lgb', self.base_models['lightgbm']),
            ('mlp', self.base_models['neural_network'])
        ])
        self.ensembles['voting'] = voting_regressor
        return voting_regressor
    
    def create_stacking_ensemble(self, X_train, y_train):
        """Stacking ensemble - Like training a head chef to combine junior chefs' work"""
        # Generate predictions from base models
        base_predictions = np.column_stack([
            model.predict(X_train) for model in self.base_models.values()
        ])
        
        # Train meta-model (head chef) on base predictions
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(base_predictions, y_train)
        
        self.ensembles['stacking'] = {
            'base_models': self.base_models,
            'meta_model': meta_model
        }
        return self.ensembles['stacking']
    
    def create_weighted_ensemble(self, X_val, y_val):
        """Weighted ensemble - Like adjusting recipe proportions based on taste tests"""
        # Calculate individual model performance weights
        weights = []
        for model in self.base_models.values():
            val_pred = model.predict(X_val)
            r2 = r2_score(y_val, val_pred)
            weights.append(max(0, r2))  # Ensure non-negative weights
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
        else:
            weights = [1/len(weights)] * len(weights)
        
        self.ensembles['weighted'] = {
            'models': list(self.base_models.values()),
            'weights': weights
        }
        return self.ensembles['weighted']

# Competition Pipeline - Our Complete Restaurant Kitchen
class CompetitionPipeline:
    """Complete competition pipeline like running a championship kitchen"""
    
    def __init__(self):
        self.base_chef = BaseModelChef()
        self.ensemble_chef = None
        self.best_model = None
        self.best_score = -np.inf
    
    def run_competition(self, X_train, X_test, y_train, y_test):
        """Run complete competition pipeline"""
        print("🍷 Starting Wine Quality Competition Pipeline...")
        
        # Prepare data
        X_train_scaled, X_test_scaled = self.base_chef.prepare_ingredients(X_train, X_test)
        
        # Split training data for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42
        )
        
        # Train base models
        print("\n📊 Training base models...")
        base_models = self.base_chef.train_base_models(X_tr, y_tr)
        
        # Evaluate base models
        print("\n🔍 Base Model Performance:")
        for name, model in base_models.items():
            val_pred = model.predict(X_val)
            r2 = r2_score(y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            print(f"{name:15}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
        
        # Create ensemble chef
        self.ensemble_chef = EnsembleChef(base_models)
        
        # Create and evaluate ensembles
        print("\n🍽️ Creating Ensemble Models...")
        
        # 1. Voting Ensemble
        voting_model = self.ensemble_chef.create_voting_ensemble()
        voting_model.fit(X_tr, y_tr)
        voting_pred = voting_model.predict(X_val)
        voting_score = r2_score(y_val, voting_pred)
        voting_rmse = np.sqrt(mean_squared_error(y_val, voting_pred))
        print(f"Voting Ensemble : R² = {voting_score:.4f}, RMSE = {voting_rmse:.4f}")
        
        # 2. Stacking Ensemble
        stacking_model = self.ensemble_chef.create_stacking_ensemble(X_tr, y_tr)
        stacking_pred = self.predict_stacking(X_val, stacking_model)
        stacking_score = r2_score(y_val, stacking_pred)
        stacking_rmse = np.sqrt(mean_squared_error(y_val, stacking_pred))
        print(f"Stacking Ensemble: R² = {stacking_score:.4f}, RMSE = {stacking_rmse:.4f}")
        
        # 3. Weighted Ensemble
        weighted_model = self.ensemble_chef.create_weighted_ensemble(X_val, y_val)
        weighted_pred = self.predict_weighted(X_val, weighted_model)
        weighted_score = r2_score(y_val, weighted_pred)
        weighted_rmse = np.sqrt(mean_squared_error(y_val, weighted_pred))
        print(f"Weighted Ensemble: R² = {weighted_score:.4f}, RMSE = {weighted_rmse:.4f}")
        
        # Select best model
        scores = {
            'voting': voting_score,
            'stacking': stacking_score,
            'weighted': weighted_score
        }
        
        best_ensemble = max(scores, key=scores.get)
        self.best_score = scores[best_ensemble]
        
        print(f"\n🏆 Best Ensemble: {best_ensemble.upper()} (R² = {self.best_score:.4f})")
        
        # Final evaluation on test set
        if best_ensemble == 'voting':
            final_pred = voting_model.predict(X_test_scaled)
        elif best_ensemble == 'stacking':
            final_pred = self.predict_stacking(X_test_scaled, stacking_model)
        else:
            final_pred = self.predict_weighted(X_test_scaled, weighted_model)
        
        final_score = r2_score(y_test, final_pred)
        final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
        
        print(f"\n🎯 Final Test Performance:")
        print(f"R² Score: {final_score:.4f}")
        print(f"RMSE: {final_rmse:.4f}")
        
        return {
            'best_model': best_ensemble,
            'test_score': final_score,
            'test_rmse': final_rmse,
            'predictions': final_pred
        }
    
    def predict_stacking(self, X, stacking_model):
        """Make predictions using stacking ensemble"""
        base_predictions = np.column_stack([
            model.predict(X) for model in stacking_model['base_models'].values()
        ])
        return stacking_model['meta_model'].predict(base_predictions)
    
    def predict_weighted(self, X, weighted_model):
        """Make predictions using weighted ensemble"""
        predictions = np.column_stack([
            model.predict(X) for model in weighted_model['models']
        ])
        return np.average(predictions, weights=weighted_model['weights'], axis=1)

# Advanced Competition Features
class CompetitionAnalyzer:
    """Analyze competition results like a food critic reviewing a restaurant"""
    
    def __init__(self, pipeline_results):
        self.results = pipeline_results
    
    def generate_competition_report(self):
        """Generate comprehensive competition analysis"""
        print("\n" + "="*60)
        print("🏆 COMPETITION PERFORMANCE REPORT")
        print("="*60)
        print(f"Best Model: {self.results['best_model'].upper()}")
        print(f"Test R² Score: {self.results['test_score']:.4f}")
        print(f"Test RMSE: {self.results['test_rmse']:.4f}")
        
        # Performance interpretation
        if self.results['test_score'] > 0.8:
            performance = "EXCELLENT - Championship Level! 🥇"
        elif self.results['test_score'] > 0.7:
            performance = "GOOD - Strong Competition Performance! 🥈"
        elif self.results['test_score'] > 0.6:
            performance = "FAIR - Room for Improvement 🥉"
        else:
            performance = "NEEDS WORK - Back to the Kitchen! 👨‍🍳"
        
        print(f"Performance Level: {performance}")
        
    def feature_importance_analysis(self, model, feature_names):
        """Analyze which features contribute most to predictions"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = sorted(zip(feature_names, importances), 
                                      key=lambda x: x[1], reverse=True)
            
            print("\n🔍 TOP FEATURE CONTRIBUTORS:")
            for i, (feature, importance) in enumerate(feature_importance[:5]):
                print(f"{i+1}. {feature}: {importance:.4f}")

# Execute Competition Pipeline
def run_wine_quality_competition():
    """Execute complete competition pipeline"""
    # Load data
    X_train, X_test, y_train, y_test = load_competition_data()
    
    # Initialize and run pipeline
    pipeline = CompetitionPipeline()
    results = pipeline.run_competition(X_train, X_test, y_train, y_test)
    
    # Analyze results
    analyzer = CompetitionAnalyzer(results)
    analyzer.generate_competition_report()
    
    return pipeline, results

# Run the competition
if __name__ == "__main__":
    competition_pipeline, final_results = run_wine_quality_competition()
```

## Advanced Features Implementation

```python
# Competition Optimization Extensions
class AdvancedCompetitionFeatures:
    """Advanced features for serious competition performance"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def feature_engineering_boost(self, X):
        """Create additional features like a chef creating signature ingredients"""
        X_enhanced = X.copy()
        
        # Polynomial features for key interactions
        X_enhanced['alcohol_acidity'] = X['alcohol'] * X['fixed acidity']
        X_enhanced['sugar_alcohol_ratio'] = X['residual sugar'] / (X['alcohol'] + 1e-6)
        X_enhanced['total_acidity'] = X['fixed acidity'] + X['volatile acidity']
        X_enhanced['sulfur_ratio'] = X['free sulfur dioxide'] / (X['total sulfur dioxide'] + 1e-6)
        
        return X_enhanced
    
    def cross_validation_optimization(self, X_train, y_train, cv_folds=5):
        """Robust cross-validation like multiple taste tests"""
        cv_scores = {}
        
        for name, model in self.pipeline.ensemble_chef.ensembles.items():
            if name == 'voting':
                scores = cross_val_score(model, X_train, y_train, 
                                       cv=cv_folds, scoring='r2')
                cv_scores[name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
        
        return cv_scores
    
    def competition_submission_generator(self, test_predictions):
        """Generate competition-ready submission file"""
        submission = pd.DataFrame({
            'id': range(len(test_predictions)),
            'quality_prediction': test_predictions
        })
        
        submission.to_csv('wine_quality_submission.csv', index=False)
        print("📤 Competition submission file generated: wine_quality_submission.csv")
        
        return submission

# Model Interpretability for Competition Insights
class CompetitionInsights:
    """Extract insights from competition models"""
    
    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
    
    def prediction_confidence_analysis(self):
        """Analyze prediction confidence across ensemble members"""
        predictions = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(self.X_test)
                predictions[name] = pred
        
        # Calculate prediction variance (uncertainty measure)
        pred_array = np.array(list(predictions.values()))
        prediction_variance = np.var(pred_array, axis=0)
        
        print(f"\n🎯 Prediction Confidence Analysis:")
        print(f"Average prediction variance: {prediction_variance.mean():.4f}")
        print(f"High uncertainty samples: {sum(prediction_variance > prediction_variance.mean() * 1.5)}")
        
        return prediction_variance
    
    def ensemble_diversity_score(self):
        """Calculate how diverse the ensemble members are"""
        correlations = []
        model_names = list(self.models.keys())
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                pred_i = self.models[model_names[i]].predict(self.X_test)
                pred_j = self.models[model_names[j]].predict(self.X_test)
                corr = np.corrcoef(pred_i, pred_j)[0,1]
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        diversity_score = 1 - avg_correlation
        
        print(f"\n🌟 Ensemble Diversity Score: {diversity_score:.4f}")
        print("(Higher scores indicate more diverse predictions)")
        
        return diversity_score
```

## Project Success Metrics

### Performance Targets
- **Minimum Acceptable**: R² > 0.65
- **Good Performance**: R² > 0.75  
- **Excellent Performance**: R² > 0.85

### Technical Requirements Checklist
- ✅ 4+ different base models implemented
- ✅ 3+ ensemble methods created
- ✅ Hyperparameter optimization applied
- ✅ Cross-validation implemented
- ✅ Competition-ready submission format
- ✅ Performance analysis and reporting

## Project Extensions

### Competition-Level Enhancements
1. **Advanced Feature Engineering**: Create domain-specific features
2. **Bayesian Optimization**: Use advanced hyperparameter tuning
3. **Model Calibration**: Ensure probability predictions are well-calibrated
4. **Ensemble Pruning**: Remove underperforming models automatically
5. **Real-time Prediction API**: Deploy model as web service

### Industry Applications
- **Finance**: Credit risk assessment ensembles
- **Healthcare**: Medical diagnosis prediction systems
- **E-commerce**: Customer behavior prediction
- **Manufacturing**: Quality control prediction
- **Marketing**: Customer lifetime value prediction

This project demonstrates production-ready ensemble modeling techniques that mirror real-world machine learning competitions and industrial applications.

## Assignment: Build a Movie Recommendation Ensemble

### Assignment Overview
Create a Django application that uses ensemble methods to predict movie ratings and recommend films to users.

### Requirements:

1. **Data Model**: Create models for `Movie`, `UserProfile`, and `MovieRating`
2. **Ensemble Implementation**: Use at least 3 different base models:
   - Random Forest (for genre preferences)
   - Gradient Boosting (for rating patterns)
   - SVM (for user similarity)

3. **Features to Include**:
   - Movie genres (encode as numerical features)
   - Release year
   - Director popularity score
   - Average rating
   - User's rating history
   - User demographic info

4. **Django Views**:
   - Training view: `/train-model/`
   - Prediction view: `/recommend-movies/`
   - Feedback view: `/rate-movie/`

5. **Evaluation Metrics**:
   - Mean Absolute Error for rating predictions
   - Precision@K for top-K recommendations
   - User satisfaction score

6. **Templates**:
   - Movie recommendation dashboard
   - Model performance metrics page
   - User rating interface

### Deliverables:
1. Complete Django application with working ensemble model
2. Sample dataset with at least 1000 movies and 500 user ratings
3. Performance comparison between individual models and ensemble
4. Documentation explaining your ensemble strategy

### Grading Criteria:
- **Functionality (40%)**: Working ensemble implementation
- **Code Quality (25%)**: Clean, well-documented Django code
- **Performance (20%)**: Model accuracy and efficiency
- **User Experience (15%)**: Intuitive web interface

---

## Course Summary

You've now mastered the art of ensemble methods - combining multiple machine learning models like a master chef orchestrates a world-class kitchen team. You understand:

- **Bagging**: Independent parallel processing (multiple chefs working simultaneously)
- **Boosting**: Sequential learning from mistakes (apprentices learning from masters)
- **Random Forest**: Specialized teams with diverse expertise
- **Gradient Boosting**: Advanced sequential training with XGBoost and LightGBM
- **Voting/Stacking**: Democratic and hierarchical decision-making systems

Your Django implementation shows how to deploy these powerful ensemble methods in production web applications, creating systems that are more accurate, robust, and reliable than any single model could achieve alone.

Remember: Just as the best restaurants have teams of specialized chefs working together, the best machine learning systems use ensembles of specialized models. You're now