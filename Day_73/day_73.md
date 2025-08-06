# AI Mastery Course: Supervised Learning Algorithms with Django
## Day 73: From Kitchen Basics to Master Chef - Supervised Learning Algorithms

**Imagine that...** you're stepping into a world-class culinary school where every dish you learn to prepare represents a different way of understanding and predicting patterns in data. Today, you'll master four fundamental cooking techniques that every data chef needs in their repertoire - each algorithm is like a different cooking method that transforms raw ingredients (data) into delicious insights.

---

## Learning Objective
By the end of this lesson, you will be able to implement and compare four core supervised learning algorithms using Django and scikit-learn, understanding when to use each "cooking technique" based on your data ingredients and desired outcomes.

---

## Lesson 1: Linear and Logistic Regression - The Foundation Sauces

Think of linear regression as making a basic roux - the fundamental sauce base that many dishes build upon. Just as a roux combines flour and butter in perfect proportions to create a smooth foundation, linear regression finds the perfect line that best fits through your data points.

Logistic regression is like making a reduction sauce - you're taking your base ingredients and transforming them into something that gives you a clear yes/no answer, just like how a reduction concentrates flavors into a decisive taste.

### Code Implementation

**models.py**
```python
from django.db import models

class HousePriceData(models.Model):
    size_sqft = models.FloatField()
    bedrooms = models.IntegerField()
    bathrooms = models.FloatField()
    age_years = models.IntegerField()
    price = models.FloatField()
    is_sold = models.BooleanField(default=False)
    
    def __str__(self):
        return f"House {self.id}: {self.size_sqft} sqft, ${self.price}"
```

**views.py**
```python
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
import numpy as np
from .models import HousePriceData

def linear_regression_view(request):
    # Get data from database - our raw ingredients
    houses = HousePriceData.objects.all().values()
    df = pd.DataFrame(houses)
    
    if len(df) < 10:  # Need enough ingredients to cook!
        return JsonResponse({'error': 'Need at least 10 house records'})
    
    # Prepare ingredients (features and target)
    X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]
    y = df['price']
    
    # Split ingredients like prep work in kitchen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create our roux (model) and cook it (fit)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Taste test (predict)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Serve the results
    context = {
        'model_type': 'Linear Regression',
        'mse': round(mse, 2),
        'coefficients': dict(zip(X.columns, model.coef_)),
        'intercept': model.intercept_,
        'sample_predictions': list(zip(y_test.iloc[:5], predictions[:5]))
    }
    
    return render(request, 'ml_results.html', context)

def logistic_regression_view(request):
    # Same prep work, different cooking method
    houses = HousePriceData.objects.all().values()
    df = pd.DataFrame(houses)
    
    if len(df) < 10:
        return JsonResponse({'error': 'Need at least 10 house records'})
    
    # Prepare ingredients for yes/no prediction (sold or not)
    X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]
    y = df['is_sold']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create reduction sauce (logistic model)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Taste and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    context = {
        'model_type': 'Logistic Regression',
        'accuracy': round(accuracy * 100, 2),
        'coefficients': dict(zip(X.columns, model.coef_[0])),
        'sample_predictions': list(zip(y_test.iloc[:5], predictions[:5]))
    }
    
    return render(request, 'ml_results.html', context)
```

### Syntax Explanation
- `from sklearn.linear_model import LinearRegression, LogisticRegression`: Importing our cooking tools
- `train_test_split()`: Dividing ingredients into practice portions and final test portions
- `model.fit(X_train, y_train)`: Teaching the model using training data (like learning a recipe)
- `model.predict()`: Using the learned recipe on new ingredients
- `mean_squared_error()` and `accuracy_score()`: Quality control measures

---

## Lesson 2: Decision Trees and Random Forests - The Recipe Decision Maker

A decision tree is like a master chef's decision-making process when tasting a dish: "Is it too salty? If yes, add acid. If no, is it too bland? If yes, add seasoning..." Each branch represents a yes/no question that leads to a final decision.

Random Forest is like having multiple expert chefs taste your dish and vote on what needs to be adjusted - the majority opinion usually gives you the best result.

### Code Implementation

**views.py (continued)**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def decision_tree_view(request):
    houses = HousePriceData.objects.all().values()
    df = pd.DataFrame(houses)
    
    if len(df) < 10:
        return JsonResponse({'error': 'Need at least 10 house records'})
    
    # Create price categories (our taste categories)
    df['price_category'] = pd.cut(df['price'], 
                                 bins=[0, 200000, 400000, float('inf')], 
                                 labels=['Budget', 'Mid-range', 'Luxury'])
    
    X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]
    y = df['price_category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Single chef making decisions
    tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    tree_model.fit(X_train, y_train)
    tree_predictions = tree_model.predict(X_test)
    tree_accuracy = accuracy_score(y_test, tree_predictions)
    
    # Committee of chef experts
    forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    forest_model.fit(X_train, y_train)
    forest_predictions = forest_model.predict(X_test)
    forest_accuracy = accuracy_score(y_test, forest_predictions)
    
    context = {
        'tree_accuracy': round(tree_accuracy * 100, 2),
        'forest_accuracy': round(forest_accuracy * 100, 2),
        'tree_feature_importance': dict(zip(X.columns, tree_model.feature_importances_)),
        'forest_feature_importance': dict(zip(X.columns, forest_model.feature_importances_)),
        'sample_predictions': list(zip(y_test.iloc[:5], tree_predictions[:5], forest_predictions[:5]))
    }
    
    return render(request, 'tree_results.html', context)
```

### Syntax Explanation
- `DecisionTreeClassifier(max_depth=5)`: Setting how many questions deep our chef can go
- `RandomForestClassifier(n_estimators=100)`: Creating 100 expert chefs to vote
- `feature_importances_`: Shows which ingredients (features) matter most in decisions
- `pd.cut()`: Creating categories like a chef organizing ingredients by type

---

## Lesson 3: Support Vector Machines - The Perfect Knife Cut

SVM is like finding the perfect knife angle and position to make the cleanest cut through ingredients. It finds the optimal boundary (hyperplane) that separates different types of data with the maximum margin - just like how a skilled chef finds the perfect cutting technique that cleanly separates ingredients with maximum efficiency.

### Code Implementation

**views.py (continued)**
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def svm_view(request):
    houses = HousePriceData.objects.all().values()
    df = pd.DataFrame(houses)
    
    if len(df) < 10:
        return JsonResponse({'error': 'Need at least 10 house records'})
    
    # Prepare ingredients - SVM needs evenly sized ingredients (scaling)
    X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]
    y = df['is_sold']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale ingredients to same size (like julienning everything uniformly)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Find the perfect cutting angle (SVM with different kernels)
    svm_linear = SVC(kernel='linear', random_state=42)
    svm_rbf = SVC(kernel='rbf', random_state=42)
    
    # Train both cutting techniques
    svm_linear.fit(X_train_scaled, y_train)
    svm_rbf.fit(X_train_scaled, y_train)
    
    # Test both techniques
    linear_predictions = svm_linear.predict(X_test_scaled)
    rbf_predictions = svm_rbf.predict(X_test_scaled)
    
    linear_accuracy = accuracy_score(y_test, linear_predictions)
    rbf_accuracy = accuracy_score(y_test, rbf_predictions)
    
    context = {
        'linear_accuracy': round(linear_accuracy * 100, 2),
        'rbf_accuracy': round(rbf_accuracy * 100, 2),
        'support_vectors_linear': svm_linear.n_support_,
        'support_vectors_rbf': svm_rbf.n_support_,
        'sample_predictions': list(zip(y_test.iloc[:5], linear_predictions[:5], rbf_predictions[:5]))
    }
    
    return render(request, 'svm_results.html', context)
```

### Syntax Explanation
- `StandardScaler()`: Ensures all ingredients are the same "size" for fair comparison
- `SVC(kernel='linear')`: Using a straight knife cut
- `SVC(kernel='rbf')`: Using a curved, flexible cutting technique
- `n_support_`: Number of critical data points that define the cutting boundary

---

## Lesson 4: k-Nearest Neighbors - The Neighborhood Taste Test

k-NN is like asking your k closest neighbor chefs what they think about a dish. If you want to know if a new recipe will be popular, you ask the 5 most similar chefs in your neighborhood, and whatever the majority says, that's your prediction. It's democracy in the kitchen!

### Code Implementation

**views.py (continued)**
```python
from sklearn.neighbors import KNeighborsClassifier

def knn_view(request):
    houses = HousePriceData.objects.all().values()
    df = pd.DataFrame(houses)
    
    if len(df) < 10:
        return JsonResponse({'error': 'Need at least 10 house records'})
    
    df['price_category'] = pd.cut(df['price'], 
                                 bins=[0, 200000, 400000, float('inf')], 
                                 labels=['Budget', 'Mid-range', 'Luxury'])
    
    X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]
    y = df['price_category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale ingredients for fair neighbor comparison
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different neighborhood sizes
    accuracies = {}
    for k in [3, 5, 7, 9]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        predictions = knn.predict(X_test_scaled)
        accuracies[f'k={k}'] = round(accuracy_score(y_test, predictions) * 100, 2)
    
    # Use best k for final model
    best_k = 5
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train_scaled, y_train)
    final_predictions = final_knn.predict(X_test_scaled)
    
    context = {
        'accuracies': accuracies,
        'best_k': best_k,
        'final_accuracy': accuracies[f'k={best_k}'],
        'sample_predictions': list(zip(y_test.iloc[:5], final_predictions[:5]))
    }
    
    return render(request, 'knn_results.html', context)
```

### Syntax Explanation
- `KNeighborsClassifier(n_neighbors=k)`: Setting how many neighbor chefs to ask
- The for loop tests different neighborhood sizes to find the best one
- Scaling is crucial here because distance matters in finding neighbors

---

## Final Quality Project: Restaurant Success Predictor

Build a Django web application that predicts restaurant success using all four algorithms. Your app should:

1. **Data Model**: Create a `Restaurant` model with features like location rating, cuisine type, price range, reviews count, etc.
2. **Algorithm Comparison Page**: Show results from all four algorithms side by side
3. **Interactive Prediction**: Allow users to input new restaurant data and get predictions
4. **Visualization Dashboard**: Display model performance metrics and feature importance

**Key Project Components:**

```python
# models.py
class Restaurant(models.Model):
    name = models.CharField(max_length=200)
    location_rating = models.FloatField()  # 1-10 scale
    cuisine_type = models.CharField(max_length=100)
    avg_price = models.FloatField()
    reviews_count = models.IntegerField()
    years_open = models.IntegerField()
    has_parking = models.BooleanField()
    is_successful = models.BooleanField()  # Our target variable
    
    def __str__(self):
        return f"{self.name} - {'Successful' if self.is_successful else 'Struggling'}"
```

This project locks in all concepts by requiring you to:
- Implement all four algorithms
- Compare their performance
- Handle real-world data preprocessing
- Create user interfaces
- Make business decisions based on model results

---

# Multi-Algorithm Classifier Comparison Project

## Final Quality Project: Restaurant Menu Classifier

Think of this project as creating an intelligent kitchen assistant that can classify different types of dishes based on their ingredients and characteristics. Just like how an experienced chef can look at ingredients and instantly know what type of cuisine they're making, our classifier will analyze data features and predict categories.

### Project Overview

We'll build a comprehensive comparison system that tests multiple classification algorithms on the same dataset, just like having different chefs compete to create the best dish using the same ingredients.

### Complete Django Implementation

```python
# models.py
from django.db import models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_wine
import joblib

class ClassifierComparison(models.Model):
    name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=50)
    accuracy = models.FloatField()
    cross_val_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.algorithm}: {self.accuracy:.4f}"

class DatasetManager:
    """Think of this as our head chef who manages all the ingredients (data)"""
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def prepare_ingredients(self):
        """Like prepping ingredients before cooking"""
        # Load wine dataset (our ingredients)
        wine = load_wine()
        X, y = wine.data, wine.target
        
        # Split ingredients for training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale ingredients (like standardizing measurements in recipes)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return {
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'features': len(wine.feature_names),
            'classes': len(wine.target_names)
        }

class ChefClassifier:
    """Each algorithm is like a different chef with their own cooking style"""
    
    def __init__(self, name, algorithm):
        self.name = name
        self.algorithm = algorithm
        self.model = None
        self.is_trained = False
        
    def train_chef(self, X_train, y_train):
        """Train each chef with the same ingredients"""
        self.model = self.algorithm
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
    def chef_prediction(self, X_test):
        """Let each chef make their prediction"""
        if not self.is_trained:
            raise ValueError(f"Chef {self.name} needs training first!")
        return self.model.predict(X_test)
    
    def evaluate_chef(self, X_test, y_test, X_train, y_train):
        """Evaluate how well each chef performed"""
        predictions = self.chef_prediction(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Cross-validation score (like multiple taste tests)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        
        return {
            'accuracy': accuracy,
            'cv_score': cv_mean,
            'cv_std': cv_scores.std(),
            'predictions': predictions,
            'classification_report': classification_report(y_test, predictions)
        }

# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from .models import ClassifierComparison, DatasetManager, ChefClassifier
import json

class KitchenCompetitionView(View):
    """Main view that orchestrates our chef competition"""
    
    def get(self, request):
        return render(request, 'classifier_comparison.html')
    
    def post(self, request):
        # Initialize our kitchen (data manager)
        kitchen = DatasetManager()
        dataset_info = kitchen.prepare_ingredients()
        
        # Our team of chef algorithms
        chef_algorithms = {
            'Precise Chef (Logistic Regression)': LogisticRegression(random_state=42, max_iter=1000),
            'Tree Master (Decision Tree)': DecisionTreeClassifier(random_state=42),
            'Forest Guardian (Random Forest)': RandomForestClassifier(n_estimators=100, random_state=42),
            'Boundary Expert (SVM)': SVC(random_state=42),
            'Neighbor Whisperer (k-NN)': KNeighborsClassifier(n_neighbors=5)
        }
        
        results = []
        best_chef = None
        best_accuracy = 0
        
        # Let each chef compete
        for chef_name, algorithm in chef_algorithms.items():
            chef = ChefClassifier(chef_name, algorithm)
            
            # Train the chef
            chef.train_chef(kitchen.X_train_scaled, kitchen.y_train)
            
            # Evaluate performance
            performance = chef.evaluate_chef(
                kitchen.X_test_scaled, 
                kitchen.y_test,
                kitchen.X_train_scaled,
                kitchen.y_train
            )
            
            # Save to database
            comparison = ClassifierComparison.objects.create(
                name=chef_name,
                algorithm=chef_name.split('(')[1].split(')')[0],
                accuracy=performance['accuracy'],
                cross_val_score=performance['cv_score']
            )
            
            results.append({
                'name': chef_name,
                'accuracy': performance['accuracy'],
                'cv_score': performance['cv_score'],
                'cv_std': performance['cv_std'],
                'classification_report': performance['classification_report']
            })
            
            # Track the winning chef
            if performance['accuracy'] > best_accuracy:
                best_accuracy = performance['accuracy']
                best_chef = chef_name
        
        # Sort results by accuracy (descending)
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return JsonResponse({
            'status': 'success',
            'dataset_info': dataset_info,
            'results': results,
            'winner': best_chef,
            'winner_accuracy': best_accuracy
        })

class ComparisonHistoryView(View):
    """View previous competition results"""
    
    def get(self, request):
        comparisons = ClassifierComparison.objects.all().order_by('-created_at')[:50]
        
        history_data = []
        for comp in comparisons:
            history_data.append({
                'algorithm': comp.algorithm,
                'accuracy': comp.accuracy,
                'cv_score': comp.cross_val_score,
                'date': comp.created_at.strftime('%Y-%m-%d %H:%M')
            })
        
        return JsonResponse({'history': history_data})

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('classifier-comparison/', views.KitchenCompetitionView.as_view(), name='classifier_comparison'),
    path('comparison-history/', views.ComparisonHistoryView.as_view(), name='comparison_history'),
]

# templates/classifier_comparison.html
<!DOCTYPE html>
<html>
<head>
    <title>AI Kitchen: Chef Algorithm Competition</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .kitchen-container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .chef-card { border: 2px solid #ddd; margin: 10px; padding: 15px; border-radius: 8px; }
        .winner { border-color: #gold; background: #fff9c4; }
        .loading { text-align: center; padding: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric-box { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
        button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        #results-chart { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="kitchen-container">
        <h1>🍳 AI Kitchen: Chef Algorithm Competition</h1>
        <p>Watch as different algorithm chefs compete to classify wine types with the highest accuracy!</p>
        
        <button onclick="startCompetition()">🏁 Start Chef Competition</button>
        <button onclick="loadHistory()">📊 View History</button>
        
        <div id="loading" class="loading" style="display: none;">
            <h3>🔥 Chefs are cooking... Please wait!</h3>
        </div>
        
        <div id="dataset-info" style="display: none;">
            <h2>🥘 Kitchen Information</h2>
            <div class="metrics" id="dataset-metrics"></div>
        </div>
        
        <div id="results-container" style="display: none;">
            <h2>🏆 Competition Results</h2>
            <div id="chef-results"></div>
            <canvas id="results-chart" width="400" height="200"></canvas>
        </div>
        
        <div id="history-container" style="display: none;">
            <h2>📈 Historical Performance</h2>
            <div id="history-content"></div>
        </div>
    </div>

    <script>
        let resultsChart = null;
        
        async function startCompetition() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results-container').style.display = 'none';
            document.getElementById('dataset-info').style.display = 'none';
            
            try {
                const response = await fetch('/classifier-comparison/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')
                    }
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    displayDatasetInfo(data.dataset_info);
                    displayResults(data.results, data.winner);
                    createChart(data.results);
                }
            } catch (error) {
                console.error('Competition failed:', error);
                alert('Kitchen disaster! Please try again.');
            }
            
            document.getElementById('loading').style.display = 'none';
        }
        
        function displayDatasetInfo(info) {
            const metricsDiv = document.getElementById('dataset-metrics');
            metricsDiv.innerHTML = `
                <div class="metric-box">
                    <h3>🍷 Wine Samples</h3>
                    <p>Training: ${info.train_size}<br>Testing: ${info.test_size}</p>
                </div>
                <div class="metric-box">
                    <h3>🧪 Features</h3>
                    <p>${info.features} characteristics</p>
                </div>
                <div class="metric-box">
                    <h3>🎯 Wine Classes</h3>
                    <p>${info.classes} types</p>
                </div>
            `;
            document.getElementById('dataset-info').style.display = 'block';
        }
        
        function displayResults(results, winner) {
            const resultsDiv = document.getElementById('chef-results');
            let html = '';
            
            results.forEach((result, index) => {
                const isWinner = result.name === winner;
                const medal = index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '👨‍🍳';
                
                html += `
                    <div class="chef-card ${isWinner ? 'winner' : ''}">
                        <h3>${medal} ${result.name}</h3>
                        <div class="metrics">
                            <div class="metric-box">
                                <strong>Accuracy</strong><br>
                                ${(result.accuracy * 100).toFixed(2)}%
                            </div>
                            <div class="metric-box">
                                <strong>Cross-Validation</strong><br>
                                ${(result.cv_score * 100).toFixed(2)}% ± ${(result.cv_std * 100).toFixed(2)}%
                            </div>
                        </div>
                        <details>
                            <summary>📝 Detailed Report</summary>
                            <pre>${result.classification_report}</pre>
                        </details>
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
            document.getElementById('results-container').style.display = 'block';
        }
        
        function createChart(results) {
            const ctx = document.getElementById('results-chart').getContext('2d');
            
            if (resultsChart) {
                resultsChart.destroy();
            }
            
            resultsChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: results.map(r => r.name.split('(')[0].trim()),
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: results.map(r => (r.accuracy * 100).toFixed(2)),
                        backgroundColor: [
                            '#FFD700', '#C0C0C0', '#CD7F32', '#87CEEB', '#DDA0DD'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: { display: true, text: 'Accuracy (%)' }
                        }
                    },
                    plugins: {
                        title: { display: true, text: 'Chef Performance Comparison' }
                    }
                }
            });
        }
        
        async function loadHistory() {
            try {
                const response = await fetch('/comparison-history/');
                const data = await response.json();
                
                let html = '<table border="1" style="width:100%; border-collapse: collapse;">';
                html += '<tr><th>Algorithm</th><th>Accuracy</th><th>CV Score</th><th>Date</th></tr>';
                
                data.history.forEach(item => {
                    html += `<tr>
                        <td>${item.algorithm}</td>
                        <td>${(item.accuracy * 100).toFixed(2)}%</td>
                        <td>${(item.cv_score * 100).toFixed(2)}%</td>
                        <td>${item.date}</td>
                    </tr>`;
                });
                
                html += '</table>';
                document.getElementById('history-content').innerHTML = html;
                document.getElementById('history-container').style.display = 'block';
            } catch (error) {
                console.error('Failed to load history:', error);
            }
        }
        
        function getCookie(name) {
            let cookieValue = null;
            if (document.cookie && document.cookie !== '') {
                const cookies = document.cookie.split(';');
                for (let i = 0; i < cookies.length; i++) {
                    const cookie = cookies[i].trim();
                    if (cookie.substring(0, name.length + 1) === (name + '=')) {
                        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                        break;
                    }
                }
            }
            return cookieValue;
        }
    </script>
</body>
</html>

## Assignment: Personal Recipe Recommender

**Task**: Create a Django application that recommends recipes based on user preferences using k-NN algorithm.

**Requirements**:
1. Create a `Recipe` model with fields: name, cooking_time, difficulty_level, main_ingredient, cuisine_type, rating
2. Create a `UserPreference` model that stores user's preferred cooking time, difficulty, and cuisine
3. Implement a k-NN classifier that recommends the top 5 most similar recipes based on user preferences
4. Create a simple web interface where users can input their preferences and see recommendations
5. Display why each recipe was recommended (which features matched)

**Deliverable**: A working Django app with at least 20 sample recipes and a functional recommendation system.

**Grading Criteria**:
- Correct k-NN implementation (40%)
- Proper Django model relationships (30%)
- User interface functionality (20%)
- Code quality and comments (10%)

This assignment differs from the main project by focusing on recommendation systems rather than classification comparison, and uses a single algorithm in depth rather than comparing multiple algorithms.