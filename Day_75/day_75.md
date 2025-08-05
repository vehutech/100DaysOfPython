# AI Mastery Course - Day 75: Unsupervised Learning with Django

## Learning Objective
By the end of this lesson, you will master the art of unsupervised learning algorithms and implement them in a Django web application, enabling you to discover hidden patterns in data without labeled examples - just like a master chef who can identify flavor profiles and create new recipes by understanding the underlying ingredients and their relationships.

---

## Introduction: Imagine That...

Imagine that you're a world-renowned chef who has just inherited a mysterious spice collection from your grandmother. The spices are unlabeled, but you need to organize them, understand their relationships, and create new recipes. You don't have a cookbook (labeled data) to guide you, but through your expertise, you can group similar spices together, identify the most important flavor compounds, and even spot unusual or exotic spices that don't fit typical patterns.

This is exactly what unsupervised learning does with data - it finds hidden patterns, groups similar items, reduces complexity, and identifies outliers, all without knowing the "correct" answers beforehand.

---

## Lesson 1: K-means Clustering - The Spice Organization System

**Kitchen Analogy**: K-means is like organizing your spice collection into specific jars. You decide you want exactly 4 jars (clusters), and you group spices based on their characteristics - sweet spices in one jar, hot spices in another, herbs in the third, and exotic spices in the fourth.

### Code Example: Basic K-means Implementation

```python
# views.py - Django view for K-means clustering
from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

def kmeans_clustering_view(request):
    """
    Django view that performs K-means clustering on sample data
    Like organizing spices into predefined jars
    """
    if request.method == 'POST':
        # Sample data: imagine these are spice characteristics
        # [sweetness, spiciness, aroma_intensity, price_per_gram]
        spice_data = np.array([
            [8, 1, 6, 12],  # Vanilla
            [9, 0, 8, 15],  # Cinnamon  
            [1, 9, 7, 8],   # Cayenne
            [0, 8, 5, 6],   # Paprika
            [2, 0, 9, 25],  # Saffron
            [1, 1, 8, 20],  # Truffle salt
            [7, 2, 4, 5],   # Brown sugar
            [8, 1, 5, 3],   # White sugar
        ])
        
        spice_names = ['Vanilla', 'Cinnamon', 'Cayenne', 'Paprika', 
                      'Saffron', 'Truffle Salt', 'Brown Sugar', 'White Sugar']
        
        # Standardize the features (like converting all measurements to same scale)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(spice_data)
        
        # Apply K-means clustering (organizing into 3 jars)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        colors = ['red', 'blue', 'green']
        
        for i in range(3):
            cluster_points = scaled_data[clusters == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i], label=f'Spice Group {i+1}', s=100)
        
        # Add spice names as labels
        for i, name in enumerate(spice_names):
            plt.annotate(name, (scaled_data[i, 0], scaled_data[i, 1]))
        
        plt.title('Spice Organization using K-means Clustering')
        plt.xlabel('Sweetness-Spiciness Profile (standardized)')
        plt.ylabel('Aroma Intensity (standardized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64 string for web display
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Organize results by cluster (like organizing spice jars)
        cluster_groups = {}
        for i, spice in enumerate(spice_names):
            cluster_id = clusters[i]
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(spice)
        
        return JsonResponse({
            'success': True,
            'plot_image': plot_data,
            'spice_groups': cluster_groups,
            'cluster_centers': kmeans.cluster_centers_.tolist()
        })
    
    return render(request, 'clustering/kmeans.html')
```

**Syntax Explanation**:
- `StandardScaler()`: Normalizes data so all features have similar scales (like converting pounds and grams to the same unit)
- `KMeans(n_clusters=3)`: Creates 3 distinct groups
- `fit_predict()`: Both learns the patterns and assigns each item to a cluster
- `random_state=42`: Ensures reproducible results
- `n_init=10`: Runs the algorithm 10 times and picks the best result

---

## Lesson 2: Hierarchical Clustering - The Recipe Family Tree

**Kitchen Analogy**: Hierarchical clustering is like creating a family tree of recipes. You start with individual dishes and gradually group them - first by cuisine type, then by cooking method, then by main ingredient - creating a tree-like structure of relationships.

### Code Example: Hierarchical Clustering Implementation

```python
# views.py - Hierarchical clustering view
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

def hierarchical_clustering_view(request):
    """
    Creates a recipe family tree using hierarchical clustering
    Like organizing recipes by their relationships and similarities
    """
    if request.method == 'POST':
        # Recipe characteristics: [prep_time, cook_time, difficulty, spice_level]
        recipe_data = np.array([
            [15, 30, 2, 1],  # Pasta Carbonara
            [20, 25, 3, 2],  # Chicken Tikka
            [10, 15, 1, 0],  # Caesar Salad
            [30, 60, 4, 3],  # Beef Curry
            [25, 45, 3, 2],  # Fish Tacos
            [5, 10, 1, 1],   # Fruit Smoothie
            [40, 90, 5, 1],  # Beef Wellington
            [20, 30, 2, 2],  # Stir Fry
        ])
        
        recipe_names = ['Pasta Carbonara', 'Chicken Tikka', 'Caesar Salad',
                       'Beef Curry', 'Fish Tacos', 'Fruit Smoothie', 
                       'Beef Wellington', 'Stir Fry']
        
        # Calculate distances between recipes (how different they are)
        distances = pdist(recipe_data, metric='euclidean')
        
        # Create hierarchical clusters using 'ward' method
        # Ward minimizes within-cluster variance (groups similar recipes)
        linkage_matrix = linkage(distances, method='ward')
        
        # Create dendrogram (family tree visualization)
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=recipe_names, 
                  orientation='top', distance_sort='descending')
        plt.title('Recipe Family Tree - Hierarchical Clustering')
        plt.xlabel('Recipes')
        plt.ylabel('Distance (Dissimilarity)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64 for web display
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return JsonResponse({
            'success': True,
            'dendrogram': plot_data,
            'linkage_matrix': linkage_matrix.tolist(),
            'recipe_relationships': 'Recipes are grouped by similarity in preparation time, difficulty, and spice level'
        })
    
    return render(request, 'clustering/hierarchical.html')
```

**Syntax Explanation**:
- `pdist()`: Calculates pairwise distances between all recipes
- `linkage()`: Creates the hierarchical structure using 'ward' method
- `method='ward'`: Minimizes within-cluster variance (keeps similar items together)
- `dendrogram()`: Visualizes the hierarchy as a tree structure
- `orientation='top'`: Makes the tree grow upward for better readability

---

## Lesson 3: Principal Component Analysis (PCA) - The Flavor Essence Extractor

**Kitchen Analogy**: PCA is like a master chef who can identify the 2-3 most important flavor compounds that define a dish's character. Instead of tracking 20 different spices, you focus on the essential flavor profiles that capture 90% of what makes each dish unique.

### Code Example: PCA Implementation

```python
# views.py - PCA analysis view
from sklearn.decomposition import PCA

def pca_analysis_view(request):
    """
    Reduces recipe complexity to essential flavor profiles
    Like identifying the core essence of what makes dishes unique
    """
    if request.method == 'POST':
        # Complex recipe data with many features (ingredients)
        # [sweet, salty, sour, bitter, umami, spicy, aromatic, texture, temperature]
        recipe_profiles = np.array([
            [7, 2, 1, 0, 2, 1, 8, 6, 8],  # Chocolate Cake
            [1, 8, 2, 0, 7, 0, 4, 3, 9],  # Beef Steak
            [2, 6, 8, 1, 3, 2, 6, 4, 2],  # Greek Salad
            [8, 1, 3, 2, 1, 0, 9, 7, 5],  # Apple Pie
            [0, 7, 1, 3, 8, 1, 5, 2, 9],  # Mushroom Risotto
            [1, 5, 0, 4, 6, 8, 3, 1, 9],  # Spicy Curry
            [6, 3, 4, 0, 2, 0, 7, 8, 3],  # Fruit Tart
            [2, 9, 0, 1, 9, 3, 2, 4, 8],  # BBQ Ribs
        ])
        
        recipe_names = ['Chocolate Cake', 'Beef Steak', 'Greek Salad', 
                       'Apple Pie', 'Mushroom Risotto', 'Spicy Curry',
                       'Fruit Tart', 'BBQ Ribs']
        
        # Apply PCA to reduce 9 dimensions to 2 main flavor profiles
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(recipe_profiles)
        
        # Create visualization of simplified flavor space
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=100, alpha=0.7)
        
        # Add recipe labels
        for i, name in enumerate(recipe_names):
            plt.annotate(name, (reduced_data[i, 0], reduced_data[i, 1]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Recipe Flavor Profiles - PCA Analysis')
        plt.xlabel(f'Primary Flavor Profile (explains {pca.explained_variance_ratio_[0]:.1%} of variation)')
        plt.ylabel(f'Secondary Flavor Profile (explains {pca.explained_variance_ratio_[1]:.1%} of variation)')
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Show feature importance (which original flavors matter most)
        feature_names = ['Sweet', 'Salty', 'Sour', 'Bitter', 'Umami', 
                        'Spicy', 'Aromatic', 'Texture', 'Temperature']
        
        components_df = pd.DataFrame(
            pca.components_.T,
            columns=['Primary Profile', 'Secondary Profile'],
            index=feature_names
        )
        
        return JsonResponse({
            'success': True,
            'plot_image': plot_data,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': sum(pca.explained_variance_ratio_),
            'feature_importance': components_df.to_dict(),
            'reduced_coordinates': reduced_data.tolist()
        })
    
    return render(request, 'clustering/pca.html')
```

**Syntax Explanation**:
- `PCA(n_components=2)`: Reduces data to 2 main dimensions
- `fit_transform()`: Learns the principal components and transforms the data
- `explained_variance_ratio_`: Shows how much information each component captures
- `components_`: Shows which original features contribute to each principal component
- `.T`: Transposes the matrix for easier interpretation

---

## Lesson 4: DBSCAN and Anomaly Detection - The Quality Control Inspector

**Kitchen Analogy**: DBSCAN is like a quality control inspector in your kitchen who can identify clusters of perfectly cooked dishes while also spotting the burnt ones, undercooked items, or unusual experimental dishes that don't fit any standard category.

### Code Example: DBSCAN Implementation

```python
# views.py - DBSCAN and anomaly detection
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def dbscan_anomaly_view(request):
    """
    Identifies natural groupings and outliers in cooking data
    Like a quality inspector finding normal dish clusters and unusual items
    """
    if request.method == 'POST':
        # Cooking data: [cooking_time, temperature, customer_rating, cost]
        cooking_data = np.array([
            [25, 180, 4.5, 12],  # Normal pasta dish
            [30, 175, 4.7, 14],  # Normal pasta dish
            [28, 185, 4.3, 13],  # Normal pasta dish
            [45, 200, 4.8, 18],  # Normal meat dish
            [50, 195, 4.6, 20],  # Normal meat dish
            [48, 205, 4.9, 19],  # Normal meat dish
            [15, 160, 4.2, 8],   # Normal salad
            [12, 155, 4.4, 9],   # Normal salad
            [10, 150, 4.1, 7],   # Normal salad
            [90, 220, 2.1, 25],  # OUTLIER: Overcooked, expensive
            [5, 300, 1.8, 30],   # OUTLIER: Burnt, very expensive
            [120, 100, 3.8, 5],  # OUTLIER: Long time, low temp, cheap
        ])
        
        dish_names = ['Pasta A', 'Pasta B', 'Pasta C', 'Meat A', 'Meat B', 
                     'Meat C', 'Salad A', 'Salad B', 'Salad C', 
                     'Overcooked Special', 'Burnt Experiment', 'Mystery Dish']
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cooking_data)
        
        # Apply DBSCAN clustering
        # eps: maximum distance between points in same cluster
        # min_samples: minimum points needed to form a cluster
        dbscan = DBSCAN(eps=0.8, min_samples=2)
        clusters = dbscan.fit_predict(scaled_data)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot normal clusters
        unique_clusters = set(clusters)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, cluster in enumerate(unique_clusters):
            if cluster == -1:  # Outliers
                outlier_points = scaled_data[clusters == cluster]
                plt.scatter(outlier_points[:, 0], outlier_points[:, 2], 
                           c='black', marker='x', s=200, label='Anomalies (Quality Issues)')
            else:  # Normal clusters
                cluster_points = scaled_data[clusters == cluster]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 2], 
                           c=colors[i % len(colors)], s=100, 
                           label=f'Normal Group {cluster + 1}')
        
        # Add dish labels
        for i, name in enumerate(dish_names):
            plt.annotate(name, (scaled_data[i, 0], scaled_data[i, 2]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title('Kitchen Quality Control - DBSCAN Anomaly Detection')
        plt.xlabel('Cooking Time (standardized)')
        plt.ylabel('Customer Rating (standardized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Identify anomalies
        anomaly_indices = np.where(clusters == -1)[0]
        anomalies = [dish_names[i] for i in anomaly_indices]
        
        # Count clusters
        normal_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        return JsonResponse({
            'success': True,
            'plot_image': plot_data,
            'anomalies_detected': anomalies,
            'normal_clusters_found': normal_clusters,
            'total_anomalies': len(anomalies),
            'cluster_assignments': clusters.tolist(),
            'quality_report': f'Found {normal_clusters} normal dish categories and {len(anomalies)} quality issues'
        })
    
    return render(request, 'clustering/dbscan.html')
```

**Syntax Explanation**:
- `DBSCAN(eps=0.8, min_samples=2)`: 
  - `eps`: Maximum distance between points in the same neighborhood
  - `min_samples`: Minimum points needed to form a dense region
- `clusters == -1`: DBSCAN labels outliers/anomalies as -1
- `np.where()`: Finds indices where condition is true
- `set(clusters)`: Gets unique cluster labels

---

## Final Quality Project: Restaurant Menu Intelligence System

Now let's combine all these techniques into a comprehensive Django application that helps restaurant managers understand their menu performance and customer preferences.

### Project Structure

```python
# models.py - Django models for our restaurant system
from django.db import models

class MenuItem(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=8, decimal_places=2)
    prep_time = models.IntegerField()  # minutes
    calories = models.IntegerField()
    spice_level = models.IntegerField(choices=[(i, i) for i in range(1, 6)])
    cuisine_type = models.CharField(max_length=50)
    created_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class OrderData(models.Model):
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE)
    quantity_ordered = models.IntegerField()
    customer_rating = models.FloatField()
    order_time = models.DateTimeField()
    customer_age_group = models.CharField(max_length=20)
    season = models.CharField(max_length=10)
    
    def __str__(self):
        return f"{self.menu_item.name} - {self.quantity_ordered} ordered"
```

```python
# views.py - Main intelligence system view
from django.shortcuts import render
from django.http import JsonResponse
from .models import MenuItem, OrderData
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def restaurant_intelligence_dashboard(request):
    """
    Complete restaurant intelligence system using all unsupervised learning techniques
    Like having a master chef analyst for your entire restaurant operation
    """
    if request.method == 'POST':
        analysis_type = request.POST.get('analysis_type', 'overview')
        
        # Get sample data (in real app, this would come from database)
        menu_features = np.array([
            [15.99, 25, 450, 2, 1],  # Italian Pasta - [price, prep_time, calories, spice, cuisine_encoded]
            [22.50, 35, 650, 3, 2],  # Indian Curry
            [12.99, 15, 350, 1, 3],  # American Salad
            [28.99, 45, 800, 4, 2],  # Spicy Indian Dish
            [18.75, 30, 500, 2, 1],  # Italian Pizza
            [14.50, 20, 300, 1, 3],  # American Sandwich
            [25.00, 40, 700, 3, 4],  # Mexican Tacos
            [19.99, 25, 450, 2, 1],  # Italian Risotto
            [35.99, 60, 900, 1, 5],  # French Fine Dining
            [8.99, 10, 200, 0, 6],   # Dessert
            [45.00, 90, 1200, 5, 7], # OUTLIER: Expensive, very spicy, long prep
        ])
        
        menu_names = ['Italian Pasta', 'Indian Curry', 'Caesar Salad', 
                     'Vindaloo Curry', 'Margherita Pizza', 'Club Sandwich',
                     'Fish Tacos', 'Mushroom Risotto', 'Coq au Vin', 
                     'Chocolate Cake', 'Chef\'s Challenge']
        
        results = {}
        
        if analysis_type == 'menu_segmentation':
            # K-means: Group menu items into pricing/complexity tiers
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(menu_features)
            
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            segments = kmeans.fit_predict(scaled_features)
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            segment_names = ['Budget-Friendly', 'Mid-Range', 'Premium']
            
            for i in range(3):
                segment_items = scaled_features[segments == i]
                plt.scatter(segment_items[:, 0], segment_items[:, 1], 
                           c=colors[i], label=segment_names[i], s=100, alpha=0.7)
            
            plt.title('Menu Segmentation Analysis')
            plt.xlabel('Price Level (standardized)')
            plt.ylabel('Preparation Time (standardized)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Group items by segment
            segmented_menu = {}
            for i, item in enumerate(menu_names):
                segment = segment_names[segments[i]]
                if segment not in segmented_menu:
                    segmented_menu[segment] = []
                segmented_menu[segment].append(item)
            
            results = {
                'plot': plot_data,
                'segments': segmented_menu,
                'insight': 'Menu items naturally fall into three pricing tiers based on complexity and cost'
            }
            
        elif analysis_type == 'menu_essence':
            # PCA: Find the core factors that differentiate menu items
            pca = PCA(n_components=2)
            essence_data = pca.fit_transform(menu_features)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(essence_data[:, 0], essence_data[:, 1], s=100, alpha=0.7)
            
            for i, name in enumerate(menu_names):
                plt.annotate(name, (essence_data[i, 0], essence_data[i, 1]),
                            xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            plt.title('Menu Item Essence Analysis (PCA)')
            plt.xlabel(f'Primary Factor ({pca.explained_variance_ratio_[0]:.1%} of variation)')
            plt.ylabel(f'Secondary Factor ({pca.explained_variance_ratio_[1]:.1%} of variation)')
            plt.grid(True, alpha=0.3)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            results = {
                'plot': plot_data,
                'variance_explained': pca.explained_variance_ratio_.tolist(),
                'insight': f'Two main factors explain {sum(pca.explained_variance_ratio_):.1%} of menu item differences'
            }
            
        elif analysis_type == 'outlier_detection':
            # DBSCAN: Find unusual menu items that might need attention
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(menu_features)
            
            dbscan = DBSCAN(eps=1.2, min_samples=2)
            clusters = dbscan.fit_predict(scaled_features)
            
            plt.figure(figsize=(10, 6))
            
            # Plot normal items and outliers
            for cluster in set(clusters):
                if cluster == -1:
                    outliers = scaled_features[clusters == cluster]
                    plt.scatter(outliers[:, 0], outliers[:, 2], 
                               c='red', marker='x', s=200, label='Outliers')
                else:
                    cluster_items = scaled_features[clusters == cluster]
                    plt.scatter(cluster_items[:, 0], cluster_items[:, 2], 
                               s=100, alpha=0.7, label=f'Group {cluster + 1}')
            
            plt.title('Menu Item Outlier Detection')
            plt.xlabel('Price (standardized)')
            plt.ylabel('Spice Level (standardized)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Identify outlier items
            outlier_indices = np.where(clusters == -1)[0]
            outlier_items = [menu_names[i] for i in outlier_indices]
            
            results = {
                'plot': plot_data,
                'outliers': outlier_items,
                'insight': f'Found {len(outlier_items)} unusual menu items that may need review'
            }
        
        return JsonResponse({'success': True, 'results': results})
    
    return render(request, 'restaurant/intelligence_dashboard.html')
```

### HTML Template

```html
<!-- templates/restaurant/intelligence_dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Restaurant Intelligence Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .dashboard { max-width: 1200px; margin: 0 auto; }
        .analysis-section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .btn { padding: 10px 20px; margin: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .result-image { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .insight-box { background: #e7f3ff; padding: 15px; border-left: 4px solid #007bff; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>🍽️ Restaurant Menu Intelligence Dashboard</h1>
        <p>Discover hidden patterns in your menu using advanced AI techniques</p>
        
        <div class="analysis-section">
            <h2>Choose Your Analysis</h2>
            <button class="btn" onclick="runAnalysis('menu_segmentation')">📊 Menu Segmentation (K-means)</button>
            <button class="btn" onclick="runAnalysis('menu_essence')">🎯 Menu Essence (PCA)</button>
            <button class="btn" onclick="runAnalysis('outlier_detection')">🔍 Outlier Detection (DBSCAN)</button>
        </div>
        
        <div id="results" class="analysis-section" style="display: none;">
            <!-- Results will be populated here -->
        </div>
    </div>

    <script>
        function runAnalysis(analysisType) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<p>🔄 Analyzing your menu data...</p>';
            resultsDiv.style.display = 'block';
            
            fetch('', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                },
                body: `analysis_type=${analysisType}`
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayResults(data.results, analysisType);
                } else {
                    resultsDiv.innerHTML = '<p>❌ Analysis failed. Please try again.</p>';
                }
            })
            .catch(error => {
                resultsDiv.innerHTML = '<p>❌ Error occurred during analysis.</p>';
                console.error('Error:', error);
            });
        }
        
        function displayResults(results, analysisType) {
            const resultsDiv = document.getElementById('results');
            let content = `<h2>📈 Analysis Results: ${getAnalysisTitle(analysisType)}</h2>`;
            
            if (results.plot) {
                content += `<img src="data:image/png;base64,${results.plot}" class="result-image" alt="Analysis Plot">`;
            }
            
            if (results.insight) {
                content += `<div class="insight-box"><strong>💡 Key Insight:</strong> ${results.insight}</div>`;
            }
            
            if (results.segments) {
                content += '<h3>🎯 Menu Segments:</h3><ul>';
                for (const [segment, items] of Object.entries(results.segments)) {
                    content += `<li><strong>${segment}:</strong> ${items.join(', ')}</li>`;
                }
                content += '</ul>';
            }
            
            if (results.outliers && results.outliers.length > 0) {
                content += `<h3>⚠️ Outlier Items:</h3><ul>`;
                results.outliers.forEach(item => {
                    content += `<li>${item}</li>`;
                });
                content += '</ul>';
            }
            
            if (results.variance_explained) {
                const total = results.variance_explained.reduce((a, b) => a + b, 0);
                content += `<p><strong>📊 Variance Explained:</strong> ${(total * 100).toFixed(1)}% of menu differences captured</p>`;
            }
            
            resultsDiv.innerHTML = content;
        }
        
        function getAnalysisTitle(type) {
            const titles = {
                'menu_segmentation': 'Menu Segmentation Analysis',
                'menu_essence': 'Menu Essence Analysis',
                'outlier_detection': 'Outlier Detection Analysis'
            };
            return titles[type] || 'Analysis';
        }
    </script>
    
    {% csrf_token %}
</body>
</html>
```

---

# Customer Segmentation System - Django Project

## Project Overview
A Django-based web application that analyzes customer data using unsupervised learning algorithms to create meaningful customer segments for business insights.

## Project Structure
```
customer_segmentation/
├── manage.py
├── customer_segmentation/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── segmentation/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   ├── ml_utils.py
│   └── templates/
│       └── segmentation/
│           ├── dashboard.html
│           ├── upload.html
│           └── results.html
├── static/
│   ├── css/
│   └── js/
├── media/
│   └── uploads/
└── requirements.txt
```

## Step 1: Django Setup and Models

### models.py
```python
from django.db import models
from django.contrib.auth.models import User
import json

class CustomerDataset(models.Model):
    name = models.CharField(max_length=200)
    file = models.FileField(upload_to='uploads/')
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class SegmentationResult(models.Model):
    ALGORITHM_CHOICES = [
        ('kmeans', 'K-Means Clustering'),
        ('hierarchical', 'Hierarchical Clustering'),
        ('dbscan', 'DBSCAN'),
        ('pca', 'PCA Analysis'),
    ]
    
    dataset = models.ForeignKey(CustomerDataset, on_delete=models.CASCADE)
    algorithm = models.CharField(max_length=20, choices=ALGORITHM_CHOICES)
    parameters = models.JSONField(default=dict)
    results = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.algorithm} - {self.dataset.name}"

class CustomerSegment(models.Model):
    segmentation_result = models.ForeignKey(SegmentationResult, on_delete=models.CASCADE)
    segment_id = models.IntegerField()
    segment_name = models.CharField(max_length=100)
    characteristics = models.JSONField(default=dict)
    customer_count = models.IntegerField()
    
    def __str__(self):
        return f"Segment {self.segment_id}: {self.segment_name}"
```

## Step 2: Machine Learning Utilities

### ml_utils.py
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

class CustomerSegmentationML:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.scaled_data = None
        
    def prepare_data(self, features=None):
        """Prepare and scale the data for analysis"""
        if features:
            self.data = self.data[features]
        
        # Handle missing values
        self.data = self.data.fillna(self.data.mean())
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(self.data)
        return self.scaled_data
    
    def kmeans_clustering(self, n_clusters=4):
        """Perform K-means clustering"""
        if self.scaled_data is None:
            self.prepare_data()
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.scaled_data)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.scaled_data, clusters)
        
        results = {
            'clusters': clusters.tolist(),
            'centroids': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette_avg,
            'n_clusters': n_clusters
        }
        
        return results
    
    def hierarchical_clustering(self, n_clusters=4, linkage='ward'):
        """Perform hierarchical clustering"""
        if self.scaled_data is None:
            self.prepare_data()
            
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage=linkage
        )
        clusters = hierarchical.fit_predict(self.scaled_data)
        
        silhouette_avg = silhouette_score(self.scaled_data, clusters)
        
        results = {
            'clusters': clusters.tolist(),
            'silhouette_score': silhouette_avg,
            'n_clusters': n_clusters,
            'linkage': linkage
        }
        
        return results
    
    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering"""
        if self.scaled_data is None:
            self.prepare_data()
            
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(self.scaled_data)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        results = {
            'clusters': clusters.tolist(),
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples
        }
        
        if n_clusters > 1:
            # Only calculate silhouette score if we have more than 1 cluster
            valid_clusters = clusters[clusters != -1]
            valid_data = self.scaled_data[clusters != -1]
            if len(set(valid_clusters)) > 1:
                results['silhouette_score'] = silhouette_score(valid_data, valid_clusters)
        
        return results
    
    def pca_analysis(self, n_components=2):
        """Perform PCA analysis"""
        if self.scaled_data is None:
            self.prepare_data()
            
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(self.scaled_data)
        
        results = {
            'pca_data': pca_data.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pca.components_.tolist(),
            'n_components': n_components
        }
        
        return results
    
    def generate_segment_characteristics(self, clusters, algorithm_type):
        """Generate characteristics for each segment"""
        segments = {}
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_mask = np.array(clusters) == cluster_id
            cluster_data = self.data[cluster_mask]
            
            characteristics = {}
            for column in self.data.columns:
                if self.data[column].dtype in ['int64', 'float64']:
                    characteristics[column] = {
                        'mean': float(cluster_data[column].mean()),
                        'std': float(cluster_data[column].std()),
                        'min': float(cluster_data[column].min()),
                        'max': float(cluster_data[column].max())
                    }
                else:
                    # For categorical data
                    characteristics[column] = cluster_data[column].value_counts().to_dict()
            
            segments[f"segment_{cluster_id}"] = {
                'id': int(cluster_id),
                'size': int(cluster_mask.sum()),
                'percentage': float(cluster_mask.sum() / len(clusters) * 100),
                'characteristics': characteristics
            }
        
        return segments
    
    def create_visualization(self, results, algorithm_type):
        """Create visualizations for the clustering results"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Customer Segmentation Analysis - {algorithm_type.upper()}', fontsize=16)
        
        # Plot 1: Cluster distribution
        clusters = np.array(results['clusters'])
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        
        axes[0, 0].bar(unique_clusters, counts, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Cluster Distribution')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Customers')
        
        # Plot 2: First two principal components colored by cluster
        if self.scaled_data.shape[1] >= 2:
            pca_temp = PCA(n_components=2)
            pca_data = pca_temp.fit_transform(self.scaled_data)
            
            scatter = axes[0, 1].scatter(pca_data[:, 0], pca_data[:, 1], 
                                       c=clusters, cmap='viridis', alpha=0.6)
            axes[0, 1].set_title('Clusters in PCA Space')
            axes[0, 1].set_xlabel('First Principal Component')
            axes[0, 1].set_ylabel('Second Principal Component')
            plt.colorbar(scatter, ax=axes[0, 1])
        
        # Plot 3: Feature importance/characteristics
        if hasattr(self, 'data') and len(self.data.columns) > 0:
            feature_means = []
            for col in self.data.select_dtypes(include=[np.number]).columns[:5]:
                feature_means.append(self.data[col].mean())
            
            if feature_means:
                axes[1, 0].bar(range(len(feature_means)), feature_means, color='lightcoral', alpha=0.7)
                axes[1, 0].set_title('Average Feature Values')
                axes[1, 0].set_xlabel('Features')
                axes[1, 0].set_ylabel('Average Value')
                axes[1, 0].set_xticks(range(len(feature_means)))
                axes[1, 0].set_xticklabels(self.data.select_dtypes(include=[np.number]).columns[:5], 
                                         rotation=45)
        
        # Plot 4: Silhouette score or other metrics
        if 'silhouette_score' in results:
            axes[1, 1].bar(['Silhouette Score'], [results['silhouette_score']], 
                          color='gold', alpha=0.7)
            axes[1, 1].set_title('Clustering Quality Metric')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        plot_url = base64.b64encode(plot_data).decode()
        return plot_url
```

## Step 3: Views and Forms

### forms.py
```python
from django import forms
from .models import CustomerDataset

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = CustomerDataset
        fields = ['name', 'file']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Dataset Name'}),
            'file': forms.FileInput(attrs={'class': 'form-control', 'accept': '.csv,.xlsx'})
        }

class ClusteringParametersForm(forms.Form):
    ALGORITHM_CHOICES = [
        ('kmeans', 'K-Means Clustering'),
        ('hierarchical', 'Hierarchical Clustering'),
        ('dbscan', 'DBSCAN'),
        ('pca', 'PCA Analysis'),
    ]
    
    algorithm = forms.ChoiceField(
        choices=ALGORITHM_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # K-means and Hierarchical parameters
    n_clusters = forms.IntegerField(
        initial=4,
        min_value=2,
        max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    # DBSCAN parameters
    eps = forms.FloatField(
        initial=0.5,
        min_value=0.1,
        max_value=2.0,
        step=0.1,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    
    min_samples = forms.IntegerField(
        initial=5,
        min_value=2,
        max_value=20,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    # PCA parameters
    n_components = forms.IntegerField(
        initial=2,
        min_value=2,
        max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
```

### views.py
```python
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import json
from .models import CustomerDataset, SegmentationResult, CustomerSegment
from .forms import DatasetUploadForm, ClusteringParametersForm
from .ml_utils import CustomerSegmentationML

@login_required
def dashboard(request):
    """Main dashboard view"""
    datasets = CustomerDataset.objects.filter(uploaded_by=request.user)
    recent_results = SegmentationResult.objects.filter(
        dataset__uploaded_by=request.user
    )[:5]
    
    context = {
        'datasets': datasets,
        'recent_results': recent_results,
    }
    return render(request, 'segmentation/dashboard.html', context)

@login_required
def upload_dataset(request):
    """Upload customer dataset"""
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.uploaded_by = request.user
            
            # Validate file format
            try:
                if dataset.file.name.endswith('.csv'):
                    df = pd.read_csv(dataset.file)
                elif dataset.file.name.endswith('.xlsx'):
                    df = pd.read_excel(dataset.file)
                else:
                    raise ValueError("Unsupported file format")
                
                # Basic validation
                if df.empty:
                    raise ValueError("File is empty")
                
                if len(df.columns) < 2:
                    raise ValueError("Dataset must have at least 2 columns")
                
                dataset.save()
                messages.success(request, f'Dataset "{dataset.name}" uploaded successfully!')
                return redirect('segmentation:dashboard')
                
            except Exception as e:
                messages.error(request, f'Error uploading file: {str(e)}')
    else:
        form = DatasetUploadForm()
    
    return render(request, 'segmentation/upload.html', {'form': form})

@login_required
def run_segmentation(request, dataset_id):
    """Run segmentation analysis"""
    dataset = get_object_or_404(CustomerDataset, id=dataset_id, uploaded_by=request.user)
    
    if request.method == 'POST':
        form = ClusteringParametersForm(request.POST)
        if form.is_valid():
            try:
                # Load dataset
                if dataset.file.name.endswith('.csv'):
                    df = pd.read_csv(dataset.file.path)
                else:
                    df = pd.read_excel(dataset.file.path)
                
                # Initialize ML class
                ml_analyzer = CustomerSegmentationML(df)
                
                # Get form data
                algorithm = form.cleaned_data['algorithm']
                parameters = {}
                
                # Run selected algorithm
                if algorithm == 'kmeans':
                    parameters['n_clusters'] = form.cleaned_data['n_clusters']
                    results = ml_analyzer.kmeans_clustering(
                        n_clusters=parameters['n_clusters']
                    )
                elif algorithm == 'hierarchical':
                    parameters['n_clusters'] = form.cleaned_data['n_clusters']
                    results = ml_analyzer.hierarchical_clustering(
                        n_clusters=parameters['n_clusters']
                    )
                elif algorithm == 'dbscan':
                    parameters['eps'] = form.cleaned_data['eps']
                    parameters['min_samples'] = form.cleaned_data['min_samples']
                    results = ml_analyzer.dbscan_clustering(
                        eps=parameters['eps'],
                        min_samples=parameters['min_samples']
                    )
                elif algorithm == 'pca':
                    parameters['n_components'] = form.cleaned_data['n_components']
                    results = ml_analyzer.pca_analysis(
                        n_components=parameters['n_components']
                    )
                
                # Generate segment characteristics
                if algorithm != 'pca':
                    segments = ml_analyzer.generate_segment_characteristics(
                        results['clusters'], algorithm
                    )
                    results['segments'] = segments
                
                # Create visualization
                plot_url = ml_analyzer.create_visualization(results, algorithm)
                results['plot'] = plot_url
                
                # Save results
                segmentation_result = SegmentationResult.objects.create(
                    dataset=dataset,
                    algorithm=algorithm,
                    parameters=parameters,
                    results=results
                )
                
                # Save individual segments
                if algorithm != 'pca' and 'segments' in results:
                    for segment_key, segment_data in results['segments'].items():
                        CustomerSegment.objects.create(
                            segmentation_result=segmentation_result,
                            segment_id=segment_data['id'],
                            segment_name=f"Segment {segment_data['id']}",
                            characteristics=segment_data['characteristics'],
                            customer_count=segment_data['size']
                        )
                
                messages.success(request, 'Segmentation analysis completed successfully!')
                return redirect('segmentation:view_results', result_id=segmentation_result.id)
                
            except Exception as e:
                messages.error(request, f'Error during analysis: {str(e)}')
    else:
        form = ClusteringParametersForm()
    
    return render(request, 'segmentation/run_analysis.html', {
        'form': form,
        'dataset': dataset
    })

@login_required
def view_results(request, result_id):
    """View segmentation results"""
    result = get_object_or_404(
        SegmentationResult,
        id=result_id,
        dataset__uploaded_by=request.user
    )
    
    segments = CustomerSegment.objects.filter(segmentation_result=result)
    
    context = {
        'result': result,
        'segments': segments,
        'results_json': json.dumps(result.results, indent=2),
    }
    
    return render(request, 'segmentation/results.html', context)

@csrf_exempt
def api_segment_details(request, segment_id):
    """API endpoint for segment details"""
    if request.method == 'GET':
        segment = get_object_or_404(CustomerSegment, id=segment_id)
        
        data = {
            'segment_name': segment.segment_name,
            'customer_count': segment.customer_count,
            'characteristics': segment.characteristics,
            'percentage': segment.characteristics.get('percentage', 0)
        }
        
        return JsonResponse(data)
```

## Step 4: URL Configuration

### segmentation/urls.py
```python
from django.urls import path
from . import views

app_name = 'segmentation'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('analyze/<int:dataset_id>/', views.run_segmentation, name='run_segmentation'),
    path('results/<int:result_id>/', views.view_results, name='view_results'),
    path('api/segment/<int:segment_id>/', views.api_segment_details, name='api_segment_details'),
]
```

### Main urls.py
```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('segmentation.urls')),
    path('accounts/', include('django.contrib.auth.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

## Step 5: Templates

### templates/segmentation/dashboard.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Segmentation Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{% url 'segmentation:dashboard' %}">
                <i class="fas fa-chart-pie me-2"></i>Customer Segmentation
            </a>
            <div class="navbar-nav ms-auto">
                <span class="navbar-text me-3">Welcome, {{ user.username }}</span>
                <a class="nav-link" href="{% url 'logout' %}">Logout</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-md-12">
                <h2>Dashboard</h2>
                <p class="text-muted">Analyze customer data to discover meaningful segments</p>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <i class="fas fa-database fa-3x text-primary mb-3"></i>
                        <h5 class="card-title">{{ datasets.count }}</h5>
                        <p class="card-text">Datasets</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <i class="fas fa-chart-line fa-3x text-success mb-3"></i>
                        <h5 class="card-title">{{ recent_results.count }}</h5>
                        <p class="card-text">Recent Analyses</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <i class="fas fa-users fa-3x text-info mb-3"></i>
                        <h5 class="card-title">Active</h5>
                        <p class="card-text">Segmentation Models</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Your Datasets</h5>
                        <a href="{% url 'segmentation:upload_dataset' %}" class="btn btn-primary btn-sm">
                            <i class="fas fa-upload me-1"></i>Upload New
                        </a>
                    </div>
                    <div class="card-body">
                        {% if datasets %}
                            <div class="list-group list-group-flush">
                                {% for dataset in datasets %}
                                <div class="list-group-item d-flex justify-content-between align-items-center">
                                    <div>
                                        <h6 class="mb-1">{{ dataset.name }}</h6>
                                        <small class="text-muted">
                                            Uploaded: {{ dataset.uploaded_at|date:"M d, Y" }}
                                        </small>
                                    </div>
                                    <a href="{% url 'segmentation:run_segmentation' dataset.id %}" 
                                       class="btn btn-outline-primary btn-sm">
                                        <i class="fas fa-play me-1"></i>Analyze
                                    </a>
                                </div>
                                {% endfor %}
                            </div>
                        {% else %}
                            <p class="text-muted text-center py-3">
                                No datasets uploaded yet. 
                                <a href="{% url 'segmentation:upload_dataset' %}">Upload your first dataset</a>
                            </p>
                        {% endif %}
                    </div>
                </div>
            </div>

            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Recent Analysis Results</h5>
                    </div>
                    <div class="card-body">
                        {% if recent_results %}
                            <div class="list-group list-group-flush">
                                {% for result in recent_results %}
                                <div class="list-group-item d-flex justify-content-between align-items-center">
                                    <div>
                                        <h6 class="mb-1">{{ result.get_algorithm_display }}</h6>
                                        <small class="text-muted">
                                            {{ result.dataset.name }} - {{ result.created_at|date:"M d, Y" }}
                                        </small>
                                    </div>
                                    <a href="{% url 'segmentation:view_results' result.id %}" 
                                       class="btn btn-outline-success btn-sm">
                                        <i class="fas fa-eye me-1"></i>View
                                    </a>
                                </div>
                                {% endfor %}
                            </div>
                        {% else %}
                            <p class="text-muted text-center py-3">
                                No analysis results yet. Run your first segmentation analysis!
                            </p>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

### templates/segmentation/results.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Segmentation Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{% url 'segmentation:dashboard' %}">
                <i class="fas fa-chart-pie me-2"></i>Customer Segmentation
            </a>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-md-12">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <div>
                        <h2>{{ result.get_algorithm_display }} Results</h2>
                        <p class="text-muted">Dataset: {{ result.dataset.name }} | 
                           Created: {{ result.created_at|date:"M d, Y H:i" }}</p>
                    </div>
                    <a href="{% url 'segmentation:dashboard' %}" class="btn btn-secondary">
                        <i class="fas fa-arrow-left me-1"></i>Back to Dashboard
                    </a>
                </div>
            </div>
        </div>

        <!-- Visualization -->
        {% if result.results.plot %}
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Visualization</h5>
                    </div>
                    <div class="card-body text-center">
                        <img src="data:image/png;base64,{{ result.results.plot }}" 
                             class="img-fluid" alt="Segmentation Visualization">
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Algorithm-specific metrics -->
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Analysis Metrics</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            {% if result.results.silhouette_score %}
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h4 class="text-primary">{{ result.results.silhouette_score|floatformat:3 }}</h4>
                                    <small class="text-muted">Silhouette Score</small>
                                </div>
                            </div>
                            {% endif %}
                            
                            {% if result.results.n_clusters %}
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h4 class="text-success">{{ result.results.n_clusters }}</h4>
                                    <small class="text-muted">Number of Clusters</small>
                                </div>
                            </div>
                            {% endif %}
                            
                            {% if result.results.inertia %}
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h4 class="text-warning">{{ result.results.inertia|floatformat:2 }}</h4>
                                    <small class="text-muted">Inertia</small>
                                </div>
                            </div>
                            {% endif %}
                            
                            {% if result.results.n_noise %}
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h4 class="text-danger">{{ result.results.n_noise }}</h4>
                                    <small class="text-muted">Noise Points</small>
                                </div>
                            </div>
                            {% endif %}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Segments Overview -->
        {% if segments %}
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Customer Segments</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            {% for segment in segments %}
                            <div class="col-md-6 mb-3">
                                <div class="card border-left-primary">
                                    <div class="card-body">
                                        <div class="d-flex justify-content-between align-items-center mb-2">
                                            <h6 class="text-primary mb-0">{{ segment.segment_name }}</h6>
                                            <span class="badge bg-primary">{{ segment.customer_count }} customers</span>
                                        </div>
                                        
                                        <!-- Key characteristics -->
                                        <div class="mt-3">
                                            <h6 class="text-muted mb-2">Key Characteristics:</h6>
                                            {% for feature, stats in segment.characteristics.items %}
                                                {% if stats.mean %}
                                                <div class="mb-1">
                                                    <small class="text-muted">{{ feature }}:</small>
                                                    <span class="fw-bold">{{ stats.mean|floatformat:2 }}</span>
                                                    <small class="text-muted">(±{{ stats.std|floatformat:2 }})</small>
                                                </div>
                                                {% endif %}
                                            {% endfor %}
                                        </div>
                                        
                                        <button class="btn btn-outline-primary btn-sm mt-2" 
                                                onclick="showSegmentDetails({{ segment.id }})">
                                            <i class="fas fa-info-circle me-1"></i>View Details
                                        </button>
                                    </div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Algorithm Parameters -->
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Algorithm Parameters</h5>
                    </div>
                    <div class="card-body">
                        <dl class="row">
                            {% for param, value in result.parameters.items %}
                            <dt class="col-sm-6">{{ param|title }}:</dt>
                            <dd class="col-sm-6">{{ value }}</dd>
                            {% endfor %}
                        </dl>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Export Options</h5>
                    </div>
                    <div class="card-body">
                        <div class="d-grid gap-2">
                            <button class="btn btn-outline-primary" onclick="exportResults('csv')">
                                <i class="fas fa-file-csv me-1"></i>Export as CSV
                            </button>
                            <button class="btn btn-outline-success" onclick="exportResults('json')">
                                <i class="fas fa-file-code me-1"></i>Export as JSON
                            </button>
                            <button class="btn btn-outline-info" onclick="window.print()">
                                <i class="fas fa-print me-1"></i>Print Report
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Raw Results (Collapsible) -->
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <button class="btn btn-link p-0 text-decoration-none" type="button" 
                                data-bs-toggle="collapse" data-bs-target="#rawResults">
                            <h5 class="mb-0">Raw Results <i class="fas fa-chevron-down"></i></h5>
                        </button>
                    </div>
                    <div class="collapse" id="rawResults">
                        <div class="card-body">
                            <pre class="bg-light p-3 rounded"><code>{{ results_json }}</code></pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Segment Details Modal -->
    <div class="modal fade" id="segmentModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Segment Details</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="segmentModalBody">
                    <!-- Content loaded via JavaScript -->
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function showSegmentDetails(segmentId) {
            fetch(`/api/segment/${segmentId}/`)
                .then(response => response.json())
                .then(data => {
                    const modalBody = document.getElementById('segmentModalBody');
                    
                    let html = `
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Segment Overview</h6>
                                <ul class="list-unstyled">
                                    <li><strong>Name:</strong> ${data.segment_name}</li>
                                    <li><strong>Size:</strong> ${data.customer_count} customers</li>
                                    <li><strong>Percentage:</strong> ${data.percentage.toFixed(1)}%</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6>Detailed Characteristics</h6>
                                <div class="table-responsive">
                                    <table class="table table-sm">
                                        <thead>
                                            <tr>
                                                <th>Feature</th>
                                                <th>Mean</th>
                                                <th>Std Dev</th>
                                                <th>Range</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                    `;
                    
                    for (const [feature, stats] of Object.entries(data.characteristics)) {
                        if (stats.mean !== undefined) {
                            html += `
                                <tr>
                                    <td>${feature}</td>
                                    <td>${stats.mean.toFixed(2)}</td>
                                    <td>${stats.std.toFixed(2)}</td>
                                    <td>${stats.min.toFixed(2)} - ${stats.max.toFixed(2)}</td>
                                </tr>
                            `;
                        }
                    }
                    
                    html += `
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    modalBody.innerHTML = html;
                    new bootstrap.Modal(document.getElementById('segmentModal')).show();
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error loading segment details');
                });
        }

        function exportResults(format) {
            const resultData = {{ results_json|safe }};
            
            if (format === 'csv') {
                // Simple CSV export for segments
                let csv = 'Segment,Customer_Count,Percentage\n';
                if (resultData.segments) {
                    for (const [segmentKey, segmentData] of Object.entries(resultData.segments)) {
                        csv += `${segmentData.id},${segmentData.size},${segmentData.percentage}\n`;
                    }
                }
                
                const blob = new Blob([csv], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'segmentation_results.csv';
                a.click();
                window.URL.revokeObjectURL(url);
                
            } else if (format === 'json') {
                const blob = new Blob([JSON.stringify(resultData, null, 2)], { type: 'application/json' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'segmentation_results.json';
                a.click();
                window.URL.revokeObjectURL(url);
            }
        }
    </script>
</body>
</html>
```

## Step 6: Settings Configuration

### settings.py
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
    'segmentation',
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

ROOT_URLCONF = 'customer_segmentation.urls'

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

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Static files
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

# Login/Logout URLs
LOGIN_URL = '/accounts/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/accounts/login/'

# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024   # 10MB
```

## Step 7: Requirements and Deployment

### requirements.txt
```
Django==4.2.7
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
openpyxl==3.1.2
Pillow==10.1.0
```

### Sample Dataset Generation Script
```python
# generate_sample_data.py
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()
np.random.seed(42)

def generate_customer_data(n_customers=1000):
    """Generate sample customer data for segmentation"""
    
    # Create different customer segments with distinct characteristics
    segments = {
        'High Value': {'size': 200, 'spending': (800, 2000), 'frequency': (15, 30), 'recency': (1, 30)},
        'Regular': {'size': 400, 'spending': (200, 800), 'frequency': (5, 15), 'recency': (15, 90)},
        'Occasional': {'size': 300, 'spending': (50, 200), 'frequency': (1, 5), 'recency': (60, 180)},
        'At Risk': {'size': 100, 'spending': (100, 500), 'frequency': (1, 3), 'recency': (120, 365)}
    }
    
    customers = []
    customer_id = 1
    
    for segment_name, params in segments.items():
        for _ in range(params['size']):
            customer = {
                'customer_id': customer_id,
                'name': fake.name(),
                'email': fake.email(),
                'age': np.random.randint(18, 75),
                'gender': np.random.choice(['M', 'F']),
                'total_spending': np.random.uniform(*params['spending']),
                'purchase_frequency': np.random.randint(*params['frequency']),
                'days_since_last_purchase': np.random.randint(*params['recency']),
                'avg_order_value': 0,  # Will calculate
                'membership_years': np.random.uniform(0.1, 10),
                'city': fake.city(),
                'preferred_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports']),
                'true_segment': segment_name  # For validation
            }
            
            # Calculate average order value
            customer['avg_order_value'] = customer['total_spending'] / max(customer['purchase_frequency'], 1)
            
            customers.append(customer)
            customer_id += 1
    
    # Shuffle the data
    np.random.shuffle(customers)
    
    df = pd.DataFrame(customers)
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_customer_data(1000)
    df.to_csv('sample_customer_data.csv', index=False)
    print(f"Generated sample data with {len(df)} customers")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
```

## Key Features Implementation

### 1. Multi-Algorithm Support
The system supports four different unsupervised learning algorithms:
- **K-Means**: For discovering spherical clusters
- **Hierarchical**: For nested cluster structures
- **DBSCAN**: For density-based clustering with noise detection
- **PCA**: For dimensionality reduction and feature analysis

### 2. Interactive Web Interface
- Dashboard for dataset management
- Real-time parameter adjustment
- Visual results with matplotlib integration
- Export functionality for results

### 3. Comprehensive Analysis
- Silhouette score for cluster quality assessment
- Detailed segment characteristics
- Visual plots for cluster interpretation
- Statistical summaries for each segment

### 4. Data Processing Pipeline
- Automatic data preprocessing and scaling
- Missing value handling
- Feature selection capabilities
- Robust error handling

### 5. Business Intelligence Features
- Customer segment profiling
- Actionable insights generation
- Export capabilities for business reporting
- Historical analysis tracking

This complete Django project demonstrates practical application of unsupervised learning algorithms in a real-world customer segmentation scenario, providing both technical implementation and business value through an intuitive web interface.

## Assignment: Customer Preference Pattern Discovery

### The Challenge
You work for a food delivery app company that wants to understand customer ordering patterns during different seasons. Your task is to implement a Django application that uses hierarchical clustering to discover natural groupings in customer behavior.

### Requirements

**Dataset**: You have customer data with these features:
- `average_order_value`: How much customers typically spend
- `order_frequency`: Orders per month  
- `preferred_cuisine_diversity`: Number of different cuisine types they order
- `peak_ordering_time`: Hour of day they most often order (0-23)
- `seasonal_variation`: How much their ordering changes between seasons (1-10 scale)

**Your Task**:
1. Create a Django view that performs hierarchical clustering on this customer data
2. Generate a dendrogram showing customer relationship patterns
3. Identify 4 distinct customer segments from the hierarchy
4. Provide business insights for each segment
5. Create an HTML template that displays the results clearly

### Sample Data to Use:
```python
customer_data = np.array([
    [25.50, 8, 3, 19, 2],   # Regular evening orderer
    [45.20, 12, 5, 20, 4],  # Premium diverse orderer  
    [15.30, 4, 2, 12, 1],   # Budget lunch customer
    [32.80, 6, 4, 13, 3],   # Moderate lunch customer
    [18.90, 15, 2, 18, 1],  # Frequent budget dinner
    [52.60, 9, 6, 21, 5],   # High-value late night
    [28.40, 7, 3, 19, 2],   # Standard evening customer
    [41.10, 10, 4, 20, 4],  # Premium evening customer
    [12.75, 3, 1, 11, 1],   # Minimal lunch customer
    [38.20, 14, 3, 18, 3],  # Frequent moderate customer
])

customer_names = ['Regular Eve', 'Premium Diverse', 'Budget Lunch', 'Moderate Lunch',
                 'Frequent Budget', 'Late Night Premium', 'Standard Eve', 'Premium Eve', 
                 'Minimal Lunch', 'Frequent Moderate']
```

### Expected Deliverables:
1. **Django view function** that processes the data and creates the dendrogram
2. **HTML template** with proper styling and interactive elements
3. **Business insights** for each of the 4 customer segments you identify
4. **Recommendations** for targeted marketing strategies based on the clusters

### Evaluation Criteria:
- **Code Quality**: Clean, well-commented Django code
- **Visualization**: Clear, properly labeled dendrogram
- **Analysis**: Meaningful interpretation of the clustering results  
- **Business Value**: Practical recommendations for the food delivery company
- **User Experience**: Professional-looking web interface

### Bonus Points:
- Add functionality to adjust the number of clusters dynamically
- Include statistical summaries for each cluster
- Suggest personalized marketing messages for each customer segment

**Submission**: Create a complete Django app with models, views, templates, and a brief explanation of your findings and recommendations.

---

## Summary: Your Unsupervised Learning Mastery

Congratulations! You've now mastered the four essential unsupervised learning techniques, just like a master chef who can:

🔹 **K-means Clustering**: Organize ingredients into specific categories for efficient kitchen management
🔹 **Hierarchical Clustering**: Understand the family relationships between recipes and cooking techniques  
🔹 **PCA**: Identify the essential flavor profiles that define your cuisine's character
🔹 **DBSCAN**: Spot quality issues and unusual dishes that need special attention

These techniques work together to give you complete insight into your data's hidden patterns, helping you make informed decisions without needing labeled examples - just like how an experienced chef can understand ingredients and create amazing dishes through intuition and pattern recognition.

**Key Syntax Reminders**:
- Always use `StandardScaler()` to normalize your data before clustering
- `fit_predict()` both learns and applies the algorithm in one step
- Set `random_state` for reproducible results
- Use `explained_variance_ratio_` in PCA to understand how much information you're capturing
- DBSCAN labels outliers as `-1`, which makes them easy to identify

Your Django application now has the power to automatically discover patterns, segment users, reduce complexity, and detect anomalies - essential skills for any modern AI-powered web application!