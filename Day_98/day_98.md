# AI Mastery Course: Python Core & Django Web Framework
## Day 98: Industry Applications

### Learning Objectives
By the end of this lesson, you will be able to:
- Identify key AI applications across healthcare, finance, autonomous systems, and sustainability sectors
- Understand the technical foundations behind real-world AI implementations
- Analyze the impact of creative AI on traditional industries
- Design AI solutions for specific industry challenges using Python and Django

---

## Introduction

Imagine that you've spent months learning to prepare individual ingredients—mastering the art of dicing vegetables, seasoning meats, and understanding flavor profiles. Now, it's time to step into a professional kitchen where these skills come together to create extraordinary dishes that serve real customers with real needs. Today, we explore how AI technologies transform industries, much like how a skilled cook transforms raw ingredients into solutions that nourish and satisfy.

---

## 1. AI in Healthcare and Finance

### Healthcare Applications

In the medical world, AI acts like a master diagnostician who never tires and can process thousands of cases simultaneously. Let's examine how Python powers these systems:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Medical diagnosis prediction system
class MedicalDiagnosisAI:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_patient_data(self, symptoms_data):
        """
        Transform raw patient symptoms into model-ready format
        Like preparing ingredients before cooking
        """
        # symptoms_data: dictionary with symptom severity scores
        feature_vector = [
            symptoms_data.get('fever', 0),
            symptoms_data.get('cough', 0),
            symptoms_data.get('fatigue', 0),
            symptoms_data.get('breathing_difficulty', 0),
            symptoms_data.get('chest_pain', 0)
        ]
        return np.array(feature_vector).reshape(1, -1)
    
    def train_model(self, training_data, labels):
        """Train the diagnostic model"""
        X_scaled = self.scaler.fit_transform(training_data)
        self.model.fit(X_scaled, labels)
        
    def predict_condition(self, patient_symptoms):
        """Make a diagnosis prediction"""
        processed_data = self.prepare_patient_data(patient_symptoms)
        scaled_data = self.scaler.transform(processed_data)
        
        prediction = self.model.predict(scaled_data)[0]
        confidence = max(self.model.predict_proba(scaled_data)[0])
        
        return {
            'predicted_condition': prediction,
            'confidence_score': round(confidence * 100, 2)
        }

# Usage example
diagnosis_ai = MedicalDiagnosisAI()

# Sample patient data
patient_symptoms = {
    'fever': 8,
    'cough': 6,
    'fatigue': 7,
    'breathing_difficulty': 4,
    'chest_pain': 2
}

# This would return a diagnosis prediction
# result = diagnosis_ai.predict_condition(patient_symptoms)
```

**Code Syntax Explanation:**
- `class MedicalDiagnosisAI:` defines a blueprint for our AI system
- `self.model = RandomForestClassifier()` creates an ensemble learning algorithm that combines multiple decision trees
- `np.array().reshape(1, -1)` converts our data into the format expected by scikit-learn (1 row, multiple columns)
- `self.scaler.transform()` normalizes input data to prevent features with larger values from dominating

### Financial Applications

In finance, AI serves as a vigilant market analyst that can process market patterns faster than any human trader:

```python
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

class FinancialRiskAssessor:
    def __init__(self):
        self.risk_factors = {}
        
    def gather_market_ingredients(self, ticker, period="1y"):
        """
        Collect market data like gathering fresh ingredients
        """
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period)
        
        # Calculate volatility (risk indicator)
        returns = hist_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate moving averages
        hist_data['MA_20'] = hist_data['Close'].rolling(window=20).mean()
        hist_data['MA_50'] = hist_data['Close'].rolling(window=50).mean()
        
        return {
            'current_price': hist_data['Close'][-1],
            'volatility': volatility,
            'price_trend': 'bullish' if hist_data['MA_20'][-1] > hist_data['MA_50'][-1] else 'bearish',
            'volume_trend': hist_data['Volume'].rolling(10).mean()[-1]
        }
    
    def assess_portfolio_risk(self, holdings):
        """
        Evaluate overall portfolio risk like tasting a complex dish
        """
        total_risk_score = 0
        total_weight = 0
        
        for ticker, weight in holdings.items():
            market_data = self.gather_market_ingredients(ticker)
            
            # Risk scoring algorithm
            risk_score = (
                market_data['volatility'] * 0.4 +  # Volatility weight
                (0.3 if market_data['price_trend'] == 'bearish' else 0.1) +  # Trend risk
                0.1  # Base risk
            )
            
            total_risk_score += risk_score * weight
            total_weight += weight
            
        overall_risk = total_risk_score / total_weight if total_weight > 0 else 0
        
        return {
            'risk_level': 'High' if overall_risk > 0.3 else 'Medium' if overall_risk > 0.15 else 'Low',
            'risk_score': round(overall_risk, 3),
            'recommendation': self._generate_recommendation(overall_risk)
        }
    
    def _generate_recommendation(self, risk_score):
        if risk_score > 0.3:
            return "Consider diversifying portfolio to reduce concentration risk"
        elif risk_score > 0.15:
            return "Portfolio shows moderate risk - monitor closely"
        else:
            return "Portfolio appears well-balanced"

# Usage
risk_assessor = FinancialRiskAssessor()
portfolio = {'AAPL': 0.4, 'GOOGL': 0.3, 'TSLA': 0.3}
# risk_analysis = risk_assessor.assess_portfolio_risk(portfolio)
```

**Code Syntax Explanation:**
- `yf.Ticker(ticker).history()` uses the yfinance library to fetch real stock market data
- `pct_change().dropna()` calculates percentage changes between periods and removes null values
- `rolling(window=20).mean()` creates a moving average over 20 periods
- Dictionary comprehensions `{key: value for item in iterable}` create efficient mappings

---

## 2. Autonomous Systems and Robotics

Think of autonomous systems as having a master navigator who can simultaneously read maps, observe traffic, and make split-second decisions. Here's how Python powers these decisions:

```python
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class EnvironmentState:
    obstacles: List[Tuple[int, int]]
    target_position: Tuple[int, int]
    current_position: Tuple[int, int]
    sensor_readings: List[float]

class AutonomousNavigationSystem:
    def __init__(self):
        self.path_history = []
        self.decision_threshold = 0.7
        
    def process_sensor_data(self, camera_feed, lidar_data):
        """
        Process multiple sensor inputs like a chef coordinating multiple burners
        """
        # Computer vision processing
        processed_image = self._detect_obstacles(camera_feed)
        
        # LiDAR processing for distance measurement
        obstacle_distances = self._process_lidar(lidar_data)
        
        # Fusion of sensor data
        environment_map = self._create_environment_map(processed_image, obstacle_distances)
        
        return environment_map
    
    def _detect_obstacles(self, image_array):
        """Use computer vision to identify obstacles"""
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours (obstacle boundaries)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            # Calculate obstacle center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                obstacles.append((cx, cy))
                
        return obstacles
    
    def _process_lidar(self, lidar_readings):
        """Process distance sensor data"""
        # Filter out noise and invalid readings
        valid_readings = [r for r in lidar_readings if 0.1 < r < 50.0]
        
        # Calculate obstacle proximity score
        proximity_score = np.mean([1/r for r in valid_readings if r > 0])
        
        return {
            'closest_obstacle': min(valid_readings) if valid_readings else float('inf'),
            'proximity_score': proximity_score,
            'safe_directions': [i for i, r in enumerate(lidar_readings) if r > 2.0]
        }
    
    def plan_navigation_path(self, current_pos, target_pos, obstacles):
        """
        Plan optimal path like planning a complex menu
        """
        # Simple A* pathfinding algorithm implementation
        def calculate_path_cost(pos1, pos2, obstacles):
            # Euclidean distance + obstacle penalty
            base_cost = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            
            # Add penalty for paths near obstacles
            obstacle_penalty = 0
            for obstacle in obstacles:
                distance_to_obstacle = np.sqrt((pos1[0] - obstacle[0])**2 + (pos1[1] - obstacle[1])**2)
                if distance_to_obstacle < 5.0:  # Safety margin
                    obstacle_penalty += (5.0 - distance_to_obstacle) * 10
                    
            return base_cost + obstacle_penalty
        
        # For simplicity, return direct path with obstacle avoidance
        if not obstacles:
            return [current_pos, target_pos]
        
        # Find safe intermediate points
        safe_points = self._find_safe_waypoints(current_pos, target_pos, obstacles)
        
        return [current_pos] + safe_points + [target_pos]
    
    def _find_safe_waypoints(self, start, end, obstacles):
        """Find intermediate points that avoid obstacles"""
        waypoints = []
        
        # Create a grid of potential waypoints
        x_steps = np.linspace(start[0], end[0], 5)[1:-1]  # Exclude start and end
        y_steps = np.linspace(start[1], end[1], 5)[1:-1]
        
        for x in x_steps:
            for y in y_steps:
                point = (x, y)
                # Check if point is safe (not too close to obstacles)
                is_safe = all(
                    np.sqrt((x - obs[0])**2 + (y - obs[1])**2) > 3.0 
                    for obs in obstacles
                )
                if is_safe:
                    waypoints.append(point)
        
        return waypoints[:2]  # Return up to 2 waypoints for simplicity

# Usage example
nav_system = AutonomousNavigationSystem()
current_position = (0, 0)
target_position = (10, 10)
detected_obstacles = [(3, 4), (7, 6), (5, 8)]

# planned_path = nav_system.plan_navigation_path(current_position, target_position, detected_obstacles)
```

**Code Syntax Explanation:**
- `@dataclass` is a decorator that automatically generates special methods like `__init__()` for data storage classes
- `cv2.Canny(gray, 50, 150)` applies edge detection with low and high thresholds of 50 and 150
- `np.linspace(start, end, num)` creates evenly spaced numbers between start and end
- List comprehensions `[expression for item in iterable if condition]` create filtered lists efficiently

---

## 3. AI for Climate and Sustainability

Environmental AI systems work like master gardeners who understand complex ecosystems and can predict how small changes affect the whole environment:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class ClimateAnalysisSystem:
    def __init__(self):
        self.emission_factors = {
            'electricity': 0.4,  # kg CO2 per kWh
            'natural_gas': 5.3,  # kg CO2 per m³
            'gasoline': 2.3,     # kg CO2 per liter
            'diesel': 2.7        # kg CO2 per liter
        }
        
    def calculate_carbon_footprint(self, consumption_data):
        """
        Calculate carbon emissions like measuring ingredients for a recipe
        """
        total_emissions = 0
        emission_breakdown = {}
        
        for energy_type, consumption in consumption_data.items():
            if energy_type in self.emission_factors:
                emissions = consumption * self.emission_factors[energy_type]
                emission_breakdown[energy_type] = emissions
                total_emissions += emissions
        
        return {
            'total_co2_kg': round(total_emissions, 2),
            'breakdown': emission_breakdown,
            'sustainability_score': self._calculate_sustainability_score(total_emissions)
        }
    
    def _calculate_sustainability_score(self, total_emissions):
        """Rate sustainability like rating a dish"""
        # Average household produces ~16,000 kg CO2/year
        if total_emissions < 8000:
            return {'score': 'Excellent', 'rating': 5}
        elif total_emissions < 12000:
            return {'score': 'Good', 'rating': 4}
        elif total_emissions < 16000:
            return {'score': 'Average', 'rating': 3}
        elif total_emissions < 20000:
            return {'score': 'Below Average', 'rating': 2}
        else:
            return {'score': 'Poor', 'rating': 1}
    
    def predict_climate_impact(self, historical_data, forecast_days=30):
        """
        Predict future environmental conditions like forecasting seasonal ingredients
        """
        # Simple linear trend analysis
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Calculate trends
        trends = {}
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # Calculate daily change rate
                daily_change = df[column].diff().mean()
                trends[column] = daily_change
        
        # Generate predictions
        last_date = df.index[-1]
        predictions = []
        
        for i in range(1, forecast_days + 1):
            future_date = last_date + timedelta(days=i)
            prediction = {}
            
            for metric, trend in trends.items():
                last_value = df[metric].iloc[-1]
                predicted_value = last_value + (trend * i)
                prediction[metric] = round(predicted_value, 2)
            
            prediction['date'] = future_date.strftime('%Y-%m-%d')
            predictions.append(prediction)
        
        return predictions
    
    def optimize_energy_usage(self, current_usage, budget_constraint=None):
        """
        Suggest optimizations like adjusting a recipe for better results
        """
        recommendations = []
        potential_savings = 0
        
        # Analyze each energy source
        for energy_type, usage in current_usage.items():
            if energy_type == 'electricity' and usage > 500:  # kWh/month
                reduction = min(usage * 0.15, 100)  # 15% reduction, max 100 kWh
                co2_saved = reduction * self.emission_factors['electricity']
                cost_saved = reduction * 0.12  # Assuming $0.12/kWh
                
                recommendations.append({
                    'energy_type': energy_type,
                    'current_usage': usage,
                    'suggested_reduction': reduction,
                    'co2_savings_kg': co2_saved,
                    'cost_savings_usd': cost_saved,
                    'actions': ['LED lighting upgrade', 'Smart thermostat', 'Energy-efficient appliances']
                })
                potential_savings += cost_saved
                
            elif energy_type == 'gasoline' and usage > 200:  # liters/month
                reduction = usage * 0.20  # 20% reduction through efficiency
                co2_saved = reduction * self.emission_factors['gasoline']
                cost_saved = reduction * 1.50  # Assuming $1.50/liter
                
                recommendations.append({
                    'energy_type': energy_type,
                    'current_usage': usage,
                    'suggested_reduction': reduction,
                    'co2_savings_kg': co2_saved,
                    'cost_savings_usd': cost_saved,
                    'actions': ['Hybrid vehicle', 'Public transport', 'Remote work', 'Trip consolidation']
                })
                potential_savings += cost_saved
        
        return {
            'recommendations': recommendations,
            'total_potential_savings_usd': round(potential_savings, 2),
            'total_co2_reduction_kg': sum(r['co2_savings_kg'] for r in recommendations)
        }

# Usage example
climate_ai = ClimateAnalysisSystem()

# Sample consumption data (monthly)
household_consumption = {
    'electricity': 750,  # kWh
    'natural_gas': 150,  # m³
    'gasoline': 300      # liters
}

# Calculate carbon footprint
# carbon_analysis = climate_ai.calculate_carbon_footprint(household_consumption)

# Get optimization recommendations
# optimization_plan = climate_ai.optimize_energy_usage(household_consumption)
```

**Code Syntax Explanation:**
- `pd.to_datetime()` converts string dates into pandas datetime objects for time series analysis
- `df.set_index('date').sort_index()` makes the date column the index and sorts chronologically
- `df[column].diff().mean()` calculates the average day-to-day change in values
- `timedelta(days=i)` creates a time offset for future date calculations

---

## 4. Creative AI Applications

Creative AI systems function like master artists who can blend different styles, understand composition, and generate novel combinations:

```python
import random
import nltk
from collections import defaultdict
import json

class CreativeContentGenerator:
    def __init__(self):
        # Download required NLTK data
        # nltk.download('punkt')
        # nltk.download('averaged_perceptron_tagger')
        
        self.style_templates = {
            'poetic': {
                'patterns': ['The {noun} {verb} like {adjective} {noun}', 
                           'In {noun} of {adjective} {noun}, {verb} the {noun}'],
                'mood': 'contemplative'
            },
            'technical': {
                'patterns': ['The {noun} demonstrates {adjective} {noun} through {verb}',
                           'Analysis reveals {adjective} {noun} patterns in {noun}'],
                'mood': 'analytical'
            },
            'narrative': {
                'patterns': ['Once upon a time, the {adjective} {noun} {verb}',
                           'The {noun} decided to {verb} the {adjective} {noun}'],
                'mood': 'storytelling'
            }
        }
        
        self.word_bank = {
            'noun': ['algorithm', 'data', 'network', 'system', 'pattern', 'model', 'intelligence', 'creativity'],
            'verb': ['learns', 'adapts', 'processes', 'generates', 'transforms', 'analyzes', 'discovers', 'creates'],
            'adjective': ['intelligent', 'adaptive', 'complex', 'elegant', 'sophisticated', 'innovative', 'dynamic', 'creative']
        }
    
    def generate_creative_content(self, content_type, style, length=3):
        """
        Generate creative content like composing a sophisticated dish
        """
        if style not in self.style_templates:
            style = 'narrative'  # Default style
            
        generated_content = []
        templates = self.style_templates[style]['patterns']
        
        for _ in range(length):
            # Select random template
            template = random.choice(templates)
            
            # Fill template with words
            content_piece = self._fill_template(template)
            generated_content.append(content_piece)
        
        return {
            'content': generated_content,
            'style': style,
            'mood': self.style_templates[style]['mood'],
            'metadata': {
                'generated_at': str(datetime.now()),
                'word_count': sum(len(piece.split()) for piece in generated_content)
            }
        }
    
    def _fill_template(self, template):
        """Fill template with appropriate words"""
        filled_template = template
        
        # Find placeholders and replace them
        for word_type, words in self.word_bank.items():
            placeholder = '{' + word_type + '}'
            while placeholder in filled_template:
                selected_word = random.choice(words)
                filled_template = filled_template.replace(placeholder, selected_word, 1)
        
        return filled_template
    
    def blend_styles(self, styles, content_length=5):
        """
        Blend multiple styles like fusion cooking
        """
        if not styles or len(styles) < 2:
            return self.generate_creative_content('text', 'narrative', content_length)
        
        blended_content = []
        
        # Alternate between styles
        for i in range(content_length):
            current_style = styles[i % len(styles)]
            piece = self.generate_creative_content('text', current_style, 1)
            blended_content.extend(piece['content'])
        
        return {
            'content': blended_content,
            'styles_used': styles,
            'blend_pattern': 'alternating',
            'total_pieces': len(blended_content)
        }
    
    def analyze_content_sentiment(self, content_list):
        """
        Analyze the emotional tone like tasting for balance
        """
        positive_words = ['elegant', 'sophisticated', 'innovative', 'creative', 'intelligent', 'discovers', 'creates']
        neutral_words = ['processes', 'analyzes', 'data', 'system', 'network']
        
        sentiment_scores = []
        
        for content in content_list:
            words = content.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            neutral_count = sum(1 for word in words if word in neutral_words)
            total_words = len(words)
            
            if total_words == 0:
                sentiment_score = 0
            else:
                sentiment_score = (positive_count * 1 + neutral_count * 0) / total_words
            
            sentiment_scores.append(sentiment_score)
        
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        return {
            'individual_scores': sentiment_scores,
            'average_sentiment': round(average_sentiment, 3),
            'overall_tone': 'Positive' if average_sentiment > 0.3 else 'Neutral' if average_sentiment > 0.1 else 'Technical'
        }

# Usage example
creative_ai = CreativeContentGenerator()

# Generate content in different styles
# poetic_content = creative_ai.generate_creative_content('text', 'poetic', 3)
# technical_content = creative_ai.generate_creative_content('text', 'technical', 2)

# Blend multiple styles
# fusion_content = creative_ai.blend_styles(['poetic', 'technical', 'narrative'], 6)

# Analyze sentiment
# sample_content = ["The algorithm learns like intelligent data", "Analysis reveals complex network patterns"]
# sentiment_analysis = creative_ai.analyze_content_sentiment(sample_content)
```

**Code Syntax Explanation:**
- `defaultdict(list)` creates a dictionary that automatically creates empty lists for new keys
- `random.choice(list)` randomly selects one item from the provided list
- `str.replace(old, new, count)` replaces occurrences of 'old' with 'new', limited by 'count'
- `sum(expression for item in iterable)` calculates the sum using a generator expression

---

## Final Project: Integrated AI Industry Solution

Now it's time to combine everything you've learned into a comprehensive solution—like preparing a complete multi-course meal that showcases all your culinary skills.

### Project: Smart City Environmental Monitor

Create a Django web application that integrates multiple AI capabilities to monitor and optimize city-wide environmental conditions.

```python
# Django models (models.py)
from django.db import models
from django.contrib.auth.models import User
import json

class EnvironmentalSensor(models.Model):
    SENSOR_TYPES = [
        ('air_quality', 'Air Quality'),
        ('noise_level', 'Noise Level'),
        ('traffic_density', 'Traffic Density'),
        ('energy_usage', 'Energy Usage')
    ]
    
    sensor_id = models.CharField(max_length=50, unique=True)
    sensor_type = models.CharField(max_length=20, choices=SENSOR_TYPES)
    location_lat = models.DecimalField(max_digits=9, decimal_places=6)
    location_lng = models.DecimalField(max_digits=9, decimal_places=6)
    installation_date = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.sensor_type} - {self.sensor_id}"

class SensorReading(models.Model):
    sensor = models.ForeignKey(EnvironmentalSensor, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    value = models.FloatField()
    unit = models.CharField(max_length=20)
    quality_score = models.FloatField(default=1.0)  # Data quality indicator
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['sensor', '-timestamp']),
        ]

class AIAnalysis(models.Model):
    analysis_type = models.CharField(max_length=50)
    input_data = models.JSONField()  # Store analysis inputs
    results = models.JSONField()     # Store analysis results
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def get_recommendations(self):
        """Extract recommendations from results"""
        if 'recommendations' in self.results:
            return self.results['recommendations']
        return []

# Django views (views.py)
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import ListView
import json
from datetime import datetime, timedelta
import numpy as np

class EnvironmentalDashboardView(ListView):
    model = EnvironmentalSensor
    template_name = 'environmental_dashboard.html'
    context_object_name = 'sensors'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get recent readings for each sensor
        sensors_data = []
        for sensor in context['sensors']:
            recent_readings = SensorReading.objects.filter(
                sensor=sensor
            ).order_by('-timestamp')[:24]  # Last 24 readings
            
            # Calculate AI-powered insights
            ai_insights = self.generate_ai_insights(recent_readings)
            
            sensors_data.append({
                'sensor': sensor,
                'recent_readings': recent_readings,
                'insights': ai_insights,
                'status': self.determine_sensor_status(recent_readings)
            })
        
        context['sensors_data'] = sensors_data
        context['city_overview'] = self.calculate_city_overview(sensors_data)
        
        return context
    
    def generate_ai_insights(self, readings):
        """Generate AI insights like creating a flavor profile"""
        if not readings:
            return {'trend': 'No data', 'forecast': 'Unable to predict'}
        
        values = [r.value for r in readings]
        
        # Trend analysis
        if len(values) >= 3:
            recent_trend = np.mean(values[:3]) - np.mean(values[-3:])
            trend = 'Improving' if recent_trend > 0 else 'Declining' if recent_trend < 0 else 'Stable'
        else:
            trend = 'Insufficient data'
        
        # Simple forecast (linear projection)
        if len(values) >= 5:
            # Calculate average change per reading
            changes = [values[i] - values[i+1] for i in range(len(values)-1)]
            avg_change = np.mean(changes)
            next_predicted = values[0] + avg_change
            forecast = f"Next reading predicted: {next_predicted:.2f}"
        else:
            forecast = "Need more data for prediction"
        
        return {
            'trend': trend,
            'forecast': forecast,
            'data_quality': np.mean([r.quality_score for r in readings]) if readings else 0
        }
    
    def determine_sensor_status(self, readings):
        """Determine sensor health like checking ingredient freshness"""
        if not readings:
            return 'No data'
        
        latest_reading = readings[0]
        time_since_last = datetime.now() - latest_reading.timestamp.replace(tzinfo=None)
        
        if time_since_last.total_seconds() > 3600:  # More than 1 hour
            return 'Offline'
        elif latest_reading.quality_score < 0.7:
            return 'Poor quality'
        else:
            return 'Online'
    
    def calculate_city_overview(self, sensors_data):
        """Calculate overall city environmental health"""
        if not sensors_data:
            return {'overall_score': 0, 'status': 'No data'}
        
        online_sensors = [s for s in sensors_data if s['status'] == 'Online']
        total_sensors = len(sensors_data)
        
        # Environmental health score
        health_scores = []
        for sensor_data in online_sensors:
            if sensor_data['recent_readings']:
                latest_value = sensor_data['recent_readings'][0].value
                sensor_type = sensor_data['sensor'].sensor_type
                
                # Normalize scores based on sensor type (0-100 scale)
                if sensor_type == 'air_quality':
                    # Lower values are better for air quality (AQI scale)
                    normalized_score = max(0, 100 - latest_value) if latest_value <= 100 else 0
                elif sensor_type == 'noise_level':
                    # Lower noise is better (dB scale)
                    normalized_score = max(0, 100 - (latest_value - 30)) if latest_value >= 30 else 100
                elif sensor_type == 'energy_usage':
                    # Lower energy usage is better (efficiency scale)
                    normalized_score = max(0, 100 - latest_value) if latest_value <= 100 else 0
                else:
                    normalized_score = 50  # Default neutral score
                
                health_scores.append(normalized_score)
        
        overall_score = np.mean(health_scores) if health_scores else 0
        
        return {
            'overall_score': round(overall_score, 1),
            'status': 'Excellent' if overall_score > 80 else 'Good' if overall_score > 60 else 'Fair' if overall_score > 40 else 'Poor',
            'online_sensors': len(online_sensors),
            'total_sensors': total_sensors,
            'uptime_percentage': round((len(online_sensors) / total_sensors) * 100, 1) if total_sensors > 0 else 0
        }

@csrf_exempt
def ai_analysis_endpoint(request):
    """
    API endpoint for AI analysis requests
    Like a service window where orders are processed
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            analysis_type = data.get('analysis_type')
            input_data = data.get('input_data', {})
            
            # Perform AI analysis based on type
            if analysis_type == 'environmental_forecast':
                results = perform_environmental_forecast(input_data)
            elif analysis_type == 'optimization_suggestions':
                results = generate_optimization_suggestions(input_data)
            elif analysis_type == 'anomaly_detection':
                results = detect_anomalies(input_data)
            else:
                return JsonResponse({'error': 'Unknown analysis type'}, status=400)
            
            # Save analysis results
            analysis = AIAnalysis.objects.create(
                analysis_type=analysis_type,
                input_data=input_data,
                results=results,
                confidence_score=results.get('confidence', 0.8)
            )
            
            return JsonResponse({
                'analysis_id': analysis.id,
                'results': results,
                'timestamp': analysis.created_at.isoformat()
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def perform_environmental_forecast(input_data):
    """Forecast environmental conditions like predicting seasonal changes"""
    sensor_readings = input_data.get('readings', [])
    forecast_hours = input_data.get('forecast_hours', 24)
    
    if not sensor_readings:
        return {
            'forecast': [],
            'confidence': 0.1,
            'error': 'Insufficient data for forecasting'
        }
    
    # Extract values and timestamps
    values = [reading['value'] for reading in sensor_readings]
    
    # Simple time series forecasting using moving average with trend
    window_size = min(5, len(values))
    if len(values) >= window_size:
        recent_avg = np.mean(values[-window_size:])
        older_avg = np.mean(values[-2*window_size:-window_size]) if len(values) >= 2*window_size else recent_avg
        
        # Calculate trend
        trend = (recent_avg - older_avg) / window_size
        
        # Generate forecasts
        forecasts = []
        for hour in range(1, forecast_hours + 1):
            predicted_value = recent_avg + (trend * hour)
            
            # Add some realistic variation
            noise = np.random.normal(0, np.std(values) * 0.1) if len(values) > 1 else 0
            predicted_value += noise
            
            forecasts.append({
                'hour': hour,
                'predicted_value': round(predicted_value, 2),
                'confidence': max(0.3, 0.9 - (hour * 0.02))  # Confidence decreases with time
            })
        
        return {
            'forecast': forecasts,
            'trend': 'increasing' if trend > 0.1 else 'decreasing' if trend < -0.1 else 'stable',
            'confidence': 0.8,
            'model_used': 'moving_average_with_trend'
        }
    
    return {
        'forecast': [],
        'confidence': 0.2,
        'error': 'Need at least 5 readings for reliable forecast'
    }

def generate_optimization_suggestions(input_data):
    """Generate optimization recommendations like suggesting recipe improvements"""
    sensor_data = input_data.get('sensor_data', {})
    city_goals = input_data.get('goals', {})
    
    suggestions = []
    
    # Analyze each sensor type
    for sensor_type, readings in sensor_data.items():
        if not readings:
            continue
            
        current_avg = np.mean([r['value'] for r in readings])
        target = city_goals.get(sensor_type, {}).get('target')
        
        if sensor_type == 'air_quality' and target:
            if current_avg > target:
                improvement_needed = current_avg - target
                suggestions.append({
                    'sensor_type': sensor_type,
                    'priority': 'High' if improvement_needed > 20 else 'Medium',
                    'current_level': current_avg,
                    'target_level': target,
                    'recommendations': [
                        'Implement low-emission zones',
                        'Increase public transportation usage',
                        'Plant more trees in high-pollution areas',
                        'Promote electric vehicle adoption'
                    ],
                    'estimated_timeline': '6-12 months',
                    'expected_improvement': f"{min(improvement_needed * 0.6, improvement_needed):.1f} point reduction"
                })
        
        elif sensor_type == 'energy_usage' and target:
            if current_avg > target:
                excess_usage = current_avg - target
                suggestions.append({
                    'sensor_type': sensor_type,
                    'priority': 'Medium',
                    'current_level': current_avg,
                    'target_level': target,
                    'recommendations': [
                        'Upgrade to LED street lighting',
                        'Install smart grid systems',
                        'Implement building energy efficiency programs',
                        'Deploy renewable energy sources'
                    ],
                    'estimated_timeline': '3-8 months',
                    'expected_improvement': f"{excess_usage * 0.25:.1f}% efficiency gain"
                })
        
        elif sensor_type == 'noise_level' and target:
            if current_avg > target:
                noise_excess = current_avg - target
                suggestions.append({
                    'sensor_type': sensor_type,
                    'priority': 'Low' if noise_excess < 5 else 'Medium',
                    'current_level': current_avg,
                    'target_level': target,
                    'recommendations': [
                        'Install noise barriers near highways',
                        'Implement quiet hours enforcement',
                        'Create more green spaces',
                        'Optimize traffic flow patterns'
                    ],
                    'estimated_timeline': '2-6 months',
                    'expected_improvement': f"{min(noise_excess * 0.4, noise_excess):.1f} dB reduction"
                })
    
    return {
        'suggestions': suggestions,
        'total_recommendations': len(suggestions),
        'confidence': 0.85,
        'generated_at': datetime.now().isoformat()
    }

def detect_anomalies(input_data):
    """Detect unusual patterns like identifying off-flavors"""
    readings = input_data.get('readings', [])
    sensitivity = input_data.get('sensitivity', 'medium')
    
    if len(readings) < 10:
        return {
            'anomalies': [],
            'confidence': 0.3,
            'error': 'Need at least 10 readings for anomaly detection'
        }
    
    values = [r['value'] for r in readings]
    timestamps = [r['timestamp'] for r in readings]
    
    # Statistical anomaly detection
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Set threshold based on sensitivity
    threshold_multiplier = {'low': 3.0, 'medium': 2.5, 'high': 2.0}
    threshold = threshold_multiplier.get(sensitivity, 2.5)
    
    anomalies = []
    for i, (value, timestamp) in enumerate(zip(values, timestamps)):
        z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
        
        if z_score > threshold:
            anomaly_type = 'spike' if value > mean_val else 'drop'
            severity = 'critical' if z_score > 4 else 'warning' if z_score > 3 else 'minor'
            
            anomalies.append({
                'index': i,
                'timestamp': timestamp,
                'value': value,
                'z_score': round(z_score, 2),
                'anomaly_type': anomaly_type,
                'severity': severity,
                'description': f"Value {value} is {z_score:.1f} standard deviations from normal"
            })
    
    return {
        'anomalies': anomalies,
        'total_anomalies': len(anomalies),
        'anomaly_rate': round((len(anomalies) / len(readings)) * 100, 1),
        'confidence': 0.9,
        'statistical_summary': {
            'mean': round(mean_val, 2),
            'std_deviation': round(std_val, 2),
            'threshold_used': threshold
        }
    }

# Django URLs configuration (urls.py)
from django.urls import path
from . import views

urlpatterns = [
    path('', views.EnvironmentalDashboardView.as_view(), name='dashboard'),
    path('api/analysis/', views.ai_analysis_endpoint, name='ai_analysis'),
    path('api/sensor/<int:sensor_id>/readings/', views.get_sensor_readings, name='sensor_readings'),
]

# Django template (environmental_dashboard.html)
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart City Environmental Monitor</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .dashboard { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }
        .city-overview { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #333; }
        .metric-label { color: #666; margin-top: 10px; }
        .sensors-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .sensor-card { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; }
        .sensor-header { padding: 20px; border-bottom: 1px solid #eee; }
        .sensor-content { padding: 20px; }
        .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }
        .status-online { background: #d4edda; color: #155724; }
        .status-offline { background: #f8d7da; color: #721c24; }
        .chart-container { height: 200px; margin: 20px 0; }
        .ai-insights { background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px; }
        .insight-item { margin: 8px 0; }
        .trend-improving { color: #28a745; }
        .trend-declining { color: #dc3545; }
        .trend-stable { color: #6c757d; }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>Smart City Environmental Monitor</h1>
            <p>AI-Powered Environmental Intelligence for Sustainable Urban Living</p>
        </div>
        
        <div class="city-overview">
            <div class="metric-card">
                <div class="metric-value">{{ city_overview.overall_score }}</div>
                <div class="metric-label">Environmental Health Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ city_overview.online_sensors }}</div>
                <div class="metric-label">Active Sensors</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ city_overview.uptime_percentage }}%</div>
                <div class="metric-label">System Uptime</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ city_overview.status }}</div>
                <div class="metric-label">Overall Status</div>
            </div>
        </div>
        
        <div class="sensors-grid">
            {% for sensor_data in sensors_data %}
            <div class="sensor-card">
                <div class="sensor-header">
                    <h3>{{ sensor_data.sensor.get_sensor_type_display }}</h3>
                    <span class="status-badge status-{{ sensor_data.status|lower }}">
                        {{ sensor_data.status }}
                    </span>
                    <p>Location: {{ sensor_data.sensor.location_lat }}, {{ sensor_data.sensor.location_lng }}</p>
                </div>
                <div class="sensor-content">
                    <div class="chart-container">
                        <canvas id="chart-{{ sensor_data.sensor.id }}"></canvas>
                    </div>
                    
                    <div class="ai-insights">
                        <h4>AI Insights</h4>
                        <div class="insight-item">
                            <strong>Trend:</strong> 
                            <span class="trend-{{ sensor_data.insights.trend|lower }}">
                                {{ sensor_data.insights.trend }}
                            </span>
                        </div>
                        <div class="insight-item">
                            <strong>Forecast:</strong> {{ sensor_data.insights.forecast }}
                        </div>
                        <div class="insight-item">
                            <strong>Data Quality:</strong> {{ sensor_data.insights.data_quality|floatformat:2 }}
                        </div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    
    <script>
        // Initialize charts for each sensor
        {% for sensor_data in sensors_data %}
        (function() {
            const ctx = document.getElementById('chart-{{ sensor_data.sensor.id }}').getContext('2d');
            const readings = {{ sensor_data.recent_readings|safe }};
            
            // Prepare chart data
            const labels = readings.map(r => new Date(r.timestamp).toLocaleTimeString());
            const values = readings.map(r => r.value);
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels.reverse(),
                    datasets: [{
                        label: '{{ sensor_data.sensor.get_sensor_type_display }}',
                        data: values.reverse(),
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            grid: {
                                color: 'rgba(0,0,0,0.1)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            });
        })();
        {% endfor %}
        
        // Auto-refresh dashboard every 5 minutes
        setInterval(() => {
            location.reload();
        }, 300000);
    </script>
</body>
</html>

# Additional Django view for API endpoint
def get_sensor_readings(request, sensor_id):
    """API endpoint to get specific sensor readings"""
    try:
        sensor = get_object_or_404(EnvironmentalSensor, id=sensor_id)
        readings = SensorReading.objects.filter(sensor=sensor).order_by('-timestamp')[:100]
        
        readings_data = [{
            'timestamp': reading.timestamp.isoformat(),
            'value': reading.value,
            'unit': reading.unit,
            'quality_score': reading.quality_score
        } for reading in readings]
        
        return JsonResponse({
            'sensor_info': {
                'id': sensor.id,
                'type': sensor.sensor_type,
                'location': {
                    'lat': float(sensor.location_lat),
                    'lng': float(sensor.location_lng)
                }
            },
            'readings': readings_data,
            'total_count': len(readings_data)
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Django settings configuration (settings.py additions)
"""
# Add to your INSTALLED_APPS
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'environmental_monitor',  # Your app name
]

# Database configuration for production
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'smart_city_db',
        'USER': 'your_db_user',
        'PASSWORD': 'your_db_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# API Rate limiting (optional)
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour'
    }
# Django Management Command (management/commands/simulate_sensor_data.py)
from django.core.management.base import BaseCommand
from environmental_monitor.models import EnvironmentalSensor, SensorReading
import random
from datetime import datetime, timedelta

class Command(BaseCommand):
    help = 'Generate simulated sensor data for testing'
    
    def add_arguments(self, parser):
        parser.add_argument('--days', type=int, default=7, help='Number of days of data to generate')
        parser.add_argument('--sensors', type=int, default=5, help='Number of sensors to create')
    
    def handle(self, *args, **options):
        """
        Generate test data like preparing sample ingredients for practice
        """
        days = options['days']
        sensor_count = options['sensors']
        
        # Create sensors if they don't exist
        sensor_types = ['air_quality', 'noise_level', 'traffic_density', 'energy_usage']
        
        for i in range(sensor_count):
            sensor, created = EnvironmentalSensor.objects.get_or_create(
                sensor_id=f'SENSOR_{i+1:03d}',
                defaults={
                    'sensor_type': sensor_types[i % len(sensor_types)],
                    'location_lat': 37.7749 + random.uniform(-0.1, 0.1),  # San Francisco area
                    'location_lng': -122.4194 + random.uniform(-0.1, 0.1),
                    'is_active': True
                }
            )
            
            if created:
                self.stdout.write(f'Created sensor: {sensor.sensor_id}')
            
            # Generate readings for the specified number of days
            start_date = datetime.now() - timedelta(days=days)
            
            for day in range(days):
                current_date = start_date + timedelta(days=day)
                
                # Generate 24 readings per day (hourly)
                for hour in range(24):
                    timestamp = current_date + timedelta(hours=hour)
                    
                    # Generate realistic values based on sensor type
                    if sensor.sensor_type == 'air_quality':
                        # AQI values (0-500)
                        base_value = 50 + random.uniform(-20, 40)
                        value = max(0, base_value + random.uniform(-10, 10))
                        unit = 'AQI'
                    elif sensor.sensor_type == 'noise_level':
                        # Decibel levels (30-90)
                        base_value = 55 + random.uniform(-15, 25)
                        value = max(30, min(90, base_value + random.uniform(-5, 5)))
                        unit = 'dB'
                    elif sensor.sensor_type == 'traffic_density':
                        # Vehicles per hour (0-200)
                        base_value = 50 + random.uniform(-30, 80)
                        value = max(0, base_value + random.uniform(-20, 20))
                        unit = 'vehicles/hour'
                    else:  # energy_usage
                        # kWh (0-100)
                        base_value = 40 + random.uniform(-20, 30)
                        value = max(0, base_value + random.uniform(-10, 10))
                        unit = 'kWh'
                    
                    # Add some quality variation
                    quality_score = random.uniform(0.8, 1.0)
                    
                    SensorReading.objects.create(
                        sensor=sensor,
                        timestamp=timestamp,
                        value=round(value, 2),
                        unit=unit,
                        quality_score=quality_score
                    )
            
            self.stdout.write(f'Generated {days * 24} readings for {sensor.sensor_id}')
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {sensor_count} sensors with {days} days of data')
        )

# Requirements.txt file
"""
Django>=4.2.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
opencv-python>=4.5.0
yfinance>=0.1.70
psycopg2-binary>=2.9.0
djangorestframework>=3.14.0
django-cors-headers>=3.13.0
celery>=5.2.0
redis>=4.3.0
"""

# Deployment script (deploy.sh)
"""
#!/bin/bash
# Smart City Environmental Monitor Deployment Script

echo "Starting deployment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser (interactive)
echo "Creating admin user..."
python manage.py createsuperuser

# Generate sample data
python manage.py simulate_sensor_data --days 30 --sensors 10

# Collect static files
python manage.py collectstatic --noinput

# Start development server
echo "Starting server..."
python manage.py runserver 0.0.0.0:8000
"""
```

**Project Code Syntax Explanation:**
- `models.ForeignKey(Model, on_delete=models.CASCADE)` creates a relationship where deleting the referenced object also deletes related objects
- `models.JSONField()` stores structured data as JSON in the database (available in Django 3.1+)
- `@csrf_exempt` decorator bypasses CSRF protection for API endpoints (use carefully in production)
- `ListView` is a Django generic view that automatically handles listing objects
- `super().get_context_data(**kwargs)` calls the parent class method and extends its functionality

---

# Day 98: Build - Industry-specific AI Solution

## Project: Smart Restaurant Analytics Platform

We'll create a comprehensive AI-powered restaurant management system that combines multiple industry applications. Think of this as building the ultimate command center for a modern restaurant - where every ingredient (data point) is tracked, every recipe (algorithm) is optimized, and every service (prediction) enhances the dining experience.

### Core Django Project Structure

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
    'rest_framework',
    'corsheaders',
    'restaurant_ai',  # Our main app
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

ROOT_URLCONF = 'restaurant_ai_project.urls'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'restaurant_ai_db',
        'USER': 'postgres',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# AI Model Settings
AI_MODELS_PATH = os.path.join(BASE_DIR, 'ai_models')
TENSORFLOW_SERVING_URL = 'http://localhost:8501'
```

### Models - The Recipe Database

```python
# restaurant_ai/models.py
from django.db import models
from django.contrib.auth.models import User
import json

class Restaurant(models.Model):
    name = models.CharField(max_length=200)
    location = models.CharField(max_length=300)
    cuisine_type = models.CharField(max_length=100)
    capacity = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class MenuItem(models.Model):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    category = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=8, decimal_places=2)
    ingredients = models.JSONField(default=list)
    nutritional_info = models.JSONField(default=dict)
    popularity_score = models.FloatField(default=0.0)
    
    def __str__(self):
        return f"{self.name} - {self.restaurant.name}"

class SalesData(models.Model):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE)
    date = models.DateField()
    quantity_sold = models.IntegerField()
    revenue = models.DecimalField(max_digits=10, decimal_places=2)
    weather_condition = models.CharField(max_length=100)
    day_of_week = models.CharField(max_length=20)
    season = models.CharField(max_length=20)
    
class CustomerFeedback(models.Model):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE, null=True, blank=True)
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])
    comment = models.TextField()
    sentiment_score = models.FloatField(null=True, blank=True)
    date_created = models.DateTimeField(auto_now_add=True)

class PredictionLog(models.Model):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    prediction_type = models.CharField(max_length=100)
    input_data = models.JSONField()
    prediction_result = models.JSONField()
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
```

### AI Services - The Master's Techniques

```python
# restaurant_ai/services.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report
import tensorflow as tf
from textblob import TextBlob
import joblib
import os
from django.conf import settings
from .models import SalesData, CustomerFeedback, MenuItem, Restaurant

class RestaurantAIService:
    """
    The master service that orchestrates all AI capabilities
    Like a head chef coordinating multiple cooking stations
    """
    
    def __init__(self):
        self.models_path = settings.AI_MODELS_PATH
        os.makedirs(self.models_path, exist_ok=True)
        
    def train_demand_forecasting_model(self, restaurant_id):
        """
        Predicts future demand like anticipating how many guests 
        will want each dish tomorrow
        """
        # Gather historical sales data
        sales_data = SalesData.objects.filter(restaurant_id=restaurant_id)
        
        if not sales_data.exists():
            raise ValueError("Insufficient sales data for training")
            
        # Prepare the data ingredients
        df = pd.DataFrame(list(sales_data.values()))
        
        # Feature engineering - like preparing mise en place
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['day_of_month'] = pd.to_datetime(df['date']).dt.day
        df['is_weekend'] = pd.to_datetime(df['date']).dt.dayofweek >= 5
        
        # Encode categorical variables
        le_weather = LabelEncoder()
        le_season = LabelEncoder()
        le_day = LabelEncoder()
        
        df['weather_encoded'] = le_weather.fit_transform(df['weather_condition'])
        df['season_encoded'] = le_season.fit_transform(df['season'])
        df['day_encoded'] = le_day.fit_transform(df['day_of_week'])
        
        # Features and target
        features = ['menu_item_id', 'month', 'day_of_month', 'weather_encoded', 
                   'season_encoded', 'day_encoded', 'is_weekend']
        X = df[features]
        y = df['quantity_sold']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model - like perfecting a signature dish
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Test the model
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        
        # Save the model and encoders
        model_path = os.path.join(self.models_path, f'demand_model_{restaurant_id}.joblib')
        joblib.dump({
            'model': model,
            'weather_encoder': le_weather,
            'season_encoder': le_season,
            'day_encoder': le_day,
            'mae': mae
        }, model_path)
        
        return {'model_accuracy': mae, 'model_path': model_path}
    
    def predict_demand(self, restaurant_id, menu_item_id, date, weather, season, day_of_week):
        """
        Predicts demand for a specific dish on a given day
        Like a seasoned chef knowing exactly how much to prep
        """
        model_path = os.path.join(self.models_path, f'demand_model_{restaurant_id}.joblib')
        
        if not os.path.exists(model_path):
            raise ValueError("Model not trained yet. Please train the model first.")
            
        # Load the saved model and encoders
        saved_data = joblib.load(model_path)
        model = saved_data['model']
        weather_encoder = saved_data['weather_encoder']
        season_encoder = saved_data['season_encoder']
        day_encoder = saved_data['day_encoder']
        
        # Prepare input data
        date_obj = pd.to_datetime(date)
        features = np.array([[
            menu_item_id,
            date_obj.month,
            date_obj.day,
            weather_encoder.transform([weather])[0],
            season_encoder.transform([season])[0],
            day_encoder.transform([day_of_week])[0],
            1 if date_obj.dayofweek >= 5 else 0
        ]])
        
        prediction = model.predict(features)[0]
        return max(0, int(prediction))  # Ensure non-negative prediction
    
    def analyze_customer_sentiment(self, feedback_text):
        """
        Analyzes customer reviews like tasting and adjusting seasoning
        """
        blob = TextBlob(feedback_text)
        sentiment_score = blob.sentiment.polarity
        
        # Classify sentiment
        if sentiment_score > 0.1:
            sentiment_category = "Positive"
        elif sentiment_score < -0.1:
            sentiment_category = "Negative"
        else:
            sentiment_category = "Neutral"
            
        return {
            'sentiment_score': sentiment_score,
            'sentiment_category': sentiment_category,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def optimize_menu_pricing(self, restaurant_id):
        """
        Suggests optimal pricing like balancing flavors in a signature dish
        """
        # Get menu items and sales data
        menu_items = MenuItem.objects.filter(restaurant_id=restaurant_id)
        pricing_suggestions = []
        
        for item in menu_items:
            sales = SalesData.objects.filter(menu_item=item)
            
            if sales.exists():
                # Calculate metrics
                avg_daily_sales = sales.aggregate(
                    avg_quantity=models.Avg('quantity_sold'),
                    avg_revenue=models.Avg('revenue')
                )
                
                # Simple price optimization logic
                current_price = float(item.price)
                avg_quantity = avg_daily_sales['avg_quantity'] or 0
                
                # If high demand, suggest slight price increase
                if avg_quantity > 20:
                    suggested_price = current_price * 1.05
                    reason = "High demand allows for premium pricing"
                # If low demand, suggest price decrease
                elif avg_quantity < 5:
                    suggested_price = current_price * 0.95
                    reason = "Lower price may increase demand"
                else:
                    suggested_price = current_price
                    reason = "Current pricing appears optimal"
                
                pricing_suggestions.append({
                    'menu_item': item.name,
                    'current_price': current_price,
                    'suggested_price': round(suggested_price, 2),
                    'avg_daily_sales': avg_quantity,
                    'reason': reason
                })
        
        return pricing_suggestions
    
    def generate_sustainability_report(self, restaurant_id):
        """
        Analyzes environmental impact like managing kitchen waste efficiently
        """
        restaurant = Restaurant.objects.get(id=restaurant_id)
        menu_items = MenuItem.objects.filter(restaurant=restaurant)
        
        sustainability_metrics = {
            'total_items': menu_items.count(),
            'vegetarian_percentage': 0,
            'carbon_footprint_estimate': 0,
            'waste_reduction_suggestions': []
        }
        
        vegetarian_count = 0
        total_carbon_score = 0
        
        for item in menu_items:
            ingredients = item.ingredients
            
            # Check if vegetarian
            meat_ingredients = ['beef', 'chicken', 'pork', 'fish', 'lamb']
            if not any(meat in str(ingredients).lower() for meat in meat_ingredients):
                vegetarian_count += 1
            
            # Estimate carbon footprint (simplified)
            carbon_score = len(ingredients) * 2.5  # Simplified calculation
            if any(meat in str(ingredients).lower() for meat in meat_ingredients):
                carbon_score *= 2.5  # Higher impact for meat
            
            total_carbon_score += carbon_score
        
        sustainability_metrics['vegetarian_percentage'] = (vegetarian_count / menu_items.count() * 100) if menu_items.count() > 0 else 0
        sustainability_metrics['carbon_footprint_estimate'] = total_carbon_score
        
        # Generate suggestions
        if sustainability_metrics['vegetarian_percentage'] < 30:
            sustainability_metrics['waste_reduction_suggestions'].append(
                "Consider adding more plant-based options to reduce environmental impact"
            )
        
        if total_carbon_score > 500:
            sustainability_metrics['waste_reduction_suggestions'].append(
                "Source ingredients locally to reduce transportation emissions"
            )
            
        return sustainability_metrics

class CreativeAIService:
    """
    Handles creative AI applications like a chef experimenting with fusion cuisine
    """
    
    def generate_recipe_suggestions(self, base_ingredients, cuisine_style="fusion"):
        """
        Creates new recipe ideas like a chef creating daily specials
        """
        # This would typically use a more sophisticated AI model
        # For demonstration, we'll use rule-based generation
        
        ingredient_combinations = {
            'italian': ['tomato', 'basil', 'mozzarella', 'olive oil'],
            'asian': ['soy sauce', 'ginger', 'garlic', 'sesame oil'],
            'mexican': ['lime', 'cilantro', 'jalapeño', 'cumin'],
            'mediterranean': ['lemon', 'oregano', 'feta', 'olive oil']
        }
        
        suggestions = []
        for style, style_ingredients in ingredient_combinations.items():
            if any(ingredient in base_ingredients for ingredient in style_ingredients):
                suggestions.append({
                    'style': style,
                    'suggested_ingredients': style_ingredients[:2],
                    'preparation_method': f"{style.capitalize()} fusion preparation"
                })
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def optimize_menu_layout(self, menu_items_data):
        """
        Suggests optimal menu organization like arranging dishes for visual appeal
        """
        # Analyze menu items by category and popularity
        categories = {}
        for item in menu_items_data:
            category = item.get('category', 'Other')
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
        
        # Sort by popularity within categories
        optimized_layout = {}
        for category, items in categories.items():
            sorted_items = sorted(items, 
                                key=lambda x: x.get('popularity_score', 0), 
                                reverse=True)
            optimized_layout[category] = sorted_items
        
        return optimized_layout
```

### Views - The Service Counter

```python
# restaurant_ai/views.py
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import json
from .models import Restaurant, MenuItem, SalesData, CustomerFeedback, PredictionLog
from .services import RestaurantAIService, CreativeAIService
from .serializers import RestaurantSerializer, MenuItemSerializer, SalesDataSerializer

@api_view(['POST'])
def train_ai_model(request, restaurant_id):
    """
    Train the AI model for demand forecasting
    Like teaching a new cook the restaurant's signature techniques
    """
    try:
        ai_service = RestaurantAIService()
        result = ai_service.train_demand_forecasting_model(restaurant_id)
        
        return Response({
            'status': 'success',
            'message': 'AI model trained successfully',
            'accuracy': result['model_accuracy']
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def predict_demand(request):
    """
    Predict demand for menu items
    Like forecasting how busy each cooking station will be
    """
    try:
        data = request.data
        ai_service = RestaurantAIService()
        
        prediction = ai_service.predict_demand(
            restaurant_id=data['restaurant_id'],
            menu_item_id=data['menu_item_id'],
            date=data['date'],
            weather=data['weather'],
            season=data['season'],
            day_of_week=data['day_of_week']
        )
        
        # Log the prediction
        PredictionLog.objects.create(
            restaurant_id=data['restaurant_id'],
            prediction_type='demand_forecast',
            input_data=data,
            prediction_result={'predicted_demand': prediction},
            confidence_score=0.85  # This would come from the model
        )
        
        return Response({
            'predicted_demand': prediction,
            'date': data['date'],
            'menu_item_id': data['menu_item_id']
        })
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def analyze_feedback(request):
    """
    Analyze customer feedback sentiment
    Like tasting each dish before it leaves the kitchen
    """
    try:
        feedback_text = request.data.get('feedback_text', '')
        ai_service = RestaurantAIService()
        
        sentiment_analysis = ai_service.analyze_customer_sentiment(feedback_text)
        
        return Response(sentiment_analysis)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def pricing_optimization(request, restaurant_id):
    """
    Get pricing optimization suggestions
    Like adjusting recipes for perfect flavor balance
    """
    try:
        ai_service = RestaurantAIService()
        suggestions = ai_service.optimize_menu_pricing(restaurant_id)
        
        return Response({
            'pricing_suggestions': suggestions
        })
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def sustainability_report(request, restaurant_id):
    """
    Generate sustainability analytics
    Like maintaining a zero-waste kitchen philosophy
    """
    try:
        ai_service = RestaurantAIService()
        report = ai_service.generate_sustainability_report(restaurant_id)
        
        return Response(report)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def creative_recipe_generation(request):
    """
    Generate creative recipe suggestions
    Like brainstorming new fusion dishes
    """
    try:
        base_ingredients = request.data.get('ingredients', [])
        cuisine_style = request.data.get('cuisine_style', 'fusion')
        
        creative_service = CreativeAIService()
        suggestions = creative_service.generate_recipe_suggestions(base_ingredients, cuisine_style)
        
        return Response({
            'recipe_suggestions': suggestions
        })
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def restaurant_dashboard(request, restaurant_id):
    """
    Comprehensive dashboard view
    Like the main command center overseeing all operations
    """
    try:
        restaurant = get_object_or_404(Restaurant, id=restaurant_id)
        ai_service = RestaurantAIService()
        
        # Get recent sales data
        recent_sales = SalesData.objects.filter(restaurant=restaurant).order_by('-date')[:10]
        
        # Get sustainability metrics
        sustainability = ai_service.generate_sustainability_report(restaurant_id)
        
        # Get pricing suggestions
        pricing = ai_service.optimize_menu_pricing(restaurant_id)
        
        # Get recent predictions
        recent_predictions = PredictionLog.objects.filter(restaurant=restaurant).order_by('-created_at')[:5]
        
        dashboard_data = {
            'restaurant': RestaurantSerializer(restaurant).data,
            'recent_sales': SalesDataSerializer(recent_sales, many=True).data,
            'sustainability_metrics': sustainability,
            'pricing_suggestions': pricing[:5],  # Top 5 suggestions
            'recent_predictions': [
                {
                    'type': p.prediction_type,
                    'result': p.prediction_result,
                    'confidence': p.confidence_score,
                    'date': p.created_at.strftime('%Y-%m-%d %H:%M')
                } for p in recent_predictions
            ]
        }
        
        return Response(dashboard_data)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)
```

### Frontend Dashboard

```html
<!-- restaurant_ai/templates/dashboard.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Restaurant AI Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .card h3 { margin-top: 0; color: #333; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-value { font-weight: bold; color: #007bff; }
        .suggestion-item { padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 5px; }
        .btn { padding: 10px 15px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .status-positive { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-negative { color: #dc3545; }
    </style>
</head>
<body>
    <h1>Smart Restaurant Analytics Platform</h1>
    
    <div class="dashboard" id="dashboard">
        <!-- Dashboard content will be loaded here -->
    </div>

    <script>
        class RestaurantDashboard {
            constructor(restaurantId) {
                this.restaurantId = restaurantId;
                this.baseUrl = '/api/restaurant';
                this.init();
            }

            async init() {
                await this.loadDashboardData();
                this.setupEventListeners();
            }

            async loadDashboardData() {
                try {
                    const response = await fetch(`${this.baseUrl}/${this.restaurantId}/dashboard/`);
                    const data = await response.json();
                    this.renderDashboard(data);
                } catch (error) {
                    console.error('Error loading dashboard data:', error);
                }
            }

            renderDashboard(data) {
                const dashboard = document.getElementById('dashboard');
                
                dashboard.innerHTML = `
                    <div class="card">
                        <h3>Restaurant Overview</h3>
                        <div class="metric">
                            <span>Name:</span>
                            <span class="metric-value">${data.restaurant.name}</span>
                        </div>
                        <div class="metric">
                            <span>Location:</span>
                            <span class="metric-value">${data.restaurant.location}</span>
                        </div>
                        <div class="metric">
                            <span>Cuisine:</span>
                            <span class="metric-value">${data.restaurant.cuisine_type}</span>
                        </div>
                        <div class="metric">
                            <span>Capacity:</span>
                            <span class="metric-value">${data.restaurant.capacity}</span>
                        </div>
                    </div>

                    <div class="card">
                        <h3>Sustainability Metrics</h3>
                        <div class="metric">
                            <span>Vegetarian Items:</span>
                            <span class="metric-value ${data.sustainability_metrics.vegetarian_percentage > 30 ? 'status-positive' : 'status-warning'}">
                                ${data.sustainability_metrics.vegetarian_percentage.toFixed(1)}%
                            </span>
                        </div>
                        <div class="metric">
                            <span>Carbon Footprint Score:</span>
                            <span class="metric-value ${data.sustainability_metrics.carbon_footprint_estimate < 300 ? 'status-positive' : 'status-warning'}">
                                ${data.sustainability_metrics.carbon_footprint_estimate}
                            </span>
                        </div>
                        <h4>Recommendations:</h4>
                        ${data.sustainability_metrics.waste_reduction_suggestions.map(suggestion => 
                            `<div class="suggestion-item">${suggestion}</div>`
                        ).join('')}
                    </div>

                    <div class="card">
                        <h3>AI Demand Predictions</h3>
                        <div id="demandChart">
                            <canvas id="demandCanvas" width="400" height="200"></canvas>
                        </div>
                        <button class="btn" onclick="dashboard.showPredictionForm()">Make New Prediction</button>
                    </div>

                    <div class="card">
                        <h3>Pricing Optimization</h3>
                        ${data.pricing_suggestions.slice(0, 5).map(item => `
                            <div class="suggestion-item">
                                <strong>${item.menu_item}</strong><br>
                                Current: $${item.current_price} → Suggested: $${item.suggested_price}<br>
                                <small>${item.reason}</small>
                            </div>
                        `).join('')}
                    </div>

                    <div class="card">
                        <h3>Recent AI Predictions</h3>
                        ${data.recent_predictions.map(pred => `
                            <div class="suggestion-item">
                                <strong>${pred.type}</strong> - Confidence: ${(pred.confidence * 100).toFixed(1)}%<br>
                                <small>${pred.date}</small>
                            </div>
                        `).join('')}
                    </div>

                    <div class="card">
                        <h3>AI Training Center</h3>
                        <button class="btn" onclick="dashboard.trainModel()">Train Demand Forecasting Model</button>
                        <button class="btn" onclick="dashboard.showFeedbackAnalysis()">Analyze Customer Feedback</button>
                        <button class="btn" onclick="dashboard.generateRecipes()">Generate Recipe Ideas</button>
                    </div>
                `;

                this.renderDemandChart(data.recent_sales);
            }

            renderDemandChart(salesData) {
                const ctx = document.getElementById('demandCanvas').getContext('2d');
                
                const chartData = {
                    labels: salesData.map(sale => sale.date),
                    datasets: [{
                        label: 'Daily Sales Quantity',
                        data: salesData.map(sale => sale.quantity_sold),
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        tension: 0.4
                    }]
                };

                new Chart(ctx, {
                    type: 'line',
                    data: chartData,
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'top',
                            },
                            title: {
                                display: true,
                                text: 'Recent Sales Trend'
                            }
                        }
                    }
                });
            }

            async trainModel() {
                try {
                    const response = await fetch(`${this.baseUrl}/${this.restaurantId}/train-ai/`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': this.getCSRFToken()
                        }
                    });
                    
                    const result = await response.json();
                    alert(`Model training completed! Accuracy: ${result.accuracy?.toFixed(3) || 'N/A'}`);
                    
                } catch (error) {
                    alert('Error training model: ' + error.message);
                }
            }

            showPredictionForm() {
                const form = prompt("Enter prediction data (format: menu_item_id,date,weather,season,day_of_week):");
                if (form) {
                    const [menuItemId, date, weather, season, dayOfWeek] = form.split(',');
                    this.makePrediction(menuItemId.trim(), date.trim(), weather.trim(), season.trim(), dayOfWeek.trim());
                }
            }

            async makePrediction(menuItemId, date, weather, season, dayOfWeek) {
                try {
                    const response = await fetch('/api/predict-demand/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': this.getCSRFToken()
                        },
                        body: JSON.stringify({
                            restaurant_id: this.restaurantId,
                            menu_item_id: parseInt(menuItemId),
                            date: date,
                            weather: weather,
                            season: season,
                            day_of_week: dayOfWeek
                        })
                    });
                    
                    const result = await response.json();
                    alert(`Predicted demand: ${result.predicted_demand} units for ${date}`);
                    this.loadDashboardData(); // Refresh dashboard
                    
                } catch (error) {
                    alert('Error making prediction: ' + error.message);
                }
            }

            showFeedbackAnalysis() {
                const feedback = prompt("Enter customer feedback text to analyze:");
                if (feedback) {
                    this.analyzeFeedback(feedback);
                }
            }

            async analyzeFeedback(feedbackText) {
                try {
                    const response = await fetch('/api/analyze-feedback/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': this.getCSRFToken()
                        },
                        body: JSON.stringify({
                            feedback_text: feedbackText
                        })
                    });
                    
                    const result = await response.json();
                    alert(`Sentiment Analysis:
                    Category: ${result.sentiment_category}
                    Score: ${result.sentiment_score.toFixed(3)}
                    Subjectivity: ${result.subjectivity.toFixed(3)}`);
                    
                } catch (error) {
                    alert('Error analyzing feedback: ' + error.message);
                }
            }

            async generateRecipes() {
                const ingredients = prompt("Enter base ingredients (comma-separated):");
                const style = prompt("Enter cuisine style (optional, default: fusion):") || "fusion";
                
                if (ingredients) {
                    try {
                        const response = await fetch('/api/creative-recipes/', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-CSRFToken': this.getCSRFToken()
                            },
                            body: JSON.stringify({
                                ingredients: ingredients.split(',').map(i => i.trim()),
                                cuisine_style: style
                            })
                        });
                        
                        const result = await response.json();
                        const suggestions = result.recipe_suggestions.map(r => 
                            `${r.style}: ${r.suggested_ingredients.join(', ')} - ${r.preparation_method}`
                        ).join('\n');
                        
                        alert(`Recipe Suggestions:\n${suggestions}`);
                        
                    } catch (error) {
                        alert('Error generating recipes: ' + error.message);
                    }
                }
            }

            getCSRFToken() {
                return document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
            }
        }

        // Initialize dashboard when page loads
        let dashboard;
        document.addEventListener('DOMContentLoaded', function() {
            // You would get the restaurant ID from the URL or user session
            const restaurantId = 1; // This should be dynamic in a real application
            dashboard = new RestaurantDashboard(restaurantId);
        });
    </script>
    


### URL Configuration

```python
# restaurant_ai/urls.py
from django.urls import path
from . import views

app_name = 'restaurant_ai'

urlpatterns = [
    # Dashboard
    path('restaurant/<int:restaurant_id>/dashboard/', views.restaurant_dashboard, name='restaurant_dashboard'),
    
    # AI Model Training and Prediction
    path('restaurant/<int:restaurant_id>/train-ai/', views.train_ai_model, name='train_ai_model'),
    path('predict-demand/', views.predict_demand, name='predict_demand'),
    
    # Analysis Endpoints
    path('analyze-feedback/', views.analyze_feedback, name='analyze_feedback'),
    path('restaurant/<int:restaurant_id>/pricing/', views.pricing_optimization, name='pricing_optimization'),
    path('restaurant/<int:restaurant_id>/sustainability/', views.sustainability_report, name='sustainability_report'),
    
    # Creative AI
    path('creative-recipes/', views.creative_recipe_generation, name='creative_recipes'),
]
```

```python
# Main project urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('restaurant_ai.urls')),
    path('', include('restaurant_ai.urls')),  # For frontend routes
]
```

### Serializers - Data Presentation Layer

```python
# restaurant_ai/serializers.py
from rest_framework import serializers
from .models import Restaurant, MenuItem, SalesData, CustomerFeedback, PredictionLog

class RestaurantSerializer(serializers.ModelSerializer):
    class Meta:
        model = Restaurant
        fields = '__all__'

class MenuItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = MenuItem
        fields = '__all__'

class SalesDataSerializer(serializers.ModelSerializer):
    menu_item_name = serializers.CharField(source='menu_item.name', read_only=True)
    
    class Meta:
        model = SalesData
        fields = '__all__'

class CustomerFeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomerFeedback
        fields = '__all__'

class PredictionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionLog
        fields = '__all__'
```

### Advanced AI Components

```python
# restaurant_ai/ai_models.py
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

class DeepDemandPredictor:
    """
    Advanced neural network for demand prediction
    Like having a master chef's intuitive understanding of timing and portions
    """
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 7  # Use 7 days of history
        
    def build_model(self, input_shape):
        """Build LSTM model for time series prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def prepare_sequences(self, data):
        """Prepare data sequences like organizing ingredients in order of use"""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(data)):
            sequences.append(data[i-self.sequence_length:i])
            targets.append(data[i])
            
        return np.array(sequences), np.array(targets)
    
    def train(self, sales_data):
        """Train the deep learning model"""
        # Normalize the data
        scaled_data = self.scaler.fit_transform(sales_data.reshape(-1, 1))
        
        # Create sequences
        X, y = self.prepare_sequences(scaled_data.flatten())
        
        # Build and train model
        self.model = self.build_model((X.shape[1], 1))
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Train the model
        history = self.model.fit(X, y, epochs=50, batch_size=32, 
                               validation_split=0.2, verbose=0)
        
        return history
    
    def predict(self, recent_data):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Scale the input data
        scaled_input = self.scaler.transform(recent_data.reshape(-1, 1))
        
        # Prepare sequence
        if len(scaled_input) >= self.sequence_length:
            input_sequence = scaled_input[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        else:
            # Pad with zeros if insufficient data
            padded = np.zeros((self.sequence_length, 1))
            padded[-len(scaled_input):] = scaled_input
            input_sequence = padded.reshape(1, self.sequence_length, 1)
        
        # Make prediction
        prediction = self.model.predict(input_sequence, verbose=0)
        
        # Scale back to original range
        return self.scaler.inverse_transform(prediction)[0][0]

class NutritionalOptimizer:
    """
    AI system for nutritional optimization
    Like a nutritionist working alongside the chef
    """
    
    def __init__(self):
        self.nutritional_database = {
            'chicken': {'protein': 31, 'carbs': 0, 'fat': 3.6, 'calories': 165},
            'beef': {'protein': 26, 'carbs': 0, 'fat': 15, 'calories': 250},
            'salmon': {'protein': 25, 'carbs': 0, 'fat': 11, 'calories': 206},
            'rice': {'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'calories': 130},
            'broccoli': {'protein': 2.8, 'carbs': 7, 'fat': 0.4, 'calories': 34},
            'spinach': {'protein': 2.9, 'carbs': 3.6, 'fat': 0.4, 'calories': 23},
            'tomato': {'protein': 0.9, 'carbs': 3.9, 'fat': 0.2, 'calories': 18},
            'olive_oil': {'protein': 0, 'carbs': 0, 'fat': 100, 'calories': 884},
        }
    
    def calculate_dish_nutrition(self, ingredients_with_quantities):
        """
        Calculate nutritional profile of a dish
        Like analyzing the perfect balance of nutrients in each creation
        """
        total_nutrition = {'protein': 0, 'carbs': 0, 'fat': 0, 'calories': 0}
        
        for ingredient, quantity_grams in ingredients_with_quantities.items():
            if ingredient.lower() in self.nutritional_database:
                nutrient_data = self.nutritional_database[ingredient.lower()]
                # Calculate per 100g, then adjust for actual quantity
                quantity_factor = quantity_grams / 100
                
                for nutrient in total_nutrition:
                    total_nutrition[nutrient] += nutrient_data[nutrient] * quantity_factor
        
        return total_nutrition
    
    def optimize_for_health_goals(self, current_recipe, health_goal="balanced"):
        """
        Suggest modifications for health optimization
        Like adjusting seasoning for perfect taste and health
        """
        current_nutrition = self.calculate_dish_nutrition(current_recipe)
        suggestions = []
        
        if health_goal == "low_calorie":
            if current_nutrition['calories'] > 400:
                suggestions.append("Consider reducing portion sizes or using lower-calorie ingredients")
            if current_nutrition['fat'] > 15:
                suggestions.append("Replace some high-fat ingredients with lean alternatives")
                
        elif health_goal == "high_protein":
            protein_ratio = current_nutrition['protein'] / current_nutrition['calories'] * 100
            if protein_ratio < 20:
                suggestions.append("Add lean protein sources like chicken breast or fish")
                
        elif health_goal == "balanced":
            # Check macro balance
            total_macros = current_nutrition['protein'] + current_nutrition['carbs'] + current_nutrition['fat']
            if total_macros > 0:
                protein_percent = (current_nutrition['protein'] * 4 / current_nutrition['calories']) * 100
                carb_percent = (current_nutrition['carbs'] * 4 / current_nutrition['calories']) * 100
                fat_percent = (current_nutrition['fat'] * 9 / current_nutrition['calories']) * 100
                
                if protein_percent < 15:
                    suggestions.append("Consider increasing protein content")
                if carb_percent > 60:
                    suggestions.append("Consider reducing carbohydrate content")
                if fat_percent > 35:
                    suggestions.append("Consider reducing fat content")
        
        return {
            'current_nutrition': current_nutrition,
            'suggestions': suggestions,
            'health_score': self.calculate_health_score(current_nutrition)
        }
    
    def calculate_health_score(self, nutrition):
        """Calculate overall health score (0-100)"""
        score = 50  # Base score
        
        # Protein bonus
        if 20 <= nutrition['protein'] <= 40:
            score += 15
        
        # Calorie appropriateness
        if 200 <= nutrition['calories'] <= 600:
            score += 15
        
        # Fat content
        if nutrition['fat'] <= 20:
            score += 10
        
        # Carb content
        if nutrition['carbs'] <= 50:
            score += 10
        
        return min(100, score)

class PredictiveMaintenanceAI:
    """
    AI for equipment maintenance prediction
    Like knowing exactly when each piece of equipment needs attention
    """
    
    def __init__(self):
        self.equipment_data = {}
        
    def log_equipment_usage(self, equipment_id, usage_hours, temperature, vibration_level):
        """Log equipment usage data"""
        if equipment_id not in self.equipment_data:
            self.equipment_data[equipment_id] = []
            
        self.equipment_data[equipment_id].append({
            'timestamp': tf.timestamp(),
            'usage_hours': usage_hours,
            'temperature': temperature,
            'vibration_level': vibration_level
        })
    
    def predict_maintenance_needs(self, equipment_id):
        """Predict when maintenance is needed"""
        if equipment_id not in self.equipment_data or len(self.equipment_data[equipment_id]) < 10:
            return {"status": "insufficient_data"}
        
        data = self.equipment_data[equipment_id]
        recent_data = data[-30:]  # Last 30 readings
        
        # Simple rule-based prediction (in production, use ML models)
        avg_temperature = np.mean([d['temperature'] for d in recent_data])
        avg_vibration = np.mean([d['vibration_level'] for d in recent_data])
        total_hours = sum([d['usage_hours'] for d in recent_data])
        
        risk_score = 0
        if avg_temperature > 80:  # High temperature
            risk_score += 30
        if avg_vibration > 5:  # High vibration
            risk_score += 25
        if total_hours > 200:  # Heavy usage
            risk_score += 20
        
        maintenance_recommendation = "low"
        if risk_score > 50:
            maintenance_recommendation = "high"
        elif risk_score > 25:
            maintenance_recommendation = "medium"
        
        return {
            "risk_score": risk_score,
            "recommendation": maintenance_recommendation,
            "avg_temperature": avg_temperature,
            "avg_vibration": avg_vibration,
            "total_usage_hours": total_hours
        }
```

### Management Commands for Data Setup

```python
# restaurant_ai/management/commands/setup_demo_data.py
from django.core.management.base import BaseCommand
from django.utils import timezone
from restaurant_ai.models import Restaurant, MenuItem, SalesData, CustomerFeedback
import random
from datetime import datetime, timedelta

class Command(BaseCommand):
    help = 'Setup demo data for the restaurant AI system'
    
    def handle(self, *args, **options):
        # Create sample restaurant
        restaurant, created = Restaurant.objects.get_or_create(
            name="AI Fusion Bistro",
            defaults={
                'location': '123 Tech Street, Silicon Valley',
                'cuisine_type': 'Modern Fusion',
                'capacity': 80
            }
        )
        
        if created:
            self.stdout.write(f'Created restaurant: {restaurant.name}')
        
        # Create sample menu items
        menu_items_data = [
            {'name': 'AI-Optimized Salmon', 'category': 'Main Course', 'price': 28.99, 
             'ingredients': ['salmon', 'quinoa', 'broccoli', 'lemon']},
            {'name': 'Smart Chicken Bowl', 'category': 'Main Course', 'price': 19.99,
             'ingredients': ['chicken', 'rice', 'vegetables', 'sauce']},
            {'name': 'Data-Driven Pasta', 'category': 'Main Course', 'price': 22.50,
             'ingredients': ['pasta', 'tomato', 'basil', 'olive_oil']},
            {'name': 'Predictive Pizza', 'category': 'Main Course', 'price': 16.99,
             'ingredients': ['dough', 'tomato_sauce', 'cheese', 'pepperoni']},
            {'name': 'Neural Network Salad', 'category': 'Appetizer', 'price': 12.99,
             'ingredients': ['spinach', 'tomato', 'cucumber', 'dressing']},
        ]
        
        menu_items = []
        for item_data in menu_items_data:
            item, created = MenuItem.objects.get_or_create(
                restaurant=restaurant,
                name=item_data['name'],
                defaults=item_data
            )
            menu_items.append(item)
            if created:
                self.stdout.write(f'Created menu item: {item.name}')
        
        # Generate sample sales data for the last 90 days
        start_date = datetime.now().date() - timedelta(days=90)
        weather_conditions = ['sunny', 'rainy', 'cloudy', 'windy']
        seasons = ['spring', 'summer', 'fall', 'winter']
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for i in range(90):
            current_date = start_date + timedelta(days=i)
            day_of_week = days_of_week[current_date.weekday()]
            
            for menu_item in menu_items:
                # Generate realistic sales patterns
                base_quantity = random.randint(5, 25)
                
                # Weekend boost
                if day_of_week in ['Friday', 'Saturday', 'Sunday']:
                    base_quantity = int(base_quantity * 1.3)
                
                # Weather impact
                weather = random.choice(weather_conditions)
                if weather == 'rainy':
                    base_quantity = int(base_quantity * 0.8)
                elif weather == 'sunny':
                    base_quantity = int(base_quantity * 1.1)
                
                SalesData.objects.get_or_create(
                    restaurant=restaurant,
                    menu_item=menu_item,
                    date=current_date,
                    defaults={
                        'quantity_sold': base_quantity,
                        'revenue': base_quantity * float(menu_item.price),
                        'weather_condition': weather,
                        'day_of_week': day_of_week,
                        'season': random.choice(seasons)
                    }
                )
        
        # Generate sample customer feedback
        feedback_samples = [
            "Excellent food and service! The AI recommendations were spot on.",
            "Good food but could use more seasoning. The portions were perfect though.",
            "Amazing experience! The smart menu suggestions saved me time.",
            "Food was okay, but the wait time was a bit long.",
            "Outstanding meal! Everything was perfectly prepared and delicious.",
            "The sustainability focus is great, but I'd like more meat options.",
            "Innovative concept and great execution. Will definitely come back!",
            "Price is a bit high for the portion size, but quality is good.",
        ]
        
        for i in range(50):
            feedback_text = random.choice(feedback_samples)
            CustomerFeedback.objects.get_or_create(
                restaurant=restaurant,
                menu_item=random.choice(menu_items) if random.random() > 0.3 else None,
                rating=random.randint(3, 5),
                comment=feedback_text,
                defaults={
                    'sentiment_score': random.uniform(-1, 1)
                }
            )
        
        self.stdout.write(self.style.SUCCESS('Demo data setup completed successfully!'))
```

### Testing the Complete System

```python
# restaurant_ai/tests.py
from django.test import TestCase, Client
from django.urls import reverse
from .models import Restaurant, MenuItem, SalesData
from .services import RestaurantAIService
import json

class RestaurantAITestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.restaurant = Restaurant.objects.create(
            name="Test Restaurant",
            location="Test Location",
            cuisine_type="Test Cuisine",
            capacity=50
        )
        
        self.menu_item = MenuItem.objects.create(
            restaurant=self.restaurant,
            name="Test Dish",
            category="Main",
            price=20.00,
            ingredients=['test_ingredient']
        )
    
    def test_ai_service_initialization(self):
        """Test that AI service initializes correctly"""
        service = RestaurantAIService()
        self.assertIsInstance(service, RestaurantAIService)
    
    def test_demand_prediction_endpoint(self):
        """Test the demand prediction API endpoint"""
        data = {
            'restaurant_id': self.restaurant.id,
            'menu_item_id': self.menu_item.id,
            'date': '2024-03-15',
            'weather': 'sunny',
            'season': 'spring',
            'day_of_week': 'Friday'
        }
        
        response = self.client.post(
            reverse('restaurant_ai:predict_demand'),
            data=json.dumps(data),
            content_type='application/json'
        )
        
        # Should return error due to insufficient training data
        self.assertEqual(response.status_code, 400)
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        service = RestaurantAIService()
        result = service.analyze_customer_sentiment("The food was absolutely amazing!")
        
        self.assertIn('sentiment_score', result)
        self.assertIn('sentiment_category', result)
        self.assertTrue(result['sentiment_score'] > 0)  # Should be positive
    
    def test_dashboard_endpoint(self):
        """Test dashboard data endpoint"""
        response = self.client.get(
            reverse('restaurant_ai:restaurant_dashboard', 
                   kwargs={'restaurant_id': self.restaurant.id})
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('restaurant', data)
        self.assertEqual(data['restaurant']['name'], 'Test Restaurant')
```

This comprehensive AI solution demonstrates industry applications across healthcare-like analytics (nutrition optimization), financial analysis (pricing optimization), autonomous systems (predictive maintenance), and sustainability tracking. The system combines multiple AI techniques including machine learning, natural language processing, and deep learning within a robust Django framework - like having a complete smart kitchen that learns, adapts, and optimizes every aspect of restaurant operations.

The project integrates real-world industry applications while maintaining the organic cooking metaphors throughout the codebase, showing how AI can transform traditional businesses into intelligent, data-driven operations.


## Assignment: Environmental Impact Calculator

Create a Python script that integrates multiple AI concepts to calculate and predict environmental impact for a small business.

### Assignment Requirements:

**Task:** Build an AI-powered environmental impact calculator that:

1. **Data Collection:** Accepts input for different business activities (electricity usage, transportation, waste production, etc.)

2. **Impact Analysis:** Calculates carbon footprint using AI algorithms similar to those in our climate analysis system

3. **Trend Prediction:** Uses basic machine learning to predict future environmental impact based on historical data

4. **Optimization Recommendations:** Provides AI-generated suggestions for reducing environmental impact

5. **Reporting:** Generates a comprehensive report with visualizations

**Specific Requirements:**
- Use at least 3 classes with inheritance
- Implement error handling for invalid inputs
- Include data validation and sanitization
- Generate at least 2 different types of visualizations
- Create a simple scoring system (0-100) for environmental performance
- Include at least 5 different optimization recommendations
- Save results to a JSON file for future analysis

**Sample Input Data Structure:**
```python
business_data = {
    'company_name': 'Green Solutions Inc.',
    'monthly_data': {
        'electricity_kwh': 1200,
        'gas_usage_m3': 150,
        'fuel_consumption_liters': 300,
        'waste_production_kg': 500,
        'water_usage_liters': 8000,
        'employee_count': 25
    },
    'historical_data': [
        # 12 months of previous data
    ]
}
```

**Expected Output:**
- Total carbon footprint (kg CO2 equivalent)
- Environmental performance score (0-100)
- Month-over-month trend analysis
- 6-month forward projection
- Ranked list of improvement recommendations
- JSON report file
- Two visualization charts (trend chart and breakdown chart)

**Evaluation Criteria:**
1. **Code Quality (25%):** Clean, well-documented code with proper error handling
2. **AI Integration (25%):** Effective use of prediction algorithms and pattern recognition
3. **Functionality (25%):** All requirements working correctly
4. **Innovation (25%):** Creative approaches to environmental analysis and recommendations

**Submission Format:**
- Single Python file named `environmental_calculator.py`
- Sample input data file `sample_business_data.json`
- README file explaining how to run the calculator
- Screenshot of generated visualizations

This assignment combines data processing, machine learning concepts, environmental science, and business intelligence—just like creating a complex dish that balances multiple flavors and techniques to create something both nutritious and satisfying.

The key is to approach this systematically: start with data preparation (like preparing your ingredients), then build your analysis engine (your cooking process), and finally present your results beautifully (like plating your final dish). Each component should work harmoniously with the others to create a comprehensive solution that demonstrates your mastery of AI concepts applied to real-world environmental challenges.