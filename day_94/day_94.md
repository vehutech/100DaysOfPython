# AI Mastery Course - Day 94: AI Specialization Choice

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand the four major AI specialization paths and their real-world applications
- Implement foundational algorithms for each specialization using Python
- Create a Django web application that demonstrates your chosen specialization
- Make an informed decision about your AI career path based on hands-on experience

---

## Introduction

Imagine that you've just graduated from culinary school and now stand before four different kitchen stations, each representing a unique culinary specialty. Just as a master chef must choose whether to perfect the art of pastry, master the flames of the grill, excel in sauce-making, or become a sommelier, an AI practitioner must eventually choose their specialization. Today, we'll explore these four "kitchen stations" of artificial intelligence, taste-test each one through hands-on coding, and help you discover where your passion and talents align.

Each specialization is like learning a different cooking technique - they all use similar ingredients (Python, mathematics, data) but combine them in distinctly different ways to create entirely different experiences for your "diners" (end users).

---

## Specialization 1: Computer Vision - The Visual Master's Station

Computer Vision specialists work like the garde manger chef who must have an exceptional eye for detail, color, and presentation. They teach machines to "see" and interpret visual information just as a chef reads the subtle changes in color that indicate perfect ripeness or doneness.

### Core Concepts
Computer Vision involves processing and analyzing visual data to extract meaningful information. Like a chef who can instantly spot when a sauce is about to break or when bread has the perfect golden crust, CV algorithms detect patterns, objects, and anomalies in images and videos.

### Code Example: Basic Object Detection

```python
import cv2
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
import base64
from io import BytesIO
from PIL import Image

# Django view for image processing
def detect_objects(request):
    """
    Process uploaded image for object detection
    Like a chef examining ingredients before cooking
    """
    if request.method == 'POST':
        # Get image from request
        image_data = request.FILES['image']
        
        # Convert to OpenCV format
        image = Image.open(image_data)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Load pre-trained face detection model
        # Think of this as the chef's trained eye for recognizing ingredients
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale (simplify the "recipe")
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (like identifying key ingredients)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,    # How much to reduce image size at each scale
            minNeighbors=5,     # Minimum neighbors required for detection
            minSize=(30, 30)    # Minimum face size
        )
        
        # Draw rectangles around faces (garnish the result)
        for (x, y, w, h) in faces:
            cv2.rectangle(opencv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Convert back to send to frontend
        _, buffer = cv2.imencode('.jpg', opencv_image)
        image_base64 = base64.b64encode(buffer).decode()
        
        return JsonResponse({
            'success': True,
            'processed_image': f'data:image/jpeg;base64,{image_base64}',
            'faces_detected': len(faces),
            'message': f'Found {len(faces)} faces in the image'
        })
    
    return render(request, 'cv_detector.html')

# Advanced medical imaging example
def analyze_medical_scan(scan_path):
    """
    Analyze medical images for anomalies
    Like a chef detecting spoilage or quality issues in ingredients
    """
    # Load medical image
    image = cv2.imread(scan_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to reduce noise (clean the workspace)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply threshold to segment regions (separate ingredients)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours (outline the regions of interest)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze each contour for anomalies
    anomalies = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Significant size threshold
            # Calculate roundness (how circular the shape is)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity < 0.5:  # Irregular shape might indicate anomaly
                x, y, w, h = cv2.boundingRect(contour)
                anomalies.append({
                    'position': (x, y),
                    'size': (w, h),
                    'area': area,
                    'irregularity_score': 1 - circularity
                })
    
    return anomalies
```

### Syntax Explanation:
- `cv2.CascadeClassifier()`: Loads a pre-trained model for detecting objects (like having a recipe card)
- `detectMultiScale()`: Scans the image at different sizes to find objects
- `cv2.rectangle()`: Draws detection boxes around found objects
- `cv2.GaussianBlur()`: Smooths the image to reduce noise
- `cv2.threshold()`: Converts grayscale to binary (black and white) for easier processing

---

## Specialization 2: Natural Language Processing - The Language Artisan's Corner

NLP specialists are like the chef who masters flavor combinations and can tell stories through food. They work with the "ingredients" of language - words, sentences, meaning, and context - to create applications that understand and generate human communication.

### Core Concepts
Natural Language Processing involves teaching machines to understand, interpret, and generate human language. Like a chef who understands that the same ingredient can taste completely different depending on preparation and context, NLP algorithms must grasp that words change meaning based on context.

### Code Example: Advanced Language Understanding

```python
import re
import nltk
from textblob import TextBlob
from collections import Counter
from django.shortcuts import render
from django.http import JsonResponse
import json

# Download required NLTK data (prepare your spice rack)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

def analyze_multilingual_text(request):
    """
    Analyze text for sentiment, language, and key insights
    Like tasting a dish and identifying all its flavor components
    """
    if request.method == 'POST':
        text = request.POST.get('text', '')
        
        # Create TextBlob object (our main cooking tool)
        blob = TextBlob(text)
        
        # Language detection (identify the cuisine type)
        detected_language = blob.detect_language()
        
        # Sentiment analysis (taste the emotional flavor)
        sentiment = blob.sentiment
        sentiment_label = 'positive' if sentiment.polarity > 0.1 else 'negative' if sentiment.polarity < -0.1 else 'neutral'
        
        # Extract key phrases (identify main ingredients)
        sentences = blob.sentences
        key_phrases = []
        
        for sentence in sentences:
            # Extract noun phrases (the main components)
            for phrase in sentence.noun_phrases:
                if len(phrase.split()) >= 2:  # Multi-word phrases are more meaningful
                    key_phrases.append(phrase)
        
        # Count word frequency (ingredient proportions)
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words).most_common(10)
        
        # Advanced: Named Entity Recognition simulation
        entities = extract_entities(text)
        
        return JsonResponse({
            'language': detected_language,
            'sentiment': {
                'label': sentiment_label,
                'polarity': round(sentiment.polarity, 3),
                'subjectivity': round(sentiment.subjectivity, 3)
            },
            'key_phrases': list(set(key_phrases))[:10],
            'word_frequency': word_freq,
            'entities': entities,
            'text_stats': {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_sentence_length': round(len(words) / len(sentences), 2)
            }
        })
    
    return render(request, 'nlp_analyzer.html')

def extract_entities(text):
    """
    Extract named entities from text
    Like identifying specific ingredients by name in a complex recipe
    """
    # Simple pattern-based entity extraction
    entities = {
        'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
        'phones': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text),
        'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
        'money': re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text),
        'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    }
    return entities

def build_chatbot_response(user_message):
    """
    Generate contextual responses for a chatbot
    Like adapting a recipe based on available ingredients and diner preferences
    """
    # Analyze user intent (understand what the customer wants)
    blob = TextBlob(user_message.lower())
    
    # Intent patterns (recipe categories)
    intents = {
        'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
        'question': ['what', 'how', 'when', 'where', 'why', 'who'],
        'complaint': ['problem', 'issue', 'wrong', 'error', 'bug', 'broken'],
        'compliment': ['great', 'awesome', 'excellent', 'perfect', 'amazing'],
        'goodbye': ['bye', 'goodbye', 'see you', 'later', 'farewell']
    }
    
    detected_intent = 'general'
    for intent, keywords in intents.items():
        if any(keyword in user_message.lower() for keyword in keywords):
            detected_intent = intent
            break
    
    # Generate appropriate response (cook the right dish)
    responses = {
        'greeting': "Hello! How can I help you today?",
        'question': f"That's a great question about '{' '.join(blob.noun_phrases)}'. Let me help you with that.",
        'complaint': "I understand your concern. Let me help resolve this issue for you.",
        'compliment': "Thank you so much! I'm glad I could help.",
        'goodbye': "Goodbye! Have a wonderful day!",
        'general': "I understand. Could you tell me more about what you're looking for?"
    }
    
    return {
        'response': responses[detected_intent],
        'intent': detected_intent,
        'confidence': calculate_confidence(user_message, detected_intent)
    }

def calculate_confidence(message, intent):
    """Calculate confidence score for intent detection"""
    # Simple confidence calculation (how sure we are about our "recipe choice")
    base_confidence = 0.7
    message_length_factor = min(len(message.split()) / 10, 0.3)
    return round(base_confidence + message_length_factor, 2)
```

### Syntax Explanation:
- `TextBlob()`: Creates an object for text analysis (like putting ingredients in a food processor)
- `detect_language()`: Identifies the language of the text
- `sentiment`: Analyzes emotional tone (polarity: -1 to 1, subjectivity: 0 to 1)
- `noun_phrases`: Extracts meaningful noun combinations
- `re.findall()`: Uses regular expressions to find patterns in text
- `Counter()`: Counts frequency of items in a list

---

## Specialization 3: Robotics - The Automation Specialist's Workshop

Robotics specialists are like the chef who designs the entire kitchen workflow, coordinating multiple stations, timing, and movements to create seamless service. They bridge the physical and digital worlds, making machines move, sense, and respond intelligently.

### Core Concepts
Robotics combines AI with physical systems, involving sensor data processing, motion planning, and real-time decision making. Like orchestrating a busy kitchen during dinner rush, robotics requires precise timing, coordination, and adaptability.

### Code Example: Robot Control and Sensor Fusion

```python
import numpy as np
import time
import json
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import threading
from collections import deque

class RobotController:
    """
    Main robot control system
    Like the head chef coordinating the entire kitchen operation
    """
    
    def __init__(self):
        # Robot state (current status of our kitchen)
        self.position = [0.0, 0.0, 0.0]  # x, y, z coordinates
        self.orientation = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        self.is_moving = False
        self.sensors = {
            'ultrasonic': 0.0,  # Distance sensor (like checking oven space)
            'gyroscope': [0.0, 0.0, 0.0],  # Orientation sensor
            'accelerometer': [0.0, 0.0, 0.0],  # Movement sensor
            'temperature': 25.0,  # Environmental sensor
            'pressure': 1013.25  # Pressure sensor
        }
        self.command_queue = deque()  # Queue of tasks to perform
        self.is_running = False
        
    def start_robot_system(self):
        """Start the main robot control loop"""
        self.is_running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()
        
    def control_loop(self):
        """
        Main control loop - like the rhythm of kitchen service
        Continuously processes sensors and executes commands
        """
        while self.is_running:
            # Read sensor data (check all stations)
            self.update_sensors()
            
            # Process command queue (handle incoming orders)
            if self.command_queue:
                command = self.command_queue.popleft()
                self.execute_command(command)
            
            # Safety checks (ensure kitchen safety)
            self.perform_safety_checks()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)
    
    def update_sensors(self):
        """
        Simulate sensor readings
        Like constantly monitoring temperature, timing, and ingredient levels
        """
        # Simulate ultrasonic sensor (distance measurement)
        # In real robot, this would read from actual hardware
        self.sensors['ultrasonic'] = np.random.normal(50.0, 5.0)  # 50cm average distance
        
        # Simulate gyroscope readings
        self.sensors['gyroscope'] = [
            np.random.normal(0, 0.1) for _ in range(3)
        ]
        
        # Simulate accelerometer
        base_accel = [0, 0, -9.81] if not self.is_moving else [0.2, 0.1, -9.81]
        self.sensors['accelerometer'] = [
            base + np.random.normal(0, 0.05) for base in base_accel
        ]
    
    def execute_command(self, command):
        """
        Execute a movement or action command
        Like following a recipe step
        """
        cmd_type = command['type']
        
        if cmd_type == 'move':
            target_pos = command['target']
            self.move_to_position(target_pos)
            
        elif cmd_type == 'rotate':
            target_angle = command['angle']
            self.rotate_to_angle(target_angle)
            
        elif cmd_type == 'pickup':
            object_id = command['object_id']
            self.pickup_object(object_id)
            
        elif cmd_type == 'place':
            location = command['location']
            self.place_object(location)
    
    def move_to_position(self, target_position):
        """
        Move robot to specified position using path planning
        Like navigating efficiently through a busy kitchen
        """
        self.is_moving = True
        start_pos = self.position.copy()
        
        # Simple linear interpolation for smooth movement
        steps = 20
        for i in range(steps + 1):
            progress = i / steps
            
            # Calculate current position (smooth movement recipe)
            current_pos = [
                start_pos[j] + (target_position[j] - start_pos[j]) * progress
                for j in range(3)
            ]
            
            self.position = current_pos
            
            # Check for obstacles using sensor fusion
            if self.detect_obstacle():
                # Stop and plan alternative path (avoid collision in kitchen)
                self.is_moving = False
                return False
            
            time.sleep(0.05)  # Simulate movement time
        
        self.is_moving = False
        return True
    
    def detect_obstacle(self):
        """
        Use sensor fusion to detect obstacles
        Like being aware of other chefs and equipment while moving
        """
        # Combine multiple sensor readings for better accuracy
        ultrasonic_distance = self.sensors['ultrasonic']
        accel_magnitude = np.linalg.norm(self.sensors['accelerometer'][:2])  # x,y acceleration
        
        # Obstacle detected if too close OR unexpected acceleration
        obstacle_detected = (ultrasonic_distance < 20.0) or (accel_magnitude > 2.0)
        
        return obstacle_detected
    
    def perform_safety_checks(self):
        """
        Continuous safety monitoring
        Like checking that stoves are off and knives are safe
        """
        # Check if robot is tilted too much
        gyro_magnitude = np.linalg.norm(self.sensors['gyroscope'])
        if gyro_magnitude > 1.0:  # Significant tilt
            self.emergency_stop("Excessive tilt detected")
        
        # Check temperature limits
        if self.sensors['temperature'] > 60.0:  # Too hot
            self.emergency_stop("Temperature limit exceeded")
    
    def emergency_stop(self, reason):
        """Emergency stop procedure"""
        self.is_moving = False
        self.command_queue.clear()
        print(f"EMERGENCY STOP: {reason}")

# Django views for robot control interface
robot = RobotController()

def robot_control_panel(request):
    """Main robot control interface"""
    return render(request, 'robot_control.html')

@csrf_exempt
def send_robot_command(request):
    """
    Send command to robot
    Like giving instructions to a kitchen assistant
    """
    if request.method == 'POST':
        try:
            command_data = json.loads(request.body)
            
            # Validate command (check if instruction makes sense)
            if validate_command(command_data):
                robot.command_queue.append(command_data)
                
                return JsonResponse({
                    'success': True,
                    'message': 'Command queued successfully',
                    'queue_length': len(robot.command_queue)
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid command format'
                })
                
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON format'
            })
    
    return JsonResponse({'error': 'POST method required'})

def validate_command(command):
    """Validate command structure and parameters"""
    required_fields = ['type']
    
    if not all(field in command for field in required_fields):
        return False
    
    cmd_type = command['type']
    
    # Validate specific command types
    if cmd_type == 'move':
        return 'target' in command and len(command['target']) == 3
    elif cmd_type == 'rotate':
        return 'angle' in command and isinstance(command['angle'], (int, float))
    
    return True

def get_robot_status(request):
    """Get current robot status and sensor readings"""
    return JsonResponse({
        'position': robot.position,
        'orientation': robot.orientation,
        'is_moving': robot.is_moving,
        'sensors': robot.sensors,
        'queue_length': len(robot.command_queue),
        'timestamp': time.time()
    })
```

### Syntax Explanation:
- `threading.Thread()`: Creates a separate thread for continuous robot control
- `deque()`: Double-ended queue for efficient command queuing
- `np.linalg.norm()`: Calculates the magnitude of a vector (sensor readings)
- `@csrf_exempt`: Django decorator to disable CSRF protection for API endpoints
- `json.loads()`: Converts JSON string to Python dictionary
- `time.sleep()`: Pauses execution for specified seconds

---

## Specialization 4: AI for Business - The Strategic Operations Manager

AI for Business specialists are like the restaurant manager who understands not just cooking, but customer preferences, inventory management, profit margins, and market trends. They apply AI to solve practical business problems and drive decision-making.

### Core Concepts
Business AI focuses on practical applications like recommendation systems, fraud detection, customer segmentation, and predictive analytics. Like a restaurant manager who must balance quality, efficiency, and profitability, business AI practitioners optimize systems for real-world impact.

### Code Example: Recommendation System and Fraud Detection

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from collections import defaultdict
import datetime

class RecommendationEngine:
    """
    Product recommendation system
    Like a sommelier who knows exactly what wine pairs with each dish and customer preference
    """
    
    def __init__(self):
        # User-item interaction matrix (customer order history)
        self.user_interactions = defaultdict(dict)  # user_id -> {item_id: rating}
        self.item_features = {}  # item_id -> {feature: value}
        self.user_profiles = {}  # user_id -> {feature: value}
        
    def add_interaction(self, user_id, item_id, rating, timestamp=None):
        """
        Record user interaction with item
        Like noting that a customer loved a particular dish
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        self.user_interactions[user_id][item_id] = {
            'rating': rating,
            'timestamp': timestamp
        }
    
    def collaborative_filtering(self, user_id, num_recommendations=5):
        """
        Recommend items based on similar users' preferences
        Like suggesting dishes that customers with similar tastes enjoyed
        """
        if user_id not in self.user_interactions:
            return self.get_popular_items(num_recommendations)
        
        user_items = set(self.user_interactions[user_id].keys())
        similar_users = []
        
        # Find users with similar preferences
        for other_user_id, other_interactions in self.user_interactions.items():
            if other_user_id == user_id:
                continue
                
            other_items = set(other_interactions.keys())
            
            # Calculate Jaccard similarity (how much overlap in preferences)
            intersection = len(user_items.intersection(other_items))
            union = len(user_items.union(other_items))
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0.2:  # Threshold for similarity
                    similar_users.append((other_user_id, similarity))
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)
        
        # Get recommendations from similar users
        recommendations = defaultdict(float)
        for similar_user_id, similarity in similar_users[:10]:  # Top 10 similar users
            for item_id, interaction in self.user_interactions[similar_user_id].items():
                if item_id not in user_items:  # Don't recommend already consumed items
                    # Weight by similarity and rating
                    recommendations[item_id] += similarity * interaction['rating']
        
        # Sort and return top recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, score in sorted_recs[:num_recommendations]]
    
    def content_based_filtering(self, user_id, num_recommendations=5):
        """
        Recommend items based on item features and user preferences
        Like recommending dishes based on ingredients and cooking style preferences
        """
        if user_id not in self.user_interactions:
            return self.get_popular_items(num_recommendations)
        
        # Build user preference profile
        user_preferences = defaultdict(float)
        total_interactions = 0
        
        for item_id, interaction in self.user_interactions[user_id].items():
            if item_id in self.item_features:
                rating = interaction['rating']
                total_interactions += 1
                
                # Weight features by rating
                for feature, value in self.item_features[item_id].items():
                    user_preferences[feature] += rating * value
        
        # Normalize preferences
        if total_interactions > 0:
            for feature in user_preferences:
                user_preferences[feature] /= total_interactions
        
        # Score all items based on user preferences
        item_scores = {}
        consumed_items = set(self.user_interactions[user_id].keys())
        
        for item_id, features in self.item_features.items():
            if item_id not in consumed_items:
                score = 0
                for feature, value in features.items():
                    if feature in user_preferences:
                        score += user_preferences[feature] * value
                item_scores[item_id] = score
        
        # Return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, score in sorted_items[:num_recommendations]]
    
    def get_popular_items(self, num_recommendations=5):
        """Fallback: recommend most popular items"""
        item_popularity = defaultdict(int)
        
        for user_interactions in self.user_interactions.values():
            for item_id in user_interactions.keys():
                item_popularity[item_id] += 1
        
        sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, popularity in sorted_items[:num_recommendations]]

class FraudDetectionSystem:
    """
    Fraud detection using machine learning
    Like a vigilant maître d' who can spot suspicious behavior and fake reservations
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
    
    def prepare_features(self, transaction_data):
        """
        Extract relevant features from transaction data
        Like identifying the key indicators that distinguish legitimate customers
        """
        features = []
        
        for transaction in transaction_data:
            # Extract temporal features (timing patterns)
            timestamp = pd.to_datetime(transaction['timestamp'])
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Transaction features
            amount = transaction['amount']
            merchant_category = transaction.get('merchant_category', 0)
            
            # User behavior features
            user_id = transaction['user_id']
            
            # Geographic features (if available)
            location_risk = transaction.get('location_risk_score', 0)
            
            # Velocity features (how quickly transactions occur)
            time_since_last = transaction.get('time_since_last_transaction', 0)
            
            feature_vector = [
                amount,
                hour_of_day,
                day_of_week,
                merchant_category,
                location_risk,
                time_since_last,
                len(str(user_id)),  # User ID length as a feature
                1 if amount > 1000 else 0,  # High amount flag
                1 if hour_of_day < 6 or hour_of_day > 22 else 0,  # Unusual time flag
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_model(self, training_data):
        """
        Train fraud detection models
        Like training staff to recognize suspicious behavior patterns
        """
        # Prepare features
        X = self.prepare_features(training_data)
        y = [transaction['is_fraud'] for transaction in training_data]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train supervised model (Random Forest)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Train unsupervised model (Isolation Forest) for anomaly detection
        # Use only legitimate transactions for training
        legitimate_transactions = X_train[np.array(y_train) == 0]
        self.isolation_forest.fit(legitimate_transactions)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def detect_fraud(self, transaction):
        """
        Detect if a transaction is fraudulent
        Like instantly recognizing if something doesn't feel right about a customer
        """
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        # Prepare features
        features = self.prepare_features([transaction])
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from both models
        fraud_probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of fraud
        anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
        
        # Combine both scores for final decision
        is_anomaly = anomaly_score < 0  # Negative score indicates anomaly
        
        # Risk assessment
        risk_level = 'low'
        if fraud_probability > 0.7 or is_anomaly:
            risk_level = 'high'
        elif fraud_probability > 0.3:
            risk_level = 'medium'
        
        return {
            'is_fraud_predicted': fraud_probability > 0.5,
            'fraud_probability': round(fraud_probability, 3),
            'anomaly_score': round(anomaly_score, 3),
            'is_anomaly': is_anomaly,
            'risk_level': risk_level,
            'confidence': round(max(fraud_probability, 1 - fraud_probability), 3)
        }

# Django views for business AI applications
recommendation_engine = RecommendationEngine()
fraud_detector = FraudDetectionSystem()

def business_ai_dashboard(request):
    """Main dashboard for business AI applications"""
    return render(request, 'business_ai_dashboard.html')

@csrf_exempt
def get_recommendations(request):
    """
    Get personalized recommendations for a user
    Like asking the sommelier for the perfect wine pairing
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_id = data.get('user_id')
            method = data.get('method', 'collaborative')  # collaborative or content_based
            num_recommendations = data.get('num_recommendations', 5)
            
            if method == 'collaborative':
                recommendations = recommendation_engine.collaborative_filtering(
                    user_id, num_recommendations
                )
            else:
                recommendations = recommendation_engine.content_based_filtering(
                    user_id, num_recommendations
                )
            
            return JsonResponse({
                'success': True,
                'user_id': user_id,
                'recommendations': recommendations,
                'method': method,
                'explanation': f'Based on {method} filtering analysis'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'error': 'POST method required'})

@csrf_exempt
def check_fraud(request):
    """
    Check if a transaction is potentially fraudulent
    Like having security verify a suspicious customer
    """
    if request.method == 'POST':
        try:
            transaction_data = json.loads(request.body)
            
            # Add timestamp if not provided
            if 'timestamp' not in transaction_data:
                transaction_data['timestamp'] = datetime.datetime.now().isoformat()
            
            # Detect fraud
            result = fraud_detector.detect_fraud(transaction_data)
            
            return JsonResponse({
                'success': True,
                'transaction_id': transaction_data.get('transaction_id', 'unknown'),
                'fraud_analysis': result
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'error': 'POST method required'})

def customer_segmentation(customer_data):
    """
    Segment customers based on behavior patterns
    Like categorizing diners by their dining preferences and frequency
    """
    # Prepare customer features
    features = []
    customer_ids = []
    
    for customer_id, data in customer_data.items():
        # RFM Analysis: Recency, Frequency, Monetary
        recency = data.get('days_since_last_purchase', 365)
        frequency = data.get('purchase_count', 0)
        monetary = data.get('total_spent', 0)
        
        # Additional behavioral features
        avg_order_value = monetary / max(frequency, 1)
        preferred_category = data.get('most_purchased_category', 0)
        seasonal_variance = data.get('seasonal_purchase_variance', 0)
        
        feature_vector = [
            recency,
            frequency,
            monetary,
            avg_order_value,
            preferred_category,
            seasonal_variance
        ]
        
        features.append(feature_vector)
        customer_ids.append(customer_id)
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Simple clustering using K-means concept (simplified)
    # In production, you'd use sklearn.cluster.KMeans
    segments = {}
    
    for i, customer_id in enumerate(customer_ids):
        feature_vec = features_scaled[i]
        
        # Simple rule-based segmentation
        if feature_vec[1] > 1 and feature_vec[2] > 1:  # High frequency & monetary
            segment = 'VIP'
        elif feature_vec[1] > 0.5:  # Medium frequency
            segment = 'Regular'
        elif feature_vec[0] < 0:  # Recent purchase
            segment = 'New'
        else:
            segment = 'At Risk'
        
        segments[customer_id] = {
            'segment': segment,
            'recency_score': round(feature_vec[0], 2),
            'frequency_score': round(feature_vec[1], 2),
            'monetary_score': round(feature_vec[2], 2)
        }
    
### Syntax Explanation:
- `defaultdict()`: Dictionary that provides default values for missing keys
- `collaborative_filtering()`: Recommends based on user similarity patterns
- `RandomForestClassifier()`: Ensemble model using multiple decision trees
- `IsolationForest()`: Unsupervised anomaly detection algorithm
- `StandardScaler()`: Normalizes features to similar scales
- `train_test_split()`: Splits data into training and testing sets
- `predict_proba()`: Returns probability estimates for each class

---

## Final Project: Multi-Specialization AI Dashboard

Now that you've tasted each specialization, it's time to create a comprehensive Django application that showcases all four areas. This project will be like designing a complete restaurant that offers multiple cuisine types under one roof.

### Project Requirements

Create a Django web application called "AI Kitchen Dashboard" that integrates all four specializations:

1. **Computer Vision Station**: Upload and analyze images
2. **NLP Counter**: Process and analyze text input
3. **Robotics Workshop**: Control simulated robot systems
4. **Business Intelligence Center**: Generate recommendations and detect anomalies

### Code Implementation:

```python
# urls.py - The restaurant's menu (routing system)
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('cv/analyze/', views.cv_analyze, name='cv_analyze'),
    path('nlp/process/', views.nlp_process, name='nlp_process'),
    path('robot/control/', views.robot_control, name='robot_control'),
    path('robot/status/', views.robot_status, name='robot_status'),
    path('business/recommend/', views.business_recommend, name='business_recommend'),
    path('business/fraud-check/', views.fraud_check, name='fraud_check'),
]

# views.py - Main coordination (like the head chef managing all stations)
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from textblob import TextBlob
import re

def dashboard(request):
    """
    Main dashboard showing all AI capabilities
    Like the main dining room where customers can see all available services
    """
    context = {
        'specializations': [
            {
                'name': 'Computer Vision',
                'description': 'Visual analysis and object detection',
                'icon': '👁️',
                'endpoint': '/cv/analyze/'
            },
            {
                'name': 'Natural Language Processing',
                'description': 'Text analysis and understanding',
                'icon': '💬',
                'endpoint': '/nlp/process/'
            },
            {
                'name': 'Robotics',
                'description': 'Motion control and sensor fusion',
                'icon': '🤖',
                'endpoint': '/robot/control/'
            },
            {
                'name': 'Business AI',
                'description': 'Recommendations and fraud detection',
                'icon': '📊',
                'endpoint': '/business/recommend/'
            }
        ]
    }
    return render(request, 'dashboard.html', context)

@csrf_exempt
def cv_analyze(request):
    """Computer Vision analysis endpoint"""
    if request.method == 'POST':
        try:
            # Handle image upload
            if 'image' in request.FILES:
                image_file = request.FILES['image']
                
                # Process image
                image = Image.open(image_file)
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Perform analysis (simplified face detection)
                gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                # Draw rectangles around faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(opencv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Convert back to base64 for frontend
                _, buffer = cv2.imencode('.jpg', opencv_image)
                processed_image = base64.b64encode(buffer).decode()
                
                return JsonResponse({
                    'success': True,
                    'faces_detected': len(faces),
                    'processed_image': f'data:image/jpeg;base64,{processed_image}',
                    'analysis': f'Detected {len(faces)} faces in the image'
                })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request method'})

@csrf_exempt
def nlp_process(request):
    """Natural Language Processing endpoint"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '')
            
            # Analyze text
            blob = TextBlob(text)
            
            # Sentiment analysis
            sentiment = blob.sentiment
            
            # Extract key information
            sentences = blob.sentences
            words = text.split()
            
            # Simple entity extraction
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            
            return JsonResponse({
                'success': True,
                'sentiment': {
                    'polarity': round(sentiment.polarity, 3),
                    'subjectivity': round(sentiment.subjectivity, 3),
                    'label': 'positive' if sentiment.polarity > 0.1 else 'negative' if sentiment.polarity < -0.1 else 'neutral'
                },
                'statistics': {
                    'word_count': len(words),
                    'sentence_count': len(sentences),
                    'avg_sentence_length': round(len(words) / len(sentences), 2) if sentences else 0
                },
                'entities': {
                    'emails': emails
                },
                'key_phrases': list(blob.noun_phrases)[:5]
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request method'})

# Add the robot and business AI endpoints from previous examples here
# (robot_control, robot_status, business_recommend, fraud_check functions)
```

### HTML Template (dashboard.html):

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Kitchen Dashboard</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
        }
        .specializations-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        .specialization-card {
            background: linear-gradient(145deg, #f8f9fa, #e9ecef);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
            border: 2px solid transparent;
        }
        .specialization-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border-color: #667eea;
        }
        .specialization-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
        .specialization-name {
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .specialization-desc {
            color: #7f8c8d;
            line-height: 1.5;
        }
        .demo-section {
            margin-top: 40px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        .demo-section.active {
            display: block;
        }
        .input-group {
            margin: 15px 0;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #2c3e50;
        }
        input[type="file"], textarea, input[type="text"] {
            width: 100%;
            padding: 10px;
            border: 2px solid #dee2e6;
            border-radius: 5px;
            font-size: 14px;
        }
        textarea {
            height: 120px;
            resize: vertical;
        }
        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s ease;
        }
        .btn:hover {
            opacity: 0.9;
        }
        .results {
            margin-top: 20px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍳 AI Kitchen Dashboard</h1>
        <p class="subtitle">Master the art of artificial intelligence across four specializations</p>
        
        <div class="specializations-grid">
            {% for spec in specializations %}
            <div class="specialization-card" onclick="showDemo('{{ spec.name|lower|cut:' ' }}')">
                <div class="specialization-icon">{{ spec.icon }}</div>
                <div class="specialization-name">{{ spec.name }}</div>
                <div class="specialization-desc">{{ spec.description }}</div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Demo sections for each specialization -->
        <div id="computer-demo" class="demo-section">
            <h3>👁️ Computer Vision Demo</h3>
            <div class="input-group">
                <label for="cv-image">Upload an image for analysis:</label>
                <input type="file" id="cv-image" accept="image/*">
            </div>
            <button class="btn" onclick="analyzeImage()">Analyze Image</button>
            <div id="cv-results" class="results" style="display: none;"></div>
        </div>
        
        <div id="natural-demo" class="demo-section">
            <h3>💬 Natural Language Processing Demo</h3>
            <div class="input-group">
                <label for="nlp-text">Enter text for analysis:</label>
                <textarea id="nlp-text" placeholder="Enter any text here..."></textarea>
            </div>
            <button class="btn" onclick="processText()">Process Text</button>
            <div id="nlp-results" class="results" style="display: none;"></div>
        </div>
        
        <div id="robotics-demo" class="demo-section">
            <h3>🤖 Robotics Demo</h3>
            <div class="input-group">
                <label>Robot Position Control:</label>
                <input type="number" id="robot-x" placeholder="X coordinate" style="width: 30%; margin-right: 5px;">
                <input type="number" id="robot-y" placeholder="Y coordinate" style="width: 30%; margin-right: 5px;">
                <input type="number" id="robot-z" placeholder="Z coordinate" style="width: 30%;">
            </div>
            <button class="btn" onclick="moveRobot()">Move Robot</button>
            <button class="btn" onclick="getRobotStatus()">Get Status</button>
            <div id="robot-results" class="results" style="display: none;"></div>
        </div>
        
        <div id="business-demo" class="demo-section">
            <h3>📊 Business AI Demo</h3>
            <div class="input-group">
                <label for="user-id">User ID for recommendations:</label>
                <input type="text" id="user-id" placeholder="Enter user ID">
            </div>
            <button class="btn" onclick="getRecommendations()">Get Recommendations</button>
            <div id="business-results" class="results" style="display: none;"></div>
        </div>
    </div>

    <script>
        function showDemo(demoType) {
            // Hide all demo sections
            const demos = document.querySelectorAll('.demo-section');
            demos.forEach(demo => demo.classList.remove('active'));
            
            // Show selected demo
            const targetDemo = document.getElementById(demoType + '-demo');
            if (targetDemo) {
                targetDemo.classList.add('active');
            }
        }
        
        async function analyzeImage() {
            const fileInput = document.getElementById('cv-image');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an image first');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', file);
            
            try {
                const response = await fetch('/cv/analyze/', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                const resultsDiv = document.getElementById('cv-results');
                
                if (result.success) {
                    resultsDiv.innerHTML = `
                        <h4>Analysis Results:</h4>
                        <p><strong>Faces Detected:</strong> ${result.faces_detected}</p>
                        <p><strong>Analysis:</strong> ${result.analysis}</p>
                        ${result.processed_image ? `<img src="${result.processed_image}" style="max-width: 100%; margin-top: 10px;">` : ''}
                    `;
                } else {
                    resultsDiv.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
                }
                
                resultsDiv.style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during analysis');
            }
        }
        
        async function processText() {
            const text = document.getElementById('nlp-text').value;
            
            if (!text.trim()) {
                alert('Please enter some text first');
                return;
            }
            
            try {
                const response = await fetch('/nlp/process/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const result = await response.json();
                const resultsDiv = document.getElementById('nlp-results');
                
                if (result.success) {
                    resultsDiv.innerHTML = `
                        <h4>Text Analysis Results:</h4>
                        <p><strong>Sentiment:</strong> ${result.sentiment.label} (${result.sentiment.polarity})</p>
                        <p><strong>Word Count:</strong> ${result.statistics.word_count}</p>
                        <p><strong>Sentences:</strong> ${result.statistics.sentence_count}</p>
                        <p><strong>Key Phrases:</strong> ${result.key_phrases.join(', ')}</p>
                        ${result.entities.emails.length > 0 ? `<p><strong>Emails Found:</strong> ${result.entities.emails.join(', ')}</p>` : ''}
                    `;
                } else {
                    resultsDiv.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
                }
                
                resultsDiv.style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during text processing');
            }
        }
        
        // Add similar functions for robotics and business AI demos
        async function moveRobot() {
            const x = document.getElementById('robot-x').value;
            const y = document.getElementById('robot-y').value;
            const z = document.getElementById('robot-z').value;
            
            if (!x || !y || !z) {
                alert('Please enter all coordinates');
                return;
            }
            
            // Implementation for robot movement
            const resultsDiv = document.getElementById('robot-results');
            resultsDiv.innerHTML = `<p>Robot movement command sent to position (${x}, ${y}, ${z})</p>`;
            resultsDiv.style.display = 'block';
        }
        
        async function getRecommendations() {
            const userId = document.getElementById('user-id').value;
            
            if (!userId) {
                alert('Please enter a user ID');
                return;
            }
            
            // Implementation for recommendations
            const resultsDiv = document.getElementById('business-results');
            resultsDiv.innerHTML = `<p>Generating recommendations for user: ${userId}</p>`;
            resultsDiv.style.display = 'block';
        }
    </script>
</body>
</html>
```

This comprehensive project demonstrates mastery of all four AI specializations within a single, cohesive application. Like a master chef who has trained in multiple cuisines, you'll have hands-on experience with the core concepts and practical applications of each specialization.

---

# Day 94: AI for Business - Advanced Fraud Detection System

## Advanced Project: Real-Time Financial Fraud Detection Platform

You'll build a comprehensive fraud detection system that monitors financial transactions in real-time, similar to how a master cook monitors every ingredient and process in their kitchen to ensure nothing spoils the final dish.

### Project Structure

```
fraud_detection_system/
├── manage.py
├── requirements.txt
├── fraud_detector/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── detection/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── ml_models.py
│   ├── fraud_analyzer.py
│   └── real_time_monitor.py
├── api/
│   ├── __init__.py
│   ├── serializers.py
│   └── views.py
└── templates/
    └── detection/
        ├── dashboard.html
        ├── transaction_detail.html
        └── alerts.html
```

### Core Models

```python
# detection/models.py
from django.db import models
from django.contrib.auth.models import User
import uuid

class Customer(models.Model):
    customer_id = models.UUIDField(default=uuid.uuid4, unique=True)
    email = models.EmailField()
    phone = models.CharField(max_length=15)
    registration_date = models.DateTimeField(auto_now_add=True)
    risk_score = models.FloatField(default=0.0)
    country = models.CharField(max_length=3)
    
    def __str__(self):
        return f"Customer {self.customer_id}"

class Transaction(models.Model):
    TRANSACTION_TYPES = [
        ('purchase', 'Purchase'),
        ('transfer', 'Transfer'),
        ('withdrawal', 'Withdrawal'),
        ('deposit', 'Deposit'),
    ]
    
    transaction_id = models.UUIDField(default=uuid.uuid4, unique=True)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    transaction_type = models.CharField(max_length=20, choices=TRANSACTION_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)
    merchant = models.CharField(max_length=200, blank=True)
    location = models.CharField(max_length=100)
    ip_address = models.GenericIPAddressField()
    device_fingerprint = models.CharField(max_length=200)
    is_fraud = models.BooleanField(default=False)
    fraud_probability = models.FloatField(default=0.0)
    flagged_by_model = models.CharField(max_length=50, blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['customer']),
            models.Index(fields=['fraud_probability']),
        ]

class FraudAlert(models.Model):
    SEVERITY_LEVELS = [
        ('low', 'Low Risk'),
        ('medium', 'Medium Risk'),
        ('high', 'High Risk'),
        ('critical', 'Critical'),
    ]
    
    transaction = models.OneToOneField(Transaction, on_delete=models.CASCADE)
    severity = models.CharField(max_length=10, choices=SEVERITY_LEVELS)
    alert_message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    reviewed = models.BooleanField(default=False)
    reviewer = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    
class CustomerBehaviorProfile(models.Model):
    customer = models.OneToOneField(Customer, on_delete=models.CASCADE)
    avg_transaction_amount = models.DecimalField(max_digits=12, decimal_places=2)
    typical_transaction_hours = models.JSONField()  # Store typical hours as list
    common_locations = models.JSONField()  # Store common locations
    transaction_frequency = models.FloatField()  # transactions per day
    last_updated = models.DateTimeField(auto_now=True)
```

### Advanced ML Fraud Detection Engine

```python
# detection/ml_models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from django.conf import settings
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionEngine:
    def __init__(self):
        self.isolation_forest = None
        self.random_forest = None
        self.scaler = StandardScaler()
        self.model_path = os.path.join(settings.BASE_DIR, 'ml_models')
        os.makedirs(self.model_path, exist_ok=True)
        
    def prepare_features(self, transaction_data):
        """
        Extract features like a chef preparing ingredients - each feature
        adds flavor to the final detection recipe
        """
        features = []
        
        for transaction in transaction_data:
            # Time-based features
            hour = transaction['timestamp'].hour
            day_of_week = transaction['timestamp'].weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Amount-based features
            amount = float(transaction['amount'])
            log_amount = np.log1p(amount)
            
            # Customer behavior features
            customer_avg = transaction.get('customer_avg_amount', amount)
            amount_deviation = abs(amount - customer_avg) / (customer_avg + 1)
            
            # Location and device features
            is_new_location = transaction.get('is_new_location', 0)
            is_new_device = transaction.get('is_new_device', 0)
            
            # Velocity features (transactions in last hour/day)
            recent_transaction_count = transaction.get('recent_transaction_count', 0)
            
            feature_vector = [
                amount, log_amount, hour, day_of_week, is_weekend,
                amount_deviation, is_new_location, is_new_device,
                recent_transaction_count
            ]
            
            features.append(feature_vector)
            
        return np.array(features)
    
    def train_models(self, transactions_queryset):
        """
        Train both anomaly detection and supervised models
        Like seasoning a dish - multiple techniques for best results
        """
        # Prepare training data
        training_data = []
        labels = []
        
        for transaction in transactions_queryset:
            # Get customer behavior stats
            customer_stats = self._get_customer_stats(transaction.customer)
            
            trans_data = {
                'timestamp': transaction.timestamp,
                'amount': transaction.amount,
                'customer_avg_amount': customer_stats['avg_amount'],
                'is_new_location': self._is_new_location(transaction),
                'is_new_device': self._is_new_device(transaction),
                'recent_transaction_count': self._get_recent_transaction_count(transaction),
            }
            
            training_data.append(trans_data)
            labels.append(transaction.is_fraud)
        
        # Prepare features
        X = self.prepare_features(training_data)
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest (unsupervised)
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(X_scaled)
        
        # Train Random Forest (supervised)
        if len(np.unique(y)) > 1:  # Only if we have both fraud and non-fraud cases
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.random_forest = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.random_forest.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.random_forest.score(X_train, y_train)
            test_score = self.random_forest.score(X_test, y_test)
            print(f"Random Forest - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")
        
        # Save models
        self.save_models()
        
    def predict_fraud_probability(self, transaction_data):
        """
        Predict fraud probability using ensemble of models
        Like tasting a dish - multiple checks ensure quality
        """
        if not self.isolation_forest and not self.random_forest:
            self.load_models()
            
        features = self.prepare_features([transaction_data])
        features_scaled = self.scaler.transform(features)
        
        probabilities = []
        
        # Anomaly detection score
        if self.isolation_forest:
            anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
            # Convert to probability (higher anomaly = higher fraud probability)
            anomaly_prob = max(0, min(1, (0.5 - anomaly_score) * 2))
            probabilities.append(anomaly_prob)
        
        # Supervised model prediction
        if self.random_forest:
            fraud_prob = self.random_forest.predict_proba(features_scaled)[0][1]
            probabilities.append(fraud_prob)
        
        # Ensemble prediction (average of available models)
        final_probability = np.mean(probabilities) if probabilities else 0.5
        
        return float(final_probability)
    
    def _get_customer_stats(self, customer):
        """Get customer's historical behavior statistics"""
        from .models import Transaction
        
        recent_transactions = Transaction.objects.filter(
            customer=customer,
            timestamp__gte=datetime.now() - timedelta(days=30)
        )
        
        if recent_transactions.exists():
            amounts = [float(t.amount) for t in recent_transactions]
            return {
                'avg_amount': np.mean(amounts),
                'std_amount': np.std(amounts),
                'transaction_count': len(amounts)
            }
        
        return {'avg_amount': 0, 'std_amount': 0, 'transaction_count': 0}
    
    def _is_new_location(self, transaction):
        """Check if transaction is from a new location"""
        from .models import Transaction
        
        similar_location_count = Transaction.objects.filter(
            customer=transaction.customer,
            location=transaction.location
        ).count()
        
        return 1 if similar_location_count <= 1 else 0
    
    def _is_new_device(self, transaction):
        """Check if transaction is from a new device"""
        from .models import Transaction
        
        same_device_count = Transaction.objects.filter(
            customer=transaction.customer,
            device_fingerprint=transaction.device_fingerprint
        ).count()
        
        return 1 if same_device_count <= 1 else 0
    
    def _get_recent_transaction_count(self, transaction):
        """Count recent transactions by the same customer"""
        from .models import Transaction
        
        recent_count = Transaction.objects.filter(
            customer=transaction.customer,
            timestamp__gte=transaction.timestamp - timedelta(hours=1)
        ).count()
        
        return recent_count
    
    def save_models(self):
        """Save trained models to disk"""
        if self.isolation_forest:
            joblib.dump(self.isolation_forest, 
                       os.path.join(self.model_path, 'isolation_forest.pkl'))
        if self.random_forest:
            joblib.dump(self.random_forest, 
                       os.path.join(self.model_path, 'random_forest.pkl'))
        joblib.dump(self.scaler, 
                   os.path.join(self.model_path, 'scaler.pkl'))
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            self.isolation_forest = joblib.load(
                os.path.join(self.model_path, 'isolation_forest.pkl'))
            self.random_forest = joblib.load(
                os.path.join(self.model_path, 'random_forest.pkl'))
            self.scaler = joblib.load(
                os.path.join(self.model_path, 'scaler.pkl'))
        except FileNotFoundError:
            print("Models not found. Please train models first.")
```

### Real-Time Fraud Analyzer

```python
# detection/fraud_analyzer.py
from .models import Transaction, Customer, FraudAlert, CustomerBehaviorProfile
from .ml_models import FraudDetectionEngine
from django.utils import timezone
from datetime import timedelta
import json

class RealTimeFraudAnalyzer:
    def __init__(self):
        self.ml_engine = FraudDetectionEngine()
        
    def analyze_transaction(self, transaction):
        """
        Analyze a transaction in real-time
        Like a chef tasting throughout cooking - constant quality checks
        """
        # Prepare transaction data for ML model
        transaction_data = self._prepare_transaction_data(transaction)
        
        # Get fraud probability from ML models
        fraud_probability = self.ml_engine.predict_fraud_probability(transaction_data)
        
        # Apply business rules
        rule_based_score = self._apply_business_rules(transaction)
        
        # Combine ML and rule-based scores
        final_score = (fraud_probability * 0.7) + (rule_based_score * 0.3)
        
        # Update transaction with fraud probability
        transaction.fraud_probability = final_score
        transaction.save()
        
        # Create alert if necessary
        if final_score > 0.7:
            self._create_fraud_alert(transaction, final_score)
        
        return {
            'fraud_probability': final_score,
            'ml_score': fraud_probability,
            'rule_score': rule_based_score,
            'risk_level': self._get_risk_level(final_score)
        }
    
    def _prepare_transaction_data(self, transaction):
        """Prepare transaction data for ML analysis"""
        customer_stats = self._get_customer_behavior_stats(transaction.customer)
        
        return {
            'timestamp': transaction.timestamp,
            'amount': transaction.amount,
            'customer_avg_amount': customer_stats.get('avg_amount', 0),
            'is_new_location': self._is_unusual_location(transaction),
            'is_new_device': self._is_new_device(transaction),
            'recent_transaction_count': self._count_recent_transactions(transaction),
        }
    
    def _apply_business_rules(self, transaction):
        """
        Apply rule-based fraud detection
        Like following a recipe - certain combinations always raise suspicion
        """
        risk_score = 0.0
        
        # Large amount rule
        if transaction.amount > 10000:
            risk_score += 0.3
        
        # Unusual time rule
        if transaction.timestamp.hour < 6 or transaction.timestamp.hour > 23:
            risk_score += 0.2
        
        # Velocity rule - too many transactions in short time
        recent_count = self._count_recent_transactions(transaction, hours=1)
        if recent_count > 5:
            risk_score += 0.4
        
        # New location rule
        if self._is_unusual_location(transaction):
            risk_score += 0.3
        
        # Amount deviation from customer's normal behavior
        customer_avg = self._get_customer_avg_amount(transaction.customer)
        if customer_avg > 0:
            deviation = abs(float(transaction.amount) - customer_avg) / customer_avg
            if deviation > 5:  # 500% deviation
                risk_score += 0.5
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def _create_fraud_alert(self, transaction, fraud_score):
        """Create fraud alert based on score severity"""
        if fraud_score >= 0.9:
            severity = 'critical'
            message = f"Critical fraud risk detected. Score: {fraud_score:.3f}"
        elif fraud_score >= 0.8:
            severity = 'high'
            message = f"High fraud risk detected. Score: {fraud_score:.3f}"
        else:
            severity = 'medium'
            message = f"Medium fraud risk detected. Score: {fraud_score:.3f}"
        
        FraudAlert.objects.create(
            transaction=transaction,
            severity=severity,
            alert_message=message
        )
    
    def _get_customer_behavior_stats(self, customer):
        """Get customer's behavior profile"""
        try:
            profile = CustomerBehaviorProfile.objects.get(customer=customer)
            return {
                'avg_amount': float(profile.avg_transaction_amount),
                'typical_hours': profile.typical_transaction_hours,
                'common_locations': profile.common_locations,
                'frequency': profile.transaction_frequency
            }
        except CustomerBehaviorProfile.DoesNotExist:
            return {}
    
    def _get_customer_avg_amount(self, customer):
        """Get customer's average transaction amount"""
        recent_transactions = Transaction.objects.filter(
            customer=customer,
            timestamp__gte=timezone.now() - timedelta(days=30)
        )
        
        if recent_transactions.exists():
            amounts = [float(t.amount) for t in recent_transactions]
            return sum(amounts) / len(amounts)
        
        return 0
    
    def _is_unusual_location(self, transaction):
        """Check if location is unusual for customer"""
        common_locations = Transaction.objects.filter(
            customer=transaction.customer,
            location=transaction.location
        ).count()
        
        return common_locations <= 1
    
    def _is_new_device(self, transaction):
        """Check if device is new for customer"""
        device_usage = Transaction.objects.filter(
            customer=transaction.customer,
            device_fingerprint=transaction.device_fingerprint
        ).count()
        
        return device_usage <= 1
    
    def _count_recent_transactions(self, transaction, hours=1):
        """Count recent transactions by customer"""
        return Transaction.objects.filter(
            customer=transaction.customer,
            timestamp__gte=timezone.now() - timedelta(hours=hours)
        ).count()
    
    def _get_risk_level(self, score):
        """Convert score to risk level"""
        if score >= 0.8:
            return 'HIGH'
        elif score >= 0.6:
            return 'MEDIUM'
        elif score >= 0.4:
            return 'LOW'
        else:
            return 'MINIMAL'
```

### Django Views for Fraud Detection

```python
# detection/views.py
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.generic import ListView, DetailView
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.db.models import Count, Avg, Q
from django.utils import timezone
from datetime import timedelta
import json

from .models import Transaction, FraudAlert, Customer
from .fraud_analyzer import RealTimeFraudAnalyzer
from .ml_models import FraudDetectionEngine

@login_required
def fraud_dashboard(request):
    """
    Main fraud detection dashboard
    Like a kitchen's command center - overview of all operations
    """
    # Get statistics for dashboard
    today = timezone.now().date()
    
    total_transactions = Transaction.objects.filter(timestamp__date=today).count()
    flagged_transactions = Transaction.objects.filter(
        timestamp__date=today,
        fraud_probability__gte=0.5
    ).count()
    
    critical_alerts = FraudAlert.objects.filter(
        created_at__date=today,
        severity='critical',
        reviewed=False
    ).count()
    
    avg_fraud_score = Transaction.objects.filter(
        timestamp__date=today
    ).aggregate(Avg('fraud_probability'))['fraud_probability__avg'] or 0
    
    # Recent high-risk transactions
    recent_high_risk = Transaction.objects.filter(
        fraud_probability__gte=0.7,
        timestamp__gte=timezone.now() - timedelta(hours=24)
    ).order_by('-timestamp')[:10]
    
    # Fraud trend data for charts
    fraud_trend = []
    for i in range(7):
        date = today - timedelta(days=i)
        fraud_count = Transaction.objects.filter(
            timestamp__date=date,
            fraud_probability__gte=0.7
        ).count()
        fraud_trend.append({
            'date': date.strftime('%Y-%m-%d'),
            'fraud_count': fraud_count
        })
    
    context = {
        'total_transactions': total_transactions,
        'flagged_transactions': flagged_transactions,
        'critical_alerts': critical_alerts,
        'avg_fraud_score': round(avg_fraud_score, 3),
        'recent_high_risk': recent_high_risk,
        'fraud_trend': json.dumps(fraud_trend),
    }
    
    return render(request, 'detection/dashboard.html', context)

@csrf_exempt
def process_transaction(request):
    """
    Process incoming transaction and analyze for fraud
    Like receiving ingredients - immediate quality inspection
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Get or create customer
            customer, created = Customer.objects.get_or_create(
                email=data['customer_email'],
                defaults={
                    'phone': data.get('phone', ''),
                    'country': data.get('country', 'US')
                }
            )
            
            # Create transaction
            transaction = Transaction.objects.create(
                customer=customer,
                amount=data['amount'],
                transaction_type=data['transaction_type'],
                merchant=data.get('merchant', ''),
                location=data.get('location', ''),
                ip_address=data['ip_address'],
                device_fingerprint=data.get('device_fingerprint', '')
            )
            
            # Analyze transaction for fraud
            analyzer = RealTimeFraudAnalyzer()
            analysis_result = analyzer.analyze_transaction(transaction)
            
            response_data = {
                'transaction_id': str(transaction.transaction_id),
                'fraud_analysis': analysis_result,
                'approved': analysis_result['fraud_probability'] < 0.8,
                'message': 'Transaction processed successfully'
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)

class TransactionDetailView(DetailView):
    model = Transaction
    template_name = 'detection/transaction_detail.html'
    slug_field = 'transaction_id'
    slug_url_kwarg = 'transaction_id'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        transaction = self.get_object()
        
        # Get related alerts
        try:
            context['fraud_alert'] = FraudAlert.objects.get(transaction=transaction)
        except FraudAlert.DoesNotExist:
            context['fraud_alert'] = None
        
        # Get customer's transaction history
        context['customer_transactions'] = Transaction.objects.filter(
            customer=transaction.customer
        ).order_by('-timestamp')[:20]
        
        return context

@login_required
def train_models(request):
    """
    Train ML models with current transaction data
    Like updating recipes based on experience
    """
    if request.method == 'POST':
        try:
            # Get training data (last 10000 transactions with fraud labels)
            training_transactions = Transaction.objects.filter(
                is_fraud__isnull=False
            ).order_by('-timestamp')[:10000]
            
            if training_transactions.count() < 100:
                return JsonResponse({
                    'error': 'Need at least 100 labeled transactions for training'
                }, status=400)
            
            # Initialize and train ML engine
            ml_engine = FraudDetectionEngine()
            ml_engine.train_models(training_transactions)
            
            return JsonResponse({
                'message': 'Models trained successfully',
                'training_samples': training_transactions.count()
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)

class AlertsListView(ListView):
    model = FraudAlert
    template_name = 'detection/alerts.html'
    context_object_name = 'alerts'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = FraudAlert.objects.select_related('transaction', 'transaction__customer')
        
        # Filter by severity if specified
        severity = self.request.GET.get('severity')
        if severity:
            queryset = queryset.filter(severity=severity)
        
        # Filter by reviewed status
        reviewed = self.request.GET.get('reviewed')
        if reviewed == 'false':
            queryset = queryset.filter(reviewed=False)
        elif reviewed == 'true':
            queryset = queryset.filter(reviewed=True)
        
        return queryset.order_by('-created_at')
```

### API Serializers and Views

```python
# api/serializers.py
from rest_framework import serializers
from detection.models import Transaction, Customer, FraudAlert

class CustomerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Customer
        fields = ['customer_id', 'email', 'phone', 'registration_date', 'risk_score', 'country']

class TransactionSerializer(serializers.ModelSerializer):
    customer = CustomerSerializer(read_only=True)
    
    class Meta:
        model = Transaction
        fields = [
            'transaction_id', 'customer', 'amount', 'transaction_type',
            'timestamp', 'merchant', 'location', 'fraud_probability',
            'is_fraud', 'flagged_by_model'
        ]

class FraudAlertSerializer(serializers.ModelSerializer):
    transaction = TransactionSerializer(read_only=True)
    
    class Meta:
        model = FraudAlert
        fields = [
            'id', 'transaction', 'severity', 'alert_message',
            'created_at', 'reviewed', 'reviewer'
        ]

# api/views.py
from rest_framework import generics, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone
from datetime import timedelta

from detection.models import Transaction, FraudAlert
from .serializers import TransactionSerializer, FraudAlertSerializer

class HighRiskTransactionsView(generics.ListAPIView):
    serializer_class = TransactionSerializer
    
    def get_queryset(self):
        threshold = float(self.request.query_params.get('threshold', 0.7))
        hours = int(self.request.query_params.get('hours', 24))
        
        return Transaction.objects.filter(
            fraud_probability__gte=threshold,
            timestamp__gte=timezone.now() - timedelta(hours=hours)
        ).order_by('-fraud_probability')

class ActiveAlertsView(generics.ListAPIView):
    serializer_class = FraudAlertSerializer
    
    def get_queryset(self):
        return FraudAlert.objects.filter(
            reviewed=False
        ).select_related('transaction', 'transaction__customer').order_by('-created_at')

@api_view(['POST'])
def mark_alert_reviewed(request, alert_id):
    try:
        alert = FraudAlert.objects.get(id=alert_id)
        alert.reviewed = True
        alert.reviewer = request.user
        alert.save()
        
        return Response({'message': 'Alert marked as reviewed'})
    except FraudAlert.DoesNotExist:
        return Response({'error': 'Alert not found'}, status=404)
```

### Templates

```html
<!-- templates/detection/dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .dashboard-container { max-width: 1200px; margin: 0 auto; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .transactions-table { width: 100%; border-collapse: collapse; background: white; }
        .transactions-table th, .transactions-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .risk-high { color: #e74c3c; font-weight: bold; }
        .risk-medium { color: #f39c12; }
        .risk-low { color: #27ae60; }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <h1>Fraud Detection Control Center</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ avg_fraud_score }}</div>
                <div class="stat-label">Average Fraud Score</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Fraud Trend (Last 7 Days)</h3>
            <canvas id="fraudTrendChart" width="400" height="200"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>Recent High-Risk Transactions</h3>
            <table class="transactions-table">
                <thead>
                    <tr>
                        <th>Transaction ID</th>
                        <th>Customer</th>
                        <th>Amount</th>
                        <th>Fraud Score</th>
                        <th>Risk Level</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {% for transaction in recent_high_risk %}
                    <tr>
                        <td><a href="{% url 'transaction_detail' transaction.transaction_id %}">{{ transaction.transaction_id|truncatechars:8 }}</a></td>
                        <td>{{ transaction.customer.email }}</td>
                        <td>${{ transaction.amount }}</td>
                        <td>{{ transaction.fraud_probability|floatformat:3 }}</td>
                        <td>
                            {% if transaction.fraud_probability >= 0.8 %}
                                <span class="risk-high">HIGH</span>
                            {% elif transaction.fraud_probability >= 0.6 %}
                                <span class="risk-medium">MEDIUM</span>
                            {% else %}
                                <span class="risk-low">LOW</span>
                            {% endif %}
                        </td>
                        <td>{{ transaction.timestamp|date:"M d, H:i" }}</td>
                    </tr>
                    {% empty %}
                    <tr><td colspan="6">No high-risk transactions in the last 24 hours</td></tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Fraud trend chart
        const ctx = document.getElementById('fraudTrendChart').getContext('2d');
        const fraudData = {{ fraud_trend|safe }};
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: fraudData.map(d => d.date),
                datasets: [{
                    label: 'Fraud Detections',
                    data: fraudData.map(d => d.fraud_count),
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        // Auto-refresh dashboard every 30 seconds
        setInterval(() => {
            location.reload();
        }, 30000);
    </script>
</body>
</html>

<!-- templates/detection/transaction_detail.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Transaction Details</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; }
        .detail-card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .detail-item { padding: 10px; border-left: 4px solid #3498db; }
        .detail-label { font-weight: bold; color: #2c3e50; }
        .detail-value { margin-top: 5px; }
        .fraud-score { font-size: 2em; font-weight: bold; }
        .score-high { color: #e74c3c; }
        .score-medium { color: #f39c12; }
        .score-low { color: #27ae60; }
        .alert-card { border-left: 4px solid #e74c3c; background: #ffeaea; }
        .history-table { width: 100%; border-collapse: collapse; }
        .history-table th, .history-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Transaction Analysis Report</h1>
        
        <div class="detail-card">
            <h2>Transaction Overview</h2>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-label">Transaction ID</div>
                    <div class="detail-value">{{ transaction.transaction_id }}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Amount</div>
                    <div class="detail-value">${{ transaction.amount }}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Type</div>
                    <div class="detail-value">{{ transaction.get_transaction_type_display }}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Timestamp</div>
                    <div class="detail-value">{{ transaction.timestamp }}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Merchant</div>
                    <div class="detail-value">{{ transaction.merchant|default:"N/A" }}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Location</div>
                    <div class="detail-value">{{ transaction.location }}</div>
                </div>
            </div>
        </div>
        
        <div class="detail-card">
            <h2>Fraud Analysis</h2>
            <div class="fraud-score 
                {% if transaction.fraud_probability >= 0.7 %}score-high
                {% elif transaction.fraud_probability >= 0.4 %}score-medium
                {% else %}score-low{% endif %}">
                Fraud Score: {{ transaction.fraud_probability|floatformat:3 }}
            </div>
            <p>
                {% if transaction.fraud_probability >= 0.8 %}
                    <strong>High Risk:</strong> This transaction shows strong indicators of potential fraud and requires immediate review.
                {% elif transaction.fraud_probability >= 0.6 %}
                    <strong>Medium Risk:</strong> This transaction shows some suspicious patterns that warrant investigation.
                {% elif transaction.fraud_probability >= 0.4 %}
                    <strong>Low Risk:</strong> This transaction shows minor irregularities but is likely legitimate.
                {% else %}
                    <strong>Normal:</strong> This transaction appears to follow normal customer patterns.
                {% endif %}
            </p>
        </div>
        
        {% if fraud_alert %}
        <div class="detail-card alert-card">
            <h2>Fraud Alert</h2>
            <p><strong>Severity:</strong> {{ fraud_alert.get_severity_display }}</p>
            <p><strong>Message:</strong> {{ fraud_alert.alert_message }}</p>
            <p><strong>Created:</strong> {{ fraud_alert.created_at }}</p>
            <p><strong>Status:</strong> 
                {% if fraud_alert.reviewed %}
                    Reviewed by {{ fraud_alert.reviewer.username }}
                {% else %}
                    Pending Review
                {% endif %}
            </p>
        </div>
        {% endif %}
        
        <div class="detail-card">
            <h2>Customer Information</h2>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-label">Customer ID</div>
                    <div class="detail-value">{{ transaction.customer.customer_id }}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Email</div>
                    <div class="detail-value">{{ transaction.customer.email }}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Risk Score</div>
                    <div class="detail-value">{{ transaction.customer.risk_score|floatformat:2 }}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Registration Date</div>
                    <div class="detail-value">{{ transaction.customer.registration_date|date:"M d, Y" }}</div>
                </div>
            </div>
        </div>
        
        <div class="detail-card">
            <h2>Customer Transaction History</h2>
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Amount</th>
                        <th>Type</th>
                        <th>Fraud Score</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for hist_transaction in customer_transactions %}
                    <tr>
                        <td>{{ hist_transaction.timestamp|date:"M d, H:i" }}</td>
                        <td>${{ hist_transaction.amount }}</td>
                        <td>{{ hist_transaction.get_transaction_type_display }}</td>
                        <td>{{ hist_transaction.fraud_probability|floatformat:3 }}</td>
                        <td>
                            {% if hist_transaction.is_fraud %}
                                <span style="color: #e74c3c;">Confirmed Fraud</span>
                            {% elif hist_transaction.fraud_probability >= 0.7 %}
                                <span style="color: #f39c12;">Flagged</span>
                            {% else %}
                                <span style="color: #27ae60;">Normal</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <a href="{% url 'fraud_dashboard' %}" style="display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 4px;">← Back to Dashboard</a>
    </div>
</body>
</html>
```

### URL Configuration

```python
# detection/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.fraud_dashboard, name='fraud_dashboard'),
    path('process-transaction/', views.process_transaction, name='process_transaction'),
    path('transaction/<uuid:transaction_id>/', views.TransactionDetailView.as_view(), name='transaction_detail'),
    path('alerts/', views.AlertsListView.as_view(), name='alerts_list'),
    path('train-models/', views.train_models, name='train_models'),
]

# fraud_detector/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('detection.urls')),
    path('api/', include('api.urls')),
]

# api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('high-risk-transactions/', views.HighRiskTransactionsView.as_view(), name='api_high_risk'),
    path('active-alerts/', views.ActiveAlertsView.as_view(), name='api_active_alerts'),
    path('alerts/<int:alert_id>/review/', views.mark_alert_reviewed, name='api_mark_reviewed'),
]
```

### Settings Configuration

```python
# fraud_detector/settings.py
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
    'rest_framework',
    'detection',
    'api',
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

ROOT_URLCONF = 'fraud_detector.urls'

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
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'fraud_detection',
        'USER': 'postgres',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20
}

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

# Celery Configuration for async processing
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
```

### Requirements File

```txt
# requirements.txt
Django==4.2.7
djangorestframework==3.14.0
psycopg2-binary==2.9.7
celery==5.3.4
redis==5.0.1
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.25.2
joblib==1.3.2
python-decouple==3.8
```

### Testing the System

```python
# detection/test_fraud_detection.py
import json
import requests
from django.test import TestCase
from django.urls import reverse
from decimal import Decimal
from .models import Customer, Transaction
from .fraud_analyzer import RealTimeFraudAnalyzer

class FraudDetectionTests(TestCase):
    def setUp(self):
        self.customer = Customer.objects.create(
            email='test@example.com',
            phone='+1234567890',
            country='US'
        )
        
        # Create normal transaction history
        for i in range(10):
            Transaction.objects.create(
                customer=self.customer,
                amount=Decimal('50.00'),
                transaction_type='purchase',
                location='New York',
                ip_address='192.168.1.1',
                device_fingerprint='device123'
            )
    
    def test_normal_transaction_analysis(self):
        """Test analysis of normal transaction"""
        transaction = Transaction.objects.create(
            customer=self.customer,
            amount=Decimal('55.00'),
            transaction_type='purchase',
            location='New York',
            ip_address='192.168.1.1',
            device_fingerprint='device123'
        )
        
        analyzer = RealTimeFraudAnalyzer()
        result = analyzer.analyze_transaction(transaction)
        
        self.assertLess(result['fraud_probability'], 0.5)
        self.assertEqual(result['risk_level'], 'MINIMAL')
    
    def test_suspicious_transaction_analysis(self):
        """Test analysis of suspicious transaction"""
        transaction = Transaction.objects.create(
            customer=self.customer,
            amount=Decimal('15000.00'),  # Unusually high amount
            transaction_type='transfer',
            location='Unknown Location',  # New location
            ip_address='10.0.0.1',  # Different IP
            device_fingerprint='new_device'  # New device
        )
        
        analyzer = RealTimeFraudAnalyzer()
        result = analyzer.analyze_transaction(transaction)
        
        self.assertGreater(result['fraud_probability'], 0.7)
        self.assertEqual(result['risk_level'], 'HIGH')
    
    def test_api_transaction_processing(self):
        """Test API endpoint for transaction processing"""
        transaction_data = {
            'customer_email': 'newcustomer@example.com',
            'amount': 25000,
            'transaction_type': 'transfer',
            'location': 'Suspicious Location',
            'ip_address': '1.2.3.4',
            'device_fingerprint': 'suspicious_device'
        }
        
        response = self.client.post(
            reverse('process_transaction'),
            data=json.dumps(transaction_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertFalse(response_data['approved'])  # Should be rejected
        self.assertGreater(response_data['fraud_analysis']['fraud_probability'], 0.8)
```

### Deployment Script

```bash
#!/bin/bash
# deploy.sh

echo "🚀 Deploying Advanced Fraud Detection System..."

# Install dependencies
pip install -r requirements.txt

# Database setup
python manage.py makemigrations
python manage.py migrate

# Create superuser (if needed)
echo "Creating superuser..."
python manage.py shell -c "
from django.contrib.auth.models import User
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print('Superuser created')
"

# Generate sample data for testing
python manage.py shell -c "
from detection.models import Customer, Transaction
from decimal import Decimal
import random
from datetime import datetime, timedelta

# Create sample customers
for i in range(100):
    customer = Customer.objects.create(
        email=f'customer{i}@example.com',
        phone=f'+123456789{i:02d}',
        country='US'
    )
    
    # Create transaction history for each customer
    for j in range(random.randint(5, 50)):
        Transaction.objects.create(
            customer=customer,
            amount=Decimal(str(random.uniform(10, 1000))),
            transaction_type=random.choice(['purchase', 'transfer', 'withdrawal']),
            location=random.choice(['New York', 'Los Angeles', 'Chicago']),
            ip_address=f'192.168.1.{random.randint(1, 255)}',
            device_fingerprint=f'device{random.randint(1, 10)}',
            is_fraud=random.random() < 0.05  # 5% fraud rate
        )

print('Sample data created')
"

# Train initial ML models
python manage.py shell -c "
from detection.ml_models import FraudDetectionEngine
from detection.models import Transaction

transactions = Transaction.objects.filter(is_fraud__isnull=False)[:1000]
if transactions.count() > 50:
    engine = FraudDetectionEngine()
    engine.train_models(transactions)
    print('ML models trained')
"

echo "✅ Deployment complete!"
echo "🌐 Access the dashboard at: http://localhost:8000"
echo "👤 Admin login: admin / admin123"

# Start development server
python manage.py runserver
```

This advanced fraud detection system demonstrates:

**Key Features:**
- **Real-time Analysis**: Every transaction is analyzed instantly using multiple AI models
- **Ensemble Learning**: Combines unsupervised (Isolation Forest) and supervised (Random Forest) models
- **Behavioral Profiling**: Tracks customer patterns like a chef knowing each customer's preferences
- **Risk Scoring**: Multi-layered scoring system with business rules and ML predictions
- **Alert Management**: Automatic alert generation with severity levels
- **Dashboard Interface**: Real-time monitoring with charts and statistics
- **API Integration**: RESTful APIs for external system integration

**Advanced Techniques:**
- **Feature Engineering**: Creates meaningful features from raw transaction data
- **Anomaly Detection**: Identifies unusual patterns without labeled data
- **Time Series Analysis**: Considers temporal patterns in fraud detection
- **Velocity Checking**: Monitors transaction frequency and patterns
- **Geolocation Analysis**: Flags unusual location-based activities
- **Device Fingerprinting**: Tracks device-based behavior patterns

The system operates like a master chef's kitchen where every ingredient (transaction) is carefully inspected using multiple quality checks (ML models) to ensure nothing harmful (fraud) reaches the final dish (approved transactions). The kitchen staff (algorithms) work together, each specializing in different aspects of quality control, while the head chef (ensemble model) makes the final decision based on all inputs.{{ total_transactions }}</div>
                <div class="stat-label">Total Transactions Today</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ flagged_transactions }}</div>
                <div class="stat-label">Flagged Transactions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ critical_alerts }}</div>
                <div class="stat-label">Critical Alerts</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">

## Assignment: AI Specialization Analysis Report

**Objective**: Write a comprehensive analysis report comparing the four AI specializations and make a recommendation for your career focus.

**Requirements**:

1. **Technical Analysis** (40 points):
   - For each specialization, explain the core algorithms and techniques used
   - Compare the types of data each specialization works with
   - Discuss the computational requirements and challenges of each field
   - Analyze the Django implementation patterns specific to each specialization

2. **Real-World Applications** (30 points):
   - Identify three specific industry applications for each specialization
   - Explain how each specialization solves different business problems
   - Provide examples of companies successfully using each approach
   - Discuss the potential impact and scalability of each specialization

3. **Personal Career Path Analysis** (20 points):
   - Based on your experience with the code examples, identify your strengths and interests
   - Choose one specialization as your primary focus and explain your reasoning
   - Outline a 6-month learning plan for your chosen specialization
   - Identify potential career opportunities and salary ranges in your chosen field

4. **Integration Strategy** (10 points):
   - Explain how your chosen specialization can work together with the other three
   - Design a hypothetical project that combines multiple specializations
   - Discuss the benefits of cross-specialization knowledge in AI careers

**Deliverables**:
- A 2,000-word written report
- One code example demonstrating advanced concepts in your chosen specialization
- A visual diagram showing the relationships between specializations

**Submission Guidelines**:
- Use proper academic formatting with citations
- Include screenshots of your working Django application
- Submit both the report and your complete code implementation

**Evaluation Criteria**:
- Depth of technical understanding
- Quality of real-world application analysis
- Clarity of career reasoning and planning
- Code quality and functionality
- Overall presentation and professionalism

This assignment ensures you not only understand the technical aspects of each specialization but can also make informed decisions about your AI career path, just like choosing your signature cooking style after mastering the fundamentals of all cuisines.