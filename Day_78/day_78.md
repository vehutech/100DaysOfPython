# AI Mastery Course - Day 78: Neural Networks Foundations
*Using Python Core and Django Web Framework*

## Learning Objective
By the end of this lesson, you will understand the fundamental building blocks of neural networks, implement core concepts from scratch using Python, and create a Django web application that demonstrates neural network predictions in action.

---

## Introduction

Imagine that you're the head chef in a world-class restaurant, and you need to train a team of sous chefs to create the perfect dish. Each sous chef (neuron) receives different ingredients (inputs), processes them with their unique cooking techniques (activation functions), and passes the result to the next chef in line. Through trial and error, feedback from customers (backpropagation), and constant recipe refinement (gradient descent), your kitchen eventually learns to create masterpiece dishes consistently.

This is exactly how neural networks work - layers of interconnected "chefs" (neurons) that learn to transform raw ingredients (data) into perfect predictions through practice and feedback.

---

## 1. Perceptrons and Multi-Layer Networks

### The Single Chef (Perceptron)

A perceptron is like having one skilled chef who takes multiple ingredients and decides whether to create a dish or not based on a simple rule.

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        # Initialize weights randomly (chef's initial recipe preferences)
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
    
    def activation_function(self, x):
        # Step function: chef decides "cook" (1) or "don't cook" (0)
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        # Chef combines ingredients based on weights and bias
        linear_output = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(linear_output)
    
    def train(self, training_inputs, labels, epochs):
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # Chef learns from mistakes (error correction)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# Example: Training a chef to recognize good ingredients
# Let's say we have 2 ingredients: freshness (0-1) and quality (0-1)
training_data = np.array([[0.8, 0.9], [0.2, 0.1], [0.9, 0.8], [0.1, 0.3]])
labels = np.array([1, 0, 1, 0])  # 1 = cook, 0 = don't cook

chef = Perceptron(input_size=2)
chef.train(training_data, labels, epochs=100)

print(f"Chef's learned weights: {chef.weights}")
print(f"Chef's bias: {chef.bias}")
```

### The Kitchen Brigade (Multi-Layer Network)

In a real kitchen, you have multiple layers of chefs working together:

```python
class MultiLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # First layer of sous chefs (input to hidden)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        
        # Head chef layer (hidden to output)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        # Smooth decision making (better than step function)
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def forward_pass(self, X):
        # Ingredients flow through the kitchen
        self.z1 = np.dot(X, self.W1) + self.b1  # Sous chefs' work
        self.a1 = self.sigmoid(self.z1)         # Sous chefs' decisions
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Head chef's work
        self.a2 = self.sigmoid(self.z2)               # Final dish quality
        
        return self.a2

# Create a kitchen with 2 ingredient stations, 4 sous chefs, 1 head chef
kitchen = MultiLayerNetwork(input_size=2, hidden_size=4, output_size=1)

# Test the untrained kitchen
test_ingredients = np.array([[0.8, 0.6]])
result = kitchen.forward_pass(test_ingredients)
print(f"Untrained kitchen output: {result[0][0]:.3f}")
```

---

## 2. Activation Functions and Backpropagation

### Activation Functions: The Chef's Decision Style

Different chefs make decisions differently:

```python
import matplotlib.pyplot as plt

def activation_functions_demo():
    x = np.linspace(-5, 5, 100)
    
    # Step function: "All or nothing" chef
    step = np.where(x >= 0, 1, 0)
    
    # Sigmoid: "Smooth decision" chef  
    sigmoid = 1 / (1 + np.exp(-x))
    
    # ReLU: "Positive vibes only" chef
    relu = np.maximum(0, x)
    
    # Tanh: "Balanced judgment" chef
    tanh = np.tanh(x)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, step, 'b-', linewidth=2)
    plt.title('Step Function: Binary Chef')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x, sigmoid, 'r-', linewidth=2)
    plt.title('Sigmoid: Smooth Chef')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x, relu, 'g-', linewidth=2)
    plt.title('ReLU: Optimistic Chef')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(x, tanh, 'm-', linewidth=2)
    plt.title('Tanh: Balanced Chef')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# activation_functions_demo()  # Uncomment to see the plots
```

### Backpropagation: Learning from Customer Feedback

```python
class NeuralNetworkWithBackprop:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        # How much the chef's decision changes with small input changes
        return x * (1 - x)
    
    def forward_pass(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward_pass(self, X, y, output):
        m = X.shape[0]  # Number of training examples
        
        # Calculate error at output (customer feedback)
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # Calculate error at hidden layer (feedback to sous chefs)
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Update weights and biases (improve recipes)
        self.W2 += self.a1.T.dot(output_delta) * self.learning_rate
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.W1 += X.T.dot(hidden_delta) * self.learning_rate
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, X, y, epochs):
        costs = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward_pass(X)
            
            # Calculate cost (how unhappy customers are)
            cost = np.mean((y - output) ** 2)
            costs.append(cost)
            
            # Backward pass (learn from feedback)
            self.backward_pass(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")
        
        return costs

# Training example: Teaching the kitchen to rate dishes
X_train = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.8], [0.9, 0.1]])
y_train = np.array([[0.2], [0.9], [0.6], [0.4]])

kitchen = NeuralNetworkWithBackprop(2, 4, 1, learning_rate=0.5)
costs = kitchen.train(X_train, y_train, epochs=1000)

print("\nTrained kitchen predictions:")
for i, x in enumerate(X_train):
    prediction = kitchen.forward_pass(x.reshape(1, -1))
    print(f"Ingredients {x} -> Predicted quality: {prediction[0][0]:.3f}, Actual: {y_train[i][0]}")
```

---

## 3. Gradient Descent Optimization

### The Recipe Improvement Process

Gradient descent is like a chef systematically improving their recipe:

```python
class GradientDescentDemo:
    def __init__(self):
        self.history = []
    
    def cost_function(self, w, X, y):
        # How "bad" our current recipe is
        predictions = X * w
        cost = np.mean((predictions - y) ** 2)
        return cost
    
    def gradient(self, w, X, y):
        # Which direction to adjust the recipe
        predictions = X * w
        gradient = 2 * np.mean(X * (predictions - y))
        return gradient
    
    def optimize(self, X, y, learning_rate=0.01, epochs=100):
        # Start with a random recipe
        w = np.random.randn()
        
        print(f"Starting weight (recipe): {w:.3f}")
        
        for epoch in range(epochs):
            cost = self.cost_function(w, X, y)
            grad = self.gradient(w, X, y)
            
            # Adjust recipe based on gradient
            w = w - learning_rate * grad
            
            self.history.append((w, cost))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Weight = {w:.3f}, Cost = {cost:.3f}")
        
        return w

# Demo: Learning the perfect cooking time multiplier
cooking_times = np.array([1, 2, 3, 4, 5])  # Base cooking times
perfect_results = np.array([2, 4, 6, 8, 10])  # Perfect results (2x multiplier)

optimizer = GradientDescentDemo()
learned_multiplier = optimizer.optimize(cooking_times, perfect_results, learning_rate=0.1, epochs=100)

print(f"\nLearned multiplier: {learned_multiplier:.3f}")
print("The kitchen learned that perfect cooking time = base_time * 2!")
```

---

## 4. Setting up TensorFlow/Keras

### Professional Kitchen Tools

```python
# First, install TensorFlow: pip install tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

class ProfessionalKitchen:
    def __init__(self):
        self.model = None
        self.history = None
    
    def build_kitchen(self, input_shape, hidden_units=64):
        """Build a professional neural network kitchen"""
        self.model = keras.Sequential([
            # Input layer: Ingredient receiving station
            layers.Dense(hidden_units, activation='relu', input_shape=(input_shape,), 
                        name='sous_chefs'),
            
            # Hidden layer: More specialized chefs
            layers.Dense(hidden_units//2, activation='relu', name='senior_chefs'),
            
            # Output layer: Head chef's final decision
            layers.Dense(1, activation='sigmoid', name='head_chef')
        ])
        
        # Configure the kitchen's learning process
        self.model.compile(
            optimizer='adam',  # Smart learning algorithm
            loss='binary_crossentropy',  # How to measure mistakes
            metrics=['accuracy']  # Success rate tracking
        )
        
        print("Professional kitchen built!")
        print(self.model.summary())
    
    def train_kitchen(self, X_train, y_train, epochs=50, validation_split=0.2):
        """Train the kitchen with customer feedback"""
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1
        )
        
    def predict_dish_quality(self, ingredients):
        """Predict how good a dish will be"""
        predictions = self.model.predict(ingredients)
        return predictions

# Example usage
# Generate synthetic restaurant data
np.random.seed(42)
n_samples = 1000

# Features: [ingredient_quality, chef_skill, kitchen_temperature, prep_time]
X_restaurant = np.random.rand(n_samples, 4)

# Target: Good dish (1) or bad dish (0)
# Good dishes need: high quality ingredients AND skilled chef
y_restaurant = ((X_restaurant[:, 0] > 0.6) & (X_restaurant[:, 1] > 0.5)).astype(int)

# Build and train the professional kitchen
kitchen = ProfessionalKitchen()
kitchen.build_kitchen(input_shape=4)
kitchen.train_kitchen(X_restaurant, y_restaurant, epochs=20)

# Test the trained kitchen
test_ingredients = np.array([[0.9, 0.8, 0.7, 0.6]])  # High quality ingredients
prediction = kitchen.predict_dish_quality(test_ingredients)
print(f"\nDish quality prediction: {prediction[0][0]:.3f}")
print("Confidence level:", "High" if prediction[0][0] > 0.7 else "Low")
```

---

## Code Syntax Explanation

### Key Python and Neural Network Concepts Used:

1. **NumPy Arrays**: `np.array()`, `np.dot()`, `np.random.randn()`
   - Efficient mathematical operations on large datasets
   - Matrix multiplication for neural network computations

2. **Class Definitions**: `class NeuralNetwork:`
   - Object-oriented programming to encapsulate network behavior
   - Methods like `__init__()`, `forward_pass()`, `backward_pass()`

3. **Broadcasting**: Automatic array dimension matching
   - Allows operations between arrays of different shapes
   - Essential for batch processing in neural networks

4. **Activation Functions**: Mathematical functions that introduce non-linearity
   - `sigmoid(x) = 1/(1 + e^(-x))`: Smooth decision making
   - `relu(x) = max(0, x)`: Simple, effective activation

5. **TensorFlow/Keras Syntax**:
   - `keras.Sequential()`: Stack layers in sequence
   - `layers.Dense()`: Fully connected layer of neurons
   - `model.compile()`: Configure learning algorithm
   - `model.fit()`: Train the network

---

# Build: Neural Network from Scratch

## Project: Kitchen Recipe Classification System

Just as a master chef learns to identify dishes by their core ingredients and preparation methods, we'll build a neural network that can classify data by learning the fundamental patterns within it. Our neural network will be the digital equivalent of a chef's trained palate - starting with raw ingredients (input data) and through layers of processing (hidden layers), arriving at a final classification (output layer).

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class NeuralNetwork:
    """
    A neural network from scratch - like building a kitchen from the ground up
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize our neural network - setting up our kitchen stations
        
        Args:
            input_size: Number of input features (ingredients)
            hidden_size: Number of neurons in hidden layer (prep stations)
            output_size: Number of output classes (final dishes)
            learning_rate: How fast our chef learns from mistakes
        """
        # Initialize weights and biases - our chef's initial recipe knowledge
        # Hidden layer weights: connections from input to hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        
        # Output layer weights: connections from hidden to output layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        
        # Track our learning progress - like a chef's journal
        self.loss_history = []
        self.accuracy_history = []
    
    def sigmoid(self, z):
        """
        Sigmoid activation function - like a chef's taste adjustment
        Smoothly transforms any input to a value between 0 and 1
        """
        # Clip z to prevent overflow - like controlling heat in cooking
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """
        Derivative of sigmoid - how sensitive our taste buds are to changes
        """
        return z * (1 - z)
    
    def softmax(self, z):
        """
        Softmax activation for output layer - like a chef's final presentation
        Converts raw scores to probabilities that sum to 1
        """
        # Subtract max for numerical stability - like balancing flavors
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Forward pass through the network - like following a recipe step by step
        
        Args:
            X: Input data (our raw ingredients)
            
        Returns:
            Tuple of (hidden_output, final_output)
        """
        # Hidden layer computation - first cooking stage
        # z1 = X * W1 + b1 (linear transformation)
        self.z1 = np.dot(X, self.W1) + self.b1
        # a1 = sigmoid(z1) (activation - like applying heat)
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer computation - final plating
        # z2 = a1 * W2 + b2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # a2 = softmax(z2) (final probabilities)
        self.a2 = self.softmax(self.z2)
        
        return self.a1, self.a2
    
    def compute_loss(self, y_true, y_pred):
        """
        Calculate cross-entropy loss - how far off our dish is from perfect
        
        Args:
            y_true: True labels (what the dish should be)
            y_pred: Predicted probabilities (what our chef thinks it is)
        """
        # Avoid log(0) by adding small epsilon - like ensuring ingredients don't burn
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy loss
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def backward_propagation(self, X, y_true):
        """
        Backward pass - learning from our cooking mistakes
        
        Args:
            X: Input data
            y_true: True labels (one-hot encoded)
        """
        m = X.shape[0]  # Number of training examples
        
        # Output layer gradients - how wrong was our final dish?
        dz2 = self.a2 - y_true  # Error in output
        dW2 = (1/m) * np.dot(self.a1.T, dz2)  # Weight gradients
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)  # Bias gradients
        
        # Hidden layer gradients - tracing the mistake back to prep
        da1 = np.dot(dz2, self.W2.T)  # Error propagated to hidden layer
        dz1 = da1 * self.sigmoid_derivative(self.a1)  # Apply derivative
        dW1 = (1/m) * np.dot(X.T, dz1)  # Weight gradients
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)  # Bias gradients
        
        # Update weights and biases - chef learns from mistakes
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def one_hot_encode(self, y):
        """
        Convert labels to one-hot encoding - like separating ingredients by type
        """
        num_classes = len(np.unique(y))
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network - our chef's intensive training program
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training iterations
            verbose: Whether to print progress
        """
        # Convert labels to one-hot encoding
        y_one_hot = self.one_hot_encode(y)
        
        for epoch in range(epochs):
            # Forward pass - cook the dish
            _, predictions = self.forward_propagation(X)
            
            # Calculate loss - taste the result
            loss = self.compute_loss(y_one_hot, predictions)
            
            # Backward pass - learn from mistakes
            self.backward_propagation(X, y_one_hot)
            
            # Calculate accuracy
            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(predicted_classes == y)
            
            # Record progress
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            
            # Print progress every 100 epochs
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
    
    def predict(self, X):
        """
        Make predictions - our trained chef classifies new dishes
        """
        _, predictions = self.forward_propagation(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities - chef's confidence in classification
        """
        _, predictions = self.forward_propagation(X)
        return predictions

# Create a realistic dataset - our ingredient combinations
print("🍳 Generating Recipe Dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=6,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

# Feature names - our cooking ingredients
feature_names = [
    'sweetness', 'saltiness', 'bitterness', 'umami',
    'temperature', 'texture', 'aroma_intensity', 'color_depth'
]

# Class names - our dish categories
class_names = ['Appetizer', 'Main Course', 'Dessert']

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['dish_type'] = [class_names[i] for i in y]

print("\n📊 Dataset Overview:")
print(f"Number of recipes: {len(df)}")
print(f"Number of ingredients (features): {len(feature_names)}")
print(f"Dish types: {class_names}")
print("\nFirst 5 recipes:")
print(df.head())

# Split the data - separate training and testing kitchens
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features - normalize our ingredient measurements
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n🔪 Training set: {X_train_scaled.shape[0]} recipes")
print(f"🍽️ Test set: {X_test_scaled.shape[0]} recipes")

# Create and train our neural network - build our master chef
print("\n🧠 Building Neural Network Chef...")
nn = NeuralNetwork(
    input_size=8,    # 8 ingredient features
    hidden_size=12,  # 12 neurons in hidden layer (prep stations)
    output_size=3,   # 3 dish categories
    learning_rate=0.01
)

print("🎓 Training the Chef...")
nn.train(X_train_scaled, y_train, epochs=1000, verbose=True)

# Test our trained chef
print("\n🧪 Testing Our Trained Chef...")
train_predictions = nn.predict(X_train_scaled)
test_predictions = nn.predict(X_test_scaled)

train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Visualize training progress
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(nn.loss_history)
plt.title('Learning Curve - Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(nn.accuracy_history)
plt.title('Learning Curve - Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

# Feature importance visualization
plt.subplot(1, 3, 3)
# Calculate average absolute weights for each input feature
feature_importance = np.mean(np.abs(nn.W1), axis=1)
plt.bar(range(len(feature_names)), feature_importance)
plt.title('Feature Importance (Ingredient Impact)')
plt.xlabel('Ingredients')
plt.ylabel('Average Weight Magnitude')
plt.xticks(range(len(feature_names)), feature_names, rotation=45)

plt.tight_layout()
plt.show()

# Demonstrate prediction with confidence scores
print("\n🔮 Chef's Predictions on New Recipes:")
print("-" * 50)

# Get prediction probabilities for first 5 test samples
test_probabilities = nn.predict_proba(X_test_scaled[:5])

for i in range(5):
    actual_class = class_names[y_test[i]]
    predicted_class = class_names[test_predictions[i]]
    confidence = test_probabilities[i]
    
    print(f"\nRecipe {i+1}:")
    print(f"Actual dish type: {actual_class}")
    print(f"Predicted dish type: {predicted_class}")
    print("Chef's confidence:")
    for j, class_name in enumerate(class_names):
        print(f"  {class_name}: {confidence[j]:.3f} ({confidence[j]*100:.1f}%)")
    
    # Show ingredient profile
    print("Ingredient profile:")
    recipe_features = X_test_scaled[i]
    for k, ingredient in enumerate(feature_names):
        print(f"  {ingredient}: {recipe_features[k]:.2f}")

# Advanced: Visualize decision boundaries (for 2D projection)
print("\n🎨 Visualizing Chef's Decision Making...")

# Project to 2D using first two principal components for visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d = pca.transform(X_test_scaled)

# Create a mesh for decision boundary
h = 0.02
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Transform mesh back to original space and predict
mesh_points = np.c_[xx.ravel(), yy.ravel()]
# Pad with zeros for other dimensions (simplified visualization)
mesh_points_full = np.zeros((mesh_points.shape[0], 8))
mesh_points_full[:, :2] = mesh_points

# Get predictions for mesh
mesh_predictions = nn.predict(mesh_points_full)
mesh_predictions = mesh_predictions.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, mesh_predictions, alpha=0.6, cmap=plt.cm.Set3)

# Plot training points
colors = ['red', 'blue', 'green']
for i, class_name in enumerate(class_names):
    mask = y_train == i
    plt.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1], 
               c=colors[i], label=f'{class_name} (train)', 
               alpha=0.7, s=50)

# Plot test points
for i, class_name in enumerate(class_names):
    mask = y_test == i
    plt.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1], 
               c=colors[i], marker='x', label=f'{class_name} (test)', 
               s=100, linewidth=2)

plt.xlabel(f'First Principal Component')
plt.ylabel(f'Second Principal Component')
plt.title('Neural Network Decision Boundaries\n(2D Projection of Recipe Space)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n✨ Neural Network Training Complete!")
print(f"🏆 Final Test Accuracy: {test_accuracy:.1%}")
print("\n🧠 Key Neural Network Components Built:")
print("• Forward Propagation: Data flows through network layers")
print("• Backward Propagation: Learning from prediction errors")
print("• Activation Functions: Non-linear transformations (sigmoid, softmax)")
print("• Gradient Descent: Optimization algorithm for learning")
print("• Weight Initialization: Starting point for network parameters")
print("• Loss Function: Cross-entropy for multi-class classification")

# Save our trained model weights (simulation)
print("\n💾 Saving Trained Chef Model...")
model_weights = {
    'W1': nn.W1,
    'b1': nn.b1,
    'W2': nn.W2,
    'b2': nn.b2,
    'feature_names': feature_names,
    'class_names': class_names,
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_
}

print("✅ Model saved! Your neural network chef is ready to classify new recipes.")
print("\n🔬 This implementation demonstrates:")
print("• Matrix operations for efficient computation")
print("• Numerical stability techniques (gradient clipping, epsilon)")
print("• Proper weight initialization and learning rate selection")
print("• Training loop with loss tracking and validation")
print("• Visualization of learning progress and decision boundaries")
```

## Project Architecture

This neural network implementation creates a complete classification system that demonstrates all fundamental concepts:

**Network Structure:**
- **Input Layer**: 8 neurons (ingredient features)
- **Hidden Layer**: 12 neurons with sigmoid activation
- **Output Layer**: 3 neurons with softmax activation (dish categories)

**Key Implementation Features:**

1. **Matrix Operations**: All computations use vectorized NumPy operations for efficiency
2. **Activation Functions**: Sigmoid for hidden layer, softmax for output probabilities
3. **Loss Function**: Cross-entropy loss for multi-class classification
4. **Optimization**: Gradient descent with backpropagation
5. **Regularization**: Gradient clipping and weight initialization for stability

**Training Process:**
- Forward propagation computes predictions
- Loss calculation measures prediction error
- Backward propagation computes gradients
- Weight updates minimize the loss function

The project generates a synthetic dataset representing recipe classifications, trains the network from scratch, and provides comprehensive visualization of the learning process and decision boundaries. This hands-on implementation solidifies understanding of neural network fundamentals without relying on high-level frameworks.

## Assignment: Restaurant Review Predictor

**Objective**: Create a neural network that predicts whether a restaurant review is positive or negative based on numerical features.

**Your Task**:
1. Create a dataset with these features for 500 restaurants:
   - Service speed (1-10)
   - Food quality (1-10) 
   - Ambiance rating (1-10)
   - Price fairness (1-10)
   - Wait time in minutes (0-60)

2. Generate labels: A review is positive (1) if:
   - Food quality ≥ 7 AND Service speed ≥ 6 AND Wait time ≤ 30
   - Otherwise negative (0)

3. Build a neural network with:
   - Input layer: 5 features
   - One hidden layer: 8 neurons with ReLU activation
   - Output layer: 1 neuron with sigmoid activation

4. Train for 50 epochs and report:
   - Final training accuracy
   - Predictions on 3 test cases you create
   - Which feature weights are highest (most important for predictions)

**Deliverable**: Python script with your implementation and a brief explanation of your results.

**Hint**: Use the `ProfessionalKitchen` class as a starting point, but modify it for your specific problem.

---

## Summary

Today you learned the foundational concepts of neural networks through our analogy:

- **Perceptrons**: Individual chefs making simple decisions
- **Multi-layer networks**: Teams of chefs working together
- **Activation functions**: Different decision-making styles
- **Backpropagation**: Learning from customer feedback
- **Gradient descent**: Systematic recipe improvement
- **TensorFlow/Keras**: Professional kitchen tools

You now have the fundamental knowledge to build neural networks from scratch and understand how they learn to transform inputs into accurate predictions, just like a kitchen transforms ingredients into perfect dishes through practice and feedback.