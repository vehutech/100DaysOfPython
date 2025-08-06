# AI Mastery Course - Day 79: Deep Neural Networks

## Learning Objective
By the end of this lesson, you will understand how to construct and optimize deep neural networks, solve gradient-related challenges, and implement advanced techniques like batch normalization and dropout to create robust, high-performing models.

---

## Imagine That...

Imagine that you're the head chef of an ambitious restaurant that's expanding from making simple dishes to creating elaborate multi-course meals. Just as a master chef layers flavors, textures, and cooking techniques across multiple courses to create an extraordinary dining experience, we're going to learn how to layer neural network components to create deep, sophisticated AI models.

In our culinary journey today, you'll discover how to stack ingredients (layers) properly, prevent your delicate sauces from breaking (vanishing gradients), maintain consistency across your entire menu (batch normalization), and know when to hold back certain ingredients to prevent overwhelming the palate (dropout). Finally, you'll master advanced cooking techniques (optimizers) that help you achieve perfect results every time.

---

## 1. Deep Network Architectures

Think of a deep neural network like preparing a complex, multi-stage meal. Each layer is like a cooking station where ingredients are transformed - the prep station (input layer), multiple cooking stages (hidden layers), and final plating (output layer).

### Basic Deep Network Structure

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepNeuralNetwork, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers - like multiple cooking stations
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        
        # Final layer - the plating station
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Example: Creating a 5-layer deep network
# Like a 5-course meal preparation process
deep_model = DeepNeuralNetwork(
    input_size=20,           # 20 raw ingredients
    hidden_sizes=[128, 64, 32, 16],  # 4 cooking stations of decreasing complexity
    output_size=2            # 2 final dishes (binary classification)
)

print("Deep Network Architecture:")
print(deep_model)
```

**Syntax Explanation:**
- `nn.Module`: Base class for all neural network modules in PyTorch
- `super().__init__()`: Calls the parent class constructor
- `nn.Linear(in_features, out_features)`: Creates a fully connected layer
- `nn.Sequential(*layers)`: Combines layers into a sequential model
- `*layers`: Unpacks the list to pass each layer as a separate argument

---

## 2. Vanishing Gradient Problem

Just like how delicate flavors can get lost when you have too many strong spices in a complex dish, gradients can vanish as they travel backward through many layers, making it hard to train the early layers effectively.

### Demonstrating the Problem

```python
def analyze_gradients(model, data_loader, device):
    """
    Analyze gradient flow through the network
    Like checking if flavors from early prep work make it to the final dish
    """
    model.train()
    gradients = {name: [] for name, param in model.named_parameters() if 'weight' in name}
    
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx > 5:  # Just analyze a few batches
            break
            
        data, target = data.to(device), target.to(device)
        
        # Clear gradients
        model.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Collect gradient norms for each layer
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradients[name].append(grad_norm)
    
    return gradients

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# Create data loader
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deep_model.to(device)

# Analyze gradients
gradients = analyze_gradients(deep_model, data_loader, device)

# Plot gradient magnitudes
layer_names = list(gradients.keys())
avg_gradients = [np.mean(gradients[name]) for name in layer_names]

plt.figure(figsize=(12, 6))
plt.plot(range(len(layer_names)), avg_gradients, 'bo-')
plt.xlabel('Layer Index (0 = earliest layer)')
plt.ylabel('Average Gradient Magnitude')
plt.title('Gradient Flow Analysis - The Vanishing Gradient Problem')
plt.xticks(range(len(layer_names)), [f'Layer {i+1}' for i in range(len(layer_names))], rotation=45)
plt.grid(True, alpha=0.3)
plt.show()
```

**Syntax Explanation:**
- `model.named_parameters()`: Returns an iterator over module parameters, yielding both name and parameter
- `param.grad.norm().item()`: Calculates the L2 norm of gradients and converts to Python scalar
- `model.zero_grad()`: Clears gradients from previous iteration
- `loss.backward()`: Computes gradients using backpropagation

---

## 3. Batch Normalization and Dropout

### Batch Normalization
Like a chef standardizing ingredient preparation to ensure consistent results across all dishes, batch normalization standardizes inputs to each layer.

### Dropout
Like a master chef who doesn't rely on just one signature ingredient but creates dishes that taste good even when some components are missing, dropout randomly removes some neurons during training.

```python
class ImprovedDeepNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(ImprovedDeepNetwork, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))  # Standardize like prep work
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))  # Sometimes skip ingredients
        
        # Hidden layers with batch norm and dropout
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer (no dropout here - we need all final touches)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create improved model
improved_model = ImprovedDeepNetwork(
    input_size=20,
    hidden_sizes=[128, 64, 32, 16],
    output_size=2,
    dropout_rate=0.3
)

print("Improved Network with Batch Normalization and Dropout:")
print(improved_model)
```

**Syntax Explanation:**
- `nn.BatchNorm1d(num_features)`: Applies batch normalization over a 2D or 3D input
- `nn.Dropout(p)`: Randomly zeroes some elements with probability p during training
- The order matters: Linear → BatchNorm → Activation → Dropout

---

## 4. Advanced Optimizers (Adam, RMSprop)

Just as different cooking techniques work better for different dishes, different optimizers work better for different neural network challenges.

```python
def train_with_different_optimizers(model, train_loader, num_epochs=50):
    """
    Compare different optimization techniques
    Like comparing different cooking methods for the same recipe
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Different optimizers - like different cooking techniques
    optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        'Adam': optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999)),
        'RMSprop': optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
    }
    
    results = {}
    
    for optimizer_name, optimizer in optimizers.items():
        print(f"\nTraining with {optimizer_name} optimizer...")
        
        # Reset model parameters
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        
        losses = []
        model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                
                # Backward pass
                loss.backward()
                
                # Update weights - like adjusting seasoning
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        results[optimizer_name] = losses
    
    return results

# Create training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train with different optimizers
training_results = train_with_different_optimizers(improved_model, train_loader)

# Plot comparison
plt.figure(figsize=(12, 8))
for optimizer_name, losses in training_results.items():
    plt.plot(losses, label=f'{optimizer_name}', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Optimizer Comparison - Different Cooking Techniques')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Syntax Explanation:**
- `optim.SGD(parameters, lr, momentum)`: Stochastic Gradient Descent with momentum
- `optim.Adam(parameters, lr, betas)`: Adaptive Moment Estimation optimizer
- `optim.RMSprop(parameters, lr, alpha)`: Root Mean Square Propagation
- `betas=(0.9, 0.999)`: Coefficients for computing running averages in Adam
- `model.apply(fn)`: Applies a function recursively to every submodule

---

## 5. Complete Training Pipeline

```python
class DeepLearningChef:
    """
    A complete deep learning training class
    Like a master chef's complete cooking methodology
    """
    
    def __init__(self, model, optimizer_name='Adam', learning_rate=0.001):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Select optimizer
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'RMSprop':
            self.optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model performance"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100 * correct / total
    
    def cook_model(self, train_loader, val_loader, epochs=100, patience=10):
        """
        Complete training process - like preparing a perfect meal
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        print("Starting the cooking process...")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_acc = self.validate(val_loader)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}: Loss = {train_loss:.4f}, Val Acc = {val_acc:.2f}%')
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc

# Create validation data
val_dataset = TensorDataset(X_test, y_test)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the chef
chef = DeepLearningChef(improved_model, optimizer_name='Adam', learning_rate=0.001)

# Cook the model
best_accuracy = chef.cook_model(train_loader, val_loader, epochs=100, patience=15)

# Plot training progress
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(chef.train_losses)
ax1.set_title('Training Loss Over Time')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(chef.val_accuracies)
ax2.set_title('Validation Accuracy Over Time')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Syntax Explanation:**
- `torch.nn.utils.clip_grad_norm_(parameters, max_norm)`: Clips gradient norm to prevent explosion
- `torch.no_grad()`: Context manager that disables gradient computation
- `torch.max(input, dim)`: Returns maximum values and indices along a dimension
- `torch.save(obj, path)`: Saves an object to a file
- `model.eval()`: Sets model to evaluation mode (affects dropout and batch norm)

---

# Deep Neural Network Image Classifier Project

## Project Overview
Build a sophisticated deep learning model that can classify images across multiple categories with high accuracy. Think of this as creating a master chef's signature dish - we'll layer multiple neural network components to create something truly exceptional.

## Project Setup

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
```

## Data Preparation

```python
# Load and prepare the CIFAR-10 dataset
def load_and_prepare_data():
    """
    Load CIFAR-10 dataset and prepare it for deep learning
    Like organizing ingredients before cooking
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to [0,1] range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    # Create validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# Load the data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_prepare_data()

print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_val.shape}")
print(f"Test data shape: {x_test.shape}")
```

## Data Augmentation

```python
def create_data_generators():
    """
    Create data generators for training augmentation
    Like having different cooking techniques for the same ingredients
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    # Validation generator (no augmentation)
    val_datagen = ImageDataGenerator()
    
    return train_datagen, val_datagen

train_datagen, val_datagen = create_data_generators()
```

## Deep Neural Network Architecture

```python
def build_deep_classifier():
    """
    Build a deep neural network for image classification
    Like creating a multi-layer recipe with various cooking techniques
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(32, 32, 3)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# Build the model
model = build_deep_classifier()
model.summary()
```

## Advanced Optimizer Configuration

```python
# Configure Adam optimizer with custom parameters
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_2_accuracy']
)
```

## Training Configuration with Callbacks

```python
def create_callbacks():
    """
    Create training callbacks for better model management
    Like having timers and temperature controls while cooking
    """
    callbacks = [
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        keras.callbacks.ModelCheckpoint(
            'best_deep_classifier.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate scheduler
        keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * 0.95 ** epoch
        )
    ]
    
    return callbacks

callbacks = create_callbacks()
```

## Model Training

```python
def train_deep_model():
    """
    Train the deep neural network
    Like the actual cooking process with careful monitoring
    """
    # Prepare data generators
    train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=32
    )
    
    val_generator = val_datagen.flow(
        x_val, y_val,
        batch_size=32
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(x_train) // 32,
        epochs=100,
        validation_data=val_generator,
        validation_steps=len(x_val) // 32,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Train the model
print("Starting deep neural network training...")
history = train_deep_model()
```

## Model Evaluation and Visualization

```python
def evaluate_and_visualize():
    """
    Evaluate model performance and create visualizations
    Like tasting and presenting the final dish
    """
    # Evaluate on test set
    test_loss, test_accuracy, test_top2 = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Top-2 Accuracy: {test_top2:.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Learning rate plot
    plt.subplot(1, 3, 3)
    plt.plot(history.history.get('lr', []), label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

evaluate_and_visualize()
```

## Prediction and Inference System

```python
def create_prediction_system():
    """
    Create a complete prediction system
    Like serving the dish with proper presentation
    """
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    def predict_image(image_array):
        """Make prediction on a single image"""
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return class_names[predicted_class], confidence, predictions[0]
    
    def predict_batch(images):
        """Make predictions on multiple images"""
        predictions = model.predict(images)
        results = []
        
        for i, pred in enumerate(predictions):
            predicted_class = np.argmax(pred)
            confidence = pred[predicted_class]
            results.append({
                'class': class_names[predicted_class],
                'confidence': confidence,
                'probabilities': pred
            })
        
        return results
    
    return predict_image, predict_batch

predict_image, predict_batch = create_prediction_system()
```

## Model Analysis and Insights

```python
def analyze_model_performance():
    """
    Analyze model performance in detail
    Like a chef analyzing the taste profile of their dish
    """
    # Get predictions for test set
    test_predictions = model.predict(x_test)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    test_true_classes = np.argmax(y_test, axis=1)
    
    # Create confusion matrix
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(test_true_classes, test_pred_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck'],
                yticklabels=['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck'])
    plt.title('Confusion Matrix - Deep Neural Network Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Print detailed classification report
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    report = classification_report(test_true_classes, test_pred_classes,
                                 target_names=class_names)
    print("Detailed Classification Report:")
    print(report)

analyze_model_performance()
```

## Sample Predictions Visualization

```python
def visualize_sample_predictions():
    """
    Visualize sample predictions with confidence scores
    Like presenting sample dishes to customers
    """
    # Select random test images
    sample_indices = np.random.choice(len(x_test), 12, replace=False)
    sample_images = x_test[sample_indices]
    sample_true_labels = np.argmax(y_test[sample_indices], axis=1)
    
    # Get predictions
    results = predict_batch(sample_images)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(sample_images[i])
        
        predicted_class = results[i]['class']
        confidence = results[i]['confidence']
        true_class = class_names[sample_true_labels[i]]
        
        # Color based on correctness
        color = 'green' if predicted_class == true_class else 'red'
        
        plt.title(f'Pred: {predicted_class}\nConf: {confidence:.3f}\nTrue: {true_class}',
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_sample_predictions()
```

## Model Deployment Preparation

```python
def prepare_for_deployment():
    """
    Prepare the model for production deployment
    Like packaging the recipe for commercial use
    """
    # Save the complete model
    model.save('deep_image_classifier_complete.h5')
    
    # Save model in TensorFlow Lite format for mobile deployment
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('deep_classifier_mobile.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # Create model architecture summary
    with open('model_architecture.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Save training configuration
    config = {
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs_trained': len(history.history['loss']),
        'final_accuracy': history.history['accuracy'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'data_augmentation': True,
        'dropout_rate': [0.25, 0.5],
        'batch_normalization': True
    }
    
    import json
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Model prepared for deployment!")
    print("Files created:")
    print("- deep_image_classifier_complete.h5 (Full model)")
    print("- deep_classifier_mobile.tflite (Mobile optimized)")
    print("- model_architecture.txt (Architecture summary)")
    print("- training_config.json (Training configuration)")

prepare_for_deployment()
```

## Performance Monitoring System

```python
def create_monitoring_system():
    """
    Create a system to monitor model performance in production
    Like having quality control checks in a restaurant
    """
    def calculate_prediction_confidence_stats(predictions):
        """Calculate confidence statistics for a batch of predictions"""
        confidences = [np.max(pred) for pred in predictions]
        
        stats = {
            'mean_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'std_confidence': np.std(confidences),
            'low_confidence_count': sum(1 for c in confidences if c < 0.7)
        }
        
        return stats
    
    def detect_data_drift(new_images, reference_images, threshold=0.1):
        """Simple data drift detection based on pixel statistics"""
        new_stats = {
            'mean_pixel': np.mean(new_images),
            'std_pixel': np.std(new_images),
            'mean_brightness': np.mean(np.mean(new_images, axis=(1,2,3)))
        }
        
        ref_stats = {
            'mean_pixel': np.mean(reference_images),
            'std_pixel': np.std(reference_images),
            'mean_brightness': np.mean(np.mean(reference_images, axis=(1,2,3)))
        }
        
        drift_detected = any(
            abs(new_stats[key] - ref_stats[key]) > threshold
            for key in new_stats
        )
        
        return drift_detected, new_stats, ref_stats
    
    return calculate_prediction_confidence_stats, detect_data_drift

# Initialize monitoring functions
calc_confidence_stats, detect_drift = create_monitoring_system()

# Example usage
sample_predictions = model.predict(x_test[:100])
confidence_stats = calc_confidence_stats(sample_predictions)
print("Confidence Statistics:", confidence_stats)
```

## Complete Project Summary

```python
def project_summary():
    """
    Provide a comprehensive summary of the project
    Like presenting the complete menu with all specialties
    """
    print("=" * 60)
    print("DEEP NEURAL NETWORK IMAGE CLASSIFIER - PROJECT COMPLETE")
    print("=" * 60)
    
    print("\n🏗️  ARCHITECTURE HIGHLIGHTS:")
    print("• Deep Convolutional Neural Network with 3 CNN blocks")
    print("• Batch Normalization for training stability")
    print("• Dropout layers for regularization")
    print("• Advanced Adam optimizer with custom parameters")
    print("• Data augmentation for improved generalization")
    
    print("\n📊 PERFORMANCE METRICS:")
    test_loss, test_accuracy, test_top2 = model.evaluate(x_test, y_test, verbose=0)
    print(f"• Test Accuracy: {test_accuracy:.4f}")
    print(f"• Test Top-2 Accuracy: {test_top2:.4f}")
    print(f"• Total Parameters: {model.count_params():,}")
    
    print("\n🚀 DEPLOYMENT READY:")
    print("• Full model saved for production")
    print("• TensorFlow Lite version for mobile")
    print("• Monitoring system implemented")
    print("• Comprehensive evaluation metrics")
    
    print("\n🎯 KEY FEATURES IMPLEMENTED:")
    print("• Real-time image classification")
    print("• Confidence score calculation")
    print("• Batch prediction capabilities")
    print("• Performance monitoring tools")
    print("• Data drift detection")
    
    print("\n" + "=" * 60)
    print("Your deep learning classifier is ready for production!")
    print("=" * 60)

project_summary()
```

This deep neural network image classifier demonstrates advanced deep learning concepts including proper architecture design, regularization techniques, advanced optimization, and production readiness. The model uses multiple layers of convolution, batch normalization, and dropout to create a robust classifier capable of handling real-world image classification tasks with high accuracy and confidence scoring.

## Assignment: Text Sentiment Analysis with Deep Networks

**Task:** Build a deep neural network to classify movie reviews as positive or negative using the concepts you've learned today.

**Requirements:**

1. **Data Preparation**: Use a small subset of movie review data (you can create synthetic data or use sklearn's text features)

2. **Network Architecture**: 
   - Create a deep network with at least 4 hidden layers
   - Use batch normalization after each hidden layer
   - Apply dropout with rate 0.3
   - Include proper activation functions

3. **Training Process**:
   - Compare Adam and RMSprop optimizers
   - Implement early stopping
   - Use gradient clipping
   - Track training metrics

4. **Analysis**:
   - Plot gradient flow analysis
   - Compare performance with and without batch normalization
   - Analyze the effect of different dropout rates (0.1, 0.3, 0.5)

**Deliverables:**
- Complete Python script with your implementation
- Performance comparison plots
- Brief analysis (2-3 paragraphs) explaining your findings

**Bonus Challenge**: Experiment with different network depths (3, 5, 7 layers) and report how depth affects training stability and final performance.

---

## Summary

Today, you've mastered the art of creating sophisticated deep neural networks, just like a chef who has learned to coordinate multiple cooking stations to create extraordinary meals. You've learned how to:

- Build deep network architectures with multiple layers
- Identify and solve the vanishing gradient problem
- Use batch normalization to standardize your "ingredients"
- Apply dropout to create robust, generalizable models
- Leverage advanced optimizers for better training

These techniques form the foundation for building state-of-the-art deep learning models that can tackle complex real-world problems. Like a master chef who combines technique with creativity, you now have the tools to build neural networks that are both deep and reliable.