# AI Mastery Course: Day 83 - Natural Language Processing with Deep Learning

## Learning Objective
By the end of this lesson, you will master the fundamental techniques of Natural Language Processing with Deep Learning, understanding how to transform raw text into meaningful numerical representations, preprocess text data effectively, and build intelligent systems that can classify and understand human language using Recurrent Neural Networks and attention mechanisms.

---

Imagine that you're running the most sophisticated recipe interpretation service in the world. Every day, thousands of handwritten recipes, food reviews, and culinary descriptions pour into your establishment from different cultures, languages, and writing styles. Your master chefs need to quickly understand not just what ingredients are mentioned, but the sentiment, cooking style, dietary restrictions, and cuisine type from each piece of text.

Just as a master chef can instantly recognize the essence of a dish from its aroma and appearance, we're going to train our AI systems to instantly understand the essence of text through sophisticated pattern recognition and memory systems.

---

## Lesson 1: Word Embeddings - The Ingredient Dictionary

In our culinary world, every ingredient has relationships with others. Tomatoes are closely related to basil, garlic pairs with onions, and certain spices belong to specific cuisine families. Word embeddings work the same way - they create a mathematical space where related words cluster together.

### Word2Vec: Learning Relationships Through Context

```python
import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Sample cooking-related sentences for training
cooking_sentences = [
    ["fresh", "tomatoes", "basil", "italian", "pasta"],
    ["spicy", "curry", "indian", "rice", "turmeric"],
    ["sweet", "dessert", "chocolate", "vanilla", "cake"],
    ["grilled", "chicken", "herbs", "protein", "healthy"],
    ["sour", "lemon", "citrus", "fish", "mediterranean"],
    ["umami", "mushrooms", "soy", "sauce", "japanese"],
    ["crispy", "fried", "golden", "texture", "delicious"]
]

# Train Word2Vec model (like teaching a chef ingredient relationships)
model = Word2Vec(sentences=cooking_sentences, vector_size=100, window=5, 
                min_count=1, workers=4, epochs=100)

# Find similar ingredients to 'tomatoes'
similar_words = model.wv.most_similar('tomatoes', topn=3)
print("Words similar to 'tomatoes':", similar_words)

# Get vector representation
tomato_vector = model.wv['tomatoes']
print(f"Tomato vector shape: {tomato_vector.shape}")
print(f"First 10 dimensions: {tomato_vector[:10]}")
```

**Code Syntax Explanation:**
- `Word2Vec()`: Creates a neural network model that learns word relationships
- `vector_size=100`: Each word becomes a 100-dimensional numerical vector
- `window=5`: The model looks at 5 words on each side for context
- `min_count=1`: Include words that appear at least once
- `model.wv.most_similar()`: Finds words with similar vector representations

### GloVe: Global Ingredient Knowledge

```python
# Simulating GloVe-style global statistics
import pandas as pd
from collections import defaultdict, Counter

def create_cooccurrence_matrix(sentences, window_size=2):
    """
    Like counting how often ingredients appear together across all recipes
    """
    vocab = set(word for sentence in sentences for word in sentence)
    vocab = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(vocab), len(vocab)))
    
    for sentence in sentences:
        for i, word in enumerate(sentence):
            # Look at surrounding words in the window
            for j in range(max(0, i - window_size), 
                          min(len(sentence), i + window_size + 1)):
                if i != j:
                    matrix[vocab[word]][vocab[sentence[j]]] += 1
    
    return matrix, vocab

# Create co-occurrence matrix
cooc_matrix, vocabulary = create_cooccurrence_matrix(cooking_sentences)
print("Co-occurrence matrix shape:", cooc_matrix.shape)
print("Vocabulary size:", len(vocabulary))

# Show how often 'tomatoes' appears with other words
if 'tomatoes' in vocabulary:
    tomato_idx = vocabulary['tomatoes']
    word_counts = [(word, cooc_matrix[tomato_idx][idx]) 
                   for word, idx in vocabulary.items() 
                   if cooc_matrix[tomato_idx][idx] > 0]
    print("Tomatoes co-occurs with:", word_counts)
```

**Code Syntax Explanation:**
- `defaultdict()`: Creates a dictionary with default values
- `enumerate()`: Provides both index and value when iterating
- `max()` and `min()`: Ensure we don't go outside sentence boundaries
- Matrix indexing `[row][column]`: Accesses specific positions in our co-occurrence table

---

## Lesson 2: Text Preprocessing - Preparing Ingredients for Cooking

Just as a chef must wash, chop, and prepare ingredients before cooking, we must clean and prepare text data before feeding it to our models.

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    """
    Like a sous chef that prepares all ingredients consistently
    """
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Remove impurities from raw text"""
        # Convert to lowercase (standardize everything)
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """Break text into individual words (like chopping vegetables)"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove common words that don't add flavor"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_words(self, tokens):
        """Reduce words to their root form (like peeling vegetables)"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_words(self, tokens):
        """Convert words to their base form (more sophisticated than stemming)"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text, use_stemming=True):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        
        if use_stemming:
            tokens = self.stem_words(tokens)
        else:
            tokens = self.lemmatize_words(tokens)
        
        return tokens

# Example usage
preprocessor = TextPreprocessor()

raw_reviews = [
    "This restaurant serves absolutely delicious Italian pasta with fresh tomatoes!",
    "The spicy curry was amazing, but the service was quite slow.",
    "I love the crispy fried chicken here. It's always perfectly seasoned."
]

processed_reviews = []
for review in raw_reviews:
    processed = preprocessor.preprocess(review)
    processed_reviews.append(processed)
    print(f"Original: {review}")
    print(f"Processed: {processed}\n")
```

**Code Syntax Explanation:**
- `re.sub(pattern, replacement, text)`: Uses regular expressions to find and replace patterns
- `r'[^a-zA-Z\s]'`: Raw string matching anything that's NOT letters or whitespace
- `class TextPreprocessor:`: Defines a blueprint for creating preprocessing objects
- `self`: Refers to the instance of the class
- List comprehension `[item for item in list if condition]`: Efficient filtering

---

## Lesson 3: RNNs for Text Classification - The Memory-Equipped Chef

Imagine a chef who remembers every ingredient added to a dish and how it affects the final flavor. RNNs work similarly - they process text word by word while maintaining memory of what came before.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# Sample restaurant review data with sentiment labels
reviews = [
    "The food was absolutely delicious and the service was perfect",  # Positive
    "Terrible experience, cold food and rude staff",  # Negative
    "Amazing flavors, fresh ingredients, highly recommended",  # Positive
    "Worst meal ever, completely overpriced and tasteless",  # Negative
    "Good food but slow service, mixed feelings",  # Neutral
    "Exceptional dining experience, will definitely return",  # Positive
    "Food was okay, nothing special but not bad either",  # Neutral
    "Outstanding chef, every dish was a masterpiece"  # Positive
]

# Sentiment labels: 0=Negative, 1=Neutral, 2=Positive
labels = [2, 0, 2, 0, 1, 2, 1, 2]

class RestaurantReviewClassifier:
    """
    A memory-equipped chef that learns to classify review sentiments
    """
    def __init__(self, max_words=1000, max_length=50):
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None
    
    def prepare_data(self, texts, labels):
        """Prepare ingredients (text) for the neural network chef"""
        # Convert text to sequences of numbers
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences to same length (like cutting vegetables to uniform size)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length,
                                       padding='post', truncating='post')
        
        return padded_sequences, np.array(labels)
    
    def build_model(self, num_classes):
        """Build the RNN architecture"""
        self.model = Sequential([
            # Embedding layer: converts word indices to dense vectors
            Embedding(input_dim=self.max_words, output_dim=64, 
                     input_length=self.max_length),
            
            # LSTM layer: the memory-equipped processor
            LSTM(64, dropout=0.2, recurrent_dropout=0.2, 
                return_sequences=False),
            
            # Dense layers for final classification
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
    
    def train(self, X, y, validation_split=0.2, epochs=10):
        """Train the chef to recognize sentiment patterns"""
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=2,
            verbose=1
        )
        return history
    
    def predict_sentiment(self, text):
        """Predict sentiment of new review"""
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_length,
                             padding='post', truncating='post')
        prediction = self.model.predict(padded)
        
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return sentiment_labels[predicted_class], confidence

# Train the model
classifier = RestaurantReviewClassifier()
X, y = classifier.prepare_data(reviews, labels)
classifier.build_model(num_classes=3)

# Train the model
history = classifier.train(X, y, epochs=20)

# Test predictions
test_reviews = [
    "The pasta was incredible, best meal of my life!",
    "Horrible service and cold food, never coming back",
    "Food was decent, service could be better"
]

for review in test_reviews:
    sentiment, confidence = classifier.predict_sentiment(review)
    print(f"Review: {review}")
    print(f"Predicted: {sentiment} (confidence: {confidence:.2f})\n")
```

**Code Syntax Explanation:**
- `Sequential([layers])`: Stacks neural network layers in order
- `Embedding()`: Converts sparse word indices to dense vectors
- `LSTM()`: Long Short-Term Memory layer that maintains context
- `dropout=0.2`: Randomly ignores 20% of connections to prevent overfitting
- `sparse_categorical_crossentropy`: Loss function for integer class labels
- `np.argmax()`: Finds the index of the highest probability

---

## Lesson 4: Attention Mechanisms - The Master Chef's Focus

Like a master chef who knows exactly which ingredients need attention at each moment during cooking, attention mechanisms help models focus on the most relevant parts of text.

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model
import numpy as np

class AttentionLayer(Layer):
    """
    Custom attention layer - like a chef's focused attention system
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create trainable weight matrices
        self.W = self.add_weight(name='attention_weight',
                               shape=(input_shape[-1], 1),
                               initializer='random_normal',
                               trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Calculate attention scores
        attention_scores = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1))
        attention_scores = tf.nn.softmax(attention_scores, axis=1)
        
        # Apply attention weights
        weighted_inputs = inputs * attention_scores
        
        # Sum weighted inputs
        output = tf.reduce_sum(weighted_inputs, axis=1)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class AttentiveReviewClassifier:
    """
    A chef with selective attention for different parts of reviews
    """
    def __init__(self, max_words=1000, max_length=50, embedding_dim=64):
        self.max_words = max_words
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None
    
    def build_attention_model(self, num_classes):
        """Build model with attention mechanism"""
        # Input layer
        inputs = tf.keras.Input(shape=(self.max_length,))
        
        # Embedding layer
        embedding = Embedding(self.max_words, self.embedding_dim, 
                            input_length=self.max_length)(inputs)
        
        # LSTM layer that returns all timesteps
        lstm_output = LSTM(64, return_sequences=True, 
                          dropout=0.2, recurrent_dropout=0.2)(embedding)
        
        # Attention layer - focuses on important words
        attention_output = AttentionLayer()(lstm_output)
        
        # Classification layers
        dense1 = Dense(32, activation='relu')(attention_output)
        dropout = Dropout(0.3)(dense1)
        outputs = Dense(num_classes, activation='softmax')(dropout)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        
        return self.model

# Demonstration of attention weights visualization
def visualize_attention(model, tokenizer, text, max_length):
    """Show which words the model pays attention to"""
    # Prepare text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, 
                          padding='post', truncating='post')
    
    # Get attention weights (simplified visualization)
    tokens = text.split()[:max_length]
    
    # Create a simple attention visualization
    print(f"Text: {text}")
    print("Word importance (simulated attention):")
    
    # This is a simplified version - real attention would require 
    # extracting intermediate layer outputs
    important_words = ['delicious', 'amazing', 'terrible', 'perfect', 
                      'horrible', 'excellent', 'awful', 'outstanding']
    
    for word in tokens:
        importance = "🔥" if word.lower() in important_words else "  "
        print(f"{importance} {word}")
    print()

# Example usage
extended_reviews = [
    "The appetizer was good but the main course was absolutely terrible",
    "Service was slow however the food quality was outstanding and delicious",
    "Mixed experience with excellent dessert but horrible main dishes",
    "Everything was perfect from start to finish, amazing chef",
    "Terrible service ruined what could have been a decent meal"
]

extended_labels = [0, 2, 1, 2, 0]  # Negative, Positive, Neutral, Positive, Negative

# Build and demonstrate attention model
attention_classifier = AttentiveReviewClassifier()
X_extended, y_extended = attention_classifier.tokenizer.fit_on_texts(extended_reviews), extended_labels
sequences = attention_classifier.tokenizer.texts_to_sequences(extended_reviews)
X_padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

attention_model = attention_classifier.build_attention_model(num_classes=3)
print(attention_model.summary())

# Visualize attention for sample texts
for review in extended_reviews[:3]:
    visualize_attention(attention_model, attention_classifier.tokenizer, 
                       review, max_length=50)
```

**Code Syntax Explanation:**
- `class AttentionLayer(Layer):`: Inherits from Keras Layer to create custom functionality
- `tf.tensordot(a, b, axes)`: Performs tensor multiplication along specified axes
- `tf.nn.softmax()`: Converts scores to probabilities that sum to 1
- `tf.reduce_sum(tensor, axis)`: Sums tensor elements along specified dimension
- `Model(inputs, outputs)`: Functional API for building complex model architectures
- `super().__init__()`: Calls parent class constructor

---

## Final Project: Cuisine Style Classifier

Now let's combine everything we've learned to create a sophisticated system that can identify cooking styles from recipe descriptions - like having a master chef who can instantly recognize the culinary tradition behind any dish.

```python
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class CuisineStyleClassifier:
    """
    Master system that combines all techniques to classify cuisine styles
    """
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.classifier = AttentiveReviewClassifier(max_words=2000, max_length=100)
        self.cuisine_labels = {
            0: 'Italian', 1: 'Asian', 2: 'Mexican', 
            3: 'Indian', 4: 'Mediterranean'
        }
    
    def prepare_training_data(self):
        """Prepare comprehensive training dataset"""
        recipe_descriptions = [
            # Italian
            "Fresh basil tomatoes mozzarella olive oil pasta parmesan garlic herbs",
            "Creamy risotto mushrooms white wine arborio rice italian seasoning",
            "Pizza margherita fresh tomatoes basil mozzarella wood fired oven",
            
            # Asian  
            "Soy sauce ginger garlic sesame oil rice noodles stir fry vegetables",
            "Miso soup tofu seaweed japanese dashi broth traditional flavors",
            "Sweet sour pork chinese cooking wine rice vinegar soy based sauce",
            
            # Mexican
            "Corn tortillas cilantro lime jalapeños cumin chili peppers salsa",
            "Black beans rice avocado chipotle peppers mexican spices traditional",
            "Tequila marinated chicken peppers onions mexican style grilled meat",
            
            # Indian
            "Turmeric cumin coriander garam masala basmati rice curry leaves",
            "Tandoor chicken yogurt spices indian bread naan traditional cooking",
            "Lentil curry coconut milk indian spices aromatic basmati rice",
            
            # Mediterranean
            "Olive oil lemon herbs mediterranean olives feta cheese greek style",
            "Grilled fish lemon herbs olive oil mediterranean vegetables fresh",
            "Hummus tahini chickpeas mediterranean diet olive oil middle eastern"
        ]
        
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        return recipe_descriptions, labels
    
    def train_complete_system(self):
        """Train the complete cuisine classification system"""
        descriptions, labels = self.prepare_training_data()
        
        # Preprocess all descriptions
        processed_descriptions = []
        for desc in descriptions:
            processed_tokens = self.preprocessor.preprocess(desc, use_stemming=False)
            processed_descriptions.append(' '.join(processed_tokens))
        
        # Prepare data for neural network
        X, y = self.classifier.tokenizer.fit_on_texts(processed_descriptions), labels
        sequences = self.classifier.tokenizer.texts_to_sequences(processed_descriptions)
        X_padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
        y_array = np.array(labels)
        
        # Build and train model
        model = self.classifier.build_attention_model(num_classes=5)
        history = model.fit(X_padded, y_array, epochs=50, batch_size=2, 
                           validation_split=0.2, verbose=1)
        
        return model, history
    
    def predict_cuisine(self, description):
        """Predict cuisine style from recipe description"""
        # Preprocess the input
        processed_tokens = self.preprocessor.preprocess(description, use_stemming=False)
        processed_text = ' '.join(processed_tokens)
        
        # Convert to sequence
        sequence = self.classifier.tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
        
        # Make prediction
        prediction = self.classifier.model.predict(padded)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return self.cuisine_labels[predicted_class], confidence
    
    def evaluate_system(self, test_descriptions, test_labels):
        """Comprehensive evaluation of the system"""
        predictions = []
        
        for desc in test_descriptions:
            cuisine, confidence = self.predict_cuisine(desc)
            # Convert cuisine name back to number for evaluation
            pred_num = [k for k, v in self.cuisine_labels.items() if v == cuisine][0]
            predictions.append(pred_num)
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(test_labels, predictions, 
                                  target_names=list(self.cuisine_labels.values())))
        
        # Create confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.cuisine_labels.values(),
                   yticklabels=self.cuisine_labels.values())
        plt.title('Cuisine Classification Confusion Matrix')
        plt.ylabel('True Cuisine')
        plt.xlabel('Predicted Cuisine')
        plt.show()

# Train and test the complete system
print("🍳 Training the Master Cuisine Classification System...")
cuisine_classifier = CuisineStyleClassifier()
model, training_history = cuisine_classifier.train_complete_system()

# Test with new recipe descriptions
test_recipes = [
    "Spaghetti carbonara with pancetta eggs parmesan cheese black pepper",
    "Pad thai noodles fish sauce tamarind peanuts asian street food style",
    "Chicken tikka masala creamy tomato curry indian spices basmati rice",
    "Fish tacos corn tortillas cabbage slaw lime cilantro mexican street food",
    "Greek salad olives feta cucumber tomatoes olive oil mediterranean herbs"
]

print("\n🎯 Testing Cuisine Predictions:")
print("=" * 60)

for recipe in test_recipes:
    cuisine, confidence = cuisine_classifier.predict_cuisine(recipe)
    print(f"Recipe: {recipe[:50]}...")
    print(f"Predicted Cuisine: {cuisine} (Confidence: {confidence:.2f})")
    print("-" * 60)
```

**Code Syntax Explanation:**
- `classification_report()`: Provides precision, recall, and F1-score metrics
- `confusion_matrix()`: Shows prediction accuracy across all classes  
- `sns.heatmap()`: Creates visual representation of confusion matrix
- `enumerate(zip(a, b))`: Simultaneously iterate over multiple lists with indices
- Dictionary comprehension `{k: v for k, v in items}`: Creates dictionaries efficiently

---

# Text Classification System Project

## Project Overview
Build a sophisticated text classification system that can analyze and categorize text documents using deep learning techniques. This system will demonstrate the power of combining traditional NLP preprocessing with modern neural network architectures.

## Core Components

### 1. Data Preparation Module

```python
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class TextPreprocessor:
    def __init__(self, max_vocab_size=10000, max_sequence_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        
    def clean_text(self, text):
        """Clean and normalize text data"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def prepare_data(self, texts, labels):
        """Prepare text data for training"""
        # Clean all texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(cleaned_texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        
        # Pad sequences to ensure uniform length
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_sequence_length, 
            padding='post', 
            truncating='post'
        )
        
        # Convert labels to categorical
        unique_labels = list(set(labels))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        indexed_labels = [label_to_index[label] for label in labels]
        categorical_labels = to_categorical(indexed_labels)
        
        return padded_sequences, categorical_labels, label_to_index

# Sample usage
preprocessor = TextPreprocessor(max_vocab_size=10000, max_sequence_length=150)
```

### 2. Neural Network Architecture

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, 
    Bidirectional, GlobalMaxPooling1D, 
    Input, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class TextClassificationModel:
    def __init__(self, vocab_size, embedding_dim=128, max_length=100, num_classes=2):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        self.model = None
        
    def build_lstm_model(self):
        """Build LSTM-based classification model"""
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length
            ),
            Bidirectional(LSTM(64, return_sequences=True)),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_attention_model(self):
        """Build model with attention mechanism"""
        inputs = Input(shape=(self.max_length,))
        
        # Embedding layer
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim
        )(inputs)
        
        # LSTM layer with return sequences for attention
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(embedding)
        
        # Multi-head attention layer
        attention_out = MultiHeadAttention(
            num_heads=4,
            key_dim=64
        )(lstm_out, lstm_out)
        
        # Layer normalization
        normalized = LayerNormalization()(attention_out + lstm_out)
        
        # Global pooling and classification layers
        pooled = GlobalMaxPooling1D()(normalized)
        dense = Dense(64, activation='relu')(pooled)
        dropout = Dropout(0.5)(dense)
        outputs = Dense(self.num_classes, activation='softmax')(dropout)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the classification model"""
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
```

### 3. Evaluation and Prediction System

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class ModelEvaluator:
    def __init__(self, model, label_to_index):
        self.model = model
        self.label_to_index = label_to_index
        self.index_to_label = {v: k for k, v in label_to_index.items()}
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        # Make predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        class_names = [self.index_to_label[i] for i in range(len(self.index_to_label))]
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        return report, cm, predicted_classes
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def predict_new_text(self, preprocessor, new_texts):
        """Predict categories for new text samples"""
        # Clean and preprocess new texts
        cleaned_texts = [preprocessor.clean_text(text) for text in new_texts]
        sequences = preprocessor.tokenizer.texts_to_sequences(cleaned_texts)
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=preprocessor.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        # Make predictions
        predictions = self.model.predict(padded_sequences)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        # Format results
        results = []
        for i, text in enumerate(new_texts):
            predicted_label = self.index_to_label[predicted_classes[i]]
            confidence = confidence_scores[i]
            results.append({
                'text': text,
                'predicted_category': predicted_label,
                'confidence': confidence
            })
        
        return results
```

### 4. Complete Implementation Example

```python
# Main execution script
def run_text_classification_project():
    """Complete text classification pipeline"""
    
    # Sample dataset creation (replace with your actual data)
    # This example uses movie reviews for sentiment analysis
    texts = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "Terrible film, waste of time and money.",
        "Average movie, nothing special but entertaining.",
        "Outstanding cinematography and brilliant performances.",
        "Boring plot, poor character development.",
        # Add more sample data here...
    ]
    
    labels = ['positive', 'negative', 'neutral', 'positive', 'negative']
    
    # Step 1: Data preprocessing
    print("Step 1: Preprocessing data...")
    preprocessor = TextPreprocessor(max_vocab_size=5000, max_sequence_length=100)
    X, y, label_mapping = preprocessor.prepare_data(texts, labels)
    
    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    # Step 3: Model building and training
    print("Step 2: Building and training model...")
    classifier = TextClassificationModel(
        vocab_size=5000,
        embedding_dim=100,
        max_length=100,
        num_classes=len(label_mapping)
    )
    
    # Build attention-based model
    model = classifier.build_attention_model()
    print(model.summary())
    
    # Train the model
    history = classifier.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=16
    )
    
    # Step 4: Model evaluation
    print("Step 3: Evaluating model...")
    evaluator = ModelEvaluator(model, label_mapping)
    report, cm, predictions = evaluator.evaluate_model(X_test, y_test)
    
    # Display results
    print("\nClassification Report:")
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{class_name}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Plot confusion matrix
    class_names = list(label_mapping.keys())
    evaluator.plot_confusion_matrix(cm, class_names)
    
    # Step 5: Test with new samples
    print("Step 4: Testing with new samples...")
    new_samples = [
        "This product exceeded my expectations! Highly recommended.",
        "Poor quality, would not buy again.",
        "Decent value for money, nothing extraordinary."
    ]
    
    results = evaluator.predict_new_text(preprocessor, new_samples)
    
    print("\nPredictions for new samples:")
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Category: {result['predicted_category']}")
        print(f"Confidence: {result['confidence']:.3f}\n")

# Run the complete project
if __name__ == "__main__":
    run_text_classification_project()
```

### 5. Django Integration Module

```python
# views.py for Django integration
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import joblib
import numpy as np

# Load trained model and preprocessor (saved during training)
model = None
preprocessor = None
evaluator = None

def load_trained_components():
    """Load pre-trained model components"""
    global model, preprocessor, evaluator
    try:
        model = tf.keras.models.load_model('best_model.h5')
        preprocessor = joblib.load('text_preprocessor.pkl')
        evaluator = joblib.load('model_evaluator.pkl')
        return True
    except:
        return False

@csrf_exempt
def classify_text_api(request):
    """API endpoint for text classification"""
    if request.method == 'POST':
        try:
            # Load components if not already loaded
            if not all([model, preprocessor, evaluator]):
                if not load_trained_components():
                    return JsonResponse({'error': 'Model not available'}, status=500)
            
            # Parse request data
            data = json.loads(request.body)
            text_input = data.get('text', '')
            
            if not text_input:
                return JsonResponse({'error': 'No text provided'}, status=400)
            
            # Make prediction
            results = evaluator.predict_new_text(preprocessor, [text_input])
            
            return JsonResponse({
                'success': True,
                'prediction': results[0]
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def classification_dashboard(request):
    """Dashboard for text classification interface"""
    return render(request, 'classification_dashboard.html')
```

### 6. Performance Optimization

```python
class OptimizedTextClassifier:
    """Optimized version with caching and batch processing"""
    
    def __init__(self, model_path, preprocessor_path):
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.prediction_cache = {}
        
    def batch_predict(self, texts, batch_size=32):
        """Process texts in batches for better performance"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Check cache first
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch_texts):
                text_hash = hash(text)
                if text_hash in self.prediction_cache:
                    cached_results.append((j, self.prediction_cache[text_hash]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Process uncached texts
            if uncached_texts:
                batch_predictions = self._predict_texts(uncached_texts)
                
                # Cache results
                for text, prediction in zip(uncached_texts, batch_predictions):
                    self.prediction_cache[hash(text)] = prediction
            
            # Combine cached and new results
            batch_results = [None] * len(batch_texts)
            for idx, result in cached_results:
                batch_results[idx] = result
            for idx, result in zip(uncached_indices, batch_predictions):
                batch_results[idx] = result
            
            results.extend(batch_results)
        
        return results
    
    def _predict_texts(self, texts):
        """Internal method for making predictions"""
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
        sequences = self.preprocessor.tokenizer.texts_to_sequences(cleaned_texts)
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.preprocessor.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        predictions = self.model.predict(padded_sequences)
        return predictions

# Usage example
optimized_classifier = OptimizedTextClassifier('best_model.h5', 'preprocessor.pkl')
```

## Project Features

1. **Advanced Text Preprocessing**: Comprehensive text cleaning and normalization with tokenization
2. **Multiple Model Architectures**: Both LSTM and attention-based models for comparison
3. **Robust Evaluation System**: Detailed metrics, confusion matrices, and performance visualization
4. **Real-time Prediction**: API endpoints for live text classification
5. **Performance Optimization**: Caching and batch processing for production use
6. **Django Integration**: Ready-to-deploy web interface

## Key Technical Concepts Demonstrated

- **Word Embeddings**: Dense vector representations of words learned during training
- **Bidirectional LSTM**: Processing sequences in both forward and backward directions
- **Attention Mechanisms**: Allowing the model to focus on relevant parts of the input
- **Sequence Padding**: Ensuring uniform input sizes for neural networks
- **Categorical Encoding**: Converting text labels to numerical format for training
- **Model Callbacks**: Early stopping and model checkpointing for optimal training

This comprehensive text classification system showcases the integration of modern NLP techniques with practical deployment considerations, providing a production-ready solution for text analysis tasks.

## Assignment: Emotion Detection in Food Reviews

**Task**: Build a system that can detect specific emotions (joy, anger, sadness, surprise, disgust) from restaurant reviews, going beyond simple positive/negative sentiment.

**Requirements**:
1. Create a dataset of 25 restaurant reviews with emotional labels
2. Implement a multi-class emotion classifier using LSTM with attention
3. Include proper text preprocessing pipeline
4. Evaluate your model with precision, recall, and F1-scores
5. Test with 5 new reviews and explain why certain emotions were detected

**Deliverables**:
- Complete Python code with comments explaining your approach
- Analysis of which words/phrases trigger different emotions
- Discussion of at least 3 challenges you encountered and how you solved them

**Bonus**: Implement a visualization that shows attention weights for emotional words in reviews.

---

## Key Takeaways

Through this comprehensive journey into Natural Language Processing with Deep Learning, you've mastered the art of transforming human language into mathematical understanding. Like a master chef who can instantly recognize flavors, techniques, and cultural influences in any dish, your AI systems can now:

- Transform words into meaningful numerical representations through embeddings
- Clean and prepare text data with sophisticated preprocessing techniques  
- Build memory-equipped neural networks that understand context and sequence
- Implement attention mechanisms that focus on the most important information
- Create complete end-to-end systems for real-world text classification tasks

The combination of Word2Vec embeddings, careful preprocessing, LSTM networks, and attention mechanisms provides you with a powerful toolkit for understanding and classifying human language at scale. These techniques form the foundation for more advanced NLP applications like machine translation, question answering, and conversational AI systems.