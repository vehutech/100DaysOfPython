# AI Mastery Course: Scikit-learn Mastery (Day 77)

## Learning Objective
By the end of this lesson, you will master advanced scikit-learn techniques to create sophisticated machine learning solutions, just like a master chef who can combine simple ingredients into complex, restaurant-quality dishes using professional techniques and workflows.

---

## Introduction: The Master Chef's Kitchen

Imagine that you've been cooking simple meals for months, and now you're ready to step into a professional kitchen. You know the basic ingredients (data preprocessing, simple models), but now you need to learn the advanced techniques that separate home cooks from master chefs. 

In scikit-learn, we're moving beyond basic recipes to create sophisticated, automated cooking processes - complete with quality control, recipe optimization, and custom techniques that make your machine learning "dishes" restaurant-quality.

Just as a master chef has specialized tools, perfected workflows, and secret techniques, today we'll learn the advanced scikit-learn methods that transform you from a beginner data cook into a machine learning master chef.

---

## Lesson 1: Advanced Scikit-learn Techniques
*The Master Chef's Advanced Techniques*

### Learning Focus
Master advanced preprocessing, feature engineering, and model selection techniques that professional ML practitioners use daily.

Think of this as learning knife skills, sauce-making, and plating techniques that elevate your cooking from basic to professional level.

### Key Concepts & Code Examples

#### 1. Feature Engineering Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Create sample restaurant review data (our ingredients)
data = {
    'review_text': [
        'Amazing food and great service', 'Terrible experience, cold food',
        'Best restaurant in town', 'Overpriced and disappointing',
        'Excellent chef, perfect dishes', 'Poor quality ingredients'
    ],
    'price_range': [3, 1, 4, 4, 5, 2],
    'service_rating': [5, 1, 5, 2, 5, 2],
    'location_score': [8.5, 3.2, 9.1, 6.8, 9.5, 4.1],
    'satisfied': [1, 0, 1, 0, 1, 0]  # Our target - customer satisfaction
}

df = pd.DataFrame(data)
print("Our Restaurant Data (Raw Ingredients):")
print(df.head())
```

**Syntax Explanation:**
- `BaseEstimator, TransformerMixin`: Base classes for creating custom transformers
- `TfidfVectorizer`: Converts text to numerical features (like turning raw herbs into seasoning)
- `warnings.filterwarnings('ignore')`: Silences warning messages for cleaner output

#### 2. Advanced Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression

# Advanced feature selection - like a chef selecting the best ingredients
class ChefFeatureSelector(BaseEstimator, TransformerMixin):
    """Custom feature selector - like a chef's ingredient quality checker"""
    
    def __init__(self, method='kbest', k=5):
        self.method = method
        self.k = k
        self.selector = None
    
    def fit(self, X, y):
        """Learn which features are the 'best ingredients'"""
        if self.method == 'kbest':
            self.selector = SelectKBest(f_classif, k=self.k)
        elif self.method == 'rfe':
            estimator = LogisticRegression(random_state=42)
            self.selector = RFE(estimator, n_features_to_select=self.k)
        
        self.selector.fit(X, y)
        return self
    
    def transform(self, X):
        """Select only the best features - our premium ingredients"""
        return self.selector.transform(X)

# Example usage
X_sample = np.random.rand(100, 10)  # 10 features
y_sample = np.random.randint(0, 2, 100)

chef_selector = ChefFeatureSelector(method='kbest', k=5)
X_selected = chef_selector.fit_transform(X_sample, y_sample)
print(f"Selected {X_selected.shape[1]} best features from {X_sample.shape[1]} total")
```

**Syntax Explanation:**
- `class ChefFeatureSelector(BaseEstimator, TransformerMixin)`: Creates a custom transformer class
- `__init__`, `fit`, `transform`: Required methods for scikit-learn transformers
- `RFE`: Recursive Feature Elimination - removes features recursively
- `f_classif`: Statistical test for feature selection

---

## Lesson 2: Pipeline Creation and Automation
*The Professional Kitchen Workflow*

### Learning Focus
Create automated, reproducible workflows that handle the entire cooking process from raw ingredients to finished dish.

### Key Concepts & Code Examples

#### 1. Basic Pipeline Creation
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Create a professional cooking pipeline - from prep to plate
chef_pipeline = Pipeline([
    ('prep_station', StandardScaler()),        # Ingredient preparation
    ('flavor_concentration', PCA(n_components=5)),  # Concentrate flavors
    ('master_chef', SVC(kernel='rbf'))         # The master chef's technique
])

# Train our pipeline (teach the kitchen workflow)
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)

chef_pipeline.fit(X_train, y_train)
accuracy = chef_pipeline.score(X_test, y_test)
print(f"Kitchen Pipeline Accuracy: {accuracy:.3f}")
```

#### 2. Advanced Pipeline with Custom Transformers
```python
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

# Custom transformer - like a chef's special sauce recipe
def create_interaction_features(X):
    """Create feature interactions - like combining flavors"""
    if X.shape[1] >= 2:
        # Create interaction between first two features
        interaction = X[:, 0:1] * X[:, 1:2]
        return np.column_stack([X, interaction])
    return X

def log_transform_features(X):
    """Apply log transformation - like reducing a sauce"""
    return np.log1p(np.abs(X))

# Professional kitchen with multiple prep stations
advanced_pipeline = Pipeline([
    ('ingredient_prep', FeatureUnion([
        ('standard_prep', StandardScaler()),
        ('special_sauce', FunctionTransformer(create_interaction_features)),
        ('reduction', FunctionTransformer(log_transform_features))
    ])),
    ('quality_control', ChefFeatureSelector(method='kbest', k=8)),
    ('master_technique', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Advanced Kitchen Pipeline Created!")
print("Stations:", [step[0] for step in advanced_pipeline.steps])
```

**Syntax Explanation:**
- `Pipeline([('name', transformer), ...])`: Creates a sequential workflow
- `FeatureUnion`: Combines multiple transformers in parallel (like multiple prep stations)
- `FunctionTransformer`: Converts functions into scikit-learn transformers
- `np.column_stack`: Stacks arrays horizontally (adds new columns)

---

## Lesson 3: Hyperparameter Tuning with GridSearch
*Perfecting the Recipe*

### Learning Focus
Systematically optimize your models like a chef perfecting a signature dish through countless iterations.

### Key Concepts & Code Examples

#### 1. Basic GridSearch
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Define our recipe variations to test
recipe_variations = {
    'master_technique__n_estimators': [50, 100, 200],      # Cooking time variations
    'master_technique__max_depth': [3, 5, 10, None],       # Complexity levels
    'master_technique__min_samples_split': [2, 5, 10],     # Precision levels
    'quality_control__k': [3, 5, 8]                        # Ingredient selection
}

# The master chef's testing kitchen
recipe_optimizer = GridSearchCV(
    advanced_pipeline,
    recipe_variations,
    cv=5,                    # 5-fold taste testing
    scoring='accuracy',      # Quality metric
    n_jobs=-1,              # Use all kitchen staff
    verbose=1               # Show progress
)

print("Starting recipe optimization...")
recipe_optimizer.fit(X_train, y_train)

print(f"\nBest Recipe Score: {recipe_optimizer.best_score_:.3f}")
print("Best Recipe Parameters:")
for param, value in recipe_optimizer.best_params_.items():
    print(f"  {param}: {value}")
```

#### 2. Advanced Hyperparameter Tuning
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# More sophisticated recipe experimentation
advanced_recipe_space = {
    'master_technique__n_estimators': randint(50, 300),
    'master_technique__max_depth': [3, 5, 7, 10, None],
    'master_technique__min_samples_split': randint(2, 20),
    'master_technique__min_samples_leaf': randint(1, 10),
    'master_technique__max_features': ['sqrt', 'log2', None],
    'quality_control__k': randint(3, 15)
}

# Randomized search - like a creative chef experimenting
creative_optimizer = RandomizedSearchCV(
    advanced_pipeline,
    advanced_recipe_space,
    n_iter=50,              # Try 50 random combinations
    cv=3,                   # 3-fold validation
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

creative_optimizer.fit(X_train, y_train)
print(f"Creative Recipe Score: {creative_optimizer.best_score_:.3f}")
```

**Syntax Explanation:**
- `GridSearchCV`: Exhaustive search over parameter grid
- `RandomizedSearchCV`: Random sampling from parameter distributions
- `cv=5`: 5-fold cross-validation
- `n_jobs=-1`: Use all available CPU cores
- `randint(50, 300)`: Random integers between 50 and 300
- `verbose=1`: Print progress information

---

## Lesson 4: Custom Transformers and Estimators
*Creating Your Signature Techniques*

### Learning Focus
Develop custom preprocessing techniques and models tailored to your specific needs, like a chef creating signature dishes.

### Key Concepts & Code Examples

#### 1. Custom Transformer for Domain-Specific Processing
```python
class RestaurantReviewProcessor(BaseEstimator, TransformerMixin):
    """Custom transformer for restaurant review data - our signature prep technique"""
    
    def __init__(self, sentiment_weights=True, length_features=True):
        self.sentiment_weights = sentiment_weights
        self.length_features = length_features
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.fitted = False
    
    def fit(self, X, y=None):
        """Learn the vocabulary - like memorizing ingredient combinations"""
        # Assume X is a DataFrame with 'review_text' column
        if hasattr(X, 'iloc'):
            texts = X.iloc[:, 0].astype(str)  # First column assumed to be text
        else:
            texts = X[:, 0] if X.ndim > 1 else X
        
        self.vectorizer.fit(texts)
        self.fitted = True
        return self
    
    def transform(self, X):
        """Process reviews - our signature preparation method"""
        if not self.fitted:
            raise ValueError("Must fit transformer before transforming")
        
        # Extract text (first column)
        if hasattr(X, 'iloc'):
            texts = X.iloc[:, 0].astype(str)
        else:
            texts = X[:, 0] if X.ndim > 1 else X
        
        # Convert text to features
        text_features = self.vectorizer.transform(texts).toarray()
        
        features = [text_features]
        
        # Add length features if requested
        if self.length_features:
            lengths = np.array([[len(text)] for text in texts])
            features.append(lengths)
        
        # Add sentiment proxy if requested
        if self.sentiment_weights:
            # Simple sentiment proxy based on positive/negative words
            positive_words = ['good', 'great', 'excellent', 'amazing', 'best', 'perfect']
            negative_words = ['bad', 'terrible', 'poor', 'worst', 'awful', 'disappointing']
            
            sentiment_scores = []
            for text in texts:
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                sentiment_scores.append([pos_count - neg_count])
            
            features.append(np.array(sentiment_scores))
        
        return np.column_stack(features)

# Test our custom processor
sample_reviews = pd.DataFrame({
    'review_text': [
        'This restaurant has amazing food and excellent service',
        'Terrible experience with poor quality dishes',
        'Good food but service could be better'
    ]
})

review_processor = RestaurantReviewProcessor()
processed_reviews = review_processor.fit_transform(sample_reviews)
print(f"Processed reviews shape: {processed_reviews.shape}")
print("Features include: text features, review length, sentiment score")
```

#### 2. Custom Estimator
```python
class ChefRecommendationSystem(BaseEstimator):
    """Custom estimator - like a master chef's decision-making process"""
    
    def __init__(self, confidence_threshold=0.7, ensemble_size=3):
        self.confidence_threshold = confidence_threshold
        self.ensemble_size = ensemble_size
        self.models = []
        self.feature_importance_ = None
    
    def fit(self, X, y):
        """Train our chef's decision-making process"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # Create ensemble of different "cooking styles"
        models = [
            ('traditional', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('modern', LogisticRegression(random_state=42, max_iter=1000)),
            ('fusion', SVC(probability=True, random_state=42))
        ]
        
        self.models = []
        for name, model in models[:self.ensemble_size]:
            model.fit(X, y)
            self.models.append((name, model))
        
        # Calculate average feature importance where possible
        importances = []
        for name, model in self.models:
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
        
        if importances:
            self.feature_importance_ = np.mean(importances, axis=0)
        
        return self
    
    def predict(self, X):
        """Make predictions like a master chef's final decision"""
        predictions = []
        
        for name, model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Ensemble voting - like getting consensus from multiple chefs
        ensemble_pred = np.array(predictions)
        final_predictions = []
        
        for i in range(X.shape[0]):
            votes = ensemble_pred[:, i]
            # Take majority vote
            final_pred = 1 if np.sum(votes) > len(votes) / 2 else 0
            final_predictions.append(final_pred)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """Return prediction confidence - like a chef's certainty level"""
        probabilities = []
        
        for name, model in self.models:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)
                probabilities.append(prob)
        
        if probabilities:
            # Average probabilities across models
            avg_prob = np.mean(probabilities, axis=0)
            return avg_prob
        else:
            # Fallback to binary predictions
            predictions = self.predict(X)
            prob_array = np.zeros((len(predictions), 2))
            prob_array[predictions == 0, 0] = 1.0
            prob_array[predictions == 1, 1] = 1.0
            return prob_array

# Test our custom estimator
chef_system = ChefRecommendationSystem(ensemble_size=3)
chef_system.fit(X_train, y_train)
chef_predictions = chef_system.predict(X_test)
chef_probabilities = chef_system.predict_proba(X_test)

print(f"Chef System Accuracy: {np.mean(chef_predictions == y_test):.3f}")
print(f"Average Confidence: {np.mean(np.max(chef_probabilities, axis=1)):.3f}")
```

**Syntax Explanation:**
- `BaseEstimator`: Base class for all estimators in scikit-learn
- `TransformerMixin`: Adds `fit_transform` method automatically
- `hasattr(object, 'attribute')`: Checks if object has specified attribute
- `np.column_stack`: Horizontally stacks arrays (combines features)
- `isinstance(X, pd.DataFrame)`: Checks if X is a pandas DataFrame
- `max_iter=1000`: Maximum iterations for convergence

---

## Final Project: Restaurant Success Predictor
*Your Master Chef Certification Dish*

Create a comprehensive machine learning system that predicts restaurant success using all the techniques you've learned. This is your signature dish that demonstrates mastery of advanced scikit-learn techniques.

### Project Requirements:

```python
# Final Project: Restaurant Success Predictor
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create comprehensive restaurant dataset
np.random.seed(42)
n_restaurants = 1000

restaurant_data = {
    'cuisine_type': np.random.choice(['Italian', 'Asian', 'Mexican', 'American', 'French'], n_restaurants),
    'location_score': np.random.normal(6.5, 2.0, n_restaurants),
    'price_range': np.random.randint(1, 6, n_restaurants),
    'chef_experience': np.random.exponential(5, n_restaurants),
    'marketing_budget': np.random.lognormal(8, 1, n_restaurants),
    'seating_capacity': np.random.randint(20, 200, n_restaurants),
    'online_reviews': np.random.normal(3.8, 1.2, n_restaurants),
    'staff_count': np.random.randint(5, 50, n_restaurants)
}

# Create realistic success labels based on features
success_probability = (
    0.3 * (restaurant_data['location_score'] / 10) +
    0.2 * (restaurant_data['chef_experience'] / 20) +
    0.2 * (restaurant_data['online_reviews'] / 5) +
    0.1 * (restaurant_data['price_range'] / 5) +
    0.1 * (restaurant_data['marketing_budget'] / 20000) +
    0.1 * np.random.random(n_restaurants)
)

restaurant_data['successful'] = (success_probability > 0.6).astype(int)

df_restaurants = pd.DataFrame(restaurant_data)

# Custom Restaurant Feature Engineer
class RestaurantFeatureEngineer(BaseEstimator, TransformerMixin):
    """Master Chef's Feature Engineering - creates the perfect ingredient mix"""
    
    def __init__(self):
        self.label_encoders = {}
        self.fitted = False
    
    def fit(self, X, y=None):
        # Fit label encoders for categorical variables
        if 'cuisine_type' in X.columns:
            self.label_encoders['cuisine_type'] = LabelEncoder()
            self.label_encoders['cuisine_type'].fit(X['cuisine_type'])
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Must fit before transform")
        
        X_transformed = X.copy()
        
        # Encode categorical variables
        if 'cuisine_type' in X_transformed.columns:
            X_transformed['cuisine_type'] = self.label_encoders['cuisine_type'].transform(X_transformed['cuisine_type'])
        
        # Create interaction features (chef's secret combinations)
        X_transformed['efficiency_ratio'] = X_transformed['seating_capacity'] / X_transformed['staff_count']
        X_transformed['marketing_per_seat'] = X_transformed['marketing_budget'] / X_transformed['seating_capacity']
        X_transformed['experience_quality'] = X_transformed['chef_experience'] * X_transformed['online_reviews']
        
        # Create categorical bins
        X_transformed['price_category'] = pd.cut(X_transformed['price_range'], 
                                               bins=[0, 2, 4, 5], 
                                               labels=[0, 1, 2]).astype(int)
        
        return X_transformed.values

# Build the Master Pipeline
master_restaurant_pipeline = Pipeline([
    ('feature_engineering', RestaurantFeatureEngineer()),
    ('preprocessing', StandardScaler()),
    ('feature_selection', ChefFeatureSelector(method='kbest', k=10)),
    ('classifier', ChefRecommendationSystem(ensemble_size=3))
])

# Prepare data
X = df_restaurants.drop('successful', axis=1)
y = df_restaurants['successful']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the master pipeline
print("Training the Master Restaurant Success Predictor...")
master_restaurant_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = master_restaurant_pipeline.predict(X_test)
y_pred_proba = master_restaurant_pipeline.predict_proba(X_test)

# Evaluate results
accuracy = np.mean(y_pred == y_test)
print(f"\nMaster Pipeline Accuracy: {accuracy:.3f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Unsuccessful', 'Successful']))

# Feature importance analysis (if available)
if hasattr(master_restaurant_pipeline.named_steps['classifier'], 'feature_importance_'):
    feature_names = ['cuisine_type', 'location_score', 'price_range', 'chef_experience',
                    'marketing_budget', 'seating_capacity', 'online_reviews', 'staff_count',
                    'efficiency_ratio', 'marketing_per_seat', 'experience_quality', 'price_category']
    
    importance = master_restaurant_pipeline.named_steps['classifier'].feature_importance_
    
    # Get indices of selected features
    selected_features = master_restaurant_pipeline.named_steps['feature_selection'].selector.get_support()
    selected_feature_names = [name for i, name in enumerate(feature_names) if selected_features[i]]
    
    print(f"\nTop Feature Importances (Selected {len(selected_feature_names)} features):")
    for name, imp in zip(selected_feature_names, importance):
        print(f"  {name}: {imp:.3f}")

print("\n🏆 Master Chef Certification Complete! 🏆")
print("You've successfully created an advanced ML pipeline with:")
print("✓ Custom transformers and feature engineering")
print("✓ Advanced preprocessing and feature selection") 
print("✓ Custom ensemble estimator")
print("✓ Professional pipeline architecture")
```

---

# End-to-End ML Pipeline Project: Restaurant Review Sentiment Analyzer

## Project Overview
Build a complete machine learning pipeline that processes restaurant reviews, cleans the data, extracts features, trains multiple models, and serves predictions through a web interface.

## Project Structure
```
restaurant_sentiment_pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   └── model_training.py
│   ├── models/
│   └── api/
├── notebooks/
├── tests/
├── requirements.txt
└── main.py
```

## Complete Implementation

### 1. Data Preprocessing Pipeline (`src/pipeline/data_preprocessing.py`)

```python
import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('stopwords')

class TextCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer for cleaning text data"""
    
    def __init__(self, remove_stopwords=True, stem_words=True):
        self.remove_stopwords = remove_stopwords
        self.stem_words = stem_words
        self.stemmer = PorterStemmer() if stem_words else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Clean and preprocess text data"""
        cleaned_texts = []
        
        for text in X:
            # Convert to lowercase
            text = str(text).lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove stopwords and stem
            if self.remove_stopwords or self.stem_words:
                words = text.split()
                
                if self.remove_stopwords:
                    words = [word for word in words if word not in self.stop_words]
                
                if self.stem_words:
                    words = [self.stemmer.stem(word) for word in words]
                
                text = ' '.join(words)
            
            cleaned_texts.append(text)
        
        return cleaned_texts

class DataPreprocessor:
    """Main data preprocessing class"""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.label_encoder = LabelEncoder()
    
    def preprocess_data(self, df, text_column='review', target_column='sentiment'):
        """Preprocess the entire dataset"""
        # Create a copy
        df_clean = df.copy()
        
        # Clean text data
        df_clean[text_column] = self.text_cleaner.fit_transform(df_clean[text_column])
        
        # Encode labels if they're strings
        if df_clean[target_column].dtype == 'object':
            df_clean[target_column] = self.label_encoder.fit_transform(df_clean[target_column])
        
        # Remove empty reviews
        df_clean = df_clean[df_clean[text_column].str.len() > 0]
        
        return df_clean
```

### 2. Feature Engineering Pipeline (`src/pipeline/feature_engineering.py`)

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """Extract text length features"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Extract various length features
        features = np.array([
            [len(text), len(text.split()), len(set(text.split()))]
            for text in X
        ])
        return features

class SentimentFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract sentiment-related features"""
    
    def __init__(self):
        # Simple positive and negative word lists
        self.positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 
                              'fantastic', 'love', 'perfect', 'best', 'delicious'}
        self.negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 
                              'hate', 'disgusting', 'poor', 'disappointing', 'nasty'}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        for text in X:
            words = set(text.lower().split())
            
            pos_count = len(words.intersection(self.positive_words))
            neg_count = len(words.intersection(self.negative_words))
            
            # Calculate ratios
            total_words = len(text.split())
            pos_ratio = pos_count / total_words if total_words > 0 else 0
            neg_ratio = neg_count / total_words if total_words > 0 else 0
            
            features.append([pos_count, neg_count, pos_ratio, neg_ratio])
        
        return np.array(features)

def create_feature_pipeline():
    """Create the complete feature engineering pipeline"""
    
    # Text vectorization
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Combine all features
    feature_pipeline = FeatureUnion([
        ('tfidf', tfidf),
        ('text_length', TextLengthExtractor()),
        ('sentiment_features', SentimentFeatureExtractor())
    ])
    
    return feature_pipeline
```

### 3. Model Training Pipeline (`src/pipeline/model_training.py`)

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

class ModelTrainer:
    """Handle model training and evaluation"""
    
    def __init__(self, feature_pipeline):
        self.feature_pipeline = feature_pipeline
        self.models = {}
        self.best_model = None
        self.best_score = 0
    
    def get_model_configs(self):
        """Define models and their hyperparameters for tuning"""
        return {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42),
                'params': {
                    'model__C': [0.1, 1, 10],
                    'model__penalty': ['l1', 'l2']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'model__n_estimators': [100, 200],
                    'model__learning_rate': [0.05, 0.1],
                    'model__max_depth': [3, 5]
                }
            }
        }
    
    def train_models(self, X_train, y_train, cv_folds=5):
        """Train multiple models with hyperparameter tuning"""
        model_configs = self.get_model_configs()
        results = {}
        
        for name, config in model_configs.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('features', self.feature_pipeline),
                ('model', config['model'])
            ])
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline,
                config['params'],
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit and store results
            grid_search.fit(X_train, y_train)
            
            self.models[name] = grid_search.best_estimator_
            results[name] = {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'cv_scores': cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=cv_folds)
            }
            
            # Track best model
            if grid_search.best_score_ > self.best_score:
                self.best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_model_name = name
            
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
        
        return results
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test set"""
        evaluation_results = {}
        
        for name, model in self.models.items():
            predictions = model.predict(X_test)
            
            evaluation_results[name] = {
                'accuracy': accuracy_score(y_test, predictions),
                'classification_report': classification_report(y_test, predictions),
                'confusion_matrix': confusion_matrix(y_test, predictions)
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"Accuracy: {evaluation_results[name]['accuracy']:.4f}")
            print("Classification Report:")
            print(evaluation_results[name]['classification_report'])
        
        return evaluation_results
    
    def save_best_model(self, filepath):
        """Save the best performing model"""
        if self.best_model:
            joblib.dump(self.best_model, filepath)
            print(f"Best model ({self.best_model_name}) saved to {filepath}")
        else:
            print("No model trained yet!")
```

### 4. Main Pipeline Runner (`main.py`)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.pipeline.data_preprocessing import DataPreprocessor
from src.pipeline.feature_engineering import create_feature_pipeline
from src.pipeline.model_training import ModelTrainer
import joblib
import os

def create_sample_data():
    """Create sample restaurant review data for demonstration"""
    reviews = [
        "The food was absolutely amazing! Great service and atmosphere.",
        "Terrible experience. Food was cold and service was slow.",
        "Good restaurant with decent prices. Would recommend.",
        "Worst meal I've ever had. Will never go back.",
        "Excellent pasta and friendly staff. Loved it!",
        "Average food, nothing special but not bad either.",
        "Outstanding cuisine and perfect ambiance. Five stars!",
        "Poor quality ingredients and rude waiters.",
        "Delicious pizza and great value for money.",
        "Disappointing meal with overpriced dishes."
    ] * 100  # Multiply to create more data
    
    # Create corresponding labels
    sentiments = [1, 0, 1, 0, 1, 1, 1, 0, 1, 0] * 100  # 1 = positive, 0 = negative
    
    return pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })

def main():
    """Main pipeline execution"""
    print("Starting Restaurant Review Sentiment Analysis Pipeline...")
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('src/models', exist_ok=True)
    
    # Step 1: Load/Create Data
    print("\n1. Loading data...")
    df = create_sample_data()
    print(f"Dataset shape: {df.shape}")
    
    # Step 2: Data Preprocessing
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.preprocess_data(df)
    
    # Save preprocessed data
    df_clean.to_csv('data/processed/cleaned_reviews.csv', index=False)
    
    # Step 3: Train-Test Split
    print("\n3. Splitting data...")
    X = df_clean['review']
    y = df_clean['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Step 4: Feature Engineering
    print("\n4. Creating feature pipeline...")
    feature_pipeline = create_feature_pipeline()
    
    # Step 5: Model Training
    print("\n5. Training models...")
    trainer = ModelTrainer(feature_pipeline)
    training_results = trainer.train_models(X_train, y_train)
    
    # Step 6: Model Evaluation
    print("\n6. Evaluating models...")
    evaluation_results = trainer.evaluate_models(X_test, y_test)
    
    # Step 7: Save Best Model
    print("\n7. Saving best model...")
    trainer.save_best_model('src/models/best_sentiment_model.pkl')
    
    # Step 8: Create Results Summary
    print("\n8. Creating results summary...")
    results_df = pd.DataFrame({
        'Model': list(training_results.keys()),
        'CV_Score': [results['best_score'] for results in training_results.values()],
        'Test_Accuracy': [evaluation_results[model]['accuracy'] for model in training_results.keys()]
    })
    
    results_df.to_csv('data/processed/model_results.csv', index=False)
    print("\nResults Summary:")
    print(results_df)
    
    # Step 9: Test Predictions
    print("\n9. Testing predictions...")
    test_reviews = [
        "This restaurant serves the best pizza in town!",
        "Awful service and terrible food quality.",
        "Great atmosphere and delicious meals."
    ]
    
    best_model = trainer.best_model
    predictions = best_model.predict(test_reviews)
    
    print("\nSample Predictions:")
    for review, pred in zip(test_reviews, predictions):
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"Review: {review}")
        print(f"Predicted Sentiment: {sentiment}\n")
    
    print("Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
```

### 5. API Service (`src/api/app.py`)

```python
from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the trained model
model_path = 'src/models/best_sentiment_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """API endpoint for sentiment prediction"""
    try:
        data = request.get_json()
        review_text = data.get('review', '')
        
        if not review_text:
            return jsonify({'error': 'No review text provided'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Make prediction
        prediction = model.predict([review_text])[0]
        probability = model.predict_proba([review_text])[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = float(max(probability))
        
        return jsonify({
            'review': review_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'prediction_value': int(prediction)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### 6. Requirements File (`requirements.txt`)

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
nltk==3.8.1
flask==2.3.2
joblib==1.3.1
matplotlib==3.7.2
seaborn==0.12.2
```

### 7. Testing Script (`tests/test_pipeline.py`)

```python
import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.data_preprocessing import DataPreprocessor, TextCleaner
from src.pipeline.feature_engineering import create_feature_pipeline

class TestPipeline(unittest.TestCase):
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'review': ['Great food!', 'Terrible service.', 'Good restaurant'],
            'sentiment': ['positive', 'negative', 'positive']
        })
    
    def test_text_cleaner(self):
        cleaner = TextCleaner()
        result = cleaner.transform(['Great Food!!! 123'])
        self.assertEqual(result[0], 'great food')
    
    def test_data_preprocessor(self):
        preprocessor = DataPreprocessor()
        result = preprocessor.preprocess_data(self.sample_data)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(x, int) for x in result['sentiment']))
    
    def test_feature_pipeline(self):
        pipeline = create_feature_pipeline()
        X = ['good food', 'bad service']
        features = pipeline.fit_transform(X)
        self.assertTrue(features.shape[0] == 2)

if __name__ == '__main__':
    unittest.main()
```

## Running the Pipeline

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline
```bash
python main.py
```

### 3. Start the API Service
```bash
python src/api/app.py
```

### 4. Test the API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "The food was absolutely delicious!"}'
```

### 5. Run Tests
```bash
python -m pytest tests/ -v
```

## Expected Outputs

1. **Cleaned dataset** saved to `data/processed/cleaned_reviews.csv`
2. **Trained models** with cross-validation scores
3. **Best model** saved to `src/models/best_sentiment_model.pkl`
4. **Results summary** with model comparison in `data/processed/model_results.csv`
5. **Working API** for real-time predictions
6. **Test predictions** on sample reviews

## Key Features Implemented

- **Custom Transformers**: TextCleaner, TextLengthExtractor, SentimentFeatureExtractor
- **Feature Union**: Combining TF-IDF, text statistics, and sentiment features
- **Automated Pipeline**: End-to-end processing with minimal manual intervention
- **Hyperparameter Tuning**: GridSearchCV for optimal model selection
- **Model Persistence**: Save/load trained models for production use
- **API Integration**: RESTful service for real-time predictions
- **Comprehensive Testing**: Unit tests for pipeline components
- **Cross-validation**: Robust model evaluation with multiple folds

This complete implementation demonstrates production-ready machine learning pipeline development with scikit-learn, including data preprocessing, feature engineering, model training, evaluation, and deployment capabilities.

## Assignment: Cuisine Classification Challenge

**The Challenge**: You are a consultant for a food delivery app. Create a machine learning system that can predict a restaurant's cuisine type based on menu item names and prices.

### Dataset Creation:
```python
# Create your dataset with menu items and prices
menu_items = {
    'Italian': ['spaghetti carbonara', 'margherita pizza', 'risotto', 'tiramisu', 'lasagna'],
    'Asian': ['pad thai', 'sushi roll', 'ramen', 'dim sum', 'fried rice'],
    'Mexican': ['tacos', 'burrito', 'quesadilla', 'guacamole', 'enchiladas'],
    'American': ['burger', 'hot dog', 'mac and cheese', 'bbq ribs', 'apple pie'],
    'French': ['coq au vin', 'croissant', 'ratatouille', 'crème brûlée', 'escargot']
}

# Your mission: Build a text classification pipeline that processes menu item names
```

### Requirements:
1. **Create a custom text preprocessor** that handles menu item names (remove numbers, standardize formatting)
2. **Build a pipeline** that includes text vectorization, feature selection, and classification
3. **Use GridSearchCV** to optimize your model parameters
4. **Achieve at least 85% accuracy** on your test set
5. **Analyze which words/features** are most important for each cuisine type

### Deliverables:
- Complete working pipeline code
- Performance evaluation with confusion matrix
- Feature importance analysis
- Brief explanation of your approach (2-3 paragraphs)

### Bonus Points:
- Create a custom transformer that adds price-based features
- Implement ensemble voting with multiple algorithms
- Add cross-validation for robust evaluation

**Submission Format**: Submit as a single Python script with comments explaining each major section.

---

## Course Summary

Congratulations! You've completed the Advanced Scikit-learn Mastery course. You've learned to:

1. **Master Advanced Techniques**: Like a chef mastering knife skills and sauce-making
2. **Create Professional Pipelines**: Building automated kitchen workflows
3. **Optimize with GridSearch**: Perfecting recipes through systematic testing  
4. **Build Custom Components**: Creating signature techniques and tools

You're now equipped with the advanced scikit-learn skills used by professional machine learning practitioners. Keep practicing these techniques, and remember - like cooking, machine learning mastery comes through continuous experimentation and refinement of your craft!

*Next up: Deploy your models to production and serve them to the world - from kitchen to customer!*