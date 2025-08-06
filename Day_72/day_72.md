# AI Mastery Course - Day 72: Data Preprocessing & Feature Engineering

## Learning Objective
By the end of this lesson, you will master the essential skills of preparing raw data for machine learning models, just like a chef transforms raw ingredients into a perfectly balanced dish ready for cooking.

---

## Introduction:

Imagine that you're the head chef at a world-renowned restaurant, and you've just received a delivery of fresh ingredients for tonight's signature dish. However, these ingredients arrive in various states - some vegetables have dirt on them, some fruits are overripe, others are underripe, and the portions are all different sizes. 

Before you can create your masterpiece, you need to clean, prepare, and standardize these ingredients. This is exactly what data preprocessing does for machine learning - it transforms raw, messy data into clean, standardized ingredients that your ML algorithms can work with effectively.

Just as a chef's prep work determines the quality of the final dish, your data preprocessing determines the success of your machine learning model.

---

## Lesson 1: Data Cleaning and Preprocessing

### The Chef's First Rule: Know Your Ingredients

A master chef always inspects their ingredients before cooking. Similarly, we must examine our data before feeding it to our models.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load our raw ingredients (data)
# Let's imagine we're working with a restaurant customer dataset
data = pd.read_csv('restaurant_customers.csv')

# First, let's inspect our ingredients
print("=== Kitchen Inspection (Data Overview) ===")
print(f"Number of customers (rows): {data.shape[0]}")
print(f"Number of attributes (columns): {data.shape[1]}")
print("\nFirst taste of our data:")
print(data.head())

print("\nData types - like checking if vegetables are fresh:")
print(data.dtypes)

print("\nBasic statistics - our ingredient quality report:")
print(data.describe())
```

**Syntax Explanation:**
- `pd.read_csv()`: Loads data from a CSV file into a pandas DataFrame
- `data.shape`: Returns tuple (rows, columns) showing data dimensions
- `data.head()`: Shows first 5 rows of data for quick inspection
- `data.dtypes`: Shows data type of each column
- `data.describe()`: Provides statistical summary for numerical columns

### Identifying Problems in Our Kitchen

```python
# Check for missing ingredients (missing values)
print("=== Missing Ingredients Report ===")
missing_data = data.isnull().sum()
print(missing_data[missing_data > 0])

# Check for duplicates - like having the same order twice
print(f"\nDuplicate orders found: {data.duplicated().sum()}")

# Visual inspection - like a chef eyeballing the ingredients
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=True, cmap='viridis')
plt.title("Missing Data Heatmap - Our Kitchen's Ingredient Status")
plt.show()
```

**Syntax Explanation:**
- `data.isnull().sum()`: Counts missing values per column
- `data.duplicated().sum()`: Counts duplicate rows
- `sns.heatmap()`: Creates visual representation of missing data
- `cmap='viridis'`: Sets color scheme for the heatmap

---

## Lesson 2: Handling Missing Data and Outliers

### The Art of Salvaging Ingredients

Just as a chef doesn't throw away slightly imperfect ingredients but finds creative ways to use them, we handle missing data strategically.

```python
# Strategy 1: Remove spoiled ingredients (drop missing values)
# Use when you have plenty of data and few missing values
def remove_spoiled_ingredients(data, threshold=0.1):
    """Remove columns with more than threshold proportion of missing values"""
    missing_percent = data.isnull().mean()
    columns_to_drop = missing_percent[missing_percent > threshold].index
    print(f"Discarding these ingredients (columns): {list(columns_to_drop)}")
    return data.drop(columns=columns_to_drop)

# Strategy 2: Fill missing ingredients (imputation)
def fill_missing_ingredients(data):
    """Fill missing values like a chef substituting ingredients"""
    data_filled = data.copy()
    
    # For numerical ingredients - use the average (like using standard portions)
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        if data[col].isnull().any():
            mean_value = data[col].mean()
            data_filled[col].fillna(mean_value, inplace=True)
            print(f"Filled missing {col} with average value: {mean_value:.2f}")
    
    # For categorical ingredients - use the most common (like chef's favorite)
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if data[col].isnull().any():
            mode_value = data[col].mode()[0]
            data_filled[col].fillna(mode_value, inplace=True)
            print(f"Filled missing {col} with most common: {mode_value}")
    
    return data_filled

# Apply our missing data strategy
cleaned_data = fill_missing_ingredients(data)
```

**Syntax Explanation:**
- `data.copy()`: Creates a copy to avoid modifying original data
- `data.select_dtypes(include=[np.number])`: Selects only numerical columns
- `fillna()`: Fills missing values with specified value
- `inplace=True`: Modifies the DataFrame directly instead of returning a copy
- `data[col].mode()[0]`: Gets the most frequent value in a column

### Handling Outliers - Dealing with Oversized Ingredients

```python
def identify_outliers(data, column):
    """Identify outliers like a chef spotting oversized vegetables"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier boundaries (like setting portion size limits)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    print(f"Found {len(outliers)} outliers in {column}")
    print(f"Normal range: {lower_bound:.2f} to {upper_bound:.2f}")
    
    return outliers, lower_bound, upper_bound

# Example: Check for outliers in customer age
age_outliers, low_bound, high_bound = identify_outliers(cleaned_data, 'age')

# Visualize outliers like arranging ingredients for inspection
plt.figure(figsize=(10, 6))
plt.boxplot(cleaned_data['age'])
plt.title("Age Distribution - Spotting the Outliers")
plt.ylabel("Age")
plt.show()
```

**Syntax Explanation:**
- `quantile(0.25)`: Calculates the 25th percentile (Q1)
- `quantile(0.75)`: Calculates the 75th percentile (Q3)
- `|` (pipe): Logical OR operator for combining conditions
- `plt.boxplot()`: Creates box plot to visualize outliers

---

## Lesson 3: Feature Scaling and Normalization

### Standardizing Portion Sizes

Imagine you're preparing a dish where you need 1 cup of flour, 2 tablespoons of sugar, and 500ml of milk. These are all different scales! Similarly, our data features often come in different scales that need standardization.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Let's say we have features with different scales
# Age (0-100), Income (0-100000), Years_experience (0-50)

def demonstrate_scaling_techniques(data):
    """Show different scaling techniques like different cooking methods"""
    
    # Select numerical features for scaling
    numerical_features = ['age', 'income', 'years_experience']
    sample_data = data[numerical_features].head(10)
    
    print("=== Original Ingredient Portions ===")
    print(sample_data)
    
    # Method 1: StandardScaler (Z-score normalization)
    # Like converting everything to "standard chef portions"
    standard_scaler = StandardScaler()
    standardized_data = standard_scaler.fit_transform(sample_data)
    standardized_df = pd.DataFrame(standardized_data, columns=numerical_features)
    
    print("\n=== Standard Chef Portions (StandardScaler) ===")
    print("Mean ≈ 0, Standard Deviation ≈ 1")
    print(standardized_df)
    
    # Method 2: MinMaxScaler (0-1 scaling)
    # Like converting everything to percentages
    minmax_scaler = MinMaxScaler()
    minmax_data = minmax_scaler.fit_transform(sample_data)
    minmax_df = pd.DataFrame(minmax_data, columns=numerical_features)
    
    print("\n=== Percentage Portions (MinMaxScaler) ===")
    print("All values between 0 and 1")
    print(minmax_df)
    
    # Method 3: RobustScaler
    # Like adjusting for unusual ingredient sizes (handles outliers better)
    robust_scaler = RobustScaler()
    robust_data = robust_scaler.fit_transform(sample_data)
    robust_df = pd.DataFrame(robust_data, columns=numerical_features)
    
    print("\n=== Outlier-Resistant Portions (RobustScaler) ===")
    print("Uses median and IQR, less affected by outliers")
    print(robust_df)
    
    return standardized_df, minmax_df, robust_df

# Apply scaling techniques
std_data, minmax_data, robust_data = demonstrate_scaling_techniques(cleaned_data)
```

**Syntax Explanation:**
- `StandardScaler()`: Creates scaler that standardizes features to mean=0, std=1
- `fit_transform()`: Learns scaling parameters and applies transformation
- `MinMaxScaler()`: Scales features to a fixed range (usually 0-1)
- `RobustScaler()`: Uses median and IQR, less sensitive to outliers

---

## Lesson 4: Feature Selection and Extraction

### Choosing the Right Ingredients for Your Recipe

Not every ingredient in your pantry belongs in every dish. A master chef selects only the ingredients that enhance the final flavor. Similarly, we select features that improve our model's performance.

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def select_best_ingredients(X, y, k=5):
    """Select the most flavorful features using statistical tests"""
    
    print("=== Selecting Best Ingredients (Features) ===")
    
    # Method 1: SelectKBest - like a taste test for ingredients
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()]
    feature_scores = selector.scores_[selector.get_support()]
    
    print(f"Top {k} ingredients selected:")
    for feature, score in zip(selected_features, feature_scores):
        print(f"  {feature}: {score:.2f} (flavor intensity)")
    
    return X_selected, selected_features

def extract_recipe_essence(X, n_components=3):
    """Extract the essence of our ingredients using PCA"""
    
    print(f"\n=== Extracting Recipe Essence (PCA) ===")
    print(f"Combining {X.shape[1]} ingredients into {n_components} flavor profiles")
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Show how much 'flavor' each component captures
    explained_variance = pca.explained_variance_ratio_
    
    print("Flavor profiles created:")
    for i, variance in enumerate(explained_variance):
        print(f"  Profile {i+1}: Captures {variance*100:.1f}% of original taste")
    
    print(f"Total flavor preserved: {sum(explained_variance)*100:.1f}%")
    
    return X_pca, pca

# Example usage (assuming we have a target variable 'customer_satisfaction')
# X = cleaned_data.select_dtypes(include=[np.number])
# y = cleaned_data['customer_satisfaction']

# selected_features, feature_names = select_best_ingredients(X, y, k=5)
# essence_features, pca_model = extract_recipe_essence(X, n_components=3)
```

**Syntax Explanation:**
- `SelectKBest()`: Selects k best features based on statistical tests
- `f_classif`: Statistical test for classification problems
- `get_support()`: Returns boolean mask of selected features
- `PCA()`: Principal Component Analysis for dimensionality reduction
- `explained_variance_ratio_`: Shows proportion of variance explained by each component

---

# Complete Data Preprocessing Pipeline Project

## The Master Chef's Kitchen Setup

Just as a master chef prepares all ingredients before cooking - washing vegetables, seasoning meats, and organizing spices - we'll build a complete preprocessing pipeline that transforms raw data into perfectly prepared ingredients for our machine learning feast.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessingPipeline:
    """
    A comprehensive data preprocessing pipeline that handles:
    - Missing data imputation
    - Outlier detection and handling
    - Feature scaling and normalization
    - Categorical encoding
    - Feature selection
    """
    
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.selected_features = None
        self.outlier_bounds = {}
    
    def detect_outliers(self, df, columns, method='iqr'):
        """
        Detect outliers using IQR method (like a chef checking for spoiled ingredients)
        """
        outliers_info = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'bounds': (lower_bound, upper_bound)
                }
                self.outlier_bounds[col] = (lower_bound, upper_bound)
        
        return outliers_info
    
    def handle_outliers(self, df, columns, method='cap'):
        """
        Handle outliers by capping or removing (like trimming excess fat from meat)
        """
        df_clean = df.copy()
        
        for col in columns:
            if col in self.outlier_bounds:
                lower_bound, upper_bound = self.outlier_bounds[col]
                
                if method == 'cap':
                    # Cap outliers to bounds
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                elif method == 'remove':
                    # Remove outlier rows
                    df_clean = df_clean[
                        (df_clean[col] >= lower_bound) & 
                        (df_clean[col] <= upper_bound)
                    ]
        
        return df_clean
    
    def encode_categorical_features(self, df, categorical_columns):
        """
        Encode categorical features (like converting recipe instructions to standard format)
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                # Handle unseen categories during transform
                unique_values = set(df_encoded[col].astype(str))
                known_values = set(self.label_encoders[col].classes_)
                new_values = unique_values - known_values
                
                if new_values:
                    # Add new categories to encoder
                    all_values = list(known_values) + list(new_values)
                    self.label_encoders[col].classes_ = np.array(all_values)
                
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def select_features(self, X, y, method='importance', n_features=10):
        """
        Select most important features (like choosing the best ingredients for a dish)
        """
        if method == 'importance':
            # Use Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.selected_features = feature_importance.head(n_features)['feature'].tolist()
            
            return X[self.selected_features], feature_importance
        
        return X, None
    
    def fit_transform(self, df, target_column, test_size=0.2):
        """
        Complete preprocessing pipeline (like a chef's complete meal prep process)
        """
        print("🍳 Starting the Master Chef's Data Preprocessing Pipeline...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Identify column types
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"📊 Found {len(numerical_columns)} numerical and {len(categorical_columns)} categorical features")
        
        # Step 1: Handle missing values (like checking for missing ingredients)
        print("\n🔍 Step 1: Handling Missing Values...")
        missing_info = X.isnull().sum()
        print(f"Missing values per column:\n{missing_info[missing_info > 0]}")
        
        if numerical_columns:
            X[numerical_columns] = self.numerical_imputer.fit_transform(X[numerical_columns])
        
        if categorical_columns:
            X[categorical_columns] = self.categorical_imputer.fit_transform(X[categorical_columns])
        
        # Step 2: Detect and handle outliers (like removing spoiled ingredients)
        if numerical_columns:
            print("\n🎯 Step 2: Detecting and Handling Outliers...")
            outliers_info = self.detect_outliers(X, numerical_columns)
            
            for col, info in outliers_info.items():
                print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")
            
            X = self.handle_outliers(X, numerical_columns, method='cap')
        
        # Step 3: Encode categorical features (like standardizing recipe formats)
        if categorical_columns:
            print("\n🏷️ Step 3: Encoding Categorical Features...")
            X = self.encode_categorical_features(X, categorical_columns)
        
        # Step 4: Scale numerical features (like standardizing measurements)
        if numerical_columns:
            print("\n⚖️ Step 4: Scaling Numerical Features...")
            X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        
        # Step 5: Feature selection (like choosing the best ingredients)
        print("\n🎖️ Step 5: Selecting Most Important Features...")
        X_selected, feature_importance = self.select_features(X, y, n_features=min(10, len(X.columns)))
        
        print(f"Selected {len(self.selected_features)} most important features:")
        for i, feature in enumerate(self.selected_features[:5], 1):
            importance = feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0]
            print(f"{i}. {feature}: {importance:.4f}")
        
        # Step 6: Split the data (like portioning ingredients for different dishes)
        print("\n🍽️ Step 6: Splitting Data for Training and Testing...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test, feature_importance
    
    def visualize_preprocessing_results(self, feature_importance, X_train, y_train):
        """
        Visualize the preprocessing results (like plating the final dish)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature importance plot
        top_features = feature_importance.head(10)
        axes[0, 0].barh(top_features['feature'], top_features['importance'])
        axes[0, 0].set_title('🎖️ Top 10 Feature Importance')
        axes[0, 0].set_xlabel('Importance Score')
        
        # Distribution of target variable
        y_train.value_counts().plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('🎯 Target Variable Distribution')
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Count')
        
        # Correlation heatmap of selected features
        correlation_matrix = X_train.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 0])
        axes[1, 0].set_title('🔥 Feature Correlation Heatmap')
        
        # Feature distribution (example with first numerical feature)
        if len(X_train.columns) > 0:
            X_train.iloc[:, 0].hist(bins=30, ax=axes[1, 1])
            axes[1, 1].set_title(f'📊 Distribution of {X_train.columns[0]}')
            axes[1, 1].set_xlabel('Values')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

# Demonstration with sample data (like preparing a sample dish)
def create_sample_restaurant_data():
    """
    Create sample restaurant customer data for demonstration
    """
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 12, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'visit_frequency': np.random.poisson(3, n_samples),
        'avg_spending': np.random.normal(45, 20, n_samples),
        'cuisine_preference': np.random.choice(['Italian', 'Chinese', 'Mexican', 'Indian'], n_samples),
        'dining_time': np.random.choice(['Breakfast', 'Lunch', 'Dinner'], n_samples),
        'party_size': np.random.choice([1, 2, 3, 4, 5, 6], n_samples),
        'satisfaction_score': np.random.uniform(1, 5, n_samples),
        'will_return': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # Target variable
    }
    
    # Add some missing values and outliers (like real-world messy data)
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df.loc[missing_indices[:50], 'income'] = np.nan
    df.loc[missing_indices[50:], 'satisfaction_score'] = np.nan
    
    # Introduce outliers
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_indices, 'avg_spending'] = np.random.uniform(200, 500, 20)
    
    return df

# Execute the complete pipeline
print("🍽️ Welcome to the Master Chef's Data Preprocessing Kitchen!")
print("=" * 60)

# Create sample data
restaurant_data = create_sample_restaurant_data()
print(f"📈 Created sample restaurant dataset with {len(restaurant_data)} customers")
print(f"Dataset shape: {restaurant_data.shape}")
print(f"Columns: {list(restaurant_data.columns)}")

# Initialize and run the preprocessing pipeline
pipeline = DataPreprocessingPipeline()

# Execute the complete preprocessing pipeline
X_train, X_test, y_train, y_test, feature_importance = pipeline.fit_transform(
    restaurant_data, 
    target_column='will_return'
)

print("\n" + "=" * 60)
print("🎉 Preprocessing Complete! Ready for Model Training")
print("=" * 60)

# Visualize results
pipeline.visualize_preprocessing_results(feature_importance, X_train, y_train)

# Test the pipeline with a simple model (like taste-testing the prepared ingredients)
print("\n🧪 Testing Preprocessed Data with Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy with Preprocessed Data: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n🏆 Pipeline Summary:")
print(f"✅ Processed {len(restaurant_data)} samples")
print(f"✅ Selected {len(pipeline.selected_features)} key features")
print(f"✅ Achieved {accuracy:.2%} accuracy")
print(f"✅ Ready for production deployment!")
```

## Key Pipeline Components Explained:

**Class Structure (`DataPreprocessingPipeline`):**
- `__init__()`: Initializes all preprocessing tools (like setting up kitchen equipment)
- Instance variables store fitted transformers for consistent preprocessing

**Outlier Detection (`detect_outliers`):**
- Uses IQR method: `Q1 - 1.5 * IQR` and `Q3 + 1.5 * IQR` for bounds
- `np.clip()` caps values to acceptable ranges
- Dictionary comprehension creates outlier summary

**Feature Encoding (`encode_categorical_features`):**
- `LabelEncoder()` converts categories to numbers
- Handles unseen categories during transform phase
- `self.label_encoders[col].classes_` stores learned categories

**Feature Selection (`select_features`):**
- `RandomForestClassifier.feature_importances_` ranks feature importance
- `pd.DataFrame.sort_values()` orders features by importance
- Returns top N features for model training

**Pipeline Integration (`fit_transform`):**
- Sequential processing: missing values → outliers → encoding → scaling → selection
- `train_test_split(stratify=y)` maintains class distribution
- Returns ready-to-use training and testing sets

This pipeline transforms raw, messy data into clean, standardized features ready for machine learning - just like a master chef transforms raw ingredients into a perfectly prepared meal!

## Assignment: The Mystery Restaurant Dataset

You've been hired as the head data chef for "The Mystery Bistro." The restaurant has been collecting customer data, but it's messy! Your task is to clean and prepare this dataset for analysis.

### Dataset Description
The dataset contains customer information with the following potential issues:
- Missing values in multiple columns
- Outliers in age and spending data
- Mixed data types that need standardization
- Too many features that might need selection

### Your Mission
Create a Python script that:

1. **Loads and inspects** the mystery dataset
2. **Identifies and handles** missing values using at least 2 different strategies
3. **Detects and addresses** outliers in numerical columns
4. **Applies appropriate scaling** to numerical features
5. **Selects the top 5 most important features** for predicting customer satisfaction
6. **Creates a summary report** showing before/after statistics

### Deliverables
- A well-commented Python script (`.py` file)
- A brief report (1-2 pages) explaining your preprocessing choices and their impact
- Visualizations showing the data before and after preprocessing

### Evaluation Criteria
- Correctness of preprocessing techniques applied
- Quality of code organization and comments
- Clarity of explanations for chosen methods
- Effectiveness of visualizations
- Insight into how preprocessing improved data quality

### Starter Code Structure
```python
# Your script should follow this structure:
def load_and_inspect_data(filepath):
    """Load data and provide initial inspection"""
    pass

def handle_missing_values(data):
    """Handle missing values with justified strategy"""
    pass

def detect_and_handle_outliers(data):
    """Identify and address outliers"""
    pass

def scale_features(data):
    """Apply appropriate scaling techniques"""
    pass

def select_important_features(data, target):
    """Select most relevant features"""
    pass

def create_preprocessing_report(original_data, processed_data):
    """Generate before/after comparison report"""
    pass

# Main execution
if __name__ == "__main__":
    # Your main preprocessing pipeline here
    pass
```

**Due Date:** Submit within one week of receiving the dataset.

---

## Course Summary

In this lesson, you've learned to be a data preprocessing chef, transforming raw, messy ingredients into clean, standardized features ready for machine learning. Remember:

1. **Always inspect your ingredients first** - understand your data before processing
2. **Handle missing values strategically** - don't just delete everything
3. **Standardize your portions** - scaling ensures fair comparison between features
4. **Choose your ingredients wisely** - feature selection improves model performance

Just as a great dish starts with proper ingredient preparation, a successful machine learning model begins with thorough data preprocessing. Master these techniques, and you'll create models that consistently deliver exceptional results!

---

*"Data is the new oil, but like crude oil, it must be refined before it becomes valuable."* - Anonymous Data Scientist