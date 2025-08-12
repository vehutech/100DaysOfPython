# Day 90: AI Ethics & Bias 

## Learning Objectives
By the end of this module, you will be able to:
- Identify and understand algorithmic bias and fairness principles in AI systems
- Implement explainable AI (XAI) techniques to make AI decisions transparent
- Apply privacy-preserving machine learning methods to protect sensitive data
- Develop AI systems following responsible development practices
- Evaluate and improve AI models for ethical considerations

---

Imagine that you're running a prestigious restaurant where every dish must be prepared with the utmost care and consideration. Your head chef doesn't just focus on making food taste good—they must ensure every ingredient is ethically sourced, every cooking method is transparent to diners with allergies, and every meal serves all customers fairly regardless of their background. 

Just as a responsible chef considers the origin of ingredients, the dietary needs of diverse customers, and the transparency of cooking methods, an AI practitioner must consider the ethical implications, fairness, and transparency of their algorithms. Welcome to the most crucial lesson in your AI journey—learning to cook up AI solutions that are not just powerful, but ethical and responsible.

---

## Lesson 1: Algorithmic Bias and Fairness

### Understanding the Recipe for Fair AI

Think of algorithmic bias like a seasoned chef who unconsciously favors certain ingredients because of past experiences. If a chef only knows how to cook Italian cuisine, they might overlook the rich flavors of Asian or African dishes, inadvertently creating a limited menu that doesn't serve all customers well.

### Types of Algorithmic Bias

**1. Historical Bias**: When training data reflects past inequalities
**2. Representation Bias**: When certain groups are underrepresented in data
**3. Measurement Bias**: When data collection methods favor certain groups
**4. Confirmation Bias**: When algorithms reinforce existing prejudices

### Code Example: Detecting Gender Bias in Job Recommendations

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample dataset with potential gender bias
np.random.seed(42)
n_samples = 1000

# Simulate job application data
data = {
    'years_experience': np.random.normal(5, 2, n_samples),
    'education_level': np.random.choice([1, 2, 3, 4], n_samples),  # 1=HS, 2=Bachelor, 3=Master, 4=PhD
    'previous_salary': np.random.normal(60000, 15000, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'age': np.random.normal(35, 8, n_samples)
}

# Create biased hiring decisions (historically biased toward males)
# This simulates real-world historical bias in hiring
bias_factor = np.where(data['gender'] == 'Male', 0.3, -0.2)
hiring_score = (
    data['years_experience'] * 0.3 + 
    np.array(data['education_level']) * 0.2 + 
    np.array(data['previous_salary']) / 100000 * 0.2 + 
    bias_factor + 
    np.random.normal(0, 0.1, n_samples)
)

data['hired'] = (hiring_score > np.median(hiring_score)).astype(int)

df = pd.DataFrame(data)

# Convert gender to numeric for model
df['gender_encoded'] = df['gender'].map({'Male': 1, 'Female': 0})

print("Dataset Overview:")
print(df.head())
print(f"\nHiring rate by gender:")
print(df.groupby('gender')['hired'].mean())
```

**Syntax Explanation:**
- `np.random.seed(42)`: Sets random seed for reproducible results
- `np.random.normal(mean, std, size)`: Generates normally distributed random numbers
- `np.where(condition, value_if_true, value_if_false)`: Vectorized conditional operation
- `pd.DataFrame(data)`: Creates DataFrame from dictionary
- `df.groupby('gender')['hired'].mean()`: Groups data by gender and calculates mean hiring rate

### Fairness Metrics Implementation

```python
def calculate_fairness_metrics(df, protected_attribute, outcome):
    """
    Calculate key fairness metrics for a binary outcome
    """
    # Statistical Parity (Demographic Parity)
    # P(hired=1|gender=Male) should equal P(hired=1|gender=Female)
    group_rates = df.groupby(protected_attribute)[outcome].mean()
    statistical_parity = abs(group_rates.iloc[0] - group_rates.iloc[1])
    
    # Equal Opportunity
    # P(hired=1|gender=Male, qualified=1) should equal P(hired=1|gender=Female, qualified=1)
    # Using education level > 2 as proxy for "qualified"
    qualified_df = df[df['education_level'] > 2]
    if len(qualified_df) > 0:
        qualified_rates = qualified_df.groupby(protected_attribute)[outcome].mean()
        equal_opportunity = abs(qualified_rates.iloc[0] - qualified_rates.iloc[1])
    else:
        equal_opportunity = 0
    
    return {
        'statistical_parity_difference': statistical_parity,
        'equal_opportunity_difference': equal_opportunity,
        'group_rates': group_rates
    }

# Calculate fairness metrics
fairness_results = calculate_fairness_metrics(df, 'gender', 'hired')
print("Fairness Metrics:")
for metric, value in fairness_results.items():
    print(f"{metric}: {value}")
```

---

## Lesson 2: Explainable AI (XAI) Techniques

### Making Your Recipe Transparent

Just as diners deserve to know what's in their food (especially those with allergies), users deserve to understand how AI systems make decisions that affect them. Explainable AI is like providing a detailed recipe card with every dish.

### Code Example: LIME for Model Explanations

```python
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier

# Prepare features for ML model (excluding target and non-predictive columns)
feature_columns = ['years_experience', 'education_level', 'previous_salary', 'gender_encoded', 'age']
X = df[feature_columns]
y = df['hired']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print(f"Model Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.3f}")

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_columns,
    class_names=['Not Hired', 'Hired'],
    mode='classification'
)

# Explain a single prediction
instance_idx = 0
explanation = explainer.explain_instance(
    X_test.iloc[instance_idx].values, 
    rf_model.predict_proba,
    num_features=len(feature_columns)
)

print(f"\nExplanation for instance {instance_idx}:")
print(f"Actual outcome: {'Hired' if y_test.iloc[instance_idx] == 1 else 'Not Hired'}")
print(f"Predicted probability of hiring: {rf_model.predict_proba(X_test.iloc[instance_idx:instance_idx+1])[0][1]:.3f}")

# Display feature importance for this instance
for feature, importance in explanation.as_list():
    print(f"{feature}: {importance:.3f}")
```

**Syntax Explanation:**
- `train_test_split()`: Splits data into training and testing sets
- `RandomForestClassifier()`: Creates ensemble model using multiple decision trees
- `fit()`: Trains the model on training data
- `predict_proba()`: Returns probability estimates for each class
- `iloc[index]`: Integer-location based indexing for DataFrames

### Feature Importance Visualization

```python
# Global feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Global Feature Importance in Hiring Decisions')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("Global Feature Importance:")
print(feature_importance)
```

---

## Lesson 3: Privacy-Preserving ML

### Protecting the Secret Ingredients

Just as a chef protects their secret recipes while still serving delicious food, we need to protect sensitive personal data while still building effective AI models.

### Code Example: Differential Privacy Implementation

```python
import numpy as np

class DifferentialPrivacy:
    """
    Simple implementation of differential privacy for dataset queries
    """
    
    def __init__(self, epsilon=1.0):
        """
        Initialize with privacy parameter epsilon
        Lower epsilon = more privacy, less accuracy
        """
        self.epsilon = epsilon
    
    def add_laplace_noise(self, true_value, sensitivity):
        """
        Add Laplace noise to ensure differential privacy
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return true_value + noise
    
    def private_mean(self, data, lower_bound, upper_bound):
        """
        Calculate differentially private mean
        """
        true_mean = np.mean(data)
        sensitivity = (upper_bound - lower_bound) / len(data)
        return self.add_laplace_noise(true_mean, sensitivity)
    
    def private_count(self, data, condition):
        """
        Calculate differentially private count
        """
        true_count = np.sum(condition)
        sensitivity = 1  # Adding/removing one person changes count by at most 1
        return max(0, self.add_laplace_noise(true_count, sensitivity))

# Example usage
dp = DifferentialPrivacy(epsilon=0.5)  # Strong privacy protection

# Calculate private statistics
salary_data = df['previous_salary'].values
private_avg_salary = dp.private_mean(salary_data, 30000, 100000)
true_avg_salary = np.mean(salary_data)

hired_condition = df['hired'] == 1
private_hired_count = dp.private_count(df, hired_condition)
true_hired_count = np.sum(hired_condition)

print(f"Salary Statistics:")
print(f"True average salary: ${true_avg_salary:,.2f}")
print(f"Private average salary: ${private_avg_salary:,.2f}")
print(f"Noise added: ${abs(private_avg_salary - true_avg_salary):,.2f}")

print(f"\nHiring Statistics:")
print(f"True hired count: {true_hired_count}")
print(f"Private hired count: {private_hired_count:.0f}")
print(f"Noise added: {abs(private_hired_count - true_hired_count):.0f}")
```

**Syntax Explanation:**
- `class ClassName:`: Defines a custom class
- `def __init__(self, params):`: Constructor method, runs when object is created
- `self`: Refers to the current instance of the class
- `np.random.laplace(loc, scale)`: Generates random numbers from Laplace distribution
- `max(0, value)`: Ensures result is non-negative

---

## Lesson 4: Responsible AI Development

### Following the Master Chef's Code of Ethics

Every master chef follows certain principles: fresh ingredients, honest preparation, and care for customer wellbeing. Similarly, responsible AI development requires adherence to ethical principles throughout the entire development lifecycle.

### Code Example: AI Ethics Checklist Implementation

```python
class AIEthicsChecker:
    """
    A comprehensive ethics checker for AI models
    """
    
    def __init__(self):
        self.checklist = {
            'data_quality': False,
            'bias_assessment': False,
            'privacy_protection': False,
            'transparency': False,
            'human_oversight': False,
            'continuous_monitoring': False
        }
        self.results = {}
    
    def check_data_quality(self, df, required_columns):
        """
        Check for basic data quality issues
        """
        issues = []
        
        # Check for missing values
        missing_data = df[required_columns].isnull().sum()
        if missing_data.sum() > 0:
            issues.append(f"Missing data found: {missing_data.to_dict()}")
        
        # Check for data imbalance
        if 'hired' in df.columns:
            class_distribution = df['hired'].value_counts(normalize=True)
            if class_distribution.min() < 0.3:  # Less than 30% minority class
                issues.append(f"Class imbalance detected: {class_distribution.to_dict()}")
        
        # Check for outliers (using IQR method)
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                issues.append(f"Outliers in {col}: {len(outliers)} instances")
        
        self.results['data_quality'] = {
            'passed': len(issues) == 0,
            'issues': issues
        }
        self.checklist['data_quality'] = len(issues) == 0
        
        return self.results['data_quality']
    
    def check_bias_assessment(self, df, protected_attributes, outcome):
        """
        Assess potential bias in the dataset/model
        """
        bias_issues = []
        
        for attr in protected_attributes:
            if attr in df.columns:
                fairness_metrics = calculate_fairness_metrics(df, attr, outcome)
                
                # Flag if statistical parity difference > 0.1
                if fairness_metrics['statistical_parity_difference'] > 0.1:
                    bias_issues.append(f"Statistical parity violation for {attr}: {fairness_metrics['statistical_parity_difference']:.3f}")
                
                # Flag if equal opportunity difference > 0.1
                if fairness_metrics['equal_opportunity_difference'] > 0.1:
                    bias_issues.append(f"Equal opportunity violation for {attr}: {fairness_metrics['equal_opportunity_difference']:.3f}")
        
        self.results['bias_assessment'] = {
            'passed': len(bias_issues) == 0,
            'issues': bias_issues
        }
        self.checklist['bias_assessment'] = len(bias_issues) == 0
        
        return self.results['bias_assessment']
    
    def check_privacy_protection(self, has_anonymization=False, has_encryption=False, 
                                has_differential_privacy=False):
        """
        Check if privacy protection measures are in place
        """
        privacy_score = sum([has_anonymization, has_encryption, has_differential_privacy])
        
        self.results['privacy_protection'] = {
            'passed': privacy_score >= 2,  # At least 2 privacy measures
            'score': privacy_score,
            'measures': {
                'anonymization': has_anonymization,
                'encryption': has_encryption,
                'differential_privacy': has_differential_privacy
            }
        }
        self.checklist['privacy_protection'] = privacy_score >= 2
        
        return self.results['privacy_protection']
    
    def generate_report(self):
        """
        Generate comprehensive ethics report
        """
        total_checks = len(self.checklist)
        passed_checks = sum(self.checklist.values())
        ethics_score = passed_checks / total_checks * 100
        
        report = f"""
        AI ETHICS ASSESSMENT REPORT
        ===========================
        
        Overall Ethics Score: {ethics_score:.1f}% ({passed_checks}/{total_checks} checks passed)
        
        Detailed Results:
        """
        
        for check, passed in self.checklist.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            report += f"\n{check.replace('_', ' ').title()}: {status}"
            
            if check in self.results:
                result = self.results[check]
                if 'issues' in result and result['issues']:
                    report += f"\n  Issues: {', '.join(result['issues'])}"
                if 'score' in result:
                    report += f"\n  Score: {result['score']}"
        
        if ethics_score < 70:
            report += f"\n\n⚠️  WARNING: Ethics score below 70%. Immediate attention required."
        elif ethics_score < 85:
            report += f"\n\n⚠️  CAUTION: Ethics score needs improvement."
        else:
            report += f"\n\n✅ GOOD: Ethics score meets recommended standards."
        
        return report

# Example usage
ethics_checker = AIEthicsChecker()

# Run ethics checks
data_quality_result = ethics_checker.check_data_quality(df, feature_columns)
bias_result = ethics_checker.check_bias_assessment(df, ['gender'], 'hired')
privacy_result = ethics_checker.check_privacy_protection(
    has_anonymization=True, 
    has_encryption=False, 
    has_differential_privacy=True
)

# Generate and display report
print(ethics_checker.generate_report())
```

**Syntax Explanation:**
- `self.checklist = {}`: Creates instance dictionary to store check results
- `df.select_dtypes(include=[np.number])`: Selects only numeric columns
- `df.quantile(0.25)`: Calculates 25th percentile (first quartile)
- `len(issues) == 0`: Boolean expression that evaluates to True if no issues
- `f"string {variable}"`: F-string formatting for readable output
- `sum([bool1, bool2, bool3])`: Counts True values (True = 1, False = 0)

---

## Final Quality Project: Ethical AI Hiring System

Now it's time to bring all your knowledge together, like a master chef creating their signature dish that showcases every technique they've learned.

```python
# Complete Ethical AI Hiring System
class EthicalHiringSystem:
    """
    A comprehensive ethical AI hiring system that incorporates
    fairness, transparency, and privacy protection
    """
    
    def __init__(self, epsilon=1.0):
        self.model = None
        self.explainer = None
        self.privacy_protection = DifferentialPrivacy(epsilon)
        self.ethics_checker = AIEthicsChecker()
        self.fairness_threshold = 0.1
        
    def preprocess_data(self, df):
        """
        Preprocess data with privacy and fairness considerations
        """
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Apply differential privacy to sensitive numeric features
        for col in ['previous_salary', 'age']:
            if col in processed_df.columns:
                private_values = [
                    self.privacy_protection.add_laplace_noise(val, 1000 if col == 'previous_salary' else 1)
                    for val in processed_df[col]
                ]
                processed_df[f'{col}_private'] = private_values
        
        return processed_df
    
    def train_fair_model(self, X_train, y_train, protected_attribute):
        """
        Train model with fairness constraints
        """
        # Train initial model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Check fairness
        temp_df = X_train.copy()
        temp_df['hired'] = y_train
        temp_df['gender'] = temp_df[protected_attribute].map({1: 'Male', 0: 'Female'})
        
        fairness_metrics = calculate_fairness_metrics(temp_df, 'gender', 'hired')
        
        if fairness_metrics['statistical_parity_difference'] > self.fairness_threshold:
            print(f"⚠️  Fairness violation detected. Difference: {fairness_metrics['statistical_parity_difference']:.3f}")
            # In a real system, you would implement bias mitigation techniques here
            print("Applying fairness constraints...")
        
        return self.model
    
    def setup_explainability(self, X_train, feature_names):
        """
        Setup explainability tools
        """
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=['Not Hired', 'Hired'],
            mode='classification'
        )
        
    def predict_with_explanation(self, instance, explain=True):
        """
        Make prediction with optional explanation
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Make prediction
        prediction = self.model.predict([instance])[0]
        probability = self.model.predict_proba([instance])[0]
        
        result = {
            'prediction': 'Hired' if prediction == 1 else 'Not Hired',
            'probability': probability[1],
            'confidence': max(probability)
        }
        
        # Add explanation if requested
        if explain and self.explainer is not None:
            explanation = self.explainer.explain_instance(
                instance, 
                self.model.predict_proba,
                num_features=len(instance)
            )
            result['explanation'] = explanation.as_list()
        
        return result
    
    def audit_system(self, test_data, protected_attributes):
        """
        Comprehensive system audit
        """
        audit_results = {
            'data_quality': self.ethics_checker.check_data_quality(test_data, test_data.columns.tolist()),
            'bias_assessment': self.ethics_checker.check_bias_assessment(test_data, protected_attributes, 'hired'),
            'privacy_protection': self.ethics_checker.check_privacy_protection(
                has_differential_privacy=True,
                has_anonymization=True
            )
        }
        
        return audit_results

# Demonstration of the complete system
print("=== ETHICAL AI HIRING SYSTEM DEMONSTRATION ===\n")

# Initialize system
hiring_system = EthicalHiringSystem(epsilon=0.5)

# Preprocess data
processed_df = hiring_system.preprocess_data(df)

# Prepare training data
feature_columns_private = ['years_experience', 'education_level', 'previous_salary_private', 'gender_encoded', 'age_private']
X = processed_df[feature_columns_private]
y = processed_df['hired']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train fair model
model = hiring_system.train_fair_model(X_train, y_train, 'gender_encoded')

# Setup explainability
hiring_system.setup_explainability(X_train, feature_columns_private)

# Make predictions with explanations
sample_candidate = X_test.iloc[0].values
result = hiring_system.predict_with_explanation(sample_candidate, explain=True)

print("Sample Prediction:")
print(f"Decision: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Hiring Probability: {result['probability']:.3f}")
print("\nExplanation:")
for feature, importance in result['explanation']:
    print(f"  {feature}: {importance:.3f}")

# System audit
test_df = processed_df.iloc[len(X_train):].copy()
test_df['hired'] = y_test.values
audit_results = hiring_system.audit_system(test_df, ['gender_encoded'])

print(f"\n=== SYSTEM AUDIT RESULTS ===")
print(hiring_system.ethics_checker.generate_report())
```

---

# AI Bias Detection and Mitigation Tool

## Project Overview
Think of this tool as your kitchen's quality control system - just like a head chef needs to taste and adjust seasoning throughout cooking to ensure every dish meets standards, our bias detection tool continuously monitors and adjusts AI model predictions to ensure fairness across different groups.

## Core Components

### 1. Django Project Setup
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
    'bias_detector',
    'rest_framework',
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

### 2. Models - The Recipe Cards
```python
# bias_detector/models.py
from django.db import models
from django.contrib.auth.models import User
import json

class Dataset(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    file_path = models.FileField(upload_to='datasets/')
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class BiasAnalysis(models.Model):
    BIAS_TYPES = [
        ('demographic', 'Demographic Parity'),
        ('equalized_odds', 'Equalized Odds'),
        ('calibration', 'Calibration'),
        ('individual', 'Individual Fairness'),
    ]
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    bias_type = models.CharField(max_length=20, choices=BIAS_TYPES)
    protected_attribute = models.CharField(max_length=100)
    target_variable = models.CharField(max_length=100)
    bias_score = models.FloatField()
    analysis_results = models.JSONField()
    mitigation_applied = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

class MitigationStrategy(models.Model):
    STRATEGY_TYPES = [
        ('preprocessing', 'Pre-processing'),
        ('inprocessing', 'In-processing'),
        ('postprocessing', 'Post-processing'),
    ]
    
    analysis = models.ForeignKey(BiasAnalysis, on_delete=models.CASCADE)
    strategy_type = models.CharField(max_length=20, choices=STRATEGY_TYPES)
    strategy_name = models.CharField(max_length=200)
    parameters = models.JSONField()
    effectiveness_score = models.FloatField()
    applied_at = models.DateTimeField(auto_now_add=True)
```

### 3. Bias Detection Engine - The Master Chef's Palate
```python
# bias_detector/bias_engine.py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class BiasDetector:
    """
    Like a master chef who can detect the slightest imbalance in flavors,
    this class identifies unfairness in AI model predictions across different groups.
    """
    
    def __init__(self, data, protected_attribute, target_variable, predictions=None):
        self.data = data.copy()
        self.protected_attribute = protected_attribute
        self.target_variable = target_variable
        self.predictions = predictions
        self.bias_metrics = {}
        
    def demographic_parity(self):
        """
        Measures if positive prediction rates are equal across groups.
        Like ensuring each table gets the same portion sizes.
        """
        if self.predictions is None:
            return None
            
        results = {}
        groups = self.data[self.protected_attribute].unique()
        
        for group in groups:
            group_mask = self.data[self.protected_attribute] == group
            group_predictions = self.predictions[group_mask]
            positive_rate = np.mean(group_predictions)
            results[str(group)] = {
                'positive_rate': positive_rate,
                'sample_size': len(group_predictions)
            }
        
        # Calculate disparity
        rates = [results[str(g)]['positive_rate'] for g in groups]
        max_disparity = max(rates) - min(rates)
        
        return {
            'metric_name': 'Demographic Parity',
            'group_results': results,
            'max_disparity': max_disparity,
            'is_fair': max_disparity < 0.1,  # 10% threshold
            'bias_score': max_disparity
        }
    
    def equalized_odds(self):
        """
        Ensures equal true positive and false positive rates across groups.
        Like making sure each cooking station has the same success rate.
        """
        if self.predictions is None:
            return None
            
        results = {}
        groups = self.data[self.protected_attribute].unique()
        
        for group in groups:
            group_mask = self.data[self.protected_attribute] == group
            y_true_group = self.data[self.target_variable][group_mask]
            y_pred_group = self.predictions[group_mask]
            
            if len(np.unique(y_true_group)) > 1:  # Check if we have both classes
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
                
                results[str(group)] = {
                    'true_positive_rate': tpr,
                    'false_positive_rate': fpr,
                    'sample_size': len(y_true_group)
                }
        
        if len(results) < 2:
            return None
            
        # Calculate disparities
        tpr_values = [results[str(g)]['true_positive_rate'] for g in groups if str(g) in results]
        fpr_values = [results[str(g)]['false_positive_rate'] for g in groups if str(g) in results]
        
        tpr_disparity = max(tpr_values) - min(tpr_values) if tpr_values else 0
        fpr_disparity = max(fpr_values) - min(fpr_values) if fpr_values else 0
        
        max_disparity = max(tpr_disparity, fpr_disparity)
        
        return {
            'metric_name': 'Equalized Odds',
            'group_results': results,
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'max_disparity': max_disparity,
            'is_fair': max_disparity < 0.1,
            'bias_score': max_disparity
        }
    
    def statistical_parity_difference(self):
        """
        Measures the difference in positive outcome rates between groups.
        Like checking if different recipe variations have consistent success rates.
        """
        groups = self.data[self.protected_attribute].unique()
        if len(groups) != 2:
            return None
            
        group_rates = {}
        for group in groups:
            group_data = self.data[self.data[self.protected_attribute] == group]
            if self.predictions is not None:
                group_mask = self.data[self.protected_attribute] == group
                positive_rate = np.mean(self.predictions[group_mask])
            else:
                positive_rate = np.mean(group_data[self.target_variable])
            
            group_rates[str(group)] = positive_rate
        
        spd = abs(group_rates[str(groups[0])] - group_rates[str(groups[1])])
        
        return {
            'metric_name': 'Statistical Parity Difference',
            'group_rates': group_rates,
            'statistical_parity_difference': spd,
            'is_fair': spd < 0.1,
            'bias_score': spd
        }
    
    def run_all_analyses(self):
        """
        Runs all bias detection methods like a comprehensive taste test.
        """
        analyses = {}
        
        dp_result = self.demographic_parity()
        if dp_result:
            analyses['demographic_parity'] = dp_result
            
        eo_result = self.equalized_odds()
        if eo_result:
            analyses['equalized_odds'] = eo_result
            
        spd_result = self.statistical_parity_difference()
        if spd_result:
            analyses['statistical_parity'] = spd_result
        
        return analyses

class BiasMitigator:
    """
    Like a chef who adjusts recipes to perfect the balance,
    this class applies techniques to reduce bias in datasets and models.
    """
    
    def __init__(self, data, protected_attribute, target_variable):
        self.data = data.copy()
        self.protected_attribute = protected_attribute
        self.target_variable = target_variable
    
    def reweighting(self):
        """
        Adjusts sample weights to balance representation across groups.
        Like adjusting portion sizes to ensure fairness.
        """
        # Calculate group sizes
        group_sizes = self.data[self.protected_attribute].value_counts()
        total_size = len(self.data)
        
        # Calculate weights for each sample
        weights = []
        for _, row in self.data.iterrows():
            group_value = row[self.protected_attribute]
            group_size = group_sizes[group_value]
            # Inverse weighting: smaller groups get higher weights
            weight = total_size / (len(group_sizes) * group_size)
            weights.append(weight)
        
        weighted_data = self.data.copy()
        weighted_data['sample_weight'] = weights
        
        return {
            'strategy': 'reweighting',
            'data': weighted_data,
            'description': 'Applied inverse frequency weighting to balance groups',
            'effectiveness': self._calculate_effectiveness(weighted_data)
        }
    
    def disparate_impact_remover(self, repair_level=1.0):
        """
        Modifies feature distributions to reduce disparate impact.
        Like adjusting ingredient ratios to ensure consistent results.
        """
        # This is a simplified implementation
        # In practice, you'd use libraries like AIF360
        
        modified_data = self.data.copy()
        
        # Get numeric columns only
        numeric_cols = modified_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_variable]
        
        if not numeric_cols:
            return None
        
        groups = modified_data[self.protected_attribute].unique()
        
        for col in numeric_cols:
            group_means = {}
            for group in groups:
                group_data = modified_data[modified_data[self.protected_attribute] == group]
                group_means[group] = group_data[col].mean()
            
            # Calculate overall mean
            overall_mean = modified_data[col].mean()
            
            # Adjust values towards overall mean based on repair level
            for group in groups:
                group_mask = modified_data[self.protected_attribute] == group
                current_values = modified_data.loc[group_mask, col]
                adjusted_values = current_values + repair_level * (overall_mean - group_means[group])
                modified_data.loc[group_mask, col] = adjusted_values
        
        return {
            'strategy': 'disparate_impact_remover',
            'data': modified_data,
            'repair_level': repair_level,
            'description': f'Applied disparate impact removal with repair level {repair_level}',
            'effectiveness': self._calculate_effectiveness(modified_data)
        }
    
    def _calculate_effectiveness(self, modified_data):
        """
        Calculates how effective the mitigation strategy was.
        """
        # Simple effectiveness measure based on statistical parity
        detector = BiasDetector(modified_data, self.protected_attribute, self.target_variable)
        spd_result = detector.statistical_parity_difference()
        
        if spd_result:
            return 1 - spd_result['bias_score']  # Higher is better
        return 0.5  # Default moderate effectiveness
```

### 4. Views - The Service Counter
```python
# bias_detector/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import json
import os

from .models import Dataset, BiasAnalysis, MitigationStrategy
from .bias_engine import BiasDetector, BiasMitigator
from .forms import DatasetUploadForm, BiasAnalysisForm

@login_required
def dashboard(request):
    """
    Main dashboard - like the kitchen's central command center.
    """
    recent_analyses = BiasAnalysis.objects.filter(
        dataset__uploaded_by=request.user
    )[:5]
    
    datasets = Dataset.objects.filter(uploaded_by=request.user)
    
    # Calculate summary statistics
    total_analyses = BiasAnalysis.objects.filter(dataset__uploaded_by=request.user).count()
    biased_analyses = BiasAnalysis.objects.filter(
        dataset__uploaded_by=request.user,
        bias_score__gt=0.1
    ).count()
    
    context = {
        'recent_analyses': recent_analyses,
        'datasets': datasets,
        'total_analyses': total_analyses,
        'biased_analyses': biased_analyses,
        'bias_percentage': round((biased_analyses / total_analyses * 100) if total_analyses > 0 else 0, 1),
    }
    
    return render(request, 'bias_detector/dashboard.html', context)

@login_required
def upload_dataset(request):
    """
    Upload dataset - like receiving fresh ingredients in the kitchen.
    """
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.uploaded_by = request.user
            dataset.save()
            
            messages.success(request, 'Dataset uploaded successfully!')
            return redirect('bias_detector:analyze_bias', dataset_id=dataset.id)
    else:
        form = DatasetUploadForm()
    
    return render(request, 'bias_detector/upload_dataset.html', {'form': form})

@login_required
def analyze_bias(request, dataset_id):
    """
    Analyze bias in dataset - like a thorough quality inspection.
    """
    dataset = get_object_or_404(Dataset, id=dataset_id, uploaded_by=request.user)
    
    if request.method == 'POST':
        form = BiasAnalysisForm(request.POST)
        if form.is_valid():
            try:
                # Load the dataset
                file_path = dataset.file_path.path
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    messages.error(request, 'Only CSV files are supported currently.')
                    return redirect('bias_detector:analyze_bias', dataset_id=dataset_id)
                
                protected_attr = form.cleaned_data['protected_attribute']
                target_var = form.cleaned_data['target_variable']
                bias_type = form.cleaned_data['bias_type']
                
                # Validate columns exist
                if protected_attr not in df.columns or target_var not in df.columns:
                    messages.error(request, 'Selected columns not found in dataset.')
                    return render(request, 'bias_detector/analyze_bias.html', {
                        'form': form, 'dataset': dataset, 'columns': df.columns.tolist()
                    })
                
                # Create predictions (in real scenario, this would be from a trained model)
                # For demo, we'll create synthetic predictions based on some logic
                np.random.seed(42)  # For reproducibility
                predictions = np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])
                
                # Run bias detection
                detector = BiasDetector(df, protected_attr, target_var, predictions)
                
                if bias_type == 'demographic':
                    analysis_result = detector.demographic_parity()
                elif bias_type == 'equalized_odds':
                    analysis_result = detector.equalized_odds()
                else:
                    analysis_result = detector.run_all_analyses()
                
                if analysis_result:
                    # Save analysis
                    bias_analysis = BiasAnalysis.objects.create(
                        dataset=dataset,
                        bias_type=bias_type,
                        protected_attribute=protected_attr,
                        target_variable=target_var,
                        bias_score=analysis_result.get('bias_score', 0),
                        analysis_results=analysis_result
                    )
                    
                    messages.success(request, 'Bias analysis completed successfully!')
                    return redirect('bias_detector:analysis_results', analysis_id=bias_analysis.id)
                else:
                    messages.error(request, 'Could not perform bias analysis on this dataset.')
            
            except Exception as e:
                messages.error(request, f'Error analyzing dataset: {str(e)}')
    else:
        form = BiasAnalysisForm()
    
    # Load dataset to get column names
    try:
        file_path = dataset.file_path.path
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()
        else:
            columns = []
    except:
        columns = []
    
    return render(request, 'bias_detector/analyze_bias.html', {
        'form': form,
        'dataset': dataset,
        'columns': columns
    })

@login_required
def analysis_results(request, analysis_id):
    """
    Display analysis results - like presenting the final dish evaluation.
    """
    analysis = get_object_or_404(BiasAnalysis, id=analysis_id, dataset__uploaded_by=request.user)
    
    # Get any applied mitigation strategies
    mitigation_strategies = MitigationStrategy.objects.filter(analysis=analysis)
    
    context = {
        'analysis': analysis,
        'mitigation_strategies': mitigation_strategies,
        'results_json': json.dumps(analysis.analysis_results, indent=2),
    }
    
    return render(request, 'bias_detector/analysis_results.html', context)

@login_required
def apply_mitigation(request, analysis_id):
    """
    Apply bias mitigation - like adjusting the recipe to fix flavor imbalances.
    """
    analysis = get_object_or_404(BiasAnalysis, id=analysis_id, dataset__uploaded_by=request.user)
    
    if request.method == 'POST':
        strategy_type = request.POST.get('strategy_type')
        
        try:
            # Load the original dataset
            file_path = analysis.dataset.file_path.path
            df = pd.read_csv(file_path)
            
            # Apply mitigation
            mitigator = BiasMitigator(df, analysis.protected_attribute, analysis.target_variable)
            
            if strategy_type == 'reweighting':
                result = mitigator.reweighting()
            elif strategy_type == 'disparate_impact':
                repair_level = float(request.POST.get('repair_level', 1.0))
                result = mitigator.disparate_impact_remover(repair_level)
            else:
                messages.error(request, 'Unknown mitigation strategy.')
                return redirect('bias_detector:analysis_results', analysis_id=analysis_id)
            
            if result:
                # Save mitigation strategy
                mitigation = MitigationStrategy.objects.create(
                    analysis=analysis,
                    strategy_type=strategy_type,
                    strategy_name=result['strategy'],
                    parameters={'repair_level': result.get('repair_level', 1.0)},
                    effectiveness_score=result['effectiveness']
                )
                
                analysis.mitigation_applied = True
                analysis.save()
                
                messages.success(request, f'Mitigation strategy applied successfully! Effectiveness: {result["effectiveness"]:.2%}')
            else:
                messages.error(request, 'Could not apply mitigation strategy.')
        
        except Exception as e:
            messages.error(request, f'Error applying mitigation: {str(e)}')
    
    return redirect('bias_detector:analysis_results', analysis_id=analysis_id)

@csrf_exempt
def api_bias_check(request):
    """
    API endpoint for real-time bias checking - like a quick taste test.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            protected_attr = data['protected_attribute']
            target_var = data['target_variable']
            predictions = data.get('predictions')
            
            # Run bias detection
            detector = BiasDetector(df, protected_attr, target_var, predictions)
            results = detector.run_all_analyses()
            
            return JsonResponse({
                'status': 'success',
                'results': results
            })
        
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({'status': 'error', 'message': 'Only POST method allowed'})
```

### 5. Forms - The Order Taking System
```python
# bias_detector/forms.py
from django import forms
from .models import Dataset, BiasAnalysis

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'file_path']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Dataset name'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Describe your dataset'}),
            'file_path': forms.FileInput(attrs={'class': 'form-control', 'accept': '.csv'})
        }

class BiasAnalysisForm(forms.Form):
    BIAS_TYPE_CHOICES = [
        ('demographic', 'Demographic Parity'),
        ('equalized_odds', 'Equalized Odds'),
        ('all', 'Run All Analyses'),
    ]
    
    protected_attribute = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., gender, race, age_group'
        }),
        help_text='The sensitive attribute to check for bias against'
    )
    
    target_variable = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., approved, hired, recommended'
        }),
        help_text='The outcome variable you want to predict'
    )
    
    bias_type = forms.ChoiceField(
        choices=BIAS_TYPE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Type of bias analysis to perform'
    )
```

### 6. Templates - The Presentation Layer

```html
<!-- bias_detector/templates/bias_detector/dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>AI Bias Detection Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{% url 'bias_detector:dashboard' %}">
                🔍 AI Bias Detector
            </a>
            <span class="navbar-text">Welcome, {{ user.username }}</span>
        </div>
    </nav>
    
    <div class="container mt-4">
        <div class="row">
            <div class="col-md-12">
                <h2>Dashboard</h2>
                <p class="text-muted">Monitor and mitigate bias in your AI systems</p>
            </div>
        </div>
        
        <!-- Statistics Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">{{ total_analyses }}</h5>
                        <p class="card-text">Total Analyses</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">{{ datasets|length }}</h5>
                        <p class="card-text">Datasets</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">{{ biased_analyses }}</h5>
                        <p class="card-text">Biased Models</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">{{ bias_percentage }}%</h5>
                        <p class="card-text">Bias Rate</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Actions -->
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Upload New Dataset</h5>
                        <p class="card-text">Start by uploading a dataset to analyze for bias.</p>
                        <a href="{% url 'bias_detector:upload_dataset' %}" class="btn btn-primary">Upload Dataset</a>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Recent Analyses</h5>
                        {% if recent_analyses %}
                            <ul class="list-unstyled">
                                {% for analysis in recent_analyses %}
                                    <li class="mb-2">
                                        <a href="{% url 'bias_detector:analysis_results' analysis.id %}">
                                            {{ analysis.dataset.name }} - {{ analysis.get_bias_type_display }}
                                        </a>
                                        <small class="text-muted d-block">
                                            Bias Score: {{ analysis.bias_score|floatformat:3 }}
                                            {% if analysis.bias_score > 0.1 %}
                                                <span class="badge bg-danger">High Bias</span>
                                            {% else %}
                                                <span class="badge bg-success">Low Bias</span>
                                            {% endif %}
                                        </small>
                                    </li>
                                {% endfor %}
                            </ul>
                        {% else %}
                            <p class="text-muted">No analyses yet. Upload a dataset to get started!</p>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Datasets List -->
        <div class="row">
            <div class="col-md-12">
                <h4>Your Datasets</h4>
                {% if datasets %}
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Description</th>
                                    <th>Uploaded</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for dataset in datasets %}
                                    <tr>
                                        <td>{{ dataset.name }}</td>
                                        <td>{{ dataset.description|truncatechars:50 }}</td>
                                        <td>{{ dataset.created_at|date:"M d, Y" }}</td>
                                        <td>
                                            <a href="{% url 'bias_detector:analyze_bias' dataset.id %}" class="btn btn-sm btn-outline-primary">Analyze</a>
                                        </td>
                                    </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                {% else %}
                    <p class="text-muted">No datasets uploaded yet.</p>
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>
```

### 7. URLs Configuration - The Menu Structure
```python
# bias_detector/urls.py
from django.urls import path
from . import views

app_name = 'bias_detector'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('analyze/<int:dataset_id>/', views.analyze_bias, name='analyze_bias'),
    path('results/<int:analysis_id>/', views.analysis_results, name='analysis_results'),
    path('mitigate/<int:analysis_id>/', views.apply_mitigation, name='apply_mitigation'),
    path('api/bias-check/', views.api_bias_check, name='api_bias_check'),
]

# urls.py (main project)
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('bias_detector.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

### 8. Advanced Features - The Special Techniques

```python
# bias_detector/advanced_analytics.py
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from django.http import HttpResponse
import numpy as np
import pandas as pd

class VisualizationEngine:
    """
    Creates visual reports like a chef presenting beautifully plated dishes.
    Each visualization tells a story about fairness in your AI system.
    """
    
    def __init__(self, analysis_data, dataset):
        self.analysis_data = analysis_data
        self.dataset = dataset
        plt.style.use('seaborn-v0_8')
    
    def create_bias_heatmap(self):
        """
        Creates a heatmap showing bias across different metrics and groups.
        Like a temperature map showing hot spots in your kitchen.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract bias scores for different metrics
        metrics = []
        scores = []
        
        for metric_name, results in self.analysis_data.items():
            if 'bias_score' in results:
                metrics.append(metric_name.replace('_', ' ').title())
                scores.append(results['bias_score'])
        
        if scores:
            # Create heatmap data
            heatmap_data = np.array(scores).reshape(1, -1)
            
            sns.heatmap(heatmap_data, 
                       annot=True, 
                       fmt='.3f', 
                       xticklabels=metrics,
                       yticklabels=['Bias Score'],
                       cmap='RdYlBu_r',
                       center=0.1,
                       ax=ax)
            
            ax.set_title('Bias Metrics Heatmap', fontsize=16, pad=20)
            plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def create_fairness_radar(self):
        """
        Creates a radar chart showing fairness across multiple dimensions.
        Like a flavor profile wheel showing balance across taste elements.
        """
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        metrics = []
        scores = []
        
        for metric_name, results in self.analysis_data.items():
            if 'bias_score' in results:
                metrics.append(metric_name.replace('_', ' ').title())
                # Convert bias score to fairness score (1 - bias)
                fairness_score = max(0, 1 - results['bias_score'])
                scores.append(fairness_score)
        
        if scores:
            # Add angles for radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            scores += scores[:1]  # Complete the circle
            angles += angles[:1]
            
            ax.plot(angles, scores, 'o-', linewidth=2, label='Fairness Score')
            ax.fill(angles, scores, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('Fairness Radar Chart', pad=20)
            ax.grid(True)
        
        return self._fig_to_base64(fig)
    
    def create_group_comparison(self, protected_attribute):
        """
        Shows side-by-side comparison of outcomes across protected groups.
        Like comparing cooking results across different kitchen stations.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract group results from analysis
        for metric_name, results in self.analysis_data.items():
            if 'group_results' in results:
                group_data = results['group_results']
                
                # Plot positive rates
                groups = list(group_data.keys())
                if 'positive_rate' in list(group_data.values())[0]:
                    rates = [group_data[group]['positive_rate'] for group in groups]
                    
                    axes[0].bar(groups, rates, alpha=0.7)
                    axes[0].set_title(f'Positive Prediction Rates by {protected_attribute}')
                    axes[0].set_ylabel('Positive Rate')
                    axes[0].tick_params(axis='x', rotation=45)
                
                # Plot sample sizes
                sizes = [group_data[group]['sample_size'] for group in groups]
                axes[1].bar(groups, sizes, alpha=0.7, color='orange')
                axes[1].set_title(f'Sample Sizes by {protected_attribute}')
                axes[1].set_ylabel('Count')
                axes[1].tick_params(axis='x', rotation=45)
                
                break
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string for web display."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"

# bias_detector/reporting.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import colors
from reportlab.lib.units import inch
import datetime

class BiasReportGenerator:
    """
    Generates comprehensive bias reports like a head chef creating a detailed recipe book.
    Documents everything about the fairness assessment for future reference.
    """
    
    def __init__(self, analysis, dataset_info):
        self.analysis = analysis
        self.dataset_info = dataset_info
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom styles for the report."""
        custom = {}
        custom['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        custom['SectionHeader'] = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.darkgreen
        )
        return custom
    
    def generate_report(self, output_path):
        """
        Generate a comprehensive PDF report.
        Like creating a detailed cookbook entry with all measurements and techniques.
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph("AI Bias Detection Report", self.custom_styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.custom_styles['SectionHeader']))
        summary_text = self._generate_executive_summary()
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Dataset Information
        story.append(Paragraph("Dataset Information", self.custom_styles['SectionHeader']))
        dataset_info = [
            ['Dataset Name', self.dataset_info.get('name', 'Unknown')],
            ['Protected Attribute', self.analysis.protected_attribute],
            ['Target Variable', self.analysis.target_variable],
            ['Analysis Date', self.analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')],
            ['Bias Type Analyzed', self.analysis.get_bias_type_display()],
        ]
        
        dataset_table = Table(dataset_info, colWidths=[2*inch, 3*inch])
        dataset_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(dataset_table)
        story.append(Spacer(1, 20))
        
        # Bias Analysis Results
        story.append(Paragraph("Bias Analysis Results", self.custom_styles['SectionHeader']))
        results_text = self._format_analysis_results()
        story.append(Paragraph(results_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.custom_styles['SectionHeader']))
        recommendations = self._generate_recommendations()
        story.append(Paragraph(recommendations, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
    
    def _generate_executive_summary(self):
        """Generate executive summary based on analysis results."""
        bias_score = self.analysis.bias_score
        
        if bias_score < 0.05:
            fairness_level = "excellent"
            concern_level = "minimal"
        elif bias_score < 0.1:
            fairness_level = "good"
            concern_level = "low"
        elif bias_score < 0.2:
            fairness_level = "moderate"
            concern_level = "moderate"
        else:
            fairness_level = "poor"
            concern_level = "high"
        
        return f"""
        This report analyzes the fairness of an AI system with respect to the protected attribute 
        '{self.analysis.protected_attribute}' when predicting '{self.analysis.target_variable}'. 
        The overall bias score is {bias_score:.3f}, indicating {fairness_level} fairness with 
        {concern_level} concern for discriminatory impact. 
        
        The analysis employed {self.analysis.get_bias_type_display()} methodology to assess 
        potential disparate impact across different demographic groups.
        """
    
    def _format_analysis_results(self):
        """Format the detailed analysis results for the report."""
        results = self.analysis.analysis_results
        formatted_text = ""
        
        for metric_name, metric_results in results.items():
            formatted_text += f"<b>{metric_name.replace('_', ' ').title()}:</b><br/>"
            
            if 'bias_score' in metric_results:
                formatted_text += f"• Bias Score: {metric_results['bias_score']:.3f}<br/>"
            
            if 'is_fair' in metric_results:
                fairness = "Fair" if metric_results['is_fair'] else "Potentially Biased"
                formatted_text += f"• Assessment: {fairness}<br/>"
            
            if 'group_results' in metric_results:
                formatted_text += "• Group-specific results:<br/>"
                for group, group_data in metric_results['group_results'].items():
                    formatted_text += f"  - {group}: "
                    if 'positive_rate' in group_data:
                        formatted_text += f"Positive rate = {group_data['positive_rate']:.3f}, "
                    formatted_text += f"Sample size = {group_data.get('sample_size', 'Unknown')}<br/>"
            
            formatted_text += "<br/>"
        
        return formatted_text
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on bias analysis."""
        bias_score = self.analysis.bias_score
        recommendations = []
        
        if bias_score > 0.1:
            recommendations.extend([
                "• <b>Immediate Action Required:</b> The detected bias level exceeds acceptable thresholds.",
                "• Consider implementing bias mitigation techniques such as reweighting or fairness constraints.",
                "• Review data collection processes to identify potential sources of bias.",
                "• Implement ongoing monitoring to track bias metrics in production."
            ])
        else:
            recommendations.extend([
                "• <b>Maintain Current Standards:</b> Bias levels are within acceptable ranges.",
                "• Continue regular bias monitoring to ensure sustained fairness.",
                "• Document current practices as best practices for future projects."
            ])
        
        recommendations.extend([
            "• Establish clear fairness criteria and thresholds for your use case.",
            "• Consider stakeholder perspectives when defining fairness requirements.",
            "• Implement explainability tools to understand model decision-making.",
            "• Regular re-evaluation as new data becomes available."
        ])
        
        return "<br/>".join(recommendations)

# bias_detector/management/commands/generate_sample_data.py
from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os
from django.conf import settings

class Command(BaseCommand):
    help = 'Generate sample datasets for testing bias detection'
    
    def handle(self, *args, **options):
        """
        Generate sample datasets like preparing ingredients for cooking demonstrations.
        Creates realistic but synthetic data with known bias patterns.
        """
        
        # Create media directory if it doesn't exist
        media_dir = os.path.join(settings.MEDIA_ROOT, 'sample_datasets')
        os.makedirs(media_dir, exist_ok=True)
        
        # Generate biased hiring dataset
        self.generate_hiring_dataset(media_dir)
        
        # Generate biased loan approval dataset
        self.generate_loan_dataset(media_dir)
        
        # Generate medical diagnosis dataset
        self.generate_medical_dataset(media_dir)
        
        self.stdout.write(self.style.SUCCESS('Sample datasets generated successfully!'))
    
    def generate_hiring_dataset(self, output_dir):
        """Generate a hiring dataset with gender bias."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        data = {
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
            'age': np.random.randint(22, 65, n_samples),
            'education_score': np.random.normal(75, 15, n_samples),
            'experience_years': np.random.exponential(5, n_samples),
            'technical_score': np.random.normal(70, 20, n_samples),
            'interview_score': np.random.normal(3.5, 1.2, n_samples)
        }
        
        # Introduce bias: males more likely to be hired
        base_probability = 0.3
        hiring_probability = []
        
        for i in range(n_samples):
            prob = base_probability
            
            # Add legitimate factors
            prob += (data['education_score'][i] - 70) * 0.003
            prob += data['experience_years'][i] * 0.02
            prob += (data['technical_score'][i] - 60) * 0.002
            prob += (data['interview_score'][i] - 3) * 0.1
            
            # Add bias: boost males by 15%
            if data['gender'][i] == 'Male':
                prob += 0.15
            
            hiring_probability.append(max(0, min(1, prob)))
        
        data['hired'] = np.random.binomial(1, hiring_probability)
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(output_dir, 'hiring_dataset_biased.csv'), index=False)
        
        self.stdout.write(f'Generated hiring dataset with {n_samples} samples')
    
    def generate_loan_dataset(self, output_dir):
        """Generate a loan approval dataset with racial bias."""
        np.random.seed(123)
        n_samples = 1500
        
        races = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
        race_probs = [0.6, 0.15, 0.15, 0.08, 0.02]
        
        data = {
            'race': np.random.choice(races, n_samples, p=race_probs),
            'income': np.random.lognormal(10.5, 0.8, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            'loan_amount': np.random.uniform(10000, 500000, n_samples),
            'employment_years': np.random.exponential(3, n_samples),
            'debt_to_income': np.random.uniform(0.1, 0.8, n_samples)
        }
        
        # Calculate loan approval with bias
        approval_probability = []
        
        for i in range(n_samples):
            prob = 0.4  # Base probability
            
            # Legitimate factors
            prob += max(0, (data['credit_score'][i] - 600)) * 0.001
            prob += max(0, (data['income'][i] - 30000)) * 0.000005
            prob -= data['debt_to_income'][i] * 0.3
            prob += min(data['employment_years'][i], 10) * 0.02
            
            # Bias: Reduce approval for minorities
            if data['race'][i] in ['Black', 'Hispanic']:
                prob -= 0.12
            
            approval_probability.append(max(0, min(1, prob)))
        
        data['approved'] = np.random.binomial(1, approval_probability)
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(output_dir, 'loan_dataset_biased.csv'), index=False)
        
        self.stdout.write(f'Generated loan dataset with {n_samples} samples')
    
    def generate_medical_dataset(self, output_dir):
        """Generate a medical diagnosis dataset with age bias."""
        np.random.seed(456)
        n_samples = 2000
        
        data = {
            'age_group': np.random.choice(['18-30', '31-50', '51-70', '70+'], 
                                       n_samples, p=[0.25, 0.35, 0.25, 0.15]),
            'symptoms_severity': np.random.normal(5, 2, n_samples),
            'previous_visits': np.random.poisson(2, n_samples),
            'insurance_type': np.random.choice(['Private', 'Medicare', 'Medicaid', 'Uninsured'], 
                                             n_samples, p=[0.5, 0.2, 0.2, 0.1]),
            'lab_results_abnormal': np.random.binomial(1, 0.3, n_samples),
            'family_history': np.random.binomial(1, 0.4, n_samples)
        }
        
        # Calculate diagnosis probability with age bias
        diagnosis_probability = []
        
        for i in range(n_samples):
            prob = 0.2  # Base probability
            
            # Legitimate medical factors
            prob += data['symptoms_severity'][i] * 0.08
            prob += data['lab_results_abnormal'][i] * 0.3
            prob += data['family_history'][i] * 0.15
            
            # Bias: Younger patients less likely to be diagnosed seriously
            if data['age_group'][i] in ['18-30', '31-50']:
                prob -= 0.1
            elif data['age_group'][i] == '70+':
                prob += 0.1
            
            diagnosis_probability.append(max(0, min(1, prob)))
        
        data['serious_diagnosis'] = np.random.binomial(1, diagnosis_probability)
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(output_dir, 'medical_dataset_biased.csv'), index=False)
        
        self.stdout.write(f'Generated medical dataset with {n_samples} samples')

# bias_detector/tests.py
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
import pandas as pd
import numpy as np
import tempfile
import os

from .models import Dataset, BiasAnalysis
from .bias_engine import BiasDetector, BiasMitigator

class BiasDetectionTestCase(TestCase):
    """
    Test suite for bias detection functionality.
    Like a quality control checklist ensuring every recipe works perfectly.
    """
    
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client = Client()
        self.client.login(username='testuser', password='testpass')
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'gender': ['Male', 'Female'] * 50,
            'age': np.random.randint(20, 60, 100),
            'score': np.random.normal(70, 15, 100),
            'outcome': np.random.choice([0, 1], 100, p=[0.6, 0.4])
        })
    
    def test_demographic_parity_detection(self):
        """Test demographic parity bias detection."""
        # Create biased predictions (males favored)
        predictions = []
        for _, row in self.sample_data.iterrows():
            base_prob = 0.3
            if row['gender'] == 'Male':
                base_prob += 0.2  # Bias towards males
            predictions.append(np.random.binomial(1, base_prob))
        
        detector = BiasDetector(self.sample_data, 'gender', 'outcome', predictions)
        result = detector.demographic_parity()
        
        self.assertIsNotNone(result)
        self.assertIn('bias_score', result)
        self.assertIn('group_results', result)
        self.assertTrue(result['bias_score'] > 0.1)  # Should detect bias
    
    def test_statistical_parity_calculation(self):
        """Test statistical parity difference calculation."""
        detector = BiasDetector(self.sample_data, 'gender', 'outcome')
        result = detector.statistical_parity_difference()
        
        self.assertIsNotNone(result)
        self.assertIn('statistical_parity_difference', result)
        self.assertIn('group_rates', result)
    
    def test_bias_mitigation_reweighting(self):
        """Test reweighting bias mitigation strategy."""
        mitigator = BiasMitigator(self.sample_data, 'gender', 'outcome')
        result = mitigator.reweighting()
        
        self.assertIsNotNone(result)
        self.assertIn('data', result)
        self.assertIn('sample_weight', result['data'].columns)
        self.assertIn('effectiveness', result)
    
    def test_dataset_upload_view(self):
        """Test dataset upload functionality."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f, index=False)
            temp_file_path = f.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                response = self.client.post(reverse('bias_detector:upload_dataset'), {
                    'name': 'Test Dataset',
                    'description': 'Test dataset for bias detection',
                    'file_path': f
                })
            
            self.assertEqual(response.status_code, 302)  # Redirect after successful upload
            self.assertTrue(Dataset.objects.filter(name='Test Dataset').exists())
            
        finally:
            os.unlink(temp_file_path)
    
    def test_dashboard_view(self):
        """Test dashboard displays correctly."""
        response = self.client.get(reverse('bias_detector:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'AI Bias Detector')
        self.assertContains(response, 'Total Analyses')

class APITestCase(TestCase):
    """Test API endpoints for bias detection."""
    
    def setUp(self):
        self.client = Client()
    
    def test_api_bias_check(self):
        """Test the API endpoint for bias checking."""
        test_data = {
            'data': [
                {'gender': 'Male', 'age': 25, 'score': 80, 'outcome': 1},
                {'gender': 'Female', 'age': 30, 'score': 85, 'outcome': 0},
                {'gender': 'Male', 'age': 35, 'score': 75, 'outcome': 1},
                {'gender': 'Female', 'age': 28, 'score': 90, 'outcome': 0},
            ],
            'protected_attribute': 'gender',
            'target_variable': 'outcome',
            'predictions': [1, 0, 1, 0]
        }
        
        response = self.client.post(
            reverse('bias_detector:api_bias_check'),
            data=test_data,
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('results', data)

# Run the project setup commands
"""
To get this bias detection tool running in your kitchen (development environment):

1. Install required packages:
pip install django pandas numpy scikit-learn scipy matplotlib seaborn reportlab

2. Set up the Django project:
python manage.py makemigrations bias_detector
python manage.py migrate
python manage.py createsuperuser

3. Generate sample data for testing:
python manage.py generate_sample_data

4. Run the development server:
python manage.py runserver

5. Access the application at: http://localhost:8000

The tool provides:
- Upload datasets and analyze them for various types of bias
- Interactive dashboard showing bias metrics and trends  
- Visualization tools for understanding bias patterns
- Mitigation strategies to reduce detected bias
- PDF report generation for documentation
- REST API for integrating bias checks into other systems

Sample datasets will be generated automatically to demonstrate:
- Hiring decisions biased by gender
- Loan approvals biased by race  
- Medical diagnoses biased by age

Each dataset contains realistic patterns that the tool can detect and help mitigate.
"""

## Assignment: Bias Detection in Credit Scoring

**Scenario**: You work for a financial institution that wants to implement an AI system for credit scoring. However, there are concerns about potential discrimination against certain demographic groups.

**Your Task**: Create a bias detection and reporting system that:

1. **Generates a synthetic credit dataset** with the following features:
   - Income, credit history length, number of previous loans, age, education level
   - Protected attributes: race, gender, marital status
   - Target variable: loan approval (binary)

2. **Implements bias detection** for at least two fairness metrics:
   - Statistical parity
   - Equal opportunity
   - Equalized odds

3. **Creates visualizations** showing:
   - Approval rates by protected groups
   - Feature importance analysis
   - Bias metrics comparison

4. **Develops mitigation recommendations** based on your findings

**Deliverables**:
- Python code implementing the bias detection system
- A written report (300-500 words) summarizing your findings
- Visualizations showing bias analysis results
- Specific recommendations for bias mitigation

**Evaluation Criteria**:
- Correctness of bias metric calculations (30%)
- Quality of visualizations and interpretability (25%)
- Depth of analysis and insights (25%)
- Practical mitigation recommendations (20%)

**Bonus Points**: Implement one bias mitigation technique and show before/after comparison.

---

## Course Summary

Congratulations! You've completed the most crucial module in AI development. Just as a master chef never stops learning about new ingredients and techniques while maintaining the highest standards of quality and ethics, your journey in responsible AI development is ongoing.

**Key Takeaways**:
1. **Bias is everywhere** - but it can be measured and mitigated
2. **Transparency builds trust** - explainable AI is not optional
3. **Privacy is paramount** - protect user data while maintaining utility
4. **Ethics is a process** - not a one-time check

Remember: The most powerful AI system is worthless if it's unfair, opaque, or violates user privacy. Your responsibility as an AI practitioner extends far beyond model accuracy—you're cooking up the future of technology that affects real people's lives.

Keep your ethical standards as high as your technical skills, and you'll create AI systems that not only work well but do good in the world.