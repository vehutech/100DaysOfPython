# Day 71: Statistics & Probability for AI
**A Complete Course with Kitchen Chef Analogies**

## Learning Objectives
By the end of this course, you will be able to:
- Apply descriptive and inferential statistics to analyze data patterns
- Work with common probability distributions in AI contexts
- Implement Bayes' theorem for probabilistic reasoning
- Conduct hypothesis testing to validate AI model assumptions
- Build practical statistical analysis tools using Python

---

## Imagine That...

Imagine that you're the head chef of a revolutionary AI restaurant called "Data Bistro." Your kitchen isn't filled with traditional ingredients—instead, you work with numbers, probabilities, and statistical insights. Just as a master chef needs to understand flavors, temperatures, and cooking techniques, an AI practitioner must master statistics and probability to create the perfect "dishes" (models and insights) from raw data "ingredients."

In your statistical kitchen, you'll learn to:
- **Taste-test your data** using descriptive statistics
- **Follow probability recipes** that predict outcomes
- **Season with Bayes' theorem** to update beliefs with new evidence
- **Quality-control your dishes** through hypothesis testing

Let's put on our chef's hat and start cooking with data!

---

## 1. Descriptive and Inferential Statistics

### The Foundation of Your Statistical Kitchen

Just as every great chef starts by understanding their ingredients, we begin with descriptive statistics—the basic "tasting" of our data.

**Descriptive Statistics: Your Data Tasting Spoons**

Think of descriptive statistics as your collection of tasting spoons. Each spoon gives you a different "taste" of your data:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Let's imagine we're analyzing customer satisfaction scores (1-10) 
# at our AI restaurant over 100 days
np.random.seed(42)
satisfaction_scores = np.random.normal(7.5, 1.2, 100)

# Basic descriptive statistics - our "tasting spoons"
print("=== Data Bistro Customer Satisfaction Analysis ===")
print(f"Mean (Average taste): {np.mean(satisfaction_scores):.2f}")
print(f"Median (Middle taste): {np.median(satisfaction_scores):.2f}")
print(f"Mode (Most common taste): {stats.mode(satisfaction_scores.round())[0][0]:.0f}")
print(f"Standard Deviation (Flavor consistency): {np.std(satisfaction_scores):.2f}")
print(f"Variance (Flavor spread): {np.var(satisfaction_scores):.2f}")
```

**Code Syntax Explanation:**
- `np.random.seed(42)`: Sets a random seed for reproducible results
- `np.random.normal(7.5, 1.2, 100)`: Generates 100 random numbers from a normal distribution with mean=7.5, std=1.2
- `stats.mode()`: Returns the most frequently occurring value
- `:.2f`: Formats numbers to 2 decimal places

**Inferential Statistics: From Sample to Population**

Now imagine you can only taste a small portion of a massive pot of soup, but you need to determine if the entire pot is seasoned correctly. That's inferential statistics!

```python
# Confidence intervals - "How confident are we about the whole pot?"
from scipy.stats import t

# Calculate 95% confidence interval for customer satisfaction
n = len(satisfaction_scores)
mean = np.mean(satisfaction_scores)
std_err = stats.sem(satisfaction_scores)  # Standard error of mean
confidence_level = 0.95
alpha = 1 - confidence_level
t_critical = t.ppf(1 - alpha/2, n-1)

margin_error = t_critical * std_err
ci_lower = mean - margin_error
ci_upper = mean + margin_error

print(f"\n=== Chef's Confidence Report ===")
print(f"We're 95% confident that our true average satisfaction is between")
print(f"{ci_lower:.2f} and {ci_upper:.2f}")
```

**Code Syntax Explanation:**
- `stats.sem()`: Calculates standard error of the mean
- `t.ppf()`: Percent point function (inverse of CDF) for t-distribution
- `alpha/2`: Two-tailed test, so we split alpha on both sides

---

## 2. Probability Distributions

### The Recipe Books of Chance

Think of probability distributions as your collection of recipe books. Each distribution is a different type of recipe that tells you how likely different outcomes are—just like how a recipe tells you the proportions of ingredients.

**Normal Distribution: The Classic Soufflé Recipe**

The normal distribution is like a perfect soufflé—most of the "rise" happens in the middle, with gentle slopes on either side.

```python
# Normal Distribution - The classic bell curve
x = np.linspace(-4, 4, 100)
normal_pdf = stats.norm.pdf(x, 0, 1)  # mean=0, std=1

plt.figure(figsize=(12, 8))

# Plot 1: Normal Distribution
plt.subplot(2, 2, 1)
plt.plot(x, normal_pdf, 'b-', linewidth=2, label='Normal Distribution')
plt.title('Normal Distribution: The Perfect Soufflé')
plt.xlabel('Standard Deviations from Mean')
plt.ylabel('Probability Density')
plt.grid(True, alpha=0.3)
plt.legend()

# Binomial Distribution - The Coin Flip Bread
n_trials = 20
p_success = 0.3
x_binomial = np.arange(0, n_trials + 1)
binomial_pmf = stats.binom.pmf(x_binomial, n_trials, p_success)

plt.subplot(2, 2, 2)
plt.bar(x_binomial, binomial_pmf, alpha=0.7, color='green')
plt.title('Binomial Distribution: Coin Flip Bread (20 flips, p=0.3)')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)

# Poisson Distribution - The Rush Hour Orders
lambda_rate = 3
x_poisson = np.arange(0, 15)
poisson_pmf = stats.poisson.pmf(x_poisson, lambda_rate)

plt.subplot(2, 2, 3)
plt.bar(x_poisson, poisson_pmf, alpha=0.7, color='orange')
plt.title('Poisson Distribution: Rush Hour Orders (λ=3)')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)

# Exponential Distribution - The Waiting Time Pasta
lambda_exp = 1.5
x_exp = np.linspace(0, 5, 100)
exp_pdf = stats.expon.pdf(x_exp, scale=1/lambda_exp)

plt.subplot(2, 2, 4)
plt.plot(x_exp, exp_pdf, 'r-', linewidth=2)
plt.title('Exponential Distribution: Waiting Time Pasta')
plt.xlabel('Time (minutes)')
plt.ylabel('Probability Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Practical example: What's the probability of getting exactly 5 satisfied customers 
# out of 10, if our satisfaction rate is 70%?
prob_exactly_5 = stats.binom.pmf(5, 10, 0.7)
print(f"\nProbability of exactly 5 satisfied customers out of 10: {prob_exactly_5:.4f}")
```

**Code Syntax Explanation:**
- `stats.norm.pdf()`: Probability density function for normal distribution
- `stats.binom.pmf()`: Probability mass function for binomial distribution
- `np.linspace(start, stop, num)`: Creates evenly spaced numbers
- `plt.subplot(rows, cols, index)`: Creates subplots in a grid

---

## 3. Bayes' Theorem and Applications

### The Master Chef's Secret Ingredient

Bayes' theorem is like the master chef's secret ingredient—it allows you to update your "recipe knowledge" based on new evidence. It's the mathematical way of saying, "Given what I just observed, how should I adjust my beliefs?"

**The Bayes' Recipe:**
P(A|B) = P(B|A) × P(A) / P(B)

Think of it as: "How likely is hypothesis A, given evidence B?"

```python
# Bayes' Theorem in Action: The Food Safety Inspector
# Scenario: A food safety test for detecting contamination

def bayes_theorem(prior, likelihood, evidence):
    """
    Calculate posterior probability using Bayes' theorem
    
    Args:
        prior: P(A) - Prior probability of hypothesis
        likelihood: P(B|A) - Likelihood of evidence given hypothesis
        evidence: P(B) - Total probability of evidence
    
    Returns:
        posterior: P(A|B) - Updated probability of hypothesis given evidence
    """
    posterior = (likelihood * prior) / evidence
    return posterior

# Kitchen contamination scenario
print("=== Bayes' Kitchen Safety Analysis ===")
print("Scenario: Food safety test results")

# Prior probabilities
prob_contaminated = 0.02  # 2% of food samples are actually contaminated
prob_clean = 1 - prob_contaminated

# Test characteristics (like a chef's palate sensitivity)
test_sensitivity = 0.95  # Test correctly identifies 95% of contaminated samples
test_specificity = 0.90  # Test correctly identifies 90% of clean samples

# Likelihoods
prob_positive_given_contaminated = test_sensitivity
prob_positive_given_clean = 1 - test_specificity

# Total probability of positive test (evidence)
prob_positive = (prob_positive_given_contaminated * prob_contaminated + 
                prob_positive_given_clean * prob_clean)

# Apply Bayes' theorem
prob_contaminated_given_positive = bayes_theorem(
    prior=prob_contaminated,
    likelihood=prob_positive_given_contaminated,
    evidence=prob_positive
)

print(f"Prior probability of contamination: {prob_contaminated:.1%}")
print(f"Test sensitivity: {test_sensitivity:.1%}")
print(f"Test specificity: {test_specificity:.1%}")
print(f"Probability of contamination given positive test: {prob_contaminated_given_positive:.1%}")

# Practical application: Email spam detection
print(f"\n=== Bayes' Email Filter (Spam Detection) ===")

def naive_bayes_spam_detector(word_frequencies, spam_prob_prior=0.3):
    """
    Simple Naive Bayes spam detector
    """
    # Example: probability of words appearing in spam vs ham
    word_probs = {
        'free': {'spam': 0.8, 'ham': 0.1},
        'money': {'spam': 0.7, 'ham': 0.05},
        'meeting': {'spam': 0.1, 'ham': 0.6},
        'urgent': {'spam': 0.6, 'ham': 0.2}
    }
    
    spam_score = spam_prob_prior
    ham_score = 1 - spam_prob_prior
    
    for word in word_frequencies:
        if word in word_probs:
            spam_score *= word_probs[word]['spam']
            ham_score *= word_probs[word]['ham']
    
    # Normalize
    total = spam_score + ham_score
    return spam_score / total

# Test email: "Free money for urgent meeting"
test_email = ['free', 'money', 'urgent', 'meeting']
spam_probability = naive_bayes_spam_detector(test_email)

print(f"Email content: {' '.join(test_email)}")
print(f"Spam probability: {spam_probability:.1%}")
```

**Code Syntax Explanation:**
- Function parameters with type hints improve code readability
- Dictionary comprehension `{'spam': 0.8, 'ham': 0.1}` creates nested data structures
- `:.1%` formats as percentage with 1 decimal place

---

## 4. Hypothesis Testing

### Quality Control in the Statistical Kitchen

Hypothesis testing is like being a quality control inspector in your kitchen. You have a theory (hypothesis) about your food, and you need to test whether the evidence supports or refutes your theory.

```python
# Hypothesis Testing: The Great Recipe Debate
print("=== The Great Recipe Debate: Hypothesis Testing ===")

# Scenario: Does our new AI-optimized recipe increase customer satisfaction?
# H0 (Null): New recipe has same satisfaction as old (μ = 7.0)
# H1 (Alternative): New recipe has higher satisfaction (μ > 7.0)

# Old recipe data (historical)
old_recipe_satisfaction = np.random.normal(7.0, 1.5, 50)

# New recipe data (test period)
np.random.seed(123)
new_recipe_satisfaction = np.random.normal(7.8, 1.4, 50)

print(f"Old recipe average: {np.mean(old_recipe_satisfaction):.2f}")
print(f"New recipe average: {np.mean(new_recipe_satisfaction):.2f}")

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(new_recipe_satisfaction, old_recipe_satisfaction)

alpha = 0.05  # Significance level (5% chance of Type I error)

print(f"\n=== Statistical Test Results ===")
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significance level (α): {alpha}")

if p_value < alpha:
    print("🎉 VERDICT: Reject null hypothesis!")
    print("The new recipe significantly improves customer satisfaction!")
else:
    print("🤔 VERDICT: Fail to reject null hypothesis")
    print("No significant difference detected between recipes")

# Effect size (Cohen's d) - How big is the practical difference?
pooled_std = np.sqrt(((len(new_recipe_satisfaction)-1) * np.var(new_recipe_satisfaction, ddof=1) + 
                      (len(old_recipe_satisfaction)-1) * np.var(old_recipe_satisfaction, ddof=1)) /
                     (len(new_recipe_satisfaction) + len(old_recipe_satisfaction) - 2))

cohens_d = (np.mean(new_recipe_satisfaction) - np.mean(old_recipe_satisfaction)) / pooled_std

print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect_interpretation = "Small effect"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "Medium effect"
else:
    effect_interpretation = "Large effect"

print(f"Practical significance: {effect_interpretation}")

# Confidence interval for the difference
diff_mean = np.mean(new_recipe_satisfaction) - np.mean(old_recipe_satisfaction)
n1, n2 = len(new_recipe_satisfaction), len(old_recipe_satisfaction)
pooled_se = pooled_std * np.sqrt(1/n1 + 1/n2)
df = n1 + n2 - 2
t_critical = stats.t.ppf(0.975, df)  # 95% CI
margin_error = t_critical * pooled_se

ci_lower = diff_mean - margin_error
ci_upper = diff_mean + margin_error

print(f"\n95% Confidence Interval for difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

**Code Syntax Explanation:**
- `stats.ttest_ind()`: Independent t-test for comparing two groups
- `ddof=1`: Delta degrees of freedom for sample variance calculation
- `np.sqrt()`: Square root function
- `stats.t.ppf(0.975, df)`: 97.5th percentile of t-distribution (for 95% CI)

---

## Final Quality Project: AI Restaurant Analytics Dashboard

Now let's combine all our statistical cooking skills to create a comprehensive analytics dashboard for our AI restaurant!

```python
# Final Project: AI Restaurant Analytics Dashboard
import warnings
warnings.filterwarnings('ignore')

class RestaurantAnalytics:
    """
    A comprehensive statistical analysis toolkit for restaurant data
    """
    
    def __init__(self, name="Data Bistro"):
        self.restaurant_name = name
        self.data = {}
        
    def generate_sample_data(self, days=365):
        """Generate realistic restaurant data"""
        np.random.seed(42)
        
        # Customer satisfaction scores
        self.data['satisfaction'] = np.random.normal(7.5, 1.2, days)
        
        # Daily revenue (in thousands)
        base_revenue = 5.0
        seasonal_effect = np.sin(np.arange(days) * 2 * np.pi / 365) * 0.5
        self.data['revenue'] = np.random.normal(base_revenue + seasonal_effect, 1.0, days)
        
        # Number of customers per day
        self.data['customers'] = np.random.poisson(150, days)
        
        # Wait times (minutes)
        self.data['wait_times'] = np.random.exponential(12, days)
        
        # Food safety scores (0-100)
        self.data['safety_scores'] = np.random.beta(8, 1, days) * 100
        
        print(f"✅ Generated {days} days of data for {self.restaurant_name}")
    
    def descriptive_analysis(self):
        """Perform comprehensive descriptive analysis"""
        print(f"\n{'='*50}")
        print(f"📊 DESCRIPTIVE ANALYTICS REPORT")
        print(f"{'='*50}")
        
        for metric, values in self.data.items():
            print(f"\n🔍 {metric.upper().replace('_', ' ')}:")
            print(f"   Mean: {np.mean(values):.2f}")
            print(f"   Median: {np.median(values):.2f}")
            print(f"   Std Dev: {np.std(values):.2f}")
            print(f"   Range: [{np.min(values):.2f}, {np.max(values):.2f}]")
            
            # Percentiles
            p25, p75 = np.percentile(values, [25, 75])
            print(f"   IQR: [{p25:.2f}, {p75:.2f}]")
    
    def probability_analysis(self):
        """Analyze probability distributions"""
        print(f"\n{'='*50}")
        print(f"🎲 PROBABILITY ANALYSIS")
        print(f"{'='*50}")
        
        # Customer satisfaction probability
        satisfaction = self.data['satisfaction']
        prob_highly_satisfied = np.mean(satisfaction >= 8.0)
        print(f"\n📈 Customer Satisfaction Probabilities:")
        print(f"   P(Satisfaction ≥ 8.0): {prob_highly_satisfied:.1%}")
        
        # Revenue probability using normal distribution
        revenue = self.data['revenue']
        revenue_mean, revenue_std = np.mean(revenue), np.std(revenue)
        prob_high_revenue = 1 - stats.norm.cdf(6.0, revenue_mean, revenue_std)
        print(f"\n💰 Revenue Probabilities:")
        print(f"   P(Daily Revenue > $6k): {prob_high_revenue:.1%}")
        
        # Wait time probability using exponential distribution
        wait_times = self.data['wait_times']
        lambda_est = 1 / np.mean(wait_times)
        prob_quick_service = stats.expon.cdf(10, scale=1/lambda_est)
        print(f"\n⏱️ Service Time Probabilities:")
        print(f"   P(Wait Time ≤ 10 min): {prob_quick_service:.1%}")
    
    def bayes_analysis(self):
        """Apply Bayes' theorem for business insights"""
        print(f"\n{'='*50}")
        print(f"🧠 BAYESIAN ANALYSIS")
        print(f"{'='*50}")
        
        # Customer retention analysis
        satisfaction = self.data['satisfaction']
        revenue = self.data['revenue']
        
        # Prior: probability of high revenue day
        high_revenue_days = revenue >= np.percentile(revenue, 75)
        prior_high_revenue = np.mean(high_revenue_days)
        
        # Likelihood: satisfaction given high revenue
        high_satisfaction_given_high_revenue = np.mean(
            satisfaction[high_revenue_days] >= 8.0
        )
        
        # Evidence: overall high satisfaction probability
        high_satisfaction_overall = np.mean(satisfaction >= 8.0)
        
        # Posterior: probability of high revenue given high satisfaction
        if high_satisfaction_overall > 0:
            posterior = (high_satisfaction_given_high_revenue * prior_high_revenue) / high_satisfaction_overall
        else:
            posterior = 0
            
        print(f"\n🎯 Customer Satisfaction → Revenue Prediction:")
        print(f"   Prior P(High Revenue): {prior_high_revenue:.1%}")
        print(f"   P(High Satisfaction | High Revenue): {high_satisfaction_given_high_revenue:.1%}")
        print(f"   Posterior P(High Revenue | High Satisfaction): {posterior:.1%}")
    
    def hypothesis_testing(self):
        """Perform hypothesis testing"""
        print(f"\n{'='*50}")
        print(f"🔬 HYPOTHESIS TESTING")
        print(f"{'='*50}")
        
        # Test if weekend revenue differs from weekday revenue
        days = len(self.data['revenue'])
        weekend_mask = np.array([(i % 7) >= 5 for i in range(days)])  # Sat, Sun
        
        weekend_revenue = self.data['revenue'][weekend_mask]
        weekday_revenue = self.data['revenue'][~weekend_mask]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(weekend_revenue, weekday_revenue)
        
        print(f"\n📅 Weekend vs Weekday Revenue Analysis:")
        print(f"   Weekend avg: ${np.mean(weekend_revenue):.2f}k")
        print(f"   Weekday avg: ${np.mean(weekday_revenue):.2f}k")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_value:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            print(f"   ✅ Significant difference detected (p < {alpha})")
        else:
            print(f"   ❌ No significant difference (p ≥ {alpha})")
    
    def generate_insights(self):
        """Generate business insights"""
        print(f"\n{'='*50}")
        print(f"💡 BUSINESS INSIGHTS & RECOMMENDATIONS")
        print(f"{'='*50}")
        
        satisfaction_avg = np.mean(self.data['satisfaction'])
        revenue_avg = np.mean(self.data['revenue'])
        wait_time_avg = np.mean(self.data['wait_times'])
        safety_avg = np.mean(self.data['safety_scores'])
        
        insights = []
        
        if satisfaction_avg >= 8.0:
            insights.append("🌟 Excellent customer satisfaction! Consider expanding operations.")
        elif satisfaction_avg >= 7.0:
            insights.append("👍 Good satisfaction levels. Focus on consistency.")
        else:
            insights.append("⚠️ Customer satisfaction needs improvement. Review operations.")
            
        if wait_time_avg <= 10:
            insights.append("⚡ Excellent service speed!")
        elif wait_time_avg <= 15:
            insights.append("⏱️ Good service times. Monitor during peak hours.")
        else:
            insights.append("🐌 Service times need improvement. Consider staffing changes.")
            
        if safety_avg >= 90:
            insights.append("🛡️ Outstanding food safety standards!")
        else:
            insights.append("🔍 Review food safety protocols.")
        
        print("\n📋 Key Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
    
    def full_analysis(self):
        """Run complete statistical analysis"""
        print(f"🏪 Welcome to {self.restaurant_name} Statistical Analysis!")
        
        self.generate_sample_data()
        self.descriptive_analysis()
        self.probability_analysis()
        self.bayes_analysis()
        self.hypothesis_testing()
        self.generate_insights()
        
        print(f"\n{'='*50}")
        print(f"📊 Analysis Complete! Check your dashboard above.")
        print(f"{'='*50}")

# Run the complete analysis
restaurant = RestaurantAnalytics("Data Bistro AI")
restaurant.full_analysis()
```

**Code Syntax Explanation:**
- **Class Definition**: `class RestaurantAnalytics:` creates a reusable data analysis object
- **Constructor**: `__init__(self, name="Data Bistro")` initializes the class with default parameters
- **Instance Variables**: `self.data = {}` stores data within the class instance
- **Method Chaining**: Methods can access `self.data` to work with the same dataset
- **Boolean Indexing**: `revenue[high_revenue_days]` filters data based on conditions
- **String Formatting**: `f"{'='*50}"` creates repeated characters for formatting
- **List Comprehension**: `[(i % 7) >= 5 for i in range(days)]` creates boolean masks efficiently

---

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chef's Statistical Kitchen - AI Analysis Tool</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .kitchen-container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }

        .chef-header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            border-radius: 15px;
            color: white;
        }

        .chef-header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .chef-header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .cooking-stations {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .station {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border: 2px solid #e0e0e0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .station:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        }

        .station h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .station-icon {
            font-size: 1.5em;
        }

        textarea, input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
            margin-bottom: 10px;
            transition: border-color 0.3s ease;
        }

        textarea:focus, input:focus, select:focus {
            outline: none;
            border-color: #4ecdc4;
            box-shadow: 0 0 0 3px rgba(78, 205, 196, 0.1);
        }

        button {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin: 5px;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        }

        button:active {
            transform: translateY(0);
        }

        #results {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            border-left: 5px solid #4ecdc4;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }

        .recipe-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }

        .ingredient-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }

        .ingredient {
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 14px;
            backdrop-filter: blur(5px);
        }

        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .visualization {
            width: 100%;
            height: 300px;
            border: 2px dashed #ddd;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-style: italic;
            border-radius: 8px;
            background: linear-gradient(45deg, #f0f0f0 25%, transparent 25%), 
                        linear-gradient(-45deg, #f0f0f0 25%, transparent 25%), 
                        linear-gradient(45deg, transparent 75%, #f0f0f0 75%), 
                        linear-gradient(-45deg, transparent 75%, #f0f0f0 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        }

        .stat-card {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            color: #333;
            font-weight: bold;
        }

        .code-snippet {
            background: #2d3748;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            margin: 10px 0;
            overflow-x: auto;
        }

        .tooltip {
            position: relative;
            cursor: help;
        }

        .tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div class="kitchen-container">
        <div class="chef-header">
            <h1>🧑‍🍳 Chef's Statistical Kitchen</h1>
            <p>Where Data Becomes Delicious Insights - Your AI Analysis Sous Chef</p>
        </div>

        <div class="recipe-section">
            <h3>🍳 Today's Statistical Recipe</h3>
            <p>Like a master chef combining ingredients to create the perfect dish, we'll blend statistical methods to extract meaningful insights from raw data. Each statistical technique is like a cooking method - some data needs gentle simmering (descriptive stats), while others require high heat analysis (hypothesis testing).</p>
            
            <div class="ingredient-list">
                <span class="ingredient">📊 Data Preparation</span>
                <span class="ingredient">📈 Descriptive Statistics</span>
                <span class="ingredient">🎲 Probability Analysis</span>
                <span class="ingredient">🔍 Hypothesis Testing</span>
                <span class="ingredient">📋 Bayesian Inference</span>
            </div>
        </div>

        <div class="cooking-stations">
            <div class="station">
                <h3><span class="station-icon">📝</span>Data Prep Station</h3>
                <p>Enter your raw data (comma-separated values):</p>
                <textarea id="dataInput" rows="4" placeholder="1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1">1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1</textarea>
                <button onclick="prepareData()">🔪 Prep Ingredients</button>
            </div>

            <div class="station">
                <h3><span class="station-icon">📊</span>Descriptive Cooking</h3>
                <p>Like tasting your dish, let's examine the basic flavors:</p>
                <button onclick="calculateDescriptive()">🍯 Calculate Basic Stats</button>
                <button onclick="generateHistogram()">📈 Create Flavor Profile</button>
            </div>

            <div class="station">
                <h3><span class="station-icon">🎲</span>Probability Oven</h3>
                <p>Test probability scenarios:</p>
                <input type="number" id="probValue" placeholder="Test value" step="0.1">
                <select id="distributionType">
                    <option value="normal">Normal Distribution</option>
                    <option value="uniform">Uniform Distribution</option>
                    <option value="binomial">Binomial Distribution</option>
                </select>
                <button onclick="calculateProbability()">🎯 Calculate Probability</button>
            </div>

            <div class="station">
                <h3><span class="station-icon">🧪</span>Hypothesis Laboratory</h3>
                <p>Test your statistical hypotheses:</p>
                <input type="number" id="hypothesisValue" placeholder="Hypothesis mean" step="0.1" value="5.0">
                <input type="number" id="alphaLevel" placeholder="Alpha level" step="0.01" value="0.05" min="0.01" max="0.1">
                <button onclick="performTTest()">🔬 T-Test Analysis</button>
            </div>

            <div class="station">
                <h3><span class="station-icon">🎯</span>Bayesian Bakery</h3>
                <p>Update beliefs with new evidence:</p>
                <input type="number" id="priorProb" placeholder="Prior probability" step="0.01" value="0.5" min="0" max="1">
                <input type="number" id="likelihood" placeholder="Likelihood" step="0.01" value="0.8" min="0" max="1">
                <input type="number" id="evidence" placeholder="Evidence" step="0.01" value="0.6" min="0" max="1">
                <button onclick="calculateBayes()">⚖️ Update Belief</button>
            </div>
        </div>

        <div class="chart-container">
            <h3>📈 Visual Kitchen Display</h3>
            <div id="visualization" class="visualization">
                🍳 Your statistical visualizations will appear here like a beautifully plated dish
            </div>
        </div>

        <div id="results">
🧑‍🍳 Chef's Analysis Results will appear here...

Welcome to your Statistical Kitchen! 
Like preparing a gourmet meal, statistical analysis requires:
1. Quality ingredients (clean data)
2. Proper techniques (statistical methods)  
3. Careful measurement (hypothesis testing)
4. Beautiful presentation (visualization)

Start by entering your data in the Data Prep Station! 👆
        </div>
    </div>

    <script>
        // Global variables - our kitchen's ingredient storage
        let currentData = [];
        let processedStats = {};

        // Data preparation - like washing and chopping vegetables
        function prepareData() {
            const input = document.getElementById('dataInput').value;
            
            try {
                // Parse the comma-separated values
                currentData = input.split(',').map(val => parseFloat(val.trim())).filter(val => !isNaN(val));
                
                if (currentData.length === 0) {
                    throw new Error("No valid numbers found");
                }

                document.getElementById('results').textContent = 
`🔪 Data Preparation Complete! 

Chef's Ingredient Analysis:
📋 Total data points: ${currentData.length}
🥕 Raw ingredients: [${currentData.slice(0, 10).join(', ')}${currentData.length > 10 ? '...' : ''}]
✨ Data type: ${typeof currentData[0]}
🧂 Ready for statistical seasoning!

Next: Try the Descriptive Cooking station to taste your data! 👨‍🍳`;

            } catch (error) {
                document.getElementById('results').textContent = 
`❌ Kitchen Error: ${error.message}
🍳 Chef's Tip: Make sure your data contains valid numbers separated by commas
Example: 1.2, 2.3, 3.4, 4.5, 5.6`;
            }
        }

        // Descriptive statistics - tasting the basic flavors
        function calculateDescriptive() {
            if (currentData.length === 0) {
                document.getElementById('results').textContent = "🚫 No ingredients prepared! Please use the Data Prep Station first.";
                return;
            }

            // Calculate statistics like a chef analyzing flavors
            const n = currentData.length;
            const sum = currentData.reduce((acc, val) => acc + val, 0);
            const mean = sum / n;
            
            const sortedData = [...currentData].sort((a, b) => a - b);
            const median = n % 2 === 0 
                ? (sortedData[n/2 - 1] + sortedData[n/2]) / 2 
                : sortedData[Math.floor(n/2)];
            
            const variance = currentData.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / (n - 1);
            const stdDev = Math.sqrt(variance);
            
            const min = Math.min(...currentData);
            const max = Math.max(...currentData);
            const range = max - min;

            // Store for later use
            processedStats = { mean, median, stdDev, min, max, variance, range, n };

            document.getElementById('results').textContent = 
`🍯 Descriptive Statistics - Chef's Flavor Profile:

📊 Basic Measurements:
   • Sample Size (Portions): ${n}
   • Mean (Average Taste): ${mean.toFixed(3)}
   • Median (Middle Flavor): ${median.toFixed(3)}
   • Standard Deviation (Flavor Spread): ${stdDev.toFixed(3)}
   • Variance (Flavor Consistency): ${variance.toFixed(3)}

🌡️ Temperature Range:
   • Minimum: ${min.toFixed(3)}
   • Maximum: ${max.toFixed(3)}
   • Range: ${range.toFixed(3)}

🧑‍🍳 Chef's Interpretation:
${mean > median ? "➡️ Slightly sweet-skewed (right-skewed distribution)" : 
  mean < median ? "⬅️ Slightly bitter-skewed (left-skewed distribution)" : 
  "⚖️ Perfectly balanced flavors (symmetric distribution)"}

Standard deviation tells us how consistent our flavors are:
${stdDev < range/4 ? "🎯 Very consistent flavors" : 
  stdDev > range/2 ? "🌪️ Highly variable flavors" : 
  "📊 Moderately variable flavors"}`;
        }

        // Probability calculations - predicting outcomes like recipe success
        function calculateProbability() {
            if (currentData.length === 0) {
                document.getElementById('results').textContent = "🚫 No ingredients prepared! Please use the Data Prep Station first.";
                return;
            }

            const testValue = parseFloat(document.getElementById('probValue').value);
            const distType = document.getElementById('distributionType').value;

            if (isNaN(testValue)) {
                document.getElementById('results').textContent = "❌ Please enter a valid test value!";
                return;
            }

            let probability = 0;
            let explanation = "";

            if (distType === 'normal') {
                // Normal distribution probability using our data's mean and std dev
                const mean = processedStats.mean || currentData.reduce((a, b) => a + b) / currentData.length;
                const stdDev = processedStats.stdDev || Math.sqrt(currentData.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / (currentData.length - 1));
                
                // Z-score calculation
                const zScore = (testValue - mean) / stdDev;
                probability = normalCDF(zScore);
                
                explanation = `Using Normal Distribution (Bell Curve):
🔔 Z-score: ${zScore.toFixed(3)}
📊 Probability of value ≤ ${testValue}: ${(probability * 100).toFixed(2)}%
📈 This means ${(probability * 100).toFixed(1)}% of our recipe batches would have this value or lower`;
                
            } else if (distType === 'uniform') {
                const min = Math.min(...currentData);
                const max = Math.max(...currentData);
                probability = testValue < min ? 0 : testValue > max ? 1 : (testValue - min) / (max - min);
                
                explanation = `Using Uniform Distribution (Equal chances):
📏 Range: ${min.toFixed(3)} to ${max.toFixed(3)}
🎲 Probability of value ≤ ${testValue}: ${(probability * 100).toFixed(2)}%
⚖️ Every value in our range has equal probability of occurring`;
            }

            document.getElementById('results').textContent = 
`🎯 Probability Analysis - Recipe Success Prediction:

${explanation}

🧑‍🍳 Chef's Probability Wisdom:
${probability < 0.1 ? "🔥 Rare occurrence - like a perfectly soufflé!" :
  probability < 0.3 ? "⭐ Uncommon but achievable - specialty dish territory" :
  probability < 0.7 ? "📊 Common occurrence - your everyday successful recipe" :
  "🍞 Very common - like perfectly baked bread, happens often!"}

💡 Probability is like predicting recipe success - we use past cooking data to forecast future results!`;
        }

        // Hypothesis testing - testing recipe theories
        function performTTest() {
            if (currentData.length === 0) {
                document.getElementById('results').textContent = "🚫 No ingredients prepared! Please use the Data Prep Station first.";
                return;
            }

            const hypothesisValue = parseFloat(document.getElementById('hypothesisValue').value);
            const alpha = parseFloat(document.getElementById('alphaLevel').value);

            if (isNaN(hypothesisValue) || isNaN(alpha)) {
                document.getElementById('results').textContent = "❌ Please enter valid hypothesis and alpha values!";
                return;
            }

            // One-sample t-test
            const n = currentData.length;
            const sampleMean = currentData.reduce((a, b) => a + b) / n;
            const sampleStd = Math.sqrt(currentData.reduce((acc, val) => acc + Math.pow(val - sampleMean, 2), 0) / (n - 1));
            const standardError = sampleStd / Math.sqrt(n);
            const tStatistic = (sampleMean - hypothesisValue) / standardError;
            const degreesOfFreedom = n - 1;
            
            // Critical t-value (approximation for two-tailed test)
            const tCritical = getTCritical(degreesOfFreedom, alpha);
            const pValue = 2 * (1 - tCDF(Math.abs(tStatistic), degreesOfFreedom));
            
            const isSignificant = pValue < alpha;
            const decision = isSignificant ? "REJECT" : "FAIL TO REJECT";

            document.getElementById('results').textContent = 
`🔬 Hypothesis Test Results - Recipe Theory Investigation:

🧪 Testing the Theory:
   H₀ (Null): Population mean = ${hypothesisValue} (Status quo recipe)
   H₁ (Alt): Population mean ≠ ${hypothesisValue} (New recipe is different)

📊 Statistical Evidence:
   • Sample Mean: ${sampleMean.toFixed(4)}
   • Sample Std Dev: ${sampleStd.toFixed(4)}
   • Standard Error: ${standardError.toFixed(4)}
   • T-Statistic: ${tStatistic.toFixed(4)}
   • Degrees of Freedom: ${degreesOfFreedom}
   • P-Value: ${pValue.toFixed(6)}
   • Alpha Level: ${alpha}

⚖️ Chef's Verdict: ${decision} the null hypothesis

🧑‍🍳 Kitchen Translation:
${isSignificant ? 
  `🔥 SIGNIFICANT DIFFERENCE! Your recipe IS different from the claimed value!
  📈 The evidence is strong enough (p = ${pValue.toFixed(4)} < ${alpha}) to conclude your cooking method produces different results.
  🎯 Like tasting two dishes and definitively knowing they're different recipes!` :
  
  `📊 NO SIGNIFICANT DIFFERENCE. Not enough evidence that your recipe differs from the claimed value.
  🤷‍♂️ The evidence isn't strong enough (p = ${pValue.toFixed(4)} > ${alpha}) to prove a difference.
  🧂 Like tasting two dishes that might be slightly different, but you can't be sure.`}

💡 P-value interpretation: If there truly was no difference, we'd see results this extreme or more ${(pValue * 100).toFixed(2)}% of the time by random chance alone.`;
        }

        // Bayesian inference - updating recipe beliefs with new evidence
        function calculateBayes() {
            const prior = parseFloat(document.getElementById('priorProb').value);
            const likelihood = parseFloat(document.getElementById('likelihood').value);
            const evidence = parseFloat(document.getElementById('evidence').value);

            if (isNaN(prior) || isNaN(likelihood) || isNaN(evidence)) {
                document.getElementById('results').textContent = "❌ Please enter valid probability values (0-1)!";
                return;
            }

            if (prior < 0 || prior > 1 || likelihood < 0 || likelihood > 1 || evidence < 0 || evidence > 1) {
                document.getElementById('results').textContent = "❌ Probabilities must be between 0 and 1!";
                return;
            }

            if (evidence === 0) {
                document.getElementById('results').textContent = "❌ Evidence cannot be zero (would cause division by zero)!";
                return;
            }

            // Bayes' Theorem: P(H|E) = P(E|H) * P(H) / P(E)
            const posterior = (likelihood * prior) / evidence;

            document.getElementById('results').textContent = 
`⚖️ Bayesian Analysis - Updating Recipe Beliefs:

🧠 Bayes' Theorem in Action:
   P(Recipe Success | New Evidence) = [P(Evidence | Recipe Success) × P(Recipe Success)] ÷ P(Evidence)

📊 Your Kitchen Data:
   • Prior Belief: ${(prior * 100).toFixed(1)}% (How confident were you before?)
   • Likelihood: ${(likelihood * 100).toFixed(1)}% (How likely is this evidence if recipe works?)
   • Evidence: ${(evidence * 100).toFixed(1)}% (How likely is this evidence overall?)

🎯 Updated Belief (Posterior): ${(posterior * 100).toFixed(1)}%

🧑‍🍳 Chef's Bayesian Wisdom:
${posterior > prior ? 
  `📈 CONFIDENCE INCREASED! Your belief in the recipe got stronger!
  🔥 From ${(prior * 100).toFixed(1)}% to ${(posterior * 100).toFixed(1)}% - the new evidence supports your recipe!
  ✨ Like tasting a dish and becoming more confident it's delicious!` :
  
  posterior < prior ?
  `📉 CONFIDENCE DECREASED. The new evidence made you less certain.
  🤔 From ${(prior * 100).toFixed(1)}% to ${(posterior * 100).toFixed(1)}% - time to reconsider the recipe?
  🧂 Like realizing your "perfect" seasoning might need adjustment.` :
  
  `⚖️ CONFIDENCE UNCHANGED. The evidence exactly matched your expectations.
  📊 Your belief stayed at ${(posterior * 100).toFixed(1)}% - rare but possible!
  🎯 Like predicting exactly how a familiar recipe would turn out.`}

💡 Bayesian thinking is like being a learning chef - you start with assumptions, gather evidence through cooking, and update your beliefs accordingly!

🔄 Change in belief: ${((posterior - prior) * 100).toFixed(1)} percentage points`;
        }

        // Generate histogram visualization
        function generateHistogram() {
            if (currentData.length === 0) {
                document.getElementById('results').textContent = "🚫 No ingredients prepared! Please use the Data Prep Station first.";
                return;
            }

            // Create a simple ASCII histogram
            const bins = 10;
            const min = Math.min(...currentData);
            const max = Math.max(...currentData);
            const binWidth = (max - min) / bins;
            const histogram = new Array(bins).fill(0);

            currentData.forEach(value => {
                const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
                histogram[binIndex]++;
            });

            const maxCount = Math.max(...histogram);
            let asciiHist = "";

            for (let i = 0; i < bins; i++) {
                const binStart = min + i * binWidth;
                const binEnd = min + (i + 1) * binWidth;
                const barLength = Math.round((histogram[i] / maxCount) * 40);
                const bar = "█".repeat(barLength);
                asciiHist += `${binStart.toFixed(1)}-${binEnd.toFixed(1)}: ${bar} (${histogram[i]})\n`;
            }

            document.getElementById('visualization').innerHTML = 
                `<div style="font-family: monospace; font-size: 12px; text-align: left; line-height: 1.5;">
                    <strong>📊 Data Distribution Histogram</strong><br><br>
                    ${asciiHist.replace(/\n/g, '<br>')}
                    <br><strong>🧑‍🍳 Chef's Visual Analysis:</strong><br>
                    Each █ represents relative frequency<br>
                    Taller bars = more common values (like popular dishes!)
                </div>`;

            document.getElementById('results').textContent = 
`📈 Histogram Generated - Visual Flavor Profile!

🎨 Your data visualization is now displayed above, showing:
• Distribution shape (symmetric, skewed, multimodal?)
• Most common value ranges (where the peaks are)
• Data spread and patterns

🧑‍🍳 Reading Your Histogram Like a Chef:
• Tall bars = Popular "flavors" in your data
• Short bars = Rare "flavors" 
• Multiple peaks = Different "recipe variations"
• Shape tells the story of your data's personality!

💡 Histograms are like plating your data - they make patterns visible and beautiful!`;
        }

        // Helper functions for statistical calculations

        // Normal CDF approximation
        function normalCDF(z) {
            // Abramowitz and Stegun approximation
            const sign = z >= 0 ? 1 : -1;
            z = Math.abs(z);
            
            const a1 =  0.254829592;
            const a2 = -0.284496736;
            const a3 =  1.421413741;
            const a4 = -1.453152027;
            const a5 =  1.061405429;
            const p  =  0.3275911;
            
            const t = 1.0 / (1.0 + p * z);
            const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);
            
            return 0.5 * (1 + sign * y);
        }

        // T-distribution CDF approximation
        function tCDF(t, df) {
            // Simple approximation for t-distribution
            if (df > 30) return normalCDF(t);
            
            // For smaller df, use a rough approximation
            const x = t / Math.sqrt(df);
            return 0.5 + (x / (2 * Math.sqrt(1 + x * x / df)));
        }

        // Get critical t-value (approximation)
        function getTCritical(df, alpha) {
            // Rough approximation for common alpha levels
            if (alpha >= 0.05) return df > 30 ? 1.96 : 2.0 + 0.5 / Math.sqrt(df);
            if (alpha >= 0.01) return df > 30 ? 2.58 : 2.6 + 1.0 / Math.sqrt(df);
            return df > 30 ? 3.29 : 3.3 + 1.5 / Math.sqrt(df);
        }

        // Initialize with welcome message
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('results').textContent = 
`🧑‍🍳 Welcome to Chef's Statistical Kitchen!

Like mastering culinary arts, statistical analysis is about:
🥕 PREPARATION: Clean, organize your data ingredients
📊 TECHNIQUE: Apply the right statistical methods  
🔬 TESTING: Verify your hypotheses like taste-testing
🎨 PRESENTATION: Visualize results beautifully

Your AI sous chef is ready to help you transform raw data into delicious insights!

👆 Start by entering your data in the Data Prep Station above.

Example data is already loaded - try clicking "🔪 Prep Ingredients" to begin!`;

## 📝 Assignment: Restaurant Performance Predictor

**Objective**: Build a statistical model to predict restaurant performance based on historical data.

**Your Challenge**:
You've been hired as a data scientist for "Probability Pizza Palace." The owner wants to understand what factors most influence customer satisfaction and revenue. 

**Task**:
1. **Data Collection**: Generate 6 months of simulated data including:
   - Daily temperature (affects customer traffic)
   - Day of week
   - Marketing spend
   - Staff count
   - Customer satisfaction scores
   - Daily revenue

2. **Statistical Analysis**: Perform the following analyses:
   - Calculate correlation coefficients between all variables
   - Test the hypothesis: "Weekend revenue is significantly higher than weekday revenue"
   - Use Bayes' theorem to calculate the probability of high revenue given high customer satisfaction
   - Identify which factors have the strongest statistical relationship with customer satisfaction

3. **Probability Modeling**: 
   - Fit appropriate probability distributions to your key metrics
   - Calculate the probability of achieving daily revenue > $8,000
   - Determine the expected wait time for customers

4. **Business Recommendations**: 
   - Provide 3 data-driven recommendations based on your statistical findings
   - Include confidence intervals for your key predictions
   - Suggest one hypothesis test the restaurant should conduct monthly

**Deliverable**: Submit a Python script with your analysis and a summary report (max 500 words) explaining your findings and recommendations.

**Evaluation Criteria**:
- Correct implementation of statistical concepts (40%)
- Proper use of probability distributions (25%)
- Quality of insights and recommendations (25%)
- Code clarity and documentation (10%)

**Bonus Challenge**: Create a simple visualization showing the relationship between customer satisfaction and revenue using matplotlib.

---

## 🎓 Course Summary

Congratulations, statistical chef! You've mastered the essential ingredients of statistics and probability for AI:

1. **Descriptive Statistics**: Your data "tasting spoons" to understand what you're working with
2. **Inferential Statistics**: Making confident predictions about the whole "pot" from a small sample
3. **Probability Distributions**: Your recipe books for understanding chance and uncertainty
4. **Bayes' Theorem**: The secret ingredient for updating beliefs with new evidence
5. **Hypothesis Testing**: Quality control for validating your statistical "recipes"

These statistical tools form the foundation of machine learning, data science, and AI decision-making. Just as a master chef combines flavors to create extraordinary dishes, you can now combine these statistical concepts to extract meaningful insights from data and build intelligent systems.

Remember: In the kitchen of data science, statistics and probability are your most essential tools. Keep practicing, keep experimenting, and keep cooking up those data insights!