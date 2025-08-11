# AI Mastery Course - Day 87: Large Language Models (LLMs)

## Learning Objective
By the end of this lesson, you will understand the fundamental architecture of Large Language Models, master prompt engineering techniques, implement fine-tuning strategies, and critically evaluate LLM limitations and biases while building practical applications using Python and Django.

---

## Introduction: Imagine That...

Imagine that you're the head chef in the world's most sophisticated restaurant where recipes aren't written in traditional cookbooks, but are learned by observing millions of meals prepared by countless cooks across the globe. Your sous chefs don't just follow instructions—they've absorbed the essence of culinary wisdom from every cuisine imaginable, understanding not just what ingredients go together, but why they create harmony on the palate.

Now imagine these sous chefs can create entirely new dishes by understanding the deep patterns of flavor, texture, and presentation they've witnessed. They can adapt any recipe to dietary restrictions, cultural preferences, or seasonal availability—all while maintaining the soul of the original creation. This is the world of Large Language Models: AI systems that have learned the art of language by observing vast collections of human communication, developing an intuitive understanding of how words, ideas, and concepts blend together to create meaningful expression.

---

## 1. Understanding GPT Architecture

### The Recipe Foundation: Transformer Architecture

Just as a master chef understands that great cuisine starts with understanding fundamental techniques, GPT (Generative Pre-trained Transformer) models are built on the transformer architecture—a revolutionary approach to processing sequential data.

**Key Components:**

#### Self-Attention Mechanism
Think of this as a chef's ability to understand how each ingredient influences every other ingredient in a dish simultaneously.

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # d_model: dimension of the model (like the number of flavor profiles a chef can work with)
        # num_heads: number of attention heads (like having multiple taste testers)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for queries, keys, and values
        # Like different ways of examining ingredients
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores (how much each ingredient matters to others)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        return context, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attention_weights
```

**Syntax Explanation:**
- `nn.Module`: Base class for all neural network modules in PyTorch
- `torch.matmul()`: Matrix multiplication function
- `.transpose(-2, -1)`: Swaps the last two dimensions of a tensor
- `.masked_fill()`: Fills elements with a value where mask condition is True
- `.view()`: Reshapes tensor dimensions
- `.contiguous()`: Ensures tensor memory is laid out contiguously

#### Positional Encoding
Like a chef understanding that the order of adding ingredients matters, models need to understand word positions.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate division term for sine and cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
```

**Syntax Explanation:**
- `torch.arange()`: Creates a sequence of numbers
- `unsqueeze(1)`: Adds a dimension at position 1
- `0::2`: Slice notation taking every 2nd element starting from 0
- `register_buffer()`: Registers a buffer that should be saved with the model

---

## 2. Fine-tuning Pre-trained Models

### Customizing the Master Chef's Palate

Fine-tuning is like taking a master chef who knows global cuisine and teaching them the specific preferences of your local diners. We start with a pre-trained model and adapt it to specific tasks.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch

class LLMFineTuner:
    def __init__(self, model_name="gpt2", learning_rate=5e-5):
        """
        Initialize the fine-tuner with a pre-trained model
        Like selecting an experienced chef to train
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.learning_rate = learning_rate
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, texts, max_length=512):
        """
        Prepare training data like organizing ingredients before cooking
        """
        def tokenize_function(examples):
            # Tokenize the text and create labels
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            # For causal language modeling, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].clone()
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def fine_tune(self, train_texts, output_dir="./fine_tuned_model", 
                  num_epochs=3, batch_size=4):
        """
        Fine-tune the model like training a chef with new recipes
        """
        # Prepare dataset
        train_dataset = self.prepare_dataset(train_texts)
        
        # Training arguments - like setting the kitchen conditions
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=500,
            learning_rate=self.learning_rate,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        print("Starting fine-tuning process...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Fine-tuning complete! Model saved to {output_dir}")
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """
        Generate text with the fine-tuned model
        Like having the trained chef create a new dish
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Example usage
if __name__ == "__main__":
    # Sample training data (like giving the chef new recipe styles to learn)
    training_texts = [
        "The art of cooking begins with understanding your ingredients.",
        "A great dish balances flavors, textures, and visual appeal.",
        "Timing in the culinary world is everything.",
        # Add more training examples...
    ]
    
    # Initialize fine-tuner
    fine_tuner = LLMFineTuner()
    
    # Fine-tune the model
    fine_tuner.fine_tune(training_texts, num_epochs=2)
    
    # Test generation
    prompt = "The secret to great cooking is"
    result = fine_tuner.generate_text(prompt)
    print(f"Generated: {result}")
```

**Syntax Explanation:**
- `AutoTokenizer.from_pretrained()`: Loads a pre-trained tokenizer
- `AutoModelForCausalLM.from_pretrained()`: Loads a pre-trained language model
- `truncation=True`: Cuts text if it exceeds max_length
- `return_tensors="pt"`: Returns PyTorch tensors
- `.clone()`: Creates a copy of the tensor
- `torch.no_grad()`: Disables gradient computation for inference
- `do_sample=True`: Enables sampling during generation

---

## 3. Prompt Engineering Techniques

### The Art of Clear Communication

Prompt engineering is like giving precise instructions to your sous chefs. The clearer and more specific your directions, the better the outcome.

```python
class PromptEngineer:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def few_shot_prompting(self, examples, query, max_length=200):
        """
        Few-shot prompting: Show examples like demonstrating techniques to apprentices
        """
        # Build prompt with examples
        prompt = "Here are some examples:\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        prompt += f"Now, please provide an output for:\nInput: {query}\nOutput:"
        
        return self._generate_response(prompt, max_length)
    
    def chain_of_thought_prompting(self, problem, max_length=300):
        """
        Chain of thought: Guide step-by-step reasoning like walking through a complex recipe
        """
        prompt = f"""
        Let's solve this step by step:

        Problem: {problem}

        Step 1: First, I need to understand what is being asked.
        Step 2: Then, I'll identify the key information.
        Step 3: Next, I'll work through the solution methodically.
        Step 4: Finally, I'll provide the answer.

        Let me work through this:
        """
        
        return self._generate_response(prompt, max_length)
    
    def role_based_prompting(self, role, task, context="", max_length=250):
        """
        Role-based prompting: Have the model assume expertise like assigning specialized chefs
        """
        prompt = f"""
        You are a {role}. {context}
        
        Your task: {task}
        
        Please provide your expert response:
        """
        
        return self._generate_response(prompt, max_length)
    
    def temperature_controlled_generation(self, prompt, temperature_values=[0.3, 0.7, 1.0]):
        """
        Demonstrate how temperature affects creativity like adjusting cooking heat
        """
        results = {}
        
        for temp in temperature_values:
            print(f"\nGenerating with temperature {temp}:")
            response = self._generate_response(prompt, max_length=150, temperature=temp)
            results[temp] = response
            print(f"Result: {response}\n")
        
        return results
    
    def _generate_response(self, prompt, max_length=200, temperature=0.7):
        """
        Internal method to generate responses
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=0.9,  # Nucleus sampling
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Return only the generated part
        return response[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()

# Example usage
prompt_engineer = PromptEngineer()

# Few-shot example
examples = [
    {"input": "The recipe calls for salt", "output": "Add salt to taste, typically 1/4 teaspoon per serving"},
    {"input": "The sauce is too thin", "output": "Thicken with cornstarch slurry or reduce by simmering"}
]
query = "The meat is overcooked"
result = prompt_engineer.few_shot_prompting(examples, query)
print(f"Few-shot result: {result}")

# Chain of thought example
problem = "How do I balance flavors in a new dish?"
cot_result = prompt_engineer.chain_of_thought_prompting(problem)
print(f"Chain of thought result: {cot_result}")

# Role-based example
role_result = prompt_engineer.role_based_prompting(
    role="experienced pastry chef",
    task="explain how to prevent cookies from spreading too much",
    context="You have 20 years of baking experience."
)
print(f"Role-based result: {role_result}")
```

**Syntax Explanation:**
- `enumerate(examples, 1)`: Creates numbered pairs starting from 1
- `f"String {variable}"`: F-string formatting for string interpolation
- `top_p=0.9`: Nucleus sampling parameter (keeps top 90% probability mass)
- `repetition_penalty=1.1`: Reduces repetitive text generation
- `.strip()`: Removes leading/trailing whitespace
- `len(inputs[0])`: Gets the length of the input sequence

---

## 4. LLM Limitations and Biases

### Understanding When the Chef Makes Mistakes

Even master chefs have limitations and unconscious biases. Understanding these helps us use LLMs more effectively and ethically.

```python
import numpy as np
from collections import Counter
import re

class BiasDetector:
    def __init__(self):
        # Common bias indicators (like recognizing when ingredients might clash)
        self.gender_terms = {
            'male': ['he', 'him', 'his', 'man', 'boy', 'father', 'brother', 'son'],
            'female': ['she', 'her', 'hers', 'woman', 'girl', 'mother', 'sister', 'daughter']
        }
        
        self.profession_stereotypes = {
            'technical': ['engineer', 'programmer', 'scientist', 'doctor'],
            'care': ['nurse', 'teacher', 'caregiver', 'social worker']
        }
    
    def analyze_gender_bias(self, generated_texts):
        """
        Analyze gender representation like checking if recipes favor certain ingredients
        """
        gender_counts = {'male': 0, 'female': 0, 'neutral': 0}
        
        for text in generated_texts:
            text_lower = text.lower()
            male_count = sum(text_lower.count(term) for term in self.gender_terms['male'])
            female_count = sum(text_lower.count(term) for term in self.gender_terms['female'])
            
            if male_count > female_count:
                gender_counts['male'] += 1
            elif female_count > male_count:
                gender_counts['female'] += 1
            else:
                gender_counts['neutral'] += 1
        
        total = len(generated_texts)
        bias_report = {
            'male_percentage': (gender_counts['male'] / total) * 100,
            'female_percentage': (gender_counts['female'] / total) * 100,
            'neutral_percentage': (gender_counts['neutral'] / total) * 100,
            'bias_detected': abs(gender_counts['male'] - gender_counts['female']) > total * 0.2
        }
        
        return bias_report
    
    def check_factual_consistency(self, prompt, model_responses):
        """
        Check if model gives consistent answers like ensuring recipes produce reliable results
        """
        # Simple consistency check by comparing key facts
        consistency_scores = []
        
        for i in range(len(model_responses)):
            for j in range(i + 1, len(model_responses)):
                # Calculate similarity between responses (simplified)
                similarity = self._calculate_similarity(model_responses[i], model_responses[j])
                consistency_scores.append(similarity)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
        
        return {
            'average_consistency': avg_consistency,
            'high_consistency': avg_consistency > 0.8,
            'responses_analyzed': len(model_responses)
        }
    
    def identify_knowledge_gaps(self, test_prompts, model, tokenizer):
        """
        Test model knowledge boundaries like knowing which techniques a chef hasn't mastered
        """
        results = {}
        
        for category, prompts in test_prompts.items():
            category_results = []
            
            for prompt in prompts:
                # Generate response
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(inputs, max_length=100, temperature=0.3)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Simple analysis of response quality
                confidence_indicators = self._analyze_confidence(response)
                category_results.append({
                    'prompt': prompt,
                    'response': response,
                    'confidence_score': confidence_indicators['score'],
                    'uncertainty_markers': confidence_indicators['markers']
                })
            
            results[category] = category_results
        
        return results
    
    def _calculate_similarity(self, text1, text2):
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def _analyze_confidence(self, text):
        """Analyze confidence markers in text"""
        uncertainty_markers = ['might', 'maybe', 'possibly', 'perhaps', 'i think', 'i believe']
        confidence_markers = ['definitely', 'certainly', 'clearly', 'obviously']
        
        text_lower = text.lower()
        uncertainty_count = sum(text_lower.count(marker) for marker in uncertainty_markers)
        confidence_count = sum(text_lower.count(marker) for marker in confidence_markers)
        
        # Simple scoring system
        score = max(0, min(1, 0.5 + (confidence_count - uncertainty_count) * 0.1))
        
        return {
            'score': score,
            'markers': {
                'uncertainty': uncertainty_count,
                'confidence': confidence_count
            }
        }

# Example usage and testing
class LLMValidator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.bias_detector = BiasDetector()
    
    def comprehensive_evaluation(self, test_scenarios):
        """
        Run comprehensive evaluation like a food critic reviewing a restaurant
        """
        results = {
            'bias_analysis': {},
            'consistency_check': {},
            'knowledge_gaps': {},
            'recommendations': []
        }
        
        # Test each scenario
        for scenario_name, scenario_data in test_scenarios.items():
            print(f"Testing scenario: {scenario_name}")
            
            # Generate multiple responses
            responses = []
            for prompt in scenario_data['prompts']:
                response = self._generate_safe_response(prompt)
                responses.append(response)
            
            # Analyze bias
            bias_results = self.bias_detector.analyze_gender_bias(responses)
            results['bias_analysis'][scenario_name] = bias_results
            
            # Check consistency
            consistency_results = self.bias_detector.check_factual_consistency(
                scenario_data['prompts'][0], responses
            )
            results['consistency_check'][scenario_name] = consistency_results
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _generate_safe_response(self, prompt, max_attempts=3):
        """Generate response with error handling"""
        for attempt in range(max_attempts):
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs, 
                        max_length=150, 
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                if attempt == max_attempts - 1:
                    return f"Error generating response: {str(e)}"
                continue
    
    def _generate_recommendations(self, results):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Check for bias issues
        for scenario, bias_data in results['bias_analysis'].items():
            if bias_data.get('bias_detected', False):
                recommendations.append(f"High gender bias detected in {scenario}. Consider diverse training data.")
        
        # Check for consistency issues
        for scenario, consistency_data in results['consistency_check'].items():
            if not consistency_data.get('high_consistency', True):
                recommendations.append(f"Low consistency in {scenario}. Review model training stability.")
        
        if not recommendations:
            recommendations.append("Model shows good bias and consistency characteristics.")
        
        return recommendations

# Example test scenarios
test_scenarios = {
    'professional_descriptions': {
        'prompts': [
            "Describe a typical day for a software engineer",
            "What does a nurse do during their shift?",
            "Tell me about a teacher's responsibilities"
        ]
    },
    'creative_tasks': {
        'prompts': [
            "Write a short story about cooking",
            "Describe preparing a family meal",
            "Explain the art of baking bread"
        ]
    }
}

# Run validation (commented out to prevent actual execution in example)
# validator = LLMValidator(model, tokenizer)
# evaluation_results = validator.comprehensive_evaluation(test_scenarios)
# print("Evaluation Results:", evaluation_results)
```

**Syntax Explanation:**
- `Counter()`: Creates a dictionary subclass for counting hashable objects
- `sum(generator)`: Sums values from a generator expression
- `abs()`: Returns absolute value
- `np.mean()`: Calculates the mean of an array
- `set()`: Creates a set (unique collection)
- `.intersection()`: Returns common elements between sets
- `.union()`: Returns all elements from both sets
- Exception handling with `try/except`: Catches and handles errors gracefully

---

# Custom AI-Powered Chatbot Project

## Project Overview
Build a sophisticated Django-powered chatbot that integrates with OpenAI's GPT models, featuring custom fine-tuning capabilities and an intuitive web interface.

## Project Structure
```
chatbot_project/
├── manage.py
├── requirements.txt
├── chatbot_project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── chat/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── migrations/
├── templates/
│   ├── base.html
│   ├── chat/
│   │   ├── home.html
│   │   └── chat_interface.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── chat.js
└── fine_tuning/
    ├── __init__.py
    ├── data_processor.py
    ├── model_trainer.py
    └── sample_data.jsonl
```

## Core Implementation

### 1. Django Models (chat/models.py)
```python
from django.db import models
from django.contrib.auth.models import User
import json

class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50, default='gpt-3.5-turbo')
    
    def __str__(self):
        return f"Session {self.session_id} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class Message(models.Model):
    MESSAGE_TYPES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]
    
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)
    tokens_used = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."

class FineTuneJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    job_id = models.CharField(max_length=100, unique=True)
    model_name = models.CharField(max_length=100)
    training_file = models.CharField(max_length=200)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    fine_tuned_model = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return f"Fine-tune Job: {self.model_name} - {self.status}"
```

### 2. OpenAI Integration Service (chat/services.py)
```python
import openai
from django.conf import settings
import json
import uuid
from .models import ChatSession, Message, FineTuneJob

class ChatService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.default_system_prompt = """You are a helpful culinary assistant. Think of yourself as an experienced head chef 
        who guides others through complex recipes and cooking techniques. Your responses should be informative, 
        encouraging, and use cooking metaphors when explaining complex concepts."""
    
    def create_session(self, user=None, model_version='gpt-3.5-turbo'):
        """Create a new chat session like preparing a fresh cooking station"""
        session_id = str(uuid.uuid4())
        session = ChatSession.objects.create(
            user=user,
            session_id=session_id,
            model_version=model_version
        )
        
        # Add system message
        Message.objects.create(
            session=session,
            content=self.default_system_prompt,
            message_type='system'
        )
        
        return session
    
    def get_chat_completion(self, session, user_message, temperature=0.7):
        """Process user input like a chef interpreting a recipe request"""
        # Save user message
        user_msg = Message.objects.create(
            session=session,
            content=user_message,
            message_type='user'
        )
        
        # Prepare conversation history
        messages = []
        for msg in session.messages.all():
            messages.append({
                "role": msg.message_type,
                "content": msg.content
            })
        
        try:
            # Get AI response
            response = openai.ChatCompletion.create(
                model=session.model_version,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            
            assistant_content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Save assistant response
            assistant_msg = Message.objects.create(
                session=session,
                content=assistant_content,
                message_type='assistant',
                tokens_used=tokens_used
            )
            
            return {
                'response': assistant_content,
                'tokens_used': tokens_used,
                'success': True
            }
            
        except Exception as e:
            return {
                'response': f"I apologize, but I encountered an issue: {str(e)}",
                'tokens_used': 0,
                'success': False
            }

class FineTuningService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
    
    def prepare_training_data(self, conversations):
        """Prepare training data like organizing ingredients before cooking"""
        training_data = []
        
        for conv in conversations:
            training_example = {
                "messages": [
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": conv['user_input']},
                    {"role": "assistant", "content": conv['expected_response']}
                ]
            }
            training_data.append(training_example)
        
        return training_data
    
    def upload_training_file(self, training_data, filename="training_data.jsonl"):
        """Upload training data like storing prepped ingredients"""
        # Convert to JSONL format
        jsonl_content = "\n".join([json.dumps(item) for item in training_data])
        
        try:
            # Create temporary file
            with open(filename, 'w') as f:
                f.write(jsonl_content)
            
            # Upload to OpenAI
            response = openai.File.create(
                file=open(filename, "rb"),
                purpose='fine-tune'
            )
            
            return response.id
            
        except Exception as e:
            raise Exception(f"Failed to upload training file: {str(e)}")
    
    def start_fine_tuning(self, training_file_id, model_name="gpt-3.5-turbo"):
        """Start fine-tuning process like beginning a slow-cooked recipe"""
        try:
            response = openai.FineTuningJob.create(
                training_file=training_file_id,
                model=model_name
            )
            
            # Save job to database
            job = FineTuneJob.objects.create(
                job_id=response.id,
                model_name=model_name,
                training_file=training_file_id,
                status='running'
            )
            
            return job
            
        except Exception as e:
            raise Exception(f"Failed to start fine-tuning: {str(e)}")
    
    def check_job_status(self, job_id):
        """Check fine-tuning status like checking if a dish is ready"""
        try:
            response = openai.FineTuningJob.retrieve(job_id)
            
            # Update database
            job = FineTuneJob.objects.get(job_id=job_id)
            job.status = response.status
            
            if response.status == 'succeeded':
                job.fine_tuned_model = response.fine_tuned_model
                job.completed_at = timezone.now()
            
            job.save()
            
            return job
            
        except Exception as e:
            raise Exception(f"Failed to check job status: {str(e)}")
```

### 3. Django Views (chat/views.py)
```python
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views.generic import View
import json
from .models import ChatSession, Message, FineTuneJob
from .services import ChatService, FineTuningService
from .forms import FineTuneForm

class ChatHomeView(View):
    """Main chat interface like the main dining area of a restaurant"""
    
    def get(self, request):
        # Get or create session
        session_id = request.session.get('chat_session_id')
        
        if session_id:
            try:
                chat_session = ChatSession.objects.get(session_id=session_id)
            except ChatSession.DoesNotExist:
                chat_session = self.create_new_session(request)
        else:
            chat_session = self.create_new_session(request)
        
        # Get conversation history
        messages = chat_session.messages.filter(message_type__in=['user', 'assistant'])
        
        context = {
            'session': chat_session,
            'messages': messages,
        }
        
        return render(request, 'chat/home.html', context)
    
    def create_new_session(self, request):
        """Create new session like setting up a new table"""
        service = ChatService()
        user = request.user if request.user.is_authenticated else None
        session = service.create_session(user=user)
        request.session['chat_session_id'] = session.session_id
        return session

@csrf_exempt
def chat_api(request):
    """API endpoint for chat messages like taking orders from customers"""
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        
        if not message:
            return JsonResponse({'error': 'Message cannot be empty'}, status=400)
        
        # Get session
        session_id = request.session.get('chat_session_id')
        if not session_id:
            return JsonResponse({'error': 'No active session'}, status=400)
        
        try:
            chat_session = ChatSession.objects.get(session_id=session_id)
            service = ChatService()
            
            # Get AI response
            result = service.get_chat_completion(
                session=chat_session,
                user_message=message,
                temperature=data.get('temperature', 0.7)
            )
            
            return JsonResponse({
                'response': result['response'],
                'tokens_used': result['tokens_used'],
                'success': result['success']
            })
            
        except ChatSession.DoesNotExist:
            return JsonResponse({'error': 'Session not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@method_decorator(login_required, name='dispatch')
class FineTuneView(View):
    """Fine-tuning management like developing new recipes"""
    
    def get(self, request):
        jobs = FineTuneJob.objects.all().order_by('-created_at')
        form = FineTuneForm()
        
        context = {
            'jobs': jobs,
            'form': form,
        }
        
        return render(request, 'chat/fine_tune.html', context)
    
    def post(self, request):
        form = FineTuneForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                service = FineTuningService()
                
                # Process uploaded training data
                training_file = request.FILES['training_file']
                training_data = self.process_training_file(training_file)
                
                # Upload and start fine-tuning
                file_id = service.upload_training_file(training_data)
                job = service.start_fine_tuning(
                    training_file_id=file_id,
                    model_name=form.cleaned_data['base_model']
                )
                
                return JsonResponse({
                    'success': True,
                    'job_id': job.job_id,
                    'message': 'Fine-tuning job started successfully!'
                })
                
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'error': str(e)
                })
        
        return JsonResponse({
            'success': False,
            'errors': form.errors
        })
    
    def process_training_file(self, file):
        """Process uploaded training file like preparing ingredients"""
        content = file.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        training_data = []
        for line in lines:
            try:
                data = json.loads(line)
                training_data.append(data)
            except json.JSONDecodeError:
                continue
        
        return training_data

def check_fine_tune_status(request, job_id):
    """Check fine-tuning progress like monitoring cooking progress"""
    try:
        service = FineTuningService()
        job = service.check_job_status(job_id)
        
        return JsonResponse({
            'status': job.status,
            'fine_tuned_model': job.fine_tuned_model,
            'created_at': job.created_at.isoformat(),
            'completed_at': job.completed_at.isoformat() if job.completed_at else None
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
```

### 4. Forms (chat/forms.py)
```python
from django import forms
from .models import FineTuneJob

class FineTuneForm(forms.Form):
    BASE_MODELS = [
        ('gpt-3.5-turbo', 'GPT-3.5 Turbo'),
        ('gpt-4', 'GPT-4'),
    ]
    
    model_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter custom model name'
        })
    )
    
    base_model = forms.ChoiceField(
        choices=BASE_MODELS,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    training_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.jsonl'
        }),
        help_text='Upload JSONL file with training conversations'
    )
    
    def clean_training_file(self):
        file = self.cleaned_data['training_file']
        
        if not file.name.endswith('.jsonl'):
            raise forms.ValidationError('Training file must be in JSONL format')
        
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            raise forms.ValidationError('File size cannot exceed 10MB')
        
        return file
```

### 5. Frontend Templates

**templates/base.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chatbot</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css" rel="stylesheet">
    <link href="{% static 'css/style.css' %}" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{% url 'chat:home' %}">🤖 AI Chatbot</a>
            <div class="navbar-nav">
                {% if user.is_authenticated %}
                    <a class="nav-link" href="{% url 'chat:fine_tune' %}">Fine-Tuning</a>
                {% endif %}
            </div>
        </div>
    </nav>

    <main class="container mt-4">
        {% block content %}
        {% endblock %}
    </main>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>
    <script src="{% static 'js/chat.js' %}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
```

**templates/chat/home.html**
```html
{% extends 'base.html' %}
{% load static %}

{% block content %}
<div class="row">
    <div class="col-md-8 mx-auto">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Chat with AI Assistant</h5>
                <small class="text-muted">Session: {{ session.session_id }}</small>
            </div>
            
            <div class="card-body chat-container" id="chatContainer" style="height: 500px; overflow-y: auto;">
                {% for message in messages %}
                    <div class="message {% if message.message_type == 'user' %}user-message{% else %}assistant-message{% endif %}">
                        <div class="message-content">
                            <strong>
                                {% if message.message_type == 'user' %}
                                    You:
                                {% else %}
                                    Assistant:
                                {% endif %}
                            </strong>
                            <p>{{ message.content|linebreaks }}</p>
                            <small class="text-muted">{{ message.timestamp|date:"H:i" }}</small>
                        </div>
                    </div>
                {% endfor %}
            </div>
            
            <div class="card-footer">
                <form id="chatForm" class="d-flex">
                    <input type="text" id="messageInput" class="form-control me-2" 
                           placeholder="Type your message..." required>
                    <button type="submit" class="btn btn-primary" id="sendButton">
                        Send
                    </button>
                </form>
            </div>
        </div>
        
        <div class="mt-3">
            <label for="temperatureRange" class="form-label">Response Creativity: 
                <span id="temperatureValue">0.7</span>
            </label>
            <input type="range" class="form-range" id="temperatureRange" 
                   min="0.1" max="1.0" step="0.1" value="0.7">
        </div>
    </div>
</div>
{% endblock %}
```

### 6. JavaScript Frontend (static/js/chat.js)
```javascript
class ChatInterface {
    constructor() {
        this.chatContainer = document.getElementById('chatContainer');
        this.chatForm = document.getElementById('chatForm');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.temperatureRange = document.getElementById('temperatureRange');
        this.temperatureValue = document.getElementById('temperatureValue');
        
        this.init();
    }
    
    init() {
        // Set up event listeners like preparing cooking stations
        this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));
        this.temperatureRange.addEventListener('input', (e) => this.updateTemperature(e));
        
        // Auto-scroll to bottom
        this.scrollToBottom();
    }
    
    async handleSubmit(e) {
        e.preventDefault();
        
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        // Disable input while processing
        this.setLoading(true);
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        
        try {
            // Send to API
            const response = await this.sendMessage(message);
            
            if (response.success) {
                this.addMessage(response.response, 'assistant');
                this.showTokenUsage(response.tokens_used);
            } else {
                this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant', true);
            }
            
        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage('Connection error. Please check your internet connection.', 'assistant', true);
        }
        
        this.setLoading(false);
    }
    
    async sendMessage(message) {
        const response = await fetch('/chat/api/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': this.getCSRFToken()
            },
            body: JSON.stringify({
                message: message,
                temperature: parseFloat(this.temperatureRange.value)
            })
        });
        
        return await response.json();
    }
    
    addMessage(content, type, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message ${isError ? 'error-message' : ''}`;
        
        const now = new Date();
        const timeString = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <strong>${type === 'user' ? 'You:' : 'Assistant:'}</strong>
                <p>${this.formatMessage(content)}</p>
                <small class="text-muted">${timeString}</small>
            </div>
        `;
        
        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    formatMessage(content) {
        // Convert line breaks and format code blocks
        return content
            .replace(/\n/g, '<br>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    }
    
    setLoading(isLoading) {
        this.sendButton.disabled = isLoading;
        this.messageInput.disabled = isLoading;
        this.sendButton.innerHTML = isLoading ? 
            '<span class="spinner-border spinner-border-sm"></span> Thinking...' : 'Send';
    }
    
    updateTemperature(e) {
        this.temperatureValue.textContent = e.target.value;
    }
    
    showTokenUsage(tokens) {
        // Create temporary notification
        const notification = document.createElement('div');
        notification.className = 'alert alert-info alert-dismissible fade show position-fixed';
        notification.style.top = '10px';
        notification.style.right = '10px';
        notification.style.zIndex = '9999';
        notification.innerHTML = `
            Tokens used: ${tokens}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }
    
    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
    
    getCSRFToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
    }
}

// Initialize chat interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChatInterface();
});
```

### 7. CSS Styling (static/css/style.css)
```css
/* Chat Interface Styling */
.chat-container {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
}

.message {
    margin-bottom: 20px;
    display: flex;
}

.user-message {
    justify-content: flex-end;
}

.assistant-message {
    justify-content: flex-start;
}

.message-content {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 18px;
    position: relative;
}

.user-message .message-content {
    background-color: #007bff;
    color: white;
    margin-left: auto;
}

.assistant-message .message-content {
    background-color: #e9ecef;
    color: #333;
    margin-right: auto;
}

.error-message .message-content {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.message-content pre {
    background-color: rgba(0, 0, 0, 0.1);
    padding: 8px;
    border-radius: 4px;
    overflow-x: auto;
    margin: 8px 0;
}

.message-content code {
    background-color: rgba(0, 0, 0, 0.1);
    padding: 2px 4px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
}

/* Fine-tuning Interface */
.fine-tune-card {
    transition: transform 0.2s;
}

.fine-tune-card:hover {
    transform: translateY(-2px);
}

.status-badge {
    font-size: 0.8em;
}

/* Responsive Design */
@media (max-width: 768px) {
    .message-content {
        max-width: 95%;
    }
    
    .chat-container {
        height: 400px !important;
    }
}

/* Loading Animation */
.spinner-border-sm {
    width: 1rem;
    height: 1rem;
}

/* Temperature Slider */
.form-range::-webkit-slider-thumb {
    background-color: #007bff;
}

.form-range::-moz-range-thumb {
    background-color: #007bff;
    border: none;
}
```

### 8. Settings Configuration (settings.py additions)
```python
# OpenAI Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-api-key-here')

# Session Configuration
SESSION_COOKIE_AGE = 86400  # 24 hours
SESSION_SAVE_EVERY_REQUEST = True

# File Upload Settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'chatbot.log',
        },
    },
    'loggers': {
        'chat': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
```

### 9. Requirements File (requirements.txt)
```txt
Django==4.2.7
openai==1.3.0
python-dotenv==1.0.0
celery==5.3.4
redis==5.0.1
gunicorn==21.2.0
whitenoise==6.6.0
psycopg2-binary==2.9.9
```

### 10. Sample Training Data (fine_tuning/sample_data.jsonl)
```jsonl
{"messages": [{"role": "system", "content": "You are a helpful culinary assistant who uses cooking metaphors to explain concepts."}, {"role": "user", "content": "How do I optimize my code?"}, {"role": "assistant", "content": "Optimizing code is like refining a recipe! Start by identifying the slowest ingredients (bottlenecks), then use efficient techniques like caching (meal prep), removing unnecessary steps (dead code), and choosing the right tools for each task. Profile your code like taste-testing - measure before and after improvements!"}]}
{"messages": [{"role": "system", "content": "You are a helpful culinary assistant who uses cooking metaphors to explain concepts."}, {"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is like teaching a chef to cook by showing them many examples! Instead of giving exact recipes (rules), you show the algorithm thousands of dishes and their ingredients. Over time, it learns patterns - like how certain spice combinations create specific flavors - and can then create new dishes or identify ingredients in unfamiliar recipes."}]}
```

## Key Features Implemented

1. **Real-time Chat Interface**: Interactive web-based chat with message history
2. **OpenAI Integration**: Direct API integration with GPT models
3. **Fine-tuning Capability**: Upload training data and create custom models
4. **Session Management**: Persistent chat sessions with history
5. **Token Usage Tracking**: Monitor API costs and usage
6. **Responsive Design**: Mobile-friendly interface
7. **Error Handling**: Robust error management and user feedback
8. **Temperature Control**: Adjustable response creativity
9. **Admin Interface**: Django admin for managing conversations and fine-tuning jobs

## Testing the Application

Run the development server:
```bash
python manage.py runserver
```

The chatbot will be available at `http://localhost:8000` with a fully functional chat interface that integrates OpenAI's GPT models and supports custom fine-tuning capabilities.


## Assignment: Bias-Aware Prompt Optimizer

**Objective:** Create a system that automatically optimizes prompts to reduce bias while maintaining response quality.

### Task Description:
Build a Django web application that allows users to input prompts, automatically detects potential biases, and suggests improved versions. Your system should:

1. **Bias Detection Module:** Identify gender, cultural, and professional biases in prompts
2. **Prompt Optimization:** Suggest alternative phrasings that reduce bias
3. **A/B Testing Interface:** Allow users to compare original vs. optimized prompts
4. **Results Dashboard:** Display bias metrics and improvement suggestions

### Requirements:
- Use Django for the web framework
- Implement at least 3 different bias detection methods
- Create a user-friendly interface for prompt testing
- Include data visualization of bias metrics
- Add export functionality for results

### Deliverables:
1. Django project with functional web interface
2. Documentation explaining your bias detection algorithms
3. Test cases demonstrating bias reduction
4. Analysis report comparing original vs. optimized prompts

### Evaluation Criteria:
- **Technical Implementation (40%):** Code quality, Django best practices, error handling
- **Bias Detection Accuracy (30%):** Effectiveness of bias identification methods
- **User Experience (20%):** Interface design and usability
- **Documentation (10%):** Clear explanation of methods and results

This assignment challenges you to think critically about AI ethics while building practical tools, combining technical skills with social responsibility—much like a chef who must balance flavor with nutrition and dietary restrictions.