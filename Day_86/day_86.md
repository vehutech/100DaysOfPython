# Day 86: Transformer Architecture - AI Mastery Course

## Learning Objective
By the end of this lesson, you will understand the fundamental components of transformer architecture, implement attention mechanisms, and work with BERT and GPT model families using Python and Django to create intelligent text processing applications.

---

Imagine you're the head chef of the world's most sophisticated restaurant where every dish must be perfectly crafted by understanding not just individual ingredients, but how each ingredient relates to every other ingredient simultaneously. Your kitchen operates with an revolutionary system where each cooking station can "pay attention" to all other stations at once, understanding context, relationships, and dependencies in ways never before possible. This is the power of transformer architecture - a cooking methodology that revolutionized how we prepare the most complex computational dishes in artificial intelligence.

---

## Lesson 1: Attention Mechanism Deep Dive

### Understanding the Master Recipe

In our culinary world, traditional recipes follow a linear sequence - first this, then that. But imagine if your sous chefs could simultaneously consider every ingredient, every cooking technique, and every flavor combination all at once, weighing their importance dynamically based on what's needed for the perfect dish.

```python
import torch
import torch.nn as nn
import numpy as np
import math

class AttentionMechanism(nn.Module):
    ---


# Text Summarization with Transformers - Django Project

## Project Overview
Build a Django web application that uses transformer models to automatically summarize long text documents. Think of this as creating a master chef who can take a lengthy recipe book and extract only the essential ingredients and steps - your transformer model will distill lengthy articles into their core essence.

## Project Structure
```
text_summarizer/
├── manage.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── summarizer/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   ├── ml_models.py
│   └── templates/
│       └── summarizer/
│           ├── index.html
│           ├── result.html
│           └── history.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
└── requirements.txt
```

## Step 1: Django Setup and Dependencies

**requirements.txt**
```txt
Django==4.2.7
transformers==4.35.2
torch==2.1.0
sentencepiece==0.1.99
accelerate==0.24.1
datasets==2.14.6
```

**config/settings.py**
```python
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
    'summarizer',
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

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
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
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

# Cache configuration for model storage
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}
```

## Step 2: Database Models

**summarizer/models.py**
```python
from django.db import models
from django.contrib.auth.models import User

class SummaryRequest(models.Model):
    SUMMARY_TYPES = [
        ('extractive', 'Extractive'),
        ('abstractive', 'Abstractive'),
        ('bullet_points', 'Bullet Points'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    original_text = models.TextField(help_text="The original text to be summarized")
    summary_type = models.CharField(max_length=20, choices=SUMMARY_TYPES, default='abstractive')
    max_length = models.IntegerField(default=150, help_text="Maximum length of summary")
    min_length = models.IntegerField(default=30, help_text="Minimum length of summary")
    created_at = models.DateTimeField(auto_now_add=True)
    processing_time = models.FloatField(null=True, blank=True, help_text="Time taken to process in seconds")
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"Summary Request {self.id} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class GeneratedSummary(models.Model):
    request = models.OneToOneField(SummaryRequest, on_delete=models.CASCADE, related_name='summary')
    summary_text = models.TextField()
    confidence_score = models.FloatField(null=True, blank=True, help_text="Model confidence score")
    model_used = models.CharField(max_length=100, default='facebook/bart-large-cnn')
    word_count_original = models.IntegerField()
    word_count_summary = models.IntegerField()
    compression_ratio = models.FloatField(help_text="Original length / Summary length")
    
    def save(self, *args, **kwargs):
        # Calculate word counts and compression ratio
        self.word_count_original = len(self.request.original_text.split())
        self.word_count_summary = len(self.summary_text.split())
        if self.word_count_summary > 0:
            self.compression_ratio = self.word_count_original / self.word_count_summary
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Summary for Request {self.request.id}"
```

## Step 3: ML Model Integration

**summarizer/ml_models.py**
```python
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline,
    BartTokenizer, 
    BartForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from django.core.cache import cache
import time
import logging

logger = logging.getLogger(__name__)

class TransformerSummarizer:
    """
    A sophisticated summarization system that acts like a master chef
    selecting the finest ingredients (key sentences) from a complex recipe (document)
    """
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
    def load_model(self, model_name='facebook/bart-large-cnn'):
        """Load and cache transformer models"""
        cache_key = f"model_{model_name.replace('/', '_')}"
        
        if cache_key in self.models:
            return self.models[cache_key], self.tokenizers[cache_key]
            
        try:
            logger.info(f"Loading model: {model_name}")
            
            if 'bart' in model_name.lower():
                tokenizer = BartTokenizer.from_pretrained(model_name)
                model = BartForConditionalGeneration.from_pretrained(model_name)
            elif 't5' in model_name.lower():
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            self.models[cache_key] = model
            self.tokenizers[cache_key] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def create_pipeline(self, model_name='facebook/bart-large-cnn'):
        """Create summarization pipeline"""
        cache_key = f"pipeline_{model_name.replace('/', '_')}"
        
        if cache_key in self.pipelines:
            return self.pipelines[cache_key]
        
        try:
            # Load model and tokenizer
            model, tokenizer = self.load_model(model_name)
            
            # Create pipeline
            summarizer_pipeline = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.pipelines[cache_key] = summarizer_pipeline
            return summarizer_pipeline
            
        except Exception as e:
            logger.error(f"Error creating pipeline: {str(e)}")
            raise
    
    def chunk_text(self, text, max_chunk_length=1000):
        """
        Break long text into manageable chunks like a chef portioning ingredients
        """
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def summarize_text(self, text, summary_type='abstractive', max_length=150, 
                      min_length=30, model_name='facebook/bart-large-cnn'):
        """
        Main summarization method - like a master chef creating the perfect reduction
        """
        start_time = time.time()
        
        try:
            # Handle different summary types
            if summary_type == 'extractive':
                return self._extractive_summarization(text, max_length, min_length)
            elif summary_type == 'bullet_points':
                return self._bullet_point_summarization(text, model_name)
            else:
                return self._abstractive_summarization(text, max_length, min_length, model_name)
                
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            processing_time = time.time() - start_time
            return {
                'summary': f"Error processing text: {str(e)}",
                'processing_time': processing_time,
                'confidence_score': 0.0,
                'model_used': model_name
            }
    
    def _abstractive_summarization(self, text, max_length, min_length, model_name):
        """Abstractive summarization using transformer models"""
        start_time = time.time()
        
        # Create pipeline
        summarizer = self.create_pipeline(model_name)
        
        # Handle long texts by chunking
        if len(text.split()) > 1000:
            chunks = self.chunk_text(text)
            chunk_summaries = []
            
            for chunk in chunks:
                try:
                    result = summarizer(
                        chunk,
                        max_length=max_length // len(chunks) + 20,
                        min_length=min_length // len(chunks),
                        do_sample=False
                    )
                    chunk_summaries.append(result[0]['summary_text'])
                except Exception as e:
                    logger.warning(f"Error summarizing chunk: {str(e)}")
                    continue
            
            # Combine and re-summarize
            combined_text = " ".join(chunk_summaries)
            if len(combined_text.split()) > max_length:
                final_result = summarizer(
                    combined_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summary = final_result[0]['summary_text']
            else:
                summary = combined_text
                
        else:
            # Direct summarization for shorter texts
            result = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            summary = result[0]['summary_text']
        
        processing_time = time.time() - start_time
        
        return {
            'summary': summary,
            'processing_time': processing_time,
            'confidence_score': self._calculate_confidence_score(text, summary),
            'model_used': model_name
        }
    
    def _extractive_summarization(self, text, max_length, min_length):
        """Simple extractive summarization by selecting key sentences"""
        sentences = text.split('. ')
        
        # Score sentences based on word frequency (simplified approach)
        word_freq = {}
        words = text.lower().split()
        
        for word in words:
            word = word.strip('.,!?";')
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = sentence.lower().split()
            score = sum(word_freq.get(word.strip('.,!?";'), 0) for word in sentence_words)
            sentence_scores[sentence] = score / len(sentence_words) if sentence_words else 0
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        
        summary_sentences = []
        current_length = 0
        target_length = max_length
        
        for sentence, score in top_sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= target_length:
                summary_sentences.append(sentence)
                current_length += sentence_length
            
            if current_length >= min_length and len(summary_sentences) >= 3:
                break
        
        summary = '. '.join(summary_sentences) + '.'
        
        return {
            'summary': summary,
            'processing_time': 0.5,  # Fast extractive method
            'confidence_score': 0.7,
            'model_used': 'extractive_algorithm'
        }
    
    def _bullet_point_summarization(self, text, model_name):
        """Create bullet-point style summary"""
        # First create an abstractive summary
        initial_summary = self._abstractive_summarization(text, 200, 50, model_name)
        
        # Convert to bullet points
        sentences = initial_summary['summary'].split('. ')
        bullet_points = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Filter out very short sentences
                bullet_points.append(f"• {sentence.strip()}")
        
        bullet_summary = '\n'.join(bullet_points)
        
        return {
            'summary': bullet_summary,
            'processing_time': initial_summary['processing_time'] + 0.1,
            'confidence_score': initial_summary['confidence_score'],
            'model_used': model_name + '_bullet_format'
        }
    
    def _calculate_confidence_score(self, original_text, summary):
        """Calculate a simple confidence score based on content preservation"""
        original_words = set(original_text.lower().split())
        summary_words = set(summary.lower().split())
        
        # Calculate overlap ratio
        if len(original_words) == 0:
            return 0.0
            
        overlap = len(original_words.intersection(summary_words))
        confidence = (overlap / len(original_words)) * 0.7 + 0.3  # Base confidence of 0.3
        
        return min(confidence, 1.0)

# Global instance
summarizer_instance = TransformerSummarizer()
```

## Step 4: Forms

**summarizer/forms.py**
```python
from django import forms
from .models import SummaryRequest

class TextSummarizationForm(forms.ModelForm):
    MODEL_CHOICES = [
        ('facebook/bart-large-cnn', 'BART Large CNN (News Articles)'),
        ('t5-base', 'T5 Base (General Purpose)'),
        ('google/pegasus-xsum', 'Pegasus XSum (Extractive)'),
    ]
    
    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        initial='facebook/bart-large-cnn',
        widget=forms.Select(attrs={
            'class': 'form-control',
            'id': 'model-select'
        })
    )
    
    class Meta:
        model = SummaryRequest
        fields = ['original_text', 'summary_type', 'max_length', 'min_length']
        widgets = {
            'original_text': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 10,
                'placeholder': 'Paste your text here for summarization...',
                'id': 'text-input'
            }),
            'summary_type': forms.Select(attrs={
                'class': 'form-control',
                'id': 'summary-type'
            }),
            'max_length': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 50,
                'max': 500,
                'id': 'max-length'
            }),
            'min_length': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 10,
                'max': 200,
                'id': 'min-length'
            }),
        }
    
    def clean_original_text(self):
        text = self.cleaned_data.get('original_text', '')
        
        if len(text.strip()) < 50:
            raise forms.ValidationError("Text must be at least 50 characters long for meaningful summarization.")
        
        if len(text.split()) > 5000:
            raise forms.ValidationError("Text is too long. Please limit to 5000 words.")
        
        return text
    
    def clean(self):
        cleaned_data = super().clean()
        max_length = cleaned_data.get('max_length')
        min_length = cleaned_data.get('min_length')
        
        if max_length and min_length and min_length >= max_length:
            raise forms.ValidationError("Minimum length must be less than maximum length.")
        
        return cleaned_data
```

## Step 5: Views

**summarizer/views.py**
```python
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views.generic import ListView
from django.core.paginator import Paginator
from .forms import TextSummarizationForm
from .models import SummaryRequest, GeneratedSummary
from .ml_models import summarizer_instance
import json
import logging

logger = logging.getLogger(__name__)

def index(request):
    """Main summarization interface"""
    if request.method == 'POST':
        form = TextSummarizationForm(request.POST)
        if form.is_valid():
            try:
                # Save the request
                summary_request = form.save(commit=False)
                if request.user.is_authenticated:
                    summary_request.user = request.user
                summary_request.save()
                
                # Get model choice from form
                model_name = form.cleaned_data.get('model_choice', 'facebook/bart-large-cnn')
                
                # Process the summarization
                result = summarizer_instance.summarize_text(
                    text=summary_request.original_text,
                    summary_type=summary_request.summary_type,
                    max_length=summary_request.max_length,
                    min_length=summary_request.min_length,
                    model_name=model_name
                )
                
                # Save processing time
                summary_request.processing_time = result['processing_time']
                summary_request.save()
                
                # Create summary object
                generated_summary = GeneratedSummary.objects.create(
                    request=summary_request,
                    summary_text=result['summary'],
                    confidence_score=result['confidence_score'],
                    model_used=result['model_used']
                )
                
                messages.success(request, 'Text summarized successfully!')
                return redirect('summarizer:result', summary_id=generated_summary.id)
                
            except Exception as e:
                logger.error(f"Summarization error: {str(e)}")
                messages.error(request, f'Error during summarization: {str(e)}')
        else:
            messages.error(request, 'Please correct the errors in the form.')
    else:
        form = TextSummarizationForm()
    
    # Get recent summaries for display
    recent_summaries = GeneratedSummary.objects.select_related('request')[:5]
    
    context = {
        'form': form,
        'recent_summaries': recent_summaries,
    }
    return render(request, 'summarizer/index.html', context)

def result(request, summary_id):
    """Display summarization result"""
    summary = get_object_or_404(GeneratedSummary, id=summary_id)
    
    # Calculate additional metrics
    original_sentences = len(summary.request.original_text.split('.'))
    summary_sentences = len(summary.summary_text.split('.'))
    
    context = {
        'summary': summary,
        'original_sentences': original_sentences,
        'summary_sentences': summary_sentences,
        'time_saved': f"{summary.compression_ratio:.1f}x faster to read",
    }
    return render(request, 'summarizer/result.html', context)

@require_http_methods(["POST"])
def quick_summarize(request):
    """AJAX endpoint for quick summarization"""
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        
        if len(text) < 50:
            return JsonResponse({
                'success': False,
                'error': 'Text too short for summarization'
            })
        
        # Quick summarization with default parameters
        result = summarizer_instance.summarize_text(
            text=text,
            summary_type='abstractive',
            max_length=100,
            min_length=30,
            model_name='facebook/bart-large-cnn'
        )
        
        return JsonResponse({
            'success': True,
            'summary': result['summary'],
            'processing_time': result['processing_time'],
            'confidence_score': result['confidence_score']
        })
        
    except Exception as e:
        logger.error(f"Quick summarization error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

class SummaryHistoryView(ListView):
    """View for displaying user's summary history"""
    model = GeneratedSummary
    template_name = 'summarizer/history.html'
    context_object_name = 'summaries'
    paginate_by = 10
    
    def get_queryset(self):
        if self.request.user.is_authenticated:
            return GeneratedSummary.objects.filter(
                request__user=self.request.user
            ).select_related('request').order_by('-request__created_at')
        else:
            return GeneratedSummary.objects.none()

def compare_models(request):
    """Compare different model outputs on the same text"""
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        
        if len(text) < 50:
            messages.error(request, 'Text too short for comparison')
            return render(request, 'summarizer/compare.html')
        
        models_to_compare = [
            'facebook/bart-large-cnn',
            't5-base',
            'google/pegasus-xsum'
        ]
        
        results = {}
        for model in models_to_compare:
            try:
                result = summarizer_instance.summarize_text(
                    text=text,
                    summary_type='abstractive',
                    max_length=150,
                    min_length=30,
                    model_name=model
                )
                results[model] = result
            except Exception as e:
                results[model] = {
                    'summary': f'Error: {str(e)}',
                    'processing_time': 0,
                    'confidence_score': 0
                }
        
        context = {
            'original_text': text,
            'results': results,
            'word_count': len(text.split())
        }
        return render(request, 'summarizer/compare_results.html', context)
    
    return render(request, 'summarizer/compare.html')
```

## Step 6: URL Configuration

**summarizer/urls.py**
```python
from django.urls import path
from . import views

app_name = 'summarizer'

urlpatterns = [
    path('', views.index, name='index'),
    path('result/<int:summary_id>/', views.result, name='result'),
    path('history/', views.SummaryHistoryView.as_view(), name='history'),
    path('compare/', views.compare_models, name='compare'),
    path('api/quick-summarize/', views.quick_summarize, name='quick_summarize'),
]
```

**config/urls.py**
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('summarizer.urls')),
]
```

## Step 7: Templates

**summarizer/templates/summarizer/index.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Text Summarizer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .card-hover:hover {
            transform: translateY(-5px);
            transition: all 0.3s ease;
        }
        .confidence-bar {
            height: 8px;
            background: linear-gradient(90deg, #ff4757, #ffa502, #26de81);
            border-radius: 4px;
        }
        .model-badge {
            font-size: 0.8em;
            border-radius: 15px;
        }
    </style>
</head>
<body class="bg-light">
    <div class="gradient-bg py-4 mb-5">
        <div class="container">
            <h1 class="display-4 text-center">
                <i class="fas fa-brain me-3"></i>AI Text Summarizer
            </h1>
            <p class="lead text-center">Transform lengthy documents into concise, meaningful summaries using advanced transformer models</p>
        </div>
    </div>

    <div class="container">
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}

        <div class="row">
            <div class="col-lg-8">
                <div class="card shadow-sm">
                    <div class="card-header bg-primary text-white">
                        <h3 class="card-title mb-0">
                            <i class="fas fa-edit me-2"></i>Summarize Your Text
                        </h3>
                    </div>
                    <div class="card-body">
                        <form method="post" id="summarization-form">
                            {% csrf_token %}
                            
                            <div class="mb-3">
                                <label for="{{ form.original_text.id_for_label }}" class="form-label">
                                    <i class="fas fa-file-text me-2"></i>Text to Summarize
                                </label>
                                {{ form.original_text }}
                                <div class="form-text">
                                    <span id="char-count">0</span> characters | 
                                    <span id="word-count">0</span> words
                                </div>
                                {% if form.original_text.errors %}
                                    <div class="invalid-feedback d-block">
                                        {% for error in form.original_text.errors %}
                                            {{ error }}
                                        {% endfor %}
                                    </div>
                                {% endif %}
                            </div>

                            <div class="row">
                                <div class="col-md-6 mb-3">
                                    <label for="{{ form.summary_type.id_for_label }}" class="form-label">
                                        <i class="fas fa-cogs me-2"></i>Summary Type
                                    </label>
                                    {{ form.summary_type }}
                                </div>
                                
                                <div class="col-md-6 mb-3">
                                    <label for="{{ form.model_choice.id_for_label }}" class="form-label">
                                        <i class="fas fa-robot me-2"></i>AI Model
                                    </label>
                                    {{ form.model_choice }}
                                </div>
                            </div>

                            <div class="row">
                                <div class="col-md-6 mb-3">
                                    <label for="{{ form.min_length.id_for_label }}" class="form-label">
                                        <i class="fas fa-compress-alt me-2"></i>Min Length (words)
                                    </label>
                                    {{ form.min_length }}
                                </div>
                                
                                <div class="col-md-6 mb-3">
                                    <label for="{{ form.max_length.id_for_label }}" class="form-label">
                                        <i class="fas fa-expand-alt me-2"></i>Max Length (words)
                                    </label>
                                    {{ form.max_length }}
                                </div>
                            </div>

                            <div class="d-grid">
                                <button type="submit" class="btn btn-primary btn-lg" id="summarize-btn">
                                    <i class="fas fa-magic me-2"></i>Generate Summary
                                    <span class="spinner-border spinner-border-sm ms-2 d-none" id="loading-spinner"></span>
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>

            <div class="col-lg-4">
                <div class="card shadow-sm mb-4">
                    <div class="card-header bg-success text-white">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-lightbulb me-2"></i>Quick Tips
                        </h5>
                    </div>
                    <div class="card-body">
                        <ul class="list-unstyled">
                            <li class="mb-2"><i class="fas fa-check text-success me-2"></i>Best results with 100+ words</li>
                            <li class="mb-2"><i class="fas fa-check text-success me-2"></i>BART works great for news articles</li>
                            <li class="mb-2"><i class="fas fa-check text-success me-2"></i>T5 is versatile for any content</li>
                            <li class="mb-2"><i class="fas fa-check text-success me-2"></i>Extractive keeps original phrases</li>
                        </ul>
                    </div>
                </div>

                {% if recent_summaries %}
                <div class="card shadow-sm">
                    <div class="card-header bg-info text-white">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-history me-2"></i>Recent Summaries
                        </h5>
                    </div>
                    <div class="card-body">
                        {% for summary in recent_summaries %}
                        <div class="border-bottom pb-2 mb-2">
                            <div class="d-flex justify-content-between align-items-start">
                                <div class="flex-grow-1">
                                    <p class="mb-1 small text-muted">
                                        {{ summary.request.created_at|date:"M d, H:i" }}
                                    </p>
                                    <p class="mb-1">{{ summary.summary_text|truncatechars:80 }}</p>
                                    <span class="badge model-badge bg-secondary">{{ summary.model_used|truncatechars:20 }}</span>
                                </div>
                                <div class="ms-2">
                                    <div class="confidence-bar" style="width: {{ summary.confidence_score|floatformat:0|add:"0" }}px"></div>
                                </div>
                            </div>
                        </div>
                        {% endfor %}
                        <a href="{% url 'summarizer:history' %}" class="btn btn-sm btn-outline-info">View All</a>
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Text counter functionality
        const textArea = document.getElementById('text-input');
        const charCount = document.getElementById('char-count');
        const wordCount = document.getElementById('word-count');
        const form = document.getElementById('summarization-form');
        const submitBtn = document.getElementById('summarize-btn');
        const loadingSpinner = document.getElementById('loading-spinner');

        function updateCounts() {
            const text = textArea.value;
            charCount.textContent = text.length;
            wordCount.textContent = text.trim() === '' ? 0 : text.trim().split(/\s+/).length;
        }

        textArea.addEventListener('input', updateCounts);

        // Form submission handler
        form.addEventListener('submit', function(e) {
            submitBtn.disabled = true;
            loadingSpinner.classList.remove('d-none');
            submitBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Processing...' + 
                                 '<span class="spinner-border spinner-border-sm ms-2"></span>';
        });

        // Initialize counts
        updateCounts();
    </script>
</body>
</html>

**summarizer/templates/summarizer/result.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summary Result - AI Text Summarizer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .metric-card {
            border-left: 4px solid #007bff;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        }
        .summary-box {
            background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        .original-text {
            background: #f8f9fa;
            border-radius: 8px;
            max-height: 300px;
            overflow-y: auto;
        }
        .confidence-indicator {
            height: 20px;
            border-radius: 10px;
            background: linear-gradient(90deg, #ff4757, #ffa502, #26de81);
            position: relative;
        }
        .confidence-pointer {
            position: absolute;
            top: -2px;
            width: 4px;
            height: 24px;
            background: #333;
            border-radius: 2px;
        }
    </style>
</head>
<body class="bg-light">
    <div class="gradient-bg py-3">
        <div class="container">
            <h1 class="h3 text-center mb-0">
                <i class="fas fa-check-circle me-2"></i>Summarization Complete
            </h1>
        </div>
    </div>

    <div class="container mt-4">
        <div class="row">
            <div class="col-12 mb-4">
                <nav aria-label="breadcrumb">
                    <ol class="breadcrumb">
                        <li class="breadcrumb-item"><a href="{% url 'summarizer:index' %}">Home</a></li>
                        <li class="breadcrumb-item active">Summary Result</li>
                    </ol>
                </nav>
            </div>
        </div>

        <div class="row">
            <!-- Summary Display -->
            <div class="col-lg-8">
                <div class="card shadow-sm mb-4">
                    <div class="card-header bg-success text-white">
                        <h3 class="card-title mb-0">
                            <i class="fas fa-file-text me-2"></i>Generated Summary
                        </h3>
                    </div>
                    <div class="card-body summary-box p-4">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <span class="badge bg-primary">{{ summary.model_used }}</span>
                            <span class="badge bg-info">{{ summary.request.summary_type|capfirst }}</span>
                        </div>
                        <div class="summary-text">
                            <p class="lead">{{ summary.summary_text|linebreaksbr }}</p>
                        </div>
                        <div class="mt-3 pt-3 border-top">
                            <small class="text-muted">
                                <i class="fas fa-clock me-1"></i>
                                Generated in {{ summary.request.processing_time|floatformat:2 }}s | 
                                <i class="fas fa-words me-1"></i>
                                {{ summary.word_count_summary }} words
                            </small>
                        </div>
                    </div>
                </div>

                <!-- Original Text -->
                <div class="card shadow-sm">
                    <div class="card-header bg-secondary text-white">
                        <h4 class="card-title mb-0">
                            <i class="fas fa-file-alt me-2"></i>Original Text
                        </h4>
                    </div>
                    <div class="card-body original-text p-3">
                        <p>{{ summary.request.original_text|linebreaksbr }}</p>
                    </div>
                </div>
            </div>

            <!-- Statistics -->
            <div class="col-lg-4">
                <!-- Performance Metrics -->
                <div class="card shadow-sm mb-4">
                    <div class="card-header bg-info text-white">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-chart-bar me-2"></i>Summary Analytics
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="metric-card p-3 mb-3">
                            <div class="d-flex justify-content-between">
                                <span><i class="fas fa-compress me-2"></i>Compression</span>
                                <strong>{{ summary.compression_ratio|floatformat:1 }}x</strong>
                            </div>
                            <small class="text-muted">{{ time_saved }}</small>
                        </div>

                        <div class="metric-card p-3 mb-3">
                            <div class="d-flex justify-content-between">
                                <span><i class="fas fa-stopwatch me-2"></i>Process Time</span>
                                <strong>{{ summary.request.processing_time|floatformat:2 }}s</strong>
                            </div>
                        </div>

                        <div class="metric-card p-3 mb-3">
                            <div class="mb-2">
                                <span><i class="fas fa-brain me-2"></i>AI Confidence</span>
                                <strong class="float-end">{{ summary.confidence_score|floatformat:1 }}%</strong>
                            </div>
                            <div class="confidence-indicator">
                                <div class="confidence-pointer" style="left: {{ summary.confidence_score|floatformat:0 }}%"></div>
                            </div>
                        </div>

                        <div class="row text-center">
                            <div class="col-6">
                                <div class="border-end">
                                    <h6 class="text-primary">{{ summary.word_count_original }}</h6>
                                    <small class="text-muted">Original Words</small>
                                </div>
                            </div>
                            <div class="col-6">
                                <h6 class="text-success">{{ summary.word_count_summary }}</h6>
                                <small class="text-muted">Summary Words</small>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="card shadow-sm">
                    <div class="card-body text-center">
                        <div class="d-grid gap-2">
                            <button class="btn btn-outline-primary" onclick="copySummary()">
                                <i class="fas fa-copy me-2"></i>Copy Summary
                            </button>
                            <button class="btn btn-outline-success" onclick="downloadSummary()">
                                <i class="fas fa-download me-2"></i>Download
                            </button>
                            <a href="{% url 'summarizer:index' %}" class="btn btn-primary">
                                <i class="fas fa-plus me-2"></i>New Summary
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function copySummary() {
            const summaryText = `{{ summary.summary_text|escapejs }}`;
            navigator.clipboard.writeText(summaryText).then(() => {
                alert('Summary copied to clipboard!');
            });
        }

        function downloadSummary() {
            const content = `AI Generated Summary\n` +
                          `Generated on: {{ summary.request.created_at|date:"F d, Y H:i" }}\n` +
                          `Model: {{ summary.model_used }}\n` +
                          `Type: {{ summary.request.summary_type }}\n\n` +
                          `SUMMARY:\n{{ summary.summary_text|escapejs }}\n\n` +
                          `ORIGINAL TEXT:\n{{ summary.request.original_text|escapejs }}`;
            
            const blob = new Blob([content], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'summary_{{ summary.id }}.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
```

**summarizer/templates/summarizer/history.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summary History - AI Text Summarizer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .summary-card {
            transition: transform 0.2s;
        }
        .summary-card:hover {
            transform: translateY(-2px);
        }
        .confidence-badge {
            font-size: 0.8em;
        }
    </style>
</head>
<body class="bg-light">
    <div class="gradient-bg py-3">
        <div class="container">
            <h1 class="h3 text-center mb-0">
                <i class="fas fa-history me-2"></i>Your Summary History
            </h1>
        </div>
    </div>

    <div class="container mt-4">
        <div class="row">
            <div class="col-12 mb-4">
                <nav aria-label="breadcrumb">
                    <ol class="breadcrumb">
                        <li class="breadcrumb-item"><a href="{% url 'summarizer:index' %}">Home</a></li>
                        <li class="breadcrumb-item active">History</li>
                    </ol>
                </nav>
            </div>
        </div>

        {% if summaries %}
            <div class="row">
                {% for summary in summaries %}
                <div class="col-lg-6 mb-4">
                    <div class="card summary-card shadow-sm h-100">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <small class="text-muted">
                                <i class="fas fa-calendar-alt me-1"></i>
                                {{ summary.request.created_at|date:"M d, Y H:i" }}
                            </small>
                            <div>
                                <span class="badge bg-primary confidence-badge">
                                    {{ summary.confidence_score|floatformat:0 }}% confidence
                                </span>
                            </div>
                        </div>
                        <div class="card-body">
                            <h6 class="card-subtitle mb-2 text-muted">
                                {{ summary.model_used }} | {{ summary.request.summary_type|capfirst }}
                            </h6>
                            <p class="card-text">{{ summary.summary_text|truncatechars:150 }}</p>
                            
                            <div class="row text-center mb-3">
                                <div class="col-4">
                                    <small class="text-muted">Original</small>
                                    <div class="fw-bold">{{ summary.word_count_original }}</div>
                                </div>
                                <div class="col-4">
                                    <small class="text-muted">Summary</small>
                                    <div class="fw-bold text-success">{{ summary.word_count_summary }}</div>
                                </div>
                                <div class="col-4">
                                    <small class="text-muted">Ratio</small>
                                    <div class="fw-bold text-primary">{{ summary.compression_ratio|floatformat:1 }}x</div>
                                </div>
                            </div>
                        </div>
                        <div class="card-footer bg-transparent">
                            <a href="{% url 'summarizer:result' summary.id %}" class="btn btn-outline-primary btn-sm">
                                <i class="fas fa-eye me-1"></i>View Full
                            </a>
                            <small class="text-muted float-end">
                                <i class="fas fa-clock me-1"></i>{{ summary.request.processing_time|floatformat:1 }}s
                            </small>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>

            <!-- Pagination -->
            {% if is_paginated %}
            <div class="d-flex justify-content-center">
                <nav aria-label="Summary pagination">
                    <ul class="pagination">
                        {% if page_obj.has_previous %}
                            <li class="page-item">
                                <a class="page-link" href="?page=1">First</a>
                            </li>
                            <li class="page-item">
                                <a class="page-link" href="?page={{ page_obj.previous_page_number }}">Previous</a>
                            </li>
                        {% endif %}

                        <li class="page-item active">
                            <span class="page-link">
                                Page {{ page_obj.number }} of {{ page_obj.paginator.num_pages }}
                            </span>
                        </li>

                        {% if page_obj.has_next %}
                            <li class="page-item">
                                <a class="page-link" href="?page={{ page_obj.next_page_number }}">Next</a>
                            </li>
                            <li class="page-item">
                                <a class="page-link" href="?page={{ page_obj.paginator.num_pages }}">Last</a>
                            </li>
                        {% endif %}
                    </ul>
                </nav>
            </div>
            {% endif %}

        {% else %}
            <div class="text-center py-5">
                <i class="fas fa-file-text fa-3x text-muted mb-3"></i>
                <h4 class="text-muted">No summaries yet</h4>
                <p class="text-muted">Start by creating your first summary!</p>
                <a href="{% url 'summarizer:index' %}" class="btn btn-primary">
                    <i class="fas fa-plus me-2"></i>Create Summary
                </a>
            </div>
        {% endif %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

## Step 8: Static Files

**static/css/style.css**
```css
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --success-gradient: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
    --warning-gradient: linear-gradient(135deg, #FFD89B 0%, #19547B 100%);
}

.btn-gradient-primary {
    background: var(--primary-gradient);
    border: none;
    color: white;
}

.btn-gradient-primary:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    color: white;
}

.card-gradient {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
}

.text-shadow {
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}

.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.fade-in {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.summary-highlight {
    background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #667eea;
}

.metric-animation {
    transition: all 0.3s ease;
}

.metric-animation:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
```

**static/js/main.js**
```javascript
document.addEventListener('DOMContentLoaded', function() {
    // Auto-resize textarea
    const textAreas = document.querySelectorAll('textarea');
    textAreas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Form validation enhancements
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const textArea = form.querySelector('textarea[required]');
            if (textArea && textArea.value.trim().length < 50) {
                e.preventDefault();
                alert('Please enter at least 50 characters for meaningful summarization.');
                textArea.focus();
            }
        });
    });

    // Tooltip initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Progress indicators
    function updateProgress(percentage) {
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = percentage + '%';
            progressBar.setAttribute('aria-valuenow', percentage);
        }
    }

    // Quick summary functionality
    function quickSummarize(text) {
        if (text.length < 50) {
            alert('Text too short for summarization');
            return;
        }

        fetch('/api/quick-summarize/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayQuickSummary(data.summary, data.processing_time, data.confidence_score);
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Network error occurred');
        });
    }

    function displayQuickSummary(summary, processingTime, confidence) {
        const modal = new bootstrap.Modal(document.getElementById('quickSummaryModal'));
        document.getElementById('quickSummaryText').textContent = summary;
        document.getElementById('quickProcessingTime').textContent = processingTime.toFixed(2) + 's';
        document.getElementById('quickConfidence').textContent = (confidence * 100).toFixed(1) + '%';
        modal.show();
    }

    // Export functionality
    window.exportSummary = function(format, summaryId) {
        const url = `/export/${format}/${summaryId}/`;
        window.open(url, '_blank');
    };

    // Theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-theme');
            localStorage.setItem('theme', 
                document.body.classList.contains('dark-theme') ? 'dark' : 'light'
            );
        });

        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-theme');
        }
    }
});
```

## Step 9: Admin Configuration

**summarizer/admin.py**
```python
from django.contrib import admin
from .models import SummaryRequest, GeneratedSummary

@admin.register(SummaryRequest)
class SummaryRequestAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'summary_type', 'created_at', 'processing_time']
    list_filter = ['summary_type', 'created_at']
    search_fields = ['user__username', 'original_text']
    readonly_fields = ['created_at', 'processing_time']
    
    fieldsets = (
        ('Request Details', {
            'fields': ('user', 'summary_type', 'original_text')
        }),
        ('Parameters', {
            'fields': ('max_length', 'min_length')
        }),
        ('Processing Info', {
            'fields': ('created_at', 'processing_time'),
            'classes': ('collapse',)
        }),
    )

@admin.register(GeneratedSummary)
class GeneratedSummaryAdmin(admin.ModelAdmin):
    list_display = ['id', 'request', 'model_used', 'word_count_summary', 'confidence_score', 'compression_ratio']
    list_filter = ['model_used']
    search_fields = ['summary_text', 'request__original_text']
    readonly_fields = ['word_count_original', 'word_count_summary', 'compression_ratio']
    
    fieldsets = (
        ('Summary Content', {
            'fields': ('request', 'summary_text')
        }),
        ('Model Information', {
            'fields': ('model_used', 'confidence_score')
        }),
        ('Statistics', {
            'fields': ('word_count_original', 'word_count_summary', 'compression_ratio'),
            'classes': ('collapse',)
        }),
    )
```

## Step 10: Management Commands

**summarizer/management/__init__.py**
```python
# Empty file
```

**summarizer/management/commands/__init__.py**
```python
# Empty file
```

**summarizer/management/commands/warm_up_models.py**
```python
from django.core.management.base import BaseCommand
from summarizer.ml_models import summarizer_instance

class Command(BaseCommand):
    help = 'Pre-load transformer models to improve first-request performance'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--models',
            nargs='+',
            default=['facebook/bart-large-cnn'],
            help='List of models to warm up'
        )
    
    def handle(self, *args, **options):
        models = options['models']
        
        self.stdout.write(
            self.style.SUCCESS(f'Warming up {len(models)} model(s)...')
        )
        
        test_text = """
        Artificial intelligence is transforming how we process and understand information.
        Modern transformer models like BART and T5 can effectively summarize long documents
        by identifying key concepts and generating coherent summaries that capture the essence
        of the original content.
        """
        
        for model_name in models:
            try:
                self.stdout.write(f'Loading model: {model_name}')
                
                # Load the model
                summarizer_instance.load_model(model_name)
                
                # Test with sample text
                result = summarizer_instance.summarize_text(
                    text=test_text,
                    model_name=model_name,
                    max_length=50,
                    min_length=20
                )
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f'✓ {model_name} loaded successfully '
                        f'(processed in {result["processing_time"]:.2f}s)'
                    )
                )
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'✗ Failed to load {model_name}: {str(e)}')
                )
        
        self.stdout.write(
            self.style.SUCCESS('Model warm-up completed!')
        )
```

## Final Project Integration

Create the Django project and run the application:

```bash
# Create project
django-admin startproject text_summarizer
cd text_summarizer
django-admin startapp summarizer

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Warm up models (optional, improves first-request performance)
python manage.py warm_up_models

# Run development server
python manage.py runserver
```

## Code Syntax Explanations

### Django Model Relationships
```python
# OneToOneField creates a 1:1 relationship between SummaryRequest and GeneratedSummary
summary = models.OneToOneField(SummaryRequest, on_delete=models.CASCADE, related_name='summary')

# ForeignKey creates a many-to-one relationship - many requests can belong to one user
user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
```

**Explanation**: The `OneToOneField` ensures each summary request has exactly one generated summary. The `related_name='summary'` allows accessing the summary from a request object using `request.summary`. The `on_delete=models.CASCADE` means when a request is deleted, its summary is also deleted.

### Transformer Model Loading
```python
# Dynamic model loading based on model name
if 'bart' in model_name.lower():
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
elif 't5' in model_name.lower():
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
```

**Explanation**: This conditional loading allows the system to use different transformer architectures. `from_pretrained()` downloads and loads pre-trained models from HuggingFace's model hub. Each model type requires specific tokenizer and model classes.

### Pipeline Creation
```python
# Create a summarization pipeline
summarizer_pipeline = pipeline(
    "summarization",  # Task type
    model=model,      # The loaded model
    tokenizer=tokenizer,  # Corresponding tokenizer
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)
```

**Explanation**: The `pipeline` function creates a high-level interface for the summarization task. `device=0` uses the first GPU, while `device=-1` uses CPU. This abstraction handles the complex tokenization and model inference steps.

### Text Chunking Algorithm
```python
def chunk_text(self, text, max_chunk_length=1000):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
```

**Explanation**: This algorithm breaks long texts into smaller chunks while preserving sentence boundaries. It accumulates sentences until the chunk would exceed `max_chunk_length`, then starts a new chunk. This prevents model input length limitations.

### Django Form Validation
```python
def clean_original_text(self):
    text = self.cleaned_data.get('original_text', '')
    
    if len(text.strip()) < 50:
        raise forms.ValidationError("Text must be at least 50 characters long")
    
    return text
```

**Explanation**: The `clean_<fieldname>()` method provides custom validation for specific fields. `cleaned_data` contains validated form data. `ValidationError` displays the error message to the user if validation fails.

### Model Property Methods
```python
def save(self, *args, **kwargs):
    # Calculate word counts automatically before saving
    self.word_count_original = len(self.request.original_text.split())
    self.word_count_summary = len(self.summary_text.split())
    if self.word_count_summary > 0:
        self.compression_ratio = self.word_count_original / self.word_count_summary
    super().save(*args, **kwargs)
```

**Explanation**: Overriding the `save()` method allows automatic calculation of derived fields. `super().save()` calls the parent class's save method to actually save to the database. This ensures data consistency.

### AJAX View Handling
```python
@require_http_methods(["POST"])
def quick_summarize(request):
    try:
        data = json.loads(request.body)  # Parse JSON request body
        text = data.get('text', '').strip()
        
        # Process and return JSON response
        return JsonResponse({
            'success': True,
            'summary': result['summary']
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
```

**Explanation**: The `@require_http_methods` decorator restricts the view to only accept POST requests. `json.loads(request.body)` parses the JSON request body. `JsonResponse` returns JSON data that JavaScript can easily consume.

### Template Context Processing
```python
context = {
    'summary': summary,
    'original_sentences': original_sentences,
    'summary_sentences': summary_sentences,
    'time_saved': f"{summary.compression_ratio:.1f}x faster to read",
}
```

**Explanation**: The context dictionary passes data from views to templates. Template variables like `{{ summary.summary_text }}` access this data. The `.1f` format specifier rounds floats to 1 decimal place.

### JavaScript Event Handling
```javascript
form.addEventListener('submit', function(e) {
    submitBtn.disabled = true;
    loadingSpinner.classList.remove('d-none');
    submitBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Processing...';
});
```

**Explanation**: `addEventListener` attaches event handlers to DOM elements. The anonymous function runs when the form is submitted. `classList.remove('d-none')` shows the loading spinner by removing Bootstrap's "display: none" class.

### CSS Custom Properties
```css
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.btn-gradient-primary {
    background: var(--primary-gradient);
}
```

**Explanation**: CSS custom properties (variables) defined in `:root` can be reused throughout the stylesheet using `var()`. This promotes consistency and makes theme changes easier.

### Bootstrap Grid System
```html
<div class="row">
    <div class="col-lg-8">
        <!-- Main content takes 8/12 columns on large screens -->
    </div>
    <div class="col-lg-4">
        <!-- Sidebar takes 4/12 columns on large screens -->
    </div>
</div>
```

**Explanation**: Bootstrap's 12-column grid system creates responsive layouts. `col-lg-8` means "take 8 columns on large screens and above". The grid automatically stacks on smaller screens.

This text summarization project demonstrates advanced Django concepts including model relationships, form validation, AJAX handling, custom management commands, and integration with machine learning models. The transformer architecture provides state-of-the-art summarization capabilities, while the Django framework handles user interface, data persistence, and web application logic.


## Assignment: Smart Recipe Analyzer

### The Challenge
Create a specialized transformer-based system that can analyze cooking recipes and provide intelligent insights. Your system should be able to:

1. **Analyze Recipe Complexity**: Use BERT to understand recipe structure and estimate cooking difficulty
2. **Generate Cooking Tips**: Use GPT-2 to generate helpful cooking suggestions based on ingredients
3. **Ingredient Relationship Mapping**: Use attention weights to visualize how different ingredients relate to each other
4. **Recipe Completion**: Given partial recipe instructions, complete the remaining steps

### Requirements

**Part 1: Recipe Complexity Analyzer (40 points)**
Create a function that takes a recipe text and returns:
- Estimated cooking time category (Quick: <30min, Medium: 30-60min, Complex: >60min)
- Difficulty score (1-10 scale)
- Required skill level (Beginner, Intermediate, Advanced)
- Key technique identification (chopping, sautéing, baking, etc.)

```python
def analyze_recipe_complexity(recipe_text):
    """
    Your implementation here
    Should return a dictionary with complexity metrics
    """
    pass
```

**Part 2: Intelligent Cooking Assistant (35 points)**
Build a GPT-based system that:
- Takes a list of available ingredients
- Generates 3 different recipe suggestions
- Provides cooking tips for each suggested recipe
- Estimates nutritional category (healthy, comfort food, etc.)

**Part 3: Attention Visualization (25 points)**
Create a visualization system that:
- Shows which words in a recipe the model pays most attention to
- Identifies the most important cooking steps
- Highlights ingredient relationships through attention patterns

### Submission Guidelines
1. **Code Implementation**: Submit clean, well-commented Python code
2. **Test Results**: Include results from testing with at least 5 different recipes
3. **Documentation**: Write a brief report (300-500 words) explaining your approach and findings
4. **Visualizations**: Include at least 2 attention visualizations showing different recipe types

### Sample Test Recipes
Use these recipes to test your system:

**Simple Recipe:**
"Heat oil in pan. Add garlic and cook 1 minute. Add tomatoes, salt, pepper. Simmer 10 minutes. Serve over pasta."

**Complex Recipe:**
"For the dough: Mix flour, yeast, salt, and water. Knead 10 minutes until smooth. Rise 1 hour. For filling: Sauté onions until caramelized, 20 minutes. Add herbs. Roll dough thin, add filling, fold into crescents. Brush with egg wash. Bake 375°F for 25 minutes until golden."

### Evaluation Criteria
- **Functionality (50%)**: Does the code work correctly and handle edge cases?
- **Insight Quality (30%)**: Are the analysis results meaningful and accurate?
- **Code Quality (20%)**: Is the code well-structured, documented, and efficient?

### Bonus Challenges (Optional)
- **Multi-language Support**: Handle recipes in different languages
- **Dietary Restriction Detection**: Identify if recipes are vegetarian, vegan, gluten-free, etc.
- **Ingredient Substitution Suggestions**: Recommend alternatives for missing ingredients

---

## Key Takeaways and Mastery Points

### What You've Mastered Today

**🧠 Transformer Architecture Deep Understanding**
You now understand how attention mechanisms work like a master chef who can simultaneously consider every ingredient, technique, and flavor combination. The scaled dot-product attention formula `Attention(Q,K,V) = softmax(QK^T/√d_k)V` is your recipe for creating models that truly understand context.

**🔧 Implementation Mastery**
You've built complete transformer components from scratch:
- Multi-head self-attention mechanisms
- Positional encoding systems
- Full transformer blocks with residual connections
- BERT and GPT model integration

**🚀 Production Integration**
Your Django application demonstrates real-world application of transformers:
- User-friendly interfaces for AI interaction
- Database storage for analysis results
- Scalable architecture for handling multiple requests
- Professional deployment-ready code structure

**📊 Advanced Concepts**
You've mastered sophisticated concepts like:
- Positional encoding mathematics and visualization
- Attention weight interpretation and analysis
- Model comparison (BERT vs GPT approaches)
- Performance optimization and error handling

### Syntax Mastery Checklist
✅ PyTorch tensor operations and neural network modules  
✅ Transformer architecture implementation patterns  
✅ Hugging Face transformers library integration  
✅ Django model-view-template architecture  
✅ Advanced Python class design for AI systems  
✅ Error handling in production AI applications  

### Real-World Applications
The skills you've developed today are directly applicable to:
- **Content Analysis Systems**: News summarization, sentiment analysis
- **Creative AI Tools**: Writing assistants, code generation
- **Educational Platforms**: Automated tutoring, content recommendation
- **Business Intelligence**: Document analysis, market research automation
- **Healthcare**: Medical text analysis, research paper summarization

### Next Steps in Your AI Journey
1. **Experiment with Fine-tuning**: Adapt pre-trained models for specific domains
2. **Explore Advanced Architectures**: Study GPT-4, T5, and other cutting-edge models
3. **Scale Your Applications**: Learn about distributed training and model serving
4. **Contribute to Open Source**: Share your implementations with the AI community

Remember, you've not just learned to use transformers - you've learned to think like the master chef of AI, understanding both the individual ingredients and how they combine to create something greater than the sum of their parts. The attention mechanism you've mastered today is the foundation for the most advanced AI systems in the world.

**🎯 Final Project Preview**: Your assignment will challenge you to apply all these concepts in a creative, practical way that demonstrates your deep understanding of transformer architecture and its real-world applications.
    The master attention system - like having chefs who can 
    focus on multiple ingredients simultaneously
    """
    def __init__(self, d_model, n_heads):
        super(AttentionMechanism, self).__init__()
        self.d_model = d_model  # Dimension of our ingredient space
        self.n_heads = n_heads  # Number of specialized chef stations
        self.d_k = d_model // n_heads  # Each station's focus capacity
        
        # Our specialized preparation stations
        self.W_q = nn.Linear(d_model, d_model)  # Query preparation
        self.W_k = nn.Linear(d_model, d_model)  # Key identification  
        self.W_v = nn.Linear(d_model, d_model)  # Value extraction
        self.W_o = nn.Linear(d_model, d_model)  # Final combination
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        The core attention recipe - how chefs decide what to focus on
        """
        # Calculate attention scores (recipe compatibility)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (some ingredients shouldn't mix)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Convert scores to probabilities (attention weights)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values (focus on important ingredients)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Prepare our queries, keys, and values
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention mechanism
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine all chef stations' outputs
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Final preparation step
        output = self.W_o(attention_output)
        
        return output, attention_weights

# Example usage - Setting up our attention system
d_model = 512  # Our ingredient space dimension
n_heads = 8    # Number of specialized stations
seq_length = 10  # Number of ingredients in our recipe
batch_size = 2   # Number of dishes we're preparing

# Create our attention mechanism
attention_layer = AttentionMechanism(d_model, n_heads)

# Sample input (ingredients encoded as vectors)
sample_input = torch.randn(batch_size, seq_length, d_model)

# Process through attention
output, weights = attention_layer(sample_input, sample_input, sample_input)

print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

**Syntax Explanation:**
- `nn.Module`: Base class for all neural network components in PyTorch
- `nn.Linear(in_features, out_features)`: Creates a linear transformation layer
- `.transpose(-2, -1)`: Swaps the last two dimensions of a tensor
- `.view()`: Reshapes tensors without changing data
- `torch.matmul()`: Matrix multiplication operation
- `torch.softmax(dim=-1)`: Applies softmax along the last dimension

---

## Lesson 2: Transformer Architecture Components

### The Complete Cooking Station Setup

Now let's build the complete cooking station - a transformer block that combines multiple specialized techniques.

```python
class TransformerBlock(nn.Module):
    """
    A complete cooking station with all necessary components
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Our main attention system
        self.attention = AttentionMechanism(d_model, n_heads)
        
        # Preparation enhancement layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Flavor enhancement network (feed-forward)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Consistency control (dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # First preparation step: attention with residual connection
        attention_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Second preparation step: flavor enhancement with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class PositionalEncoding(nn.Module):
    """
    Adds position information - like knowing the order of cooking steps
    """
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create position encoding table
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create the encoding pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even positions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd positions
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Building our complete transformer setup
class SimpleTransformer(nn.Module):
    """
    A complete cooking facility with multiple stations
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length):
        super(SimpleTransformer, self).__init__()
        
        # Ingredient vocabulary (word embeddings)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Position tracking system
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Multiple cooking stations (transformer blocks)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        # Final preparation layer
        self.ln_final = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Convert ingredients to rich representations
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        
        # Add position information
        x = self.pos_encoding(x)
        
        # Process through all cooking stations
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
            
        # Final preparation
        return self.ln_final(x)

# Example setup
vocab_size = 10000
d_model = 512
n_heads = 8
n_layers = 6
d_ff = 2048
max_seq_length = 1000

model = SimpleTransformer(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length)

# Test with sample data
sample_tokens = torch.randint(0, vocab_size, (2, 50))  # 2 recipes, 50 ingredients each
output = model(sample_tokens)

print(f"Model output shape: {output.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Syntax Explanation:**
- `nn.LayerNorm`: Normalizes layer inputs for stable training
- `nn.Sequential`: Chains multiple layers together
- `nn.ReLU()`: Rectified Linear Unit activation function
- `nn.Dropout(p)`: Randomly zeros some elements during training
- `nn.ModuleList`: Container for multiple sub-modules
- `register_buffer`: Stores tensors that aren't parameters but need to be part of the model

---

## Lesson 3: BERT and GPT Model Families

### Two Master Chef Approaches

In our culinary world, we have two legendary chef schools - each with their own philosophy about how to create the perfect dish.

```python
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
import torch

class BERTChef:
    """
    The master chef who sees the entire recipe at once
    Specializes in understanding context from all directions
    """
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
    def analyze_recipe(self, text):
        """
        Analyzes a complete recipe understanding all ingredients simultaneously
        """
        # Prepare the ingredients (tokenize)
        inputs = self.tokenizer(text, return_tensors='pt', 
                              padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            # The chef examines everything at once
            outputs = self.model(**inputs)
            
        # Extract the understanding (last hidden states)
        recipe_understanding = outputs.last_hidden_state
        pooled_understanding = outputs.pooler_output
        
        return {
            'detailed_analysis': recipe_understanding,
            'overall_understanding': pooled_understanding,
            'attention_weights': outputs.attentions if hasattr(outputs, 'attentions') else None
        }

class GPTChef:
    """
    The master chef who creates recipes one ingredient at a time
    Specializes in generating new combinations based on what came before
    """
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def create_recipe(self, prompt, max_length=100, temperature=0.8):
        """
        Creates a new recipe by predicting the next best ingredient
        """
        # Start with the initial idea
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            # Generate new recipe step by step
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=1
            )
            
        # Decode the created recipe
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Demonstration of both chef approaches
def demonstrate_chef_approaches():
    """
    Shows how both chef styles work on the same culinary challenge
    """
    # Initialize our master chefs
    bert_chef = BERTChef()
    gpt_chef = GPTChef()
    
    # A sample recipe to analyze
    sample_recipe = """
    To create the perfect pasta dish, you need fresh tomatoes, basil, and garlic. 
    The secret is in balancing the acidity of tomatoes with the aromatic herbs.
    """
    
    # BERT Chef Analysis (understanding the complete recipe)
    print("=== BERT Chef Analysis ===")
    bert_analysis = bert_chef.analyze_recipe(sample_recipe)
    print(f"Recipe understanding shape: {bert_analysis['detailed_analysis'].shape}")
    print(f"Overall comprehension: {bert_analysis['overall_understanding'].shape}")
    
    # GPT Chef Creation (generating new content)
    print("\n=== GPT Chef Creation ===")
    recipe_start = "The secret to perfect pasta is"
    generated_recipe = gpt_chef.create_recipe(recipe_start, max_length=80)
    print(f"Generated recipe: {generated_recipe}")
    
    return bert_analysis, generated_recipe

# Run the demonstration
bert_result, gpt_result = demonstrate_chef_approaches()
```

**Syntax Explanation:**
- `from_pretrained()`: Loads pre-trained models from Hugging Face
- `return_tensors='pt'`: Returns PyTorch tensors
- `torch.no_grad()`: Disables gradient computation for inference
- `.eval()`: Sets model to evaluation mode (disables dropout, batch norm updates)
- `**inputs`: Unpacks dictionary as keyword arguments
- `hasattr(obj, 'attr')`: Checks if object has specified attribute

---

## Lesson 4: Positional Encoding and Self-Attention

### The Art of Sequence and Focus

In our culinary masterpiece, understanding both the order of operations and which elements to focus on is crucial for creating the perfect dish.

```python
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedPositionalEncoding(nn.Module):
    """
    Sophisticated position tracking system for our cooking sequences
    """
    def __init__(self, d_model, max_seq_length=10000, dropout=0.1):
        super(AdvancedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create the position encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create different frequency patterns for position encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine and cosine functions with different frequencies
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add position information to our ingredient representations
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    def visualize_encoding(self, seq_length=100):
        """
        Visualize how position encoding looks like
        """
        pe_sample = self.pe[:seq_length, :64].squeeze().numpy()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pe_sample.T, cmap='RdYlBu', center=0)
        plt.title('Positional Encoding Visualization\n(Each row is a dimension, each column is a position)')
        plt.xlabel('Position in Sequence')
        plt.ylabel('Encoding Dimension')
        plt.tight_layout()
        plt.show()
        
        return pe_sample

class MultiHeadSelfAttention(nn.Module):
    """
    Advanced attention system where each head specializes in different aspects
    Like having specialized chefs for different types of ingredient relationships
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Specialized transformation stations
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None, dropout=None):
        """
        The core attention mechanism with detailed tracking
        """
        d_k = Q.size(-1)
        
        # Calculate compatibility scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Convert to attention probabilities
        p_attn = torch.softmax(scores, dim=-1)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
            
        # Apply attention to values
        output = torch.matmul(p_attn, V)
        
        return output, p_attn
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Transform inputs through our specialized stations
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention mechanism
        x, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        
        # Combine all head outputs
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_k
        )
        
        # Final transformation
        output = self.W_o(x)
        
        return output, attention_weights
    
    def visualize_attention(self, input_sequence, tokens=None):
        """
        Visualize what the attention mechanism is focusing on
        """
        self.eval()
        with torch.no_grad():
            output, attention_weights = self.forward(input_sequence, input_sequence, input_sequence)
            
        # Get attention weights from the first head
        attn_weights = attention_weights[0, 0].cpu().numpy()  # [seq_len, seq_len]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights, 
                    xticklabels=tokens if tokens else range(attn_weights.shape[1]),
                    yticklabels=tokens if tokens else range(attn_weights.shape[0]),
                    cmap='Blues')
        plt.title('Self-Attention Visualization\n(Darker = More Attention)')
        plt.xlabel('Keys (What we attend to)')
        plt.ylabel('Queries (What is attending)')
        plt.tight_layout()
        plt.show()
        
        return attn_weights

# Comprehensive example demonstrating both concepts
def demonstrate_position_and_attention():
    """
    Shows how position encoding and self-attention work together
    """
    # Model parameters
    d_model = 512
    n_heads = 8
    seq_length = 20
    batch_size = 1
    
    # Create components
    pos_encoder = AdvancedPositionalEncoding(d_model)
    attention = MultiHeadSelfAttention(d_model, n_heads)
    
    # Create sample input (like a sequence of ingredient embeddings)
    sample_embeddings = torch.randn(seq_length, batch_size, d_model)
    
    # Apply positional encoding
    positioned_embeddings = pos_encoder(sample_embeddings)
    
    # Transpose for attention (batch_first format)
    positioned_embeddings = positioned_embeddings.transpose(0, 1)
    
    # Apply self-attention
    attended_output, attn_weights = attention(
        positioned_embeddings, positioned_embeddings, positioned_embeddings
    )
    
    print("=== Position and Attention Integration ===")
    print(f"Original embeddings shape: {sample_embeddings.shape}")
    print(f"After position encoding: {positioned_embeddings.shape}")
    print(f"After self-attention: {attended_output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Visualize position encoding patterns (optional, requires matplotlib)
    try:
        pos_encoder.visualize_encoding(seq_length=50)
    except:
        print("Visualization skipped (matplotlib not available)")
    
    return positioned_embeddings, attended_output, attn_weights

# Run demonstration
embeddings, output, weights = demonstrate_position_and_attention()
```

**Syntax Explanation:**
- `assert condition`: Raises AssertionError if condition is False
- `.register_buffer()`: Registers a tensor that should be part of the module but not a parameter
- `.contiguous()`: Ensures tensor memory is contiguous for certain operations
- `.transpose(dim1, dim2)`: Swaps two dimensions
- `torch.arange()`: Creates a tensor with evenly spaced values
- `.unsqueeze(dim)`: Adds a dimension of size 1 at specified position

---

## Django Integration Project: Intelligent Text Processor

### Building a Production-Ready Culinary AI System

Now let's create a Django application that brings all our transformer knowledge together in a real-world cooking platform.

```python
# models.py - Our data recipe book
from django.db import models
from django.contrib.auth.models import User
import json

class TextAnalysisRequest(models.Model):
    """
    Stores requests for our AI cooking analysis
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    input_text = models.TextField(help_text="The recipe text to analyze")
    analysis_type = models.CharField(
        max_length=20,
        choices=[
            ('bert_analysis', 'Deep Understanding (BERT)'),
            ('gpt_generation', 'Creative Generation (GPT)'),
            ('attention_viz', 'Attention Visualization'),
            ('similarity', 'Recipe Similarity Analysis')
        ],
        default='bert_analysis'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.analysis_type} - {self.created_at}"

class AnalysisResult(models.Model):
    """
    Stores the results of our AI analysis
    """
    request = models.OneToOneField(TextAnalysisRequest, on_delete=models.CASCADE)
    result_data = models.JSONField(help_text="Analysis results in JSON format")
    confidence_score = models.FloatField(default=0.0)
    processing_time = models.FloatField(help_text="Time taken in seconds")
    
    def get_formatted_result(self):
        """Helper method to format results for display"""
        return json.dumps(self.result_data, indent=2)

# views.py - Our cooking service handlers
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.contrib import messages
import torch
import time
import json
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel

class TransformerService:
    """
    Our master AI cooking service that handles all transformer operations
    """
    def __init__(self):
        # Initialize our chef models
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Set models to evaluation mode
        self.bert_model.eval()
        self.gpt2_model.eval()
        
        # Handle GPT-2 padding token
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
    
    def analyze_with_bert(self, text):
        """
        Deep understanding analysis using BERT
        """
        start_time = time.time()
        
        # Tokenize input
        inputs = self.bert_tokenizer(
            text, 
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        # Extract key information
        last_hidden_states = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # Calculate some meaningful metrics
        attention_summary = self._summarize_attention(last_hidden_states)
        
        processing_time = time.time() - start_time
        
        result = {
            'analysis_type': 'BERT Deep Understanding',
            'text_length': len(text),
            'token_count': inputs['input_ids'].size(1),
            'embedding_dimensions': last_hidden_states.size(-1),
            'attention_summary': attention_summary,
            'confidence_indicators': {
                'text_complexity': min(len(text.split()) / 100.0, 1.0),
                'semantic_richness': float(torch.mean(torch.std(last_hidden_states, dim=1)).item())
            }
        }
        
        return result, processing_time
    
    def generate_with_gpt(self, prompt, max_length=150):
        """
        Creative generation using GPT-2
        """
        start_time = time.time()
        
        # Encode input
        inputs = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.gpt2_model.generate(
                inputs,
                max_length=max_length,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.gpt2_tokenizer.pad_token_id,
                num_return_sequences=1,
                repetition_penalty=1.2
            )
        
        generated_text = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        processing_time = time.time() - start_time
        
        result = {
            'analysis_type': 'GPT Creative Generation',
            'original_prompt': prompt,
            'generated_text': generated_text,
            'generation_stats': {
                'input_tokens': len(inputs[0]),
                'output_tokens': len(outputs[0]),
                'creativity_score': 0.8  # Placeholder for actual creativity measurement
            }
        }
        
        return result, processing_time
    
    def _summarize_attention(self, hidden_states):
        """
        Creates a summary of attention patterns
        """
        # Calculate basic statistics about the hidden representations
        mean_activation = torch.mean(hidden_states).item()
        max_activation = torch.max(hidden_states).item()
        min_activation = torch.min(hidden_states).item()
        std_activation = torch.std(hidden_states).item()
        
        return {
            'mean_activation': round(mean_activation, 4),
            'max_activation': round(max_activation, 4),
            'min_activation': round(min_activation, 4),
            'std_activation': round(std_activation, 4),
            'activation_range': round(max_activation - min_activation, 4)
        }

# Initialize our service
transformer_service = TransformerService()

@login_required
def text_analysis_dashboard(request):
    """
    Main dashboard for our AI cooking platform
    """
    user_requests = TextAnalysisRequest.objects.filter(user=request.user).order_by('-created_at')[:10]
    
    context = {
        'recent_requests': user_requests,
        'analysis_types': TextAnalysisRequest._meta.get_field('analysis_type').choices
    }
    
    return render(request, 'analysis/dashboard.html', context)

@login_required
def submit_analysis(request):
    """
    Handle submission of new analysis requests
    """
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '').strip()
        analysis_type = request.POST.get('analysis_type', 'bert_analysis')
        
        if not input_text:
            messages.error(request, 'Please provide text to analyze.')
            return redirect('text_analysis_dashboard')
        
        # Create the analysis request
        analysis_request = TextAnalysisRequest.objects.create(
            user=request.user,
            input_text=input_text,
            analysis_type=analysis_type
        )
        
        # Process the request
        try:
            if analysis_type == 'bert_analysis':
                result_data, processing_time = transformer_service.analyze_with_bert(input_text)
                confidence_score = result_data['confidence_indicators']['semantic_richness']
                
            elif analysis_type == 'gpt_generation':
                result_data, processing_time = transformer_service.generate_with_gpt(input_text)
                confidence_score = result_data['generation_stats']['creativity_score']
                
            else:
                result_data = {'error': 'Unsupported analysis type'}
                processing_time = 0
                confidence_score = 0
            
            # Save the results
            AnalysisResult.objects.create(
                request=analysis_request,
                result_data=result_data,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
            # Mark request as processed
            analysis_request.processed_at = timezone.now()
            analysis_request.save()
            
            messages.success(request, f'Analysis completed in {processing_time:.2f} seconds!')
            return redirect('view_result', request_id=analysis_request.id)
            
        except Exception as e:
            messages.error(request, f'Analysis failed: {str(e)}')
            return redirect('text_analysis_dashboard')
    
    return redirect('text_analysis_dashboard')

@login_required
def view_result(request, request_id):
    """
    Display analysis results
    """
    analysis_request = get_object_or_404(TextAnalysisRequest, id=request_id, user=request.user)
    
    try:
        result = AnalysisResult.objects.get(request=analysis_request)
    except AnalysisResult.DoesNotExist:
        messages.error(request, 'Analysis result not found.')
        return redirect('text_analysis_dashboard')
    
    context = {
        'analysis_request': analysis_request,
        'result': result,
        'formatted_result': result.get_formatted_result()
    }
    
    return render(request, 'analysis/result.html', context)

# urls.py - Our routing system
from django.urls import path
from . import views

urlpatterns = [
    path('', views.text_analysis_dashboard, name='text_analysis_dashboard'),
    path('submit/', views.submit_analysis, name='submit_analysis'),
    path('result/<int:request_id>/', views.view_result, name='view_result'),
]

# settings.py additions - Configuration for our AI cooking platform
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'analysis',  # Our transformer analysis app
]

# Templates for our Django application

# templates/analysis/dashboard.html
"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Text Analysis - Transformer Kitchen</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1200px; margin: 0 auto; 
            background: white; padding: 30px; 
            border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header { text-align: center; margin-bottom: 30px; }
        .analysis-form { 
            background: #f8f9fa; padding: 25px; 
            border-radius: 10px; margin-bottom: 30px;
            border-left: 5px solid #667eea;
        }
        textarea { 
            width: 100%; min-height: 150px; padding: 15px; 
            border: 2px solid #e9ecef; border-radius: 8px;
            font-family: inherit; font-size: 14px;
        }
        select, button { 
            padding: 12px 20px; margin: 10px 5px 10px 0; 
            border: 2px solid #667eea; border-radius: 8px;
        }
        button { 
            background: #667eea; color: white; 
            cursor: pointer; font-weight: bold;
            transition: all 0.3s ease;
        }
        button:hover { 
            background: #764ba2; transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .recent-requests { margin-top: 30px; }
        .request-item { 
            background: #f8f9fa; padding: 15px; 
            margin: 10px 0; border-radius: 8px;
            border-left: 4px solid #28a745;
        }
        .messages { margin: 20px 0; }
        .alert { 
            padding: 15px; border-radius: 8px; margin: 10px 0;
        }
        .alert-success { 
            background: #d4edda; color: #155724; 
            border: 1px solid #c3e6cb;
        }
        .alert-error { 
            background: #f8d7da; color: #721c24; 
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 AI Transformer Kitchen</h1>
            <p>Master the art of text understanding with BERT and GPT</p>
        </div>
        
        {% if messages %}
            <div class="messages">
                {% for message in messages %}
                    <div class="alert alert-{{ message.tags }}">
                        {{ message }}
                    </div>
                {% endfor %}
            </div>
        {% endif %}
        
        <div class="analysis-form">
            <h3>🔍 Submit New Analysis</h3>
            <form method="post" action="{% url 'submit_analysis' %}">
                {% csrf_token %}
                
                <div style="margin-bottom: 20px;">
                    <label for="input_text"><strong>Text to Analyze:</strong></label>
                    <textarea name="input_text" id="input_text" 
                              placeholder="Enter your text here for AI analysis... (e.g., a recipe, article, or any text you want to understand better)"
                              required></textarea>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <label for="analysis_type"><strong>Analysis Type:</strong></label>
                    <select name="analysis_type" id="analysis_type">
                        {% for value, label in analysis_types %}
                            <option value="{{ value }}">{{ label }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <button type="submit">🚀 Start Analysis</button>
            </form>
        </div>
        
        <div class="recent-requests">
            <h3>📊 Recent Analyses</h3>
            {% for request in recent_requests %}
                <div class="request-item">
                    <strong>{{ request.get_analysis_type_display }}</strong>
                    <br>
                    <small>{{ request.created_at|date:"M d, Y H:i" }}</small>
                    <br>
                    <em>{{ request.input_text|truncatewords:15 }}</em>
                    {% if request.processed_at %}
                        <br>
                        <a href="{% url 'view_result' request.id %}" 
                           style="color: #667eea; text-decoration: none; font-weight: bold;">
                            → View Results
                        </a>
                    {% else %}
                        <br>
                        <span style="color: #ffc107;">⏳ Processing...</span>
                    {% endif %}
                </div>
            {% empty %}
                <p style="text-align: center; color: #6c757d;">
                    No analyses yet. Submit your first text above! 🎯
                </p>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

# templates/analysis/result.html
"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results - Transformer Kitchen</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1200px; margin: 0 auto; 
            background: white; padding: 30px; 
            border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header { text-align: center; margin-bottom: 30px; }
        .result-section { 
            background: #f8f9fa; padding: 25px; 
            border-radius: 10px; margin: 20px 0;
            border-left: 5px solid #28a745;
        }
        .original-text { 
            background: #fff3cd; padding: 20px; 
            border-radius: 8px; margin: 20px 0;
            border-left: 5px solid #ffc107;
        }
        .metrics { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; margin: 20px 0;
        }
        .metric-card { 
            background: white; padding: 20px; 
            border-radius: 8px; text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-value { 
            font-size: 2em; font-weight: bold; 
            color: #667eea; margin-bottom: 5px;
        }
        .json-output { 
            background: #2d3748; color: #e2e8f0; 
            padding: 20px; border-radius: 8px; 
            overflow-x: auto; font-family: 'Courier New', monospace;
            font-size: 14px; line-height: 1.4;
        }
        .back-button { 
            display: inline-block; padding: 12px 20px; 
            background: #667eea; color: white; 
            text-decoration: none; border-radius: 8px;
            font-weight: bold; margin: 20px 0;
            transition: all 0.3s ease;
        }
        .back-button:hover { 
            background: #764ba2; transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Analysis Results</h1>
            <p>{{ analysis_request.get_analysis_type_display }}</p>
        </div>
        
        <a href="{% url 'text_analysis_dashboard' %}" class="back-button">
            ← Back to Dashboard
        </a>
        
        <div class="original-text">
            <h3>📝 Original Text</h3>
            <p>{{ analysis_request.input_text }}</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{{ result.confidence_score|floatformat:3 }}</div>
                <div>Confidence Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ result.processing_time|floatformat:2 }}s</div>
                <div>Processing Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ analysis_request.created_at|date:"M d" }}</div>
                <div>Analysis Date</div>
            </div>
        </div>
        
        <div class="result-section">
            <h3>🔍 Detailed Results</h3>
            <div class="json-output">{{ formatted_result }}</div>
        </div>
    </div>
</body>
</html>
"""