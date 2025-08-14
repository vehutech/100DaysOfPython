# Day 92: Advanced NLP Applications
## AI Mastery Course - Python Core & Django Web Framework

### Learning Objective
By the end of this lesson, you will master advanced Natural Language Processing techniques including Named Entity Recognition, question-answering systems, and chatbot development. You'll understand how to combine these components like a master chef orchestrating multiple cooking techniques to create sophisticated language applications.

---

## Imagine That...

Imagine you're the head chef of a prestigious restaurant where every dish tells a story. Your diners don't just want food—they want conversations, answers to their questions, and personalized experiences. As a culinary artist, you need to understand not just what ingredients your guests mention, but also extract the essence of their requests, answer their queries about your menu, and engage them in delightful conversation.

Just as a chef must identify premium ingredients (entities), understand complex recipe requests (questions), and maintain engaging dialogue with guests, today we'll learn how to build intelligent systems that can extract meaningful information from text, answer questions accurately, and create conversational experiences that feel natural and helpful.

---

## 1. Named Entity Recognition (NER)

Think of NER as the ability to identify and categorize the finest ingredients in any recipe or conversation. Just as a skilled chef can instantly recognize saffron, truffle oil, or aged parmesan in a complex dish, NER helps our applications identify and classify important entities in text.

### Code Example: Building a NER System

```python
# install required packages first
# pip install spacy transformers torch
# python -m spacy download en_core_web_sm

import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import django
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

class EntityExtractor:
    """
    A sophisticated ingredient identification system for text analysis
    Like a chef's palate that can identify every component in a complex sauce
    """
    
    def __init__(self):
        # Load pre-trained models - our trained palate
        self.spacy_nlp = spacy.load("en_core_web_sm")
        
        # Initialize transformer model for more complex entity recognition
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.ner_pipeline = pipeline("ner", 
                                   model=self.model, 
                                   tokenizer=self.tokenizer,
                                   aggregation_strategy="simple")
    
    def extract_basic_entities(self, text):
        """
        Extract basic entities using spaCy
        Like identifying common ingredients at first glance
        """
        doc = self.spacy_nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_),
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def extract_advanced_entities(self, text):
        """
        Extract entities using transformer models
        Like a master chef detecting subtle flavor notes
        """
        entities = self.ner_pipeline(text)
        
        # Group and clean the results
        grouped_entities = []
        for entity in entities:
            grouped_entities.append({
                'text': entity['word'],
                'label': entity['entity_group'],
                'confidence': round(entity['score'], 4),
                'start': entity['start'],
                'end': entity['end']
            })
        
        return grouped_entities

# Django view for NER API
@csrf_exempt
def extract_entities(request):
    """
    Django view to handle entity extraction requests
    Like a service window where guests can ask about ingredients
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '')
            
            extractor = EntityExtractor()
            
            # Extract using both methods
            basic_entities = extractor.extract_basic_entities(text)
            advanced_entities = extractor.extract_advanced_entities(text)
            
            return JsonResponse({
                'success': True,
                'text': text,
                'basic_entities': basic_entities,
                'advanced_entities': advanced_entities,
                'entity_count': len(basic_entities) + len(advanced_entities)
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'error': 'Method not allowed'})
```

**Syntax Explanation:**
- `spacy.load()`: Loads a pre-trained language model
- `pipeline()`: Creates a Hugging Face transformer pipeline for NER
- `@csrf_exempt`: Django decorator to bypass CSRF protection for API endpoints
- `json.loads()`: Parses JSON string into Python dictionary
- `JsonResponse()`: Django's JSON response class

---

## 2. Question Answering Systems

Building a question-answering system is like training a sommelier who can instantly provide detailed information about any wine in an extensive cellar. The system must understand the context (the wine collection) and provide accurate, relevant answers to specific queries.

### Code Example: Question Answering System

```python
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from django.views.decorators.http import require_http_methods
import re

class IntelligentSommelier:
    """
    An AI sommelier that can answer questions about any topic
    Like a wine expert with encyclopedic knowledge
    """
    
    def __init__(self):
        # Initialize the question-answering pipeline
        self.qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
        
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.qa_model,
            tokenizer=self.qa_tokenizer,
            return_confidence_score=True
        )
    
    def find_answer(self, question, context, max_length=512):
        """
        Find answers within given context
        Like consulting your knowledge base for the perfect response
        """
        # Truncate context if too long
        if len(context) > max_length:
            context = context[:max_length]
        
        try:
            result = self.qa_pipeline({
                'question': question,
                'context': context
            })
            
            return {
                'answer': result['answer'],
                'confidence': round(result['score'], 4),
                'start': result['start'],
                'end': result['end']
            }
            
        except Exception as e:
            return {
                'answer': 'I apologize, but I cannot find a suitable answer in the provided context.',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def enhance_answer(self, answer_data, question):
        """
        Enhance the answer with additional context
        Like garnishing a dish with complementary elements
        """
        if answer_data['confidence'] > 0.7:
            confidence_level = "highly confident"
        elif answer_data['confidence'] > 0.4:
            confidence_level = "moderately confident"
        else:
            confidence_level = "less confident"
        
        enhanced_response = {
            'original_question': question,
            'answer': answer_data['answer'],
            'confidence_score': answer_data['confidence'],
            'confidence_level': confidence_level,
            'reliable': answer_data['confidence'] > 0.5
        }
        
        return enhanced_response

# Django view for question answering
@require_http_methods(["POST"])
@csrf_exempt
def answer_question(request):
    """
    Django endpoint for question answering
    Like a knowledgeable waiter ready to answer any menu question
    """
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        context = data.get('context', '').strip()
        
        if not question or not context:
            return JsonResponse({
                'success': False,
                'error': 'Both question and context are required'
            })
        
        sommelier = IntelligentSommelier()
        answer_data = sommelier.find_answer(question, context)
        enhanced_answer = sommelier.enhance_answer(answer_data, question)
        
        return JsonResponse({
            'success': True,
            'data': enhanced_answer
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Processing error: {str(e)}'
        })
```

**Syntax Explanation:**
- `AutoTokenizer.from_pretrained()`: Loads a pre-trained tokenizer
- `AutoModelForQuestionAnswering.from_pretrained()`: Loads a pre-trained Q&A model
- `return_confidence_score=True`: Enables confidence scoring in pipeline
- `require_http_methods(["POST"])`: Django decorator limiting allowed HTTP methods
- `strip()`: Removes whitespace from string ends

---

## 3. Dialogue Systems and Chatbots

Creating a chatbot is like training a host who can engage guests in meaningful conversation, remember context, and respond appropriately to various conversational cues. Our chatbot will maintain conversation flow while providing helpful and contextually relevant responses.

### Code Example: Conversational Chatbot

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from django.views.decorators.cache import cache_page
import uuid
from datetime import datetime

class ConversationHost:
    """
    An AI conversation host that maintains engaging dialogue
    Like a skilled host who remembers guests and adapts to their preferences
    """
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.conversation_generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Memory system - like remembering regular guests' preferences
        self.conversation_memory = {}
    
    def initialize_conversation(self, user_id=None):
        """
        Start a new conversation session
        Like greeting a new guest and preparing for their visit
        """
        if not user_id:
            user_id = str(uuid.uuid4())
        
        self.conversation_memory[user_id] = {
            'history': [],
            'context': '',
            'started_at': datetime.now(),
            'turn_count': 0
        }
        
        return user_id
    
    def generate_response(self, user_input, user_id, max_length=100):
        """
        Generate contextual response
        Like crafting the perfect reply based on the conversation flow
        """
        if user_id not in self.conversation_memory:
            user_id = self.initialize_conversation(user_id)
        
        conversation = self.conversation_memory[user_id]
        
        # Build conversation context
        context = conversation['context']
        if context:
            full_input = f"{context} {user_input}"
        else:
            full_input = user_input
        
        try:
            # Generate response using the model
            inputs = self.tokenizer.encode(full_input + self.tokenizer.eos_token, 
                                         return_tensors='pt')
            
            # Generate with controlled parameters
            chat_history_ids = self.model.generate(
                inputs, 
                max_length=max_length,
                num_beams=5,
                no_repeat_ngram_size=2,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode response
            response = self.tokenizer.decode(
                chat_history_ids[:, inputs.shape[-1]:][0], 
                skip_special_tokens=True
            )
            
            # Clean and validate response
            response = self.clean_response(response, user_input)
            
            # Update conversation memory
            self.update_conversation_memory(user_id, user_input, response)
            
            return response
            
        except Exception as e:
            return f"I apologize, but I'm having trouble processing that. Could you rephrase?"
    
    def clean_response(self, response, user_input):
        """
        Clean and improve the generated response
        Like a chef adjusting seasoning before serving
        """
        # Remove empty responses
        if not response.strip():
            return "That's interesting! Tell me more."
        
        # Remove repetitions of user input
        if response.lower().strip() == user_input.lower().strip():
            return "Could you elaborate on that?"
        
        # Truncate very long responses
        if len(response) > 200:
            response = response[:200] + "..."
        
        return response.strip()
    
    def update_conversation_memory(self, user_id, user_input, bot_response):
        """
        Update conversation context for continuity
        Like remembering what dishes a guest enjoyed
        """
        conversation = self.conversation_memory[user_id]
        
        # Add to history
        conversation['history'].append({
            'user': user_input,
            'bot': bot_response,
            'timestamp': datetime.now()
        })
        
        # Update context (keep last 3 exchanges)
        recent_history = conversation['history'][-3:]
        context_parts = []
        
        for exchange in recent_history:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Bot: {exchange['bot']}")
        
        conversation['context'] = " ".join(context_parts)
        conversation['turn_count'] += 1

# Django view for chatbot
@csrf_exempt
@cache_page(60 * 5)  # Cache for 5 minutes
def chat_endpoint(request):
    """
    Main chatbot endpoint
    Like the main service point where conversations happen
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '').strip()
            user_id = data.get('user_id', str(uuid.uuid4()))
            
            if not message:
                return JsonResponse({
                    'success': False,
                    'error': 'Message cannot be empty'
                })
            
            # Initialize conversation host
            host = ConversationHost()
            
            # Generate response
            response = host.generate_response(message, user_id)
            
            # Get conversation stats
            conversation_stats = host.conversation_memory.get(user_id, {})
            
            return JsonResponse({
                'success': True,
                'response': response,
                'user_id': user_id,
                'turn_count': conversation_stats.get('turn_count', 0),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Chat processing error: {str(e)}'
            })
    
    return JsonResponse({'error': 'Method not allowed'})
```

**Syntax Explanation:**
- `AutoModelForCausalLM`: Model class for text generation
- `uuid.uuid4()`: Generates unique identifiers for users
- `cache_page(60 * 5)`: Django decorator for caching responses
- `datetime.now()`: Gets current timestamp
- `model.generate()`: Generates text using transformer model parameters
- `max_length`, `num_beams`, `temperature`: Text generation control parameters

---

## 4. Language Translation Models

Building a translation system is like having a polyglot chef who can adapt any recipe to different culinary traditions while maintaining the essence and flavor of the original dish.

### Code Example: Translation System

```python
from transformers import MarianMTModel, MarianTokenizer, pipeline
from langdetect import detect
from django.core.cache import cache

class PolyglotTranslator:
    """
    A multilingual translator that bridges language barriers
    Like a chef who can adapt recipes for different cultural palates
    """
    
    def __init__(self):
        # Initialize translation models for popular language pairs
        self.translation_models = {}
        self.tokenizers = {}
        
        # Common translation pairs
        self.supported_pairs = [
            ('en', 'fr'),  # English to French
            ('en', 'es'),  # English to Spanish  
            ('en', 'de'),  # English to German
            ('fr', 'en'),  # French to English
            ('es', 'en'),  # Spanish to English
            ('de', 'en')   # German to English
        ]
        
        self.initialize_models()
    
    def initialize_models(self):
        """
        Initialize translation models for supported language pairs
        Like preparing different cooking techniques for various cuisines
        """
        for source_lang, target_lang in self.supported_pairs:
            model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
            
            try:
                # Load model and tokenizer
                self.translation_models[f'{source_lang}-{target_lang}'] = MarianMTModel.from_pretrained(model_name)
                self.tokenizers[f'{source_lang}-{target_lang}'] = MarianTokenizer.from_pretrained(model_name)
                
            except Exception as e:
                print(f"Could not load model for {source_lang}-{target_lang}: {str(e)}")
    
    def detect_language(self, text):
        """
        Detect the language of input text
        Like identifying the cuisine style of a dish
        """
        try:
            detected_lang = detect(text)
            confidence = 0.9  # langdetect doesn't provide confidence, so we estimate
            
            return {
                'language': detected_lang,
                'confidence': confidence,
                'text_length': len(text)
            }
            
        except Exception as e:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def translate_text(self, text, source_lang=None, target_lang='en'):
        """
        Translate text between languages
        Like adapting a recipe from one cuisine to another
        """
        # Auto-detect source language if not provided
        if not source_lang:
            detection = self.detect_language(text)
            source_lang = detection['language']
            
            if source_lang == 'unknown':
                return {
                    'success': False,
                    'error': 'Could not detect source language',
                    'original_text': text
                }
        
        # Check if translation pair is supported
        model_key = f'{source_lang}-{target_lang}'
        
        if model_key not in self.translation_models:
            return {
                'success': False,
                'error': f'Translation from {source_lang} to {target_lang} is not supported',
                'supported_pairs': [pair for pair in self.supported_pairs]
            }
        
        try:
            # Get model and tokenizer
            model = self.translation_models[model_key]
            tokenizer = self.tokenizers[model_key]
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate translation
            translated_tokens = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            
            # Decode translation
            translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            
            return {
                'success': True,
                'original_text': text,
                'translated_text': translated_text,
                'source_language': source_lang,
                'target_language': target_lang,
                'model_used': model_key
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Translation failed: {str(e)}',
                'original_text': text
            }
    
    def batch_translate(self, texts, source_lang=None, target_lang='en'):
        """
        Translate multiple texts efficiently
        Like preparing multiple dishes with the same technique
        """
        results = []
        
        for i, text in enumerate(texts):
            result = self.translate_text(text, source_lang, target_lang)
            result['batch_index'] = i
            results.append(result)
        
        return {
            'batch_size': len(texts),
            'successful_translations': len([r for r in results if r['success']]),
            'results': results
        }

# Django views for translation
@csrf_exempt
def translate_endpoint(request):
    """
    Main translation endpoint
    Like a translation service counter
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '').strip()
            source_lang = data.get('source_lang')  # Optional
            target_lang = data.get('target_lang', 'en')
            
            if not text:
                return JsonResponse({
                    'success': False,
                    'error': 'Text to translate is required'
                })
            
            # Check cache first
            cache_key = f"translate_{hash(text)}_{source_lang}_{target_lang}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                cached_result['from_cache'] = True
                return JsonResponse(cached_result)
            
            # Initialize translator
            translator = PolyglotTranslator()
            
            # Perform translation
            result = translator.translate_text(text, source_lang, target_lang)
            
            # Cache successful translations
            if result['success']:
                cache.set(cache_key, result, timeout=3600)  # Cache for 1 hour
            
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Translation service error: {str(e)}'
            })
    
    return JsonResponse({'error': 'Method not allowed'})

@csrf_exempt
def detect_language_endpoint(request):
    """
    Language detection endpoint
    Like identifying the cuisine type of an unknown dish
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '').strip()
            
            if not text:
                return JsonResponse({
                    'success': False,
                    'error': 'Text for language detection is required'
                })
            
            translator = PolyglotTranslator()
            detection_result = translator.detect_language(text)
            
            return JsonResponse({
                'success': True,
                'detection': detection_result,
                'text': text
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Language detection error: {str(e)}'
            })
    
    return JsonResponse({'error': 'Method not allowed'})
```

**Syntax Explanation:**
- `MarianMTModel`: Facebook's Marian neural machine translation model
- `MarianTokenizer`: Tokenizer for Marian translation models
- `langdetect.detect()`: Automatically detects text language
- `cache.get()` and `cache.set()`: Django cache system for storing results
- `hash()`: Creates hash for cache keys
- `truncation=True, max_length=512`: Limits input length for model processing

---

## Final Project: Smart Customer Service Assistant

Now let's combine all these techniques to create a comprehensive customer service assistant that can understand customer queries, extract important information, provide answers, maintain conversations, and even handle multilingual support.

### Complete Integration Project

```python
# smart_assistant/models.py
from django.db import models
from django.utils import timezone
import uuid

class CustomerConversation(models.Model):
    """
    Track customer conversations like a restaurant's guest book
    """
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)
    customer_name = models.CharField(max_length=100, blank=True)
    language = models.CharField(max_length=10, default='en')
    started_at = models.DateTimeField(default=timezone.now)
    last_activity = models.DateTimeField(auto_now=True)
    is_resolved = models.BooleanField(default=False)
    satisfaction_score = models.IntegerField(null=True, blank=True)
    
    def __str__(self):
        return f"Conversation {self.session_id}"

class ConversationMessage(models.Model):
    """
    Individual messages in conversations
    """
    conversation = models.ForeignKey(CustomerConversation, on_delete=models.CASCADE)
    message_type = models.CharField(max_length=20, choices=[
        ('customer', 'Customer'),
        ('assistant', 'Assistant'),
        ('system', 'System')
    ])
    content = models.TextField()
    entities_detected = models.JSONField(default=dict, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['timestamp']

# smart_assistant/services.py
from .models import CustomerConversation, ConversationMessage
import json

class SmartAssistant:
    """
    The master chef of customer service - combining all our techniques
    Like a head chef coordinating multiple stations to create perfect dishes
    """
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.qa_system = IntelligentSommelier()
        self.chatbot = ConversationHost()
        self.translator = PolyglotTranslator()
        
        # Knowledge base for customer service
        self.knowledge_base = """
        Our restaurant offers fine dining experiences with seasonal menus.
        We are open Tuesday through Sunday, 5 PM to 11 PM.
        Reservations can be made online or by calling (555) 123-4567.
        We offer vegetarian, vegan, and gluten-free options.
        Private dining rooms are available for special events.
        Our sommelier can recommend wine pairings for any meal.
        We accept all major credit cards and cash.
        Dress code is smart casual to formal.
        """
    
    def process_customer_query(self, message, session_id=None, customer_language='en'):
        """
        Process a complete customer query using all available techniques
        Like orchestrating all cooking stations to fulfill a complex order
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get or create conversation
        conversation, created = CustomerConversation.objects.get_or_create(
            session_id=session_id,
            defaults={'language': customer_language}
        )
        
        # Translate if needed
        original_message = message
        if customer_language != 'en':
            translation_result = self.translator.translate_text(message, customer_language, 'en')
            if translation_result['success']:
                message = translation_result['translated_text']
        
        # Extract entities from the message
        entities = self.entity_extractor.extract_basic_entities(message)
        
        # Try to answer using knowledge base
        qa_result = self.qa_system.find_answer(message, self.knowledge_base)
        
        # Generate conversational response
        conversational_response = self.chatbot.generate_response(message, str(session_id))
        
        # Determine best response strategy
        if qa_result['confidence'] > 0.5:
            # Use knowledge base answer
            response = qa_result['answer']
            response_type = 'knowledge_based'
            confidence = qa_result['confidence']
        else:
            # Use conversational response
            response = conversational_response
            response_type = 'conversational'
            confidence = 0.7  # Estimated confidence for conversational responses
        
        # Translate response back if needed
        final_response = response
        if customer_language != 'en':
            response_translation = self.translator.translate_text(response, 'en', customer_language)
            if response_translation['success']:
                final_response = response_translation['translated_text']
        
        # Save conversation message
        ConversationMessage.objects.create(
            conversation=conversation,
            message_type='customer',
            content=original_message,
            entities_detected={'entities': entities},
            confidence_score=confidence
        )
        
        ConversationMessage.objects.create(
            conversation=conversation,
            message_type='assistant',
            content=final_response,
            confidence_score=confidence
        )
        
        # Prepare comprehensive response
        return {
            'session_id': str(session_id),
            'response': final_response,
            'response_type': response_type,
            'confidence': confidence,
            'entities_found': entities,
            'language': customer_language,
            'conversation_turn': conversation.conversationmessage_set.filter(message_type='customer').count()
        }

# smart_assistant/views.py
@csrf_exempt
def smart_assistant_endpoint(request):
    """
    Main endpoint for the smart assistant
    Like the main ordering system that handles all customer requests
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '').strip()
            session_id = data.get('session_id')
            language = data.get('language', 'en')
            
            if not message:
                return JsonResponse({
                    'success': False,
                    'error': 'Message is required'
                })
            
            # Process with smart assistant
            assistant = SmartAssistant()
            result = assistant.process_customer_query(message, session_id, language)
            
            return JsonResponse({
                'success': True,
                'data': result
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Assistant processing error: {str(e)}'
            })
    
    return JsonResponse({'error': 'Method not allowed'})

# smart_assistant/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.smart_assistant_endpoint, name='smart_chat'),
    path('extract-entities/', views.extract_entities, name='extract_entities'),
    path('answer-question/', views.answer_question, name='answer_question'),
    path('translate/', views.translate_endpoint, name='translate'),
    path('detect-language/', views.detect_language_endpoint, name='detect_language'),
]
```

---

# Multi-task NLP Application - Django Project

## Project Overview
Build a comprehensive Django web application that serves multiple NLP functionalities in one unified platform - like a master chef's kitchen that can prepare various dishes from the same ingredients.

## Project Structure
```
nlp_kitchen/
├── manage.py
├── nlp_kitchen/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── nlp_app/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── templates/
│       └── nlp_app/
│           ├── base.html
│           ├── dashboard.html
│           └── results.html
├── static/
│   ├── css/
│   └── js/
└── requirements.txt
```

## Core Implementation

### 1. Django Settings Configuration
```python
# nlp_kitchen/settings.py
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
    'nlp_app',
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

ROOT_URLCONF = 'nlp_kitchen.urls'

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
```

### 2. Models for Data Storage
```python
# nlp_app/models.py
from django.db import models
from django.contrib.auth.models import User
import json

class NLPTask(models.Model):
    TASK_TYPES = [
        ('ner', 'Named Entity Recognition'),
        ('qa', 'Question Answering'),
        ('chat', 'Chatbot Response'),
        ('translate', 'Language Translation'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    task_type = models.CharField(max_length=20, choices=TASK_TYPES)
    input_text = models.TextField()
    output_result = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    processing_time = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.get_task_type_display()} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class ChatHistory(models.Model):
    session_id = models.CharField(max_length=100)
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
```

### 3. NLP Processing Engine
```python
# nlp_app/nlp_processor.py
import spacy
import re
import time
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from googletrans import Translator
import random

class NLPKitchen:
    """The main cooking station where all NLP recipes are prepared"""
    
    def __init__(self):
        # Load the base ingredients (models)
        self.nlp = spacy.load("en_core_web_sm")
        self.qa_pipeline = None
        self.translator = Translator()
        self._initialize_qa_model()
    
    def _initialize_qa_model(self):
        """Prepare the question-answering station"""
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
        except Exception as e:
            print(f"QA model initialization failed: {e}")
    
    def extract_entities(self, text):
        """Extract named entities like a chef identifying key ingredients"""
        start_time = time.time()
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_),
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        # Add custom pattern matching for additional entities
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'URL': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        }
        
        for label, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': label,
                    'description': f'Custom {label.lower()} pattern',
                    'start': match.start(),
                    'end': match.end()
                })
        
        processing_time = time.time() - start_time
        
        return {
            'entities': entities,
            'total_entities': len(entities),
            'processing_time': processing_time,
            'entity_types': list(set([e['label'] for e in entities]))
        }
    
    def answer_question(self, context, question):
        """Serve up answers like a knowledgeable chef explaining recipes"""
        start_time = time.time()
        
        if not self.qa_pipeline:
            return {
                'answer': 'Question answering service temporarily unavailable',
                'confidence': 0.0,
                'processing_time': 0.0
            }
        
        try:
            result = self.qa_pipeline(question=question, context=context)
            processing_time = time.time() - start_time
            
            return {
                'answer': result['answer'],
                'confidence': round(result['score'], 4),
                'start_position': result['start'],
                'end_position': result['end'],
                'processing_time': processing_time
            }
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'answer': f'Error processing question: {str(e)}',
                'confidence': 0.0,
                'processing_time': processing_time
            }
    
    def generate_chat_response(self, message, chat_history=None):
        """Generate conversational responses like a friendly chef chatting with customers"""
        start_time = time.time()
        
        # Simple rule-based chatbot with personality
        message_lower = message.lower()
        
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        farewells = ['bye', 'goodbye', 'see you', 'farewell', 'take care']
        
        if any(greeting in message_lower for greeting in greetings):
            responses = [
                "Hello! Welcome to our NLP kitchen. How can I assist you today?",
                "Hi there! Ready to cook up some amazing text analysis?",
                "Greetings! What linguistic dish shall we prepare together?"
            ]
        elif any(farewell in message_lower for farewell in farewells):
            responses = [
                "Goodbye! Thanks for visiting our NLP kitchen!",
                "Take care! Come back anytime for more text processing!",
                "See you later! Happy cooking with your data!"
            ]
        elif 'help' in message_lower or 'what can you do' in message_lower:
            responses = [
                "I can help you with named entity recognition, question answering, language translation, and general conversation. What interests you most?",
            ]
        elif 'nlp' in message_lower or 'natural language' in message_lower:
            responses = [
                "NLP is like cooking - we take raw text ingredients and transform them into something meaningful and useful!",
            ]
        else:
            # Generate contextual responses based on entities in the message
            doc = self.nlp(message)
            entities = [ent.text for ent in doc.ents]
            
            if entities:
                responses = [
                    f"I noticed you mentioned {', '.join(entities[:3])}. That's interesting! Would you like me to analyze this text further?",
                    f"I see some key entities in your message: {', '.join(entities[:2])}. How can I help you process this information?",
                ]
            else:
                responses = [
                    "That's interesting! Can you tell me more about what you'd like to analyze?",
                    "I'm here to help with your text processing needs. What would you like to explore?",
                    "Fascinating! Would you like me to perform any specific NLP tasks on this text?",
                ]
        
        processing_time = time.time() - start_time
        
        return {
            'response': random.choice(responses),
            'processing_time': processing_time,
            'detected_entities': [ent.text for ent in self.nlp(message).ents]
        }
    
    def translate_text(self, text, target_language='es'):
        """Translate text like adapting a recipe for different cultural tastes"""
        start_time = time.time()
        
        try:
            # Detect source language
            detection = self.translator.detect(text)
            source_lang = detection.lang
            confidence = detection.confidence
            
            # Translate text
            translation = self.translator.translate(text, dest=target_language)
            
            processing_time = time.time() - start_time
            
            return {
                'original_text': text,
                'translated_text': translation.text,
                'source_language': source_lang,
                'target_language': target_language,
                'detection_confidence': confidence,
                'processing_time': processing_time
            }
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'original_text': text,
                'translated_text': f'Translation error: {str(e)}',
                'source_language': 'unknown',
                'target_language': target_language,
                'detection_confidence': 0.0,
                'processing_time': processing_time
            }

# Initialize the global NLP processor
nlp_kitchen = NLPKitchen()
```

### 4. Django Forms
```python
# nlp_app/forms.py
from django import forms

class NERForm(forms.Form):
    text = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 6,
            'placeholder': 'Enter text for named entity recognition...',
            'class': 'form-control'
        }),
        label="Text to Analyze"
    )

class QAForm(forms.Form):
    context = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 8,
            'placeholder': 'Enter the context/passage here...',
            'class': 'form-control'
        }),
        label="Context"
    )
    question = forms.CharField(
        widget=forms.TextInput(attrs={
            'placeholder': 'Ask a question about the context...',
            'class': 'form-control'
        }),
        label="Question"
    )

class ChatForm(forms.Form):
    message = forms.CharField(
        widget=forms.TextInput(attrs={
            'placeholder': 'Type your message...',
            'class': 'form-control',
            'id': 'chat-input'
        }),
        label="Message"
    )

class TranslationForm(forms.Form):
    LANGUAGE_CHOICES = [
        ('es', 'Spanish'),
        ('fr', 'French'),
        ('de', 'German'),
        ('it', 'Italian'),
        ('pt', 'Portuguese'),
        ('ru', 'Russian'),
        ('ja', 'Japanese'),
        ('ko', 'Korean'),
        ('zh', 'Chinese (Simplified)'),
        ('ar', 'Arabic'),
    ]
    
    text = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 4,
            'placeholder': 'Enter text to translate...',
            'class': 'form-control'
        }),
        label="Text to Translate"
    )
    target_language = forms.ChoiceField(
        choices=LANGUAGE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label="Target Language"
    )
```

### 5. Django Views
```python
# nlp_app/views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
import json
import uuid

from .forms import NERForm, QAForm, ChatForm, TranslationForm
from .models import NLPTask, ChatHistory
from .nlp_processor import nlp_kitchen

def dashboard(request):
    """Main dashboard - the central command center of our NLP kitchen"""
    recent_tasks = NLPTask.objects.all()[:10]
    
    # Calculate statistics
    stats = {
        'total_tasks': NLPTask.objects.count(),
        'ner_tasks': NLPTask.objects.filter(task_type='ner').count(),
        'qa_tasks': NLPTask.objects.filter(task_type='qa').count(),
        'chat_tasks': NLPTask.objects.filter(task_type='chat').count(),
        'translation_tasks': NLPTask.objects.filter(task_type='translate').count(),
    }
    
    return render(request, 'nlp_app/dashboard.html', {
        'recent_tasks': recent_tasks,
        'stats': stats,
        'ner_form': NERForm(),
        'qa_form': QAForm(),
        'chat_form': ChatForm(),
        'translation_form': TranslationForm(),
    })

def process_ner(request):
    """Process Named Entity Recognition requests"""
    if request.method == 'POST':
        form = NERForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            
            # Process through our NLP kitchen
            result = nlp_kitchen.extract_entities(text)
            
            # Save to database
            task = NLPTask.objects.create(
                user=request.user if request.user.is_authenticated else None,
                task_type='ner',
                input_text=text,
                output_result=result,
                processing_time=result['processing_time']
            )
            
            return render(request, 'nlp_app/results.html', {
                'task_type': 'Named Entity Recognition',
                'input_text': text,
                'result': result,
                'task_id': task.id
            })
    
    return redirect('dashboard')

def process_qa(request):
    """Process Question Answering requests"""
    if request.method == 'POST':
        form = QAForm(request.POST)
        if form.is_valid():
            context = form.cleaned_data['context']
            question = form.cleaned_data['question']
            
            # Process through our NLP kitchen
            result = nlp_kitchen.answer_question(context, question)
            result['context'] = context
            result['question'] = question
            
            # Save to database
            task = NLPTask.objects.create(
                user=request.user if request.user.is_authenticated else None,
                task_type='qa',
                input_text=f"Context: {context}\nQuestion: {question}",
                output_result=result,
                processing_time=result['processing_time']
            )
            
            return render(request, 'nlp_app/results.html', {
                'task_type': 'Question Answering',
                'input_text': f"Q: {question}",
                'result': result,
                'task_id': task.id
            })
    
    return redirect('dashboard')

@csrf_exempt
def process_chat(request):
    """Process Chat requests"""
    if request.method == 'POST':
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            message = data.get('message', '')
            session_id = data.get('session_id', str(uuid.uuid4()))
        else:
            form = ChatForm(request.POST)
            if form.is_valid():
                message = form.cleaned_data['message']
                session_id = request.session.session_key or str(uuid.uuid4())
            else:
                return JsonResponse({'error': 'Invalid form data'})
        
        # Get chat history for context
        recent_history = ChatHistory.objects.filter(
            session_id=session_id
        ).order_by('-timestamp')[:5]
        
        history = [
            {'message': h.message, 'response': h.response}
            for h in reversed(recent_history)
        ]
        
        # Process through our NLP kitchen
        result = nlp_kitchen.generate_chat_response(message, history)
        
        # Save chat history
        ChatHistory.objects.create(
            session_id=session_id,
            message=message,
            response=result['response']
        )
        
        # Save task
        task = NLPTask.objects.create(
            user=request.user if request.user.is_authenticated else None,
            task_type='chat',
            input_text=message,
            output_result=result,
            processing_time=result['processing_time']
        )
        
        if request.content_type == 'application/json':
            return JsonResponse({
                'response': result['response'],
                'session_id': session_id,
                'detected_entities': result.get('detected_entities', []),
                'processing_time': result['processing_time']
            })
        else:
            return render(request, 'nlp_app/results.html', {
                'task_type': 'Chat Response',
                'input_text': message,
                'result': result,
                'task_id': task.id
            })
    
    return redirect('dashboard')

def process_translation(request):
    """Process Translation requests"""
    if request.method == 'POST':
        form = TranslationForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            target_language = form.cleaned_data['target_language']
            
            # Process through our NLP kitchen
            result = nlp_kitchen.translate_text(text, target_language)
            
            # Save to database
            task = NLPTask.objects.create(
                user=request.user if request.user.is_authenticated else None,
                task_type='translate',
                input_text=text,
                output_result=result,
                processing_time=result['processing_time']
            )
            
            return render(request, 'nlp_app/results.html', {
                'task_type': 'Language Translation',
                'input_text': text,
                'result': result,
                'task_id': task.id
            })
    
    return redirect('dashboard')

def task_history(request):
    """View task history"""
    tasks = NLPTask.objects.all()
    
    # Filter by task type if requested
    task_type = request.GET.get('type')
    if task_type:
        tasks = tasks.filter(task_type=task_type)
    
    return render(request, 'nlp_app/history.html', {
        'tasks': tasks,
        'filter_type': task_type
    })
```

### 6. URL Configuration
```python
# nlp_kitchen/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('nlp_app.urls')),
]

# nlp_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('ner/', views.process_ner, name='process_ner'),
    path('qa/', views.process_qa, name='process_qa'),
    path('chat/', views.process_chat, name='process_chat'),
    path('translate/', views.process_translation, name='process_translation'),
    path('history/', views.task_history, name='task_history'),
]
```

### 7. Frontend Templates

#### Base Template
```html
<!-- nlp_app/templates/nlp_app/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}NLP Kitchen#### Results Template
```html
<!-- nlp_app/templates/nlp_app/results.html -->
{% extends 'nlp_app/base.html' %}

{% block content %}
<div class="row">
    <div class="col-md-12">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2><i class="fas fa-chart-line me-2"></i>{{ task_type }} Results</h2>
            <a href="{% url 'dashboard' %}" class="btn btn-secondary">
                <i class="fas fa-arrow-left me-2"></i>Back to Kitchen
            </a>
        </div>
    </div>
</div>

<div class="row">
    <!-- Input Section -->
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-edit me-2"></i>Input</h5>
            </div>
            <div class="card-body">
                <pre class="bg-light p-3 rounded">{{ input_text }}</pre>
            </div>
        </div>
    </div>

    <!-- Results Section -->
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-magic me-2"></i>Results</h5>
            </div>
            <div class="card-body">
                {% if task_type == "Named Entity Recognition" %}
                    <div class="mb-3">
                        <strong>Processing Time:</strong> {{ result.processing_time|floatformat:3 }}s<br>
                        <strong>Total Entities:</strong> {{ result.total_entities }}<br>
                        <strong>Entity Types:</strong> {{ result.entity_types|join:", " }}
                    </div>
                    
                    <h6>Detected Entities:</h6>
                    {% for entity in result.entities %}
                        <span class="entity-tag entity-{{ entity.label }}">
                            {{ entity.text }} ({{ entity.label }})
                        </span>
                    {% empty %}
                        <p class="text-muted">No entities detected.</p>
                    {% endfor %}

                {% elif task_type == "Question Answering" %}
                    <div class="mb-3">
                        <strong>Processing Time:</strong> {{ result.processing_time|floatformat:3 }}s<br>
                        <strong>Confidence:</strong> {{ result.confidence|floatformat:2 }}
                    </div>
                    
                    <div class="alert alert-success">
                        <h6><i class="fas fa-lightbulb me-2"></i>Answer:</h6>
                        <p class="mb-0">{{ result.answer }}</p>
                    </div>

                {% elif task_type == "Chat Response" %}
                    <div class="mb-3">
                        <strong>Processing Time:</strong> {{ result.processing_time|floatformat:3 }}s
                    </div>
                    
                    <div class="alert alert-info">
                        <h6><i class="fas fa-robot me-2"></i>Response:</h6>
                        <p class="mb-0">{{ result.response }}</p>
                    </div>
                    
                    {% if result.detected_entities %}
                        <h6>Detected in your message:</h6>
                        {% for entity in result.detected_entities %}
                            <span class="badge bg-secondary me-1">{{ entity }}</span>
                        {% endfor %}
                    {% endif %}

                {% elif task_type == "Language Translation" %}
                    <div class="mb-3">
                        <strong>Processing Time:</strong> {{ result.processing_time|floatformat:3 }}s<br>
                        <strong>Source Language:</strong> {{ result.source_language|upper }}<br>
                        <strong>Target Language:</strong> {{ result.target_language|upper }}<br>
                        <strong>Detection Confidence:</strong> {{ result.detection_confidence|floatformat:2 }}
                    </div>
                    
                    <div class="alert alert-primary">
                        <h6><i class="fas fa-language me-2"></i>Translation:</h6>
                        <p class="mb-0">{{ result.translated_text }}</p>
                    </div>
                {% endif %}
            </div>
        </div>
    </div>
</div>

<!-- JSON Debug (for development) -->
<div class="row mt-4">
    <div class="col-md-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-code me-2"></i>Raw Results (JSON)</h5>
            </div>
            <div class="card-body">
                <pre class="bg-dark text-light p-3 rounded"><code>{{ result|pprint }}</code></pre>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

### 8. Requirements and Installation

#### Requirements File
```text
# requirements.txt
Django==4.2.7
spacy==3.7.2
transformers==4.35.2
torch==2.1.0
googletrans==4.0.0rc1
pandas==2.1.3
numpy==1.24.3
requests==2.31.0
```

#### Installation Commands
```bash
# Create virtual environment
python -m venv nlp_kitchen_env
source nlp_kitchen_env/bin/activate  # On Windows: nlp_kitchen_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Django setup
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic
python manage.py createsuperuser  # Optional: create admin user

# Run the server
python manage.py runserver
```

### 9. Advanced Features & Enhancements

#### API Endpoint for External Integration
```python
# Add to nlp_app/views.py
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
@require_http_methods(["POST"])
def api_process_text(request):
    """API endpoint for external applications"""
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        task_type = data.get('task_type', 'ner')
        
        if not text:
            return JsonResponse({'error': 'Text is required'}, status=400)
        
        if task_type == 'ner':
            result = nlp_kitchen.extract_entities(text)
        elif task_type == 'qa':
            context = data.get('context', '')
            result = nlp_kitchen.answer_question(context, text)
        elif task_type == 'chat':
            result = nlp_kitchen.generate_chat_response(text)
        elif task_type == 'translate':
            target_lang = data.get('target_language', 'es')
            result = nlp_kitchen.translate_text(text, target_lang)
        else:
            return JsonResponse({'error': 'Invalid task type'}, status=400)
        
        # Save task
        NLPTask.objects.create(
            task_type=task_type,
            input_text=text,
            output_result=result,
            processing_time=result.get('processing_time', 0.0)
        )
        
        return JsonResponse({
            'status': 'success',
            'result': result
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Add to urlpatterns in nlp_app/urls.py
path('api/process/', views.api_process_text, name='api_process'),
```

#### Batch Processing Feature
```python
# Add to nlp_app/views.py
def batch_process(request):
    """Handle batch processing of multiple texts"""
    if request.method == 'POST':
        texts = request.POST.getlist('texts')
        task_type = request.POST.get('task_type', 'ner')
        
        results = []
        for text in texts:
            if task_type == 'ner':
                result = nlp_kitchen.extract_entities(text)
            elif task_type == 'translate':
                target_lang = request.POST.get('target_language', 'es')
                result = nlp_kitchen.translate_text(text, target_lang)
            # Add other task types as needed
            
            results.append({
                'input': text,
                'output': result
            })
            
            # Save each task
            NLPTask.objects.create(
                task_type=task_type,
                input_text=text,
                output_result=result,
                processing_time=result.get('processing_time', 0.0)
            )
        
        return render(request, 'nlp_app/batch_results.html', {
            'results': results,
            'task_type': task_type
        })
    
    return render(request, 'nlp_app/batch_form.html')
```

### 10. Performance Optimization

#### Caching for Repeated Queries
```python
# Add to nlp_app/nlp_processor.py
from django.core.cache import cache
import hashlib

class NLPKitchen:
    def _get_cache_key(self, text, operation):
        """Generate cache key for results"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"nlp_{operation}_{text_hash}"
    
    def extract_entities(self, text):
        """Extract entities with caching"""
        cache_key = self._get_cache_key(text, 'ner')
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # Process as before...
        result = self._process_entities(text)
        
        # Cache for 1 hour
        cache.set(cache_key, result, 3600)
        return result
```

### 11. Testing the Application

#### Sample Test Cases
```python
# nlp_app/tests.py
from django.test import TestCase, Client
from django.urls import reverse
from .models import NLPTask
from .nlp_processor import nlp_kitchen

class NLPKitchenTests(TestCase):
    def setUp(self):
        self.client = Client()
    
    def test_ner_processing(self):
        """Test Named Entity Recognition"""
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
        result = nlp_kitchen.extract_entities(text)
        
        self.assertIn('entities', result)
        self.assertGreater(len(result['entities']), 0)
        self.assertIn('processing_time', result)
    
    def test_qa_processing(self):
        """Test Question Answering"""
        context = "Django is a Python web framework. It was created in 2003."
        question = "When was Django created?"
        
        result = nlp_kitchen.answer_question(context, question)
        
        self.assertIn('answer', result)
        self.assertIn('2003', result['answer'])
    
    def test_dashboard_view(self):
        """Test dashboard loads correctly"""
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'NLP Kitchen')
    
    def test_ner_form_submission(self):
        """Test NER form submission"""
        response = self.client.post(reverse('process_ner'), {
            'text': 'Barack Obama was the President of the United States.'
        })
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(NLPTask.objects.count(), 1)
        
        task = NLPTask.objects.first()
        self.assertEqual(task.task_type, 'ner')

# Run tests with: python manage.py test
```

### 12. Deployment Considerations

#### Production Settings
```python
# nlp_kitchen/settings_prod.py
from .settings import *

DEBUG = False
ALLOWED_HOSTS = ['your-domain.com', 'www.your-domain.com']

# Use PostgreSQL for production
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'nlp_kitchen_db',
        'USER': 'nlp_user',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# Cache configuration
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}

# Static files for production
STATIC_ROOT = '/path/to/static/files/'

# Security settings
SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
```

This multi-task NLP application demonstrates how to integrate various natural language processing capabilities into a single Django web application. Like a well-organized restaurant kitchen, each component has its specific role while working together to create a comprehensive text processing experience.

The application showcases Django's MVC architecture, demonstrates proper separation of concerns, implements caching for performance, includes comprehensive error handling, and provides both web interface and API access for maximum flexibility.

{% endblock %}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .task-card {
            transition: transform 0.2s;
            border-left: 4px solid #007bff;
        }
        .task-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
        }
        .entity-tag {
            display: inline-block;
            padding: 2px 8px;
            margin: 2px;
            border-radius: 12px;
            font-size: 0.8em;
        }
        .entity-PERSON { background-color: #e3f2fd; color: #1565c0; }
        .entity-ORG { background-color: #f3e5f5; color: #7b1fa2; }
        .entity-GPE { background-color: #e8f5e8; color: #388e3c; }
        .entity-MONEY { background-color: #fff3e0; color: #f57c00; }
        .entity-DATE { background-color: #fce4ec; color: #c2185b; }
        .entity-EMAIL { background-color: #e0f2f1; color: #00695c; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{% url 'dashboard' %}">
                <i class="fas fa-brain me-2"></i>NLP Kitchen
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="{% url 'dashboard' %}">Dashboard</a>
                <a class="nav-link" href="{% url 'task_history' %}">History</a>
            </div>
        </div>
    </nav>

    <main class="container mt-4">
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}

        {% block content %}{% endblock %}
    </main>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
```

#### Dashboard Template
```html
<!-- nlp_app/templates/nlp_app/dashboard.html -->
{% extends 'nlp_app/base.html' %}

{% block content %}
<div class="row mb-4">
    <div class="col-md-12">
        <h1 class="display-4">
            <i class="fas fa-utensils me-3"></i>Welcome to the NLP Kitchen
        </h1>
        <p class="lead">Where raw text becomes delicious insights through the art of natural language processing.</p>
    </div>
</div>

<!-- Statistics Cards -->
<div class="row mb-4">
    <div class="col-md-3">
        <div class="card stat-card">
            <div class="card-body text-center">
                <i class="fas fa-tasks fa-2x mb-2"></i>
                <h3>{{ stats.total_tasks }}</h3>
                <p class="mb-0">Total Tasks</p>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card stat-card">
            <div class="card-body text-center">
                <i class="fas fa-tags fa-2x mb-2"></i>
                <h3>{{ stats.ner_tasks }}</h3>
                <p class="mb-0">NER Tasks</p>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card stat-card">
            <div class="card-body text-center">
                <i class="fas fa-question-circle fa-2x mb-2"></i>
                <h3>{{ stats.qa_tasks }}</h3>
                <p class="mb-0">Q&A Tasks</p>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card stat-card">
            <div class="card-body text-center">
                <i class="fas fa-comments fa-2x mb-2"></i>
                <h3>{{ stats.chat_tasks }}</h3>
                <p class="mb-0">Chat Tasks</p>
            </div>
        </div>
    </div>
</div>

<!-- NLP Tasks -->
<div class="row">
    <!-- Named Entity Recognition -->
    <div class="col-md-6 mb-4">
        <div class="card task-card">
            <div class="card-header">
                <h5><i class="fas fa-search me-2"></i>Named Entity Recognition</h5>
            </div>
            <div class="card-body">
                <p>Extract and identify named entities from text - like finding the key ingredients in a recipe.</p>
                <form method="post" action="{% url 'process_ner' %}">
                    {% csrf_token %}
                    {{ ner_form.text }}
                    <button type="submit" class="btn btn-primary mt-2">
                        <i class="fas fa-magic me-2"></i>Extract Entities
                    </button>
                </form>
            </div>
        </div>
    </div>

    <!-- Question Answering -->
    <div class="col-md-6 mb-4">
        <div class="card task-card">
            <div class="card-header">
                <h5><i class="fas fa-question me-2"></i>Question Answering</h5>
            </div>
            <div class="card-body">
                <p>Get precise answers from text - like asking the chef about their secret recipe.</p>
                <form method="post" action="{% url 'process_qa' %}">
                    {% csrf_token %}
                    {{ qa_form.context }}
                    {{ qa_form.question }}
                    <button type="submit" class="btn btn-success mt-2">
                        <i class="fas fa-brain me-2"></i>Get Answer
                    </button>
                </form>
            </div>
        </div>
    </div>

    <!-- Chatbot -->
    <div class="col-md-6 mb-4">
        <div class="card task-card">
            <div class="card-header">
                <h5><i class="fas fa-robot me-2"></i>AI Assistant</h5>
            </div>
            <div class="card-body">
                <p>Have a conversation with our AI assistant - like chatting with a knowledgeable chef.</p>
                <div id="chat-messages" class="chat-container mb-3 p-2 border rounded" style="min-height: 200px;">
                    <div class="text-muted text-center">Start a conversation...</div>
                </div>
                <div class="input-group">
                    <input type="text" id="chat-input" class="form-control" placeholder="Type your message...">
                    <button class="btn btn-info" onclick="sendMessage()">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Language Translation -->
    <div class="col-md-6 mb-4">
        <div class="card task-card">
            <div class="card-header">
                <h5><i class="fas fa-language me-2"></i>Language Translation</h5>
            </div>
            <div class="card-body">
                <p>Translate text between languages - like adapting recipes for different cultures.</p>
                <form method="post" action="{% url 'process_translation' %}">
                    {% csrf_token %}
                    {{ translation_form.text }}
                    {{ translation_form.target_language }}
                    <button type="submit" class="btn btn-warning mt-2">
                        <i class="fas fa-globe me-2"></i>Translate
                    </button>
                </form>
            </div>
        </div>
    </div>
</div>

<!-- Recent Tasks -->
{% if recent_tasks %}
<div class="row mt-5">
    <div class="col-md-12">
        <h3>Recent Kitchen Activity</h3>
        <div class="row">
            {% for task in recent_tasks %}
            <div class="col-md-4 mb-3">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">{{ task.get_task_type_display }}</h6>
                        <p class="card-text text-truncate">{{ task.input_text|slice:":50" }}...</p>
                        <small class="text-muted">
                            {{ task.created_at|date:"M d, Y H:i" }} | 
                            {{ task.processing_time|floatformat:3 }}s
                        </small>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</div>
{% endif %}
{% endblock %}

{% block extra_js %}
<script>
let sessionId = null;

function sendMessage() {
    const input = document.getElementById('chat-input');
    const messagesContainer = document.getElementById('chat-messages');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Clear the placeholder text on first message
    if (messagesContainer.children.length === 1 && 
        messagesContainer.children[0].classList.contains('text-muted')) {
        messagesContainer.innerHTML = '';
    }
    
    // Add user message
    addMessage(message, 'user');
    input.value = '';
    
    // Add loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'mb-2 text-start';
    loadingDiv.innerHTML = `
        <div class="d-inline-block bg-light p-2 rounded">
            <i class="fas fa-spinner fa-spin"></i> Thinking...
        </div>
    `;
    messagesContainer.appendChild(loadingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Send message to server
    fetch('/chat/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
        },
        body: JSON.stringify({
            message: message,
            session_id: sessionId
        })
    })
    .then(response => response.json())
    .then(data => {
        // Remove loading indicator
        messagesContainer.removeChild(loadingDiv);
        
        // Add bot response
        addMessage(data.response, 'bot');
        sessionId = data.session_id;
        
        // Show detected entities if any
        if (data.detected_entities && data.detected_entities.length > 0) {
            const entitiesDiv = document.createElement('div');
            entitiesDiv.className = 'mb-2 text-start';
            entitiesDiv.innerHTML = `
                <div class="d-inline-block bg-info bg-opacity-10 p-2 rounded">
                    <small><strong>Detected:</strong> ${data.detected_entities.join(', ')}</small>
                </div>
            `;
            messagesContainer.appendChild(entitiesDiv);
        }
        
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    })
    .catch(error => {
        messagesContainer.removeChild(loadingDiv);
        addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        console.error('Error:', error);
    });
}

function addMessage(message, sender) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `mb-2 ${sender === 'user' ? 'text-end' : 'text-start'}`;
    
    const bgClass = sender === 'user' ? 'bg-primary text-white' : 'bg-light';
    messageDiv.innerHTML = `
        <div class="d-inline-block ${bgClass} p-2 rounded" style="max-width: 80%;">
            ${message}
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Allow Enter key to send messages
document.getElementById('chat-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
</script>
{% endblock %}


## Assignment: Build a Specialized Knowledge Assistant

Create a specialized knowledge assistant for a specific domain (e.g., medical FAQ, legal assistance, technical support, or educational tutoring). Your assistant should:

### Requirements:

1. **Domain-Specific Entity Recognition**: Identify at least 5 types of entities relevant to your chosen domain
2. **Curated Knowledge Base**: Create a comprehensive knowledge base with at least 50 Q&A pairs
3. **Contextual Conversation**: Maintain conversation history and provide context-aware responses
4. **Multi-turn Dialogue**: Handle follow-up questions and clarifications
5. **Confidence Scoring**: Implement confidence thresholds and appropriate fallback responses

### Technical Specifications:

- Use Django REST framework for API endpoints
- Implement proper error handling and logging
- Include unit tests for core functionality
- Create a simple web interface for testing
- Document your API endpoints with clear examples

### Deliverables:

1. **Django Application**: Complete working application with all endpoints
2. **Custom Knowledge Base**: Domain-specific data in JSON or CSV format
3. **Web Interface**: Simple HTML/JavaScript frontend for testing
4. **Documentation**: README with setup instructions and API documentation
5. **Test Cases**: At least 10 test scenarios with expected outputs

### Example Domain Choices:

- **Medical Assistant**: Handle symptoms, medication queries, appointment scheduling
- **Legal Help Desk**: Answer questions about common legal procedures and rights  
- **Technical Support**: Troubleshoot software/hardware issues with step-by-step guidance
- **Educational Tutor**: Explain concepts, provide practice problems, track learning progress

### Evaluation Criteria:

- **Functionality** (40%): All features work as specified
- **Code Quality** (25%): Clean, well-documented, maintainable code
- **Domain Expertise** (20%): Accuracy and relevance of domain knowledge
- **User Experience** (15%): Intuitive interface and helpful responses

### Submission Timeline:
- **Planning Phase**: Choose domain and outline knowledge base (Day 1-2)
- **Development Phase**: Build core functionality (Day 3-5)
- **Testing & Documentation**: Complete testing and documentation (Day 6-7)
- **Final Submission**: Due by end of Day 7

---

## Code Syntax Summary

Throughout this lesson, we've used several key Python and Django patterns:

### Python Syntax Elements:
```python
# Class inheritance and initialization
class MyClass:
    def __init__(self):
        self.attribute = value

# Dictionary comprehensions and JSON handling
entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
data = json.loads(request.body)

# Exception handling with try-except
try:
    result = process_data()
    return result
except Exception as e:
    return {'error': str(e)}

# String methods and formatting
text = data.get('message', '').strip()
response = f"Processing error: {str(e)}"

# List slicing and manipulation
recent_history = conversation['history'][-3:]
context = context[:max_length]
```

### Django Framework Patterns:
```python
# View decorators
@csrf_exempt  # Disable CSRF protection for API
@require_http_methods(["POST"])  # Limit HTTP methods
@cache_page(60 * 5)  # Cache response for 5 minutes

# Model definitions with relationships
class MyModel(models.Model):
    field = models.CharField(max_length=100)
    foreign_key = models.ForeignKey(OtherModel, on_delete=models.CASCADE)

# JSON responses
return JsonResponse({
    'success': True,
    'data': result
})

# URL patterns
path('endpoint/', views.my_view, name='endpoint_name')
```

### Transformers Library Patterns:
```python
# Model and tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained("model_name")
model = AutoModelForQuestionAnswering.from_pretrained("model_name")

# Pipeline creation
pipeline = pipeline("task_name", model=model, tokenizer=tokenizer)

# Text generation parameters
result = model.generate(
    inputs,
    max_length=100,
    num_beams=5,
    temperature=0.7,
    do_sample=True
)
```

### Key Concepts Explained:

1. **Entity Extraction**: Using spaCy and transformers to identify and classify named entities in text
2. **Question Answering**: Leveraging BERT-based models to find answers within given contexts  
3. **Conversational AI**: Implementing dialogue systems with memory and context awareness
4. **Machine Translation**: Using Marian models for multilingual support
5. **Django Integration**: Building robust web APIs with proper error handling and caching

---

## Conclusion

Just as a master chef orchestrates multiple cooking techniques to create an extraordinary dining experience, we've learned to combine various NLP techniques to build sophisticated language applications. You now understand how to extract meaningful information from text, answer complex questions, maintain engaging conversations, and bridge language barriers.

The integration of these techniques in our Smart Customer Service Assistant demonstrates how individual NLP components work together like a well-coordinated team, each contributing their specialized skills to deliver exceptional user experiences.

Remember: the key to mastering these advanced NLP applications lies in understanding not just how each technique works individually, but how they complement each other to solve real-world problems. Practice combining these approaches in different ways, experiment with various models and parameters, and always keep the end-user experience at the heart of your development process.

Your assignment will challenge you to apply all these concepts in a focused domain, helping you develop the practical skills needed to build production-ready NLP applications that can truly understand and assist users in meaningful ways.