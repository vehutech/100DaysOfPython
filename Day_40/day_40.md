# Day 40: Django Templates & Static Files
## Building Beautiful, Dynamic Web Pages Like a Master Chef

### Learning Objective
By the end of this lesson, you will understand how to create dynamic, visually appealing web pages using Django's template system, implement template inheritance for consistent layouts, and properly manage static files like CSS and JavaScript - transforming raw data into beautifully presented web experiences.

---

## Introduction: The Restaurant Kitchen Analogy

Imagine that you're running a high-end restaurant. You've mastered the art of cooking (your Django views and models), but now you need to present your culinary masterpieces to your customers. You can't just dump ingredients on a plate and expect diners to be impressed. You need proper plating, garnishes, and presentation - that's exactly what Django templates do for your web applications.

Just as a master chef uses specialized tools, elegant plateware, and consistent presentation standards across all dishes, Django provides you with a powerful template system to transform your raw data into beautiful, consistent web pages.

---

## Part 1: Understanding Django Template Language (DTL)

### The Template as Your Restaurant's Plating System

Think of Django Template Language (DTL) as your restaurant's standardized plating system. Just as every dish that leaves your kitchen follows certain presentation rules, DTL provides a consistent way to display your data.

In our kitchen analogy:
- **Variables** are like the main ingredients you place on the plate
- **Template tags** are like your plating techniques (how you arrange things)
- **Filters** are like your garnishes and seasonings (how you modify the presentation)

### Basic Template Syntax

```html
<!-- blog/templates/blog/post_detail.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{{ post.title }}</title>
</head>
<body>
    <h1>{{ post.title }}</h1>
    <p>Published on {{ post.created_at|date:"F d, Y" }}</p>
    <div>
        {{ post.content|linebreaks }}
    </div>
    
    {% if post.author %}
        <p>Written by: {{ post.author.username }}</p>
    {% endif %}
    
    <h3>Comments ({{ post.comments.count }})</h3>
    {% for comment in post.comments.all %}
        <div>
            <strong>{{ comment.author.username }}</strong>
            <p>{{ comment.content }}</p>
        </div>
    {% empty %}
        <p>No comments yet.</p>
    {% endfor %}
</body>
</html>
```

### Setting Up Your Template Directory

First, let's organize our kitchen properly:

```python
# settings.py
import os

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
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
```

---

## Part 2: Template Inheritance - Your Restaurant's Signature Style

### The Base Template as Your Restaurant's Standard Service

Imagine that your restaurant has a signature style - every table gets the same quality linens, the same style of plates, and the same level of service. This consistency is what makes your restaurant recognizable and professional.

Template inheritance works exactly like this. You create a base template that defines your "house style," and then each specific page inherits this foundation while adding its own unique elements.

### Creating the Base Template

```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}My Blog{% endblock %}</title>
    {% load static %}
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
    {% block extra_css %}{% endblock %}
</head>
<body>
    <header>
        <nav>
            <h1><a href="{% url 'blog:index' %}">My Blog</a></h1>
            <ul>
                <li><a href="{% url 'blog:index' %}">Home</a></li>
                <li><a href="{% url 'blog:about' %}">About</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        {% block content %}
        {% endblock %}
    </main>
    
    <footer>
        <p>&copy; 2024 My Blog. All rights reserved.</p>
    </footer>
    
    {% block extra_js %}{% endblock %}
</body>
</html>
```

### Child Templates - Specialized Dishes

Just as each dish on your menu has its own unique presentation while maintaining your restaurant's signature style, child templates inherit the base structure while adding their own content.

```html
<!-- blog/templates/blog/post_list.html -->
{% extends 'base.html' %}
{% load static %}

{% block title %}Latest Posts - My Blog{% endblock %}

{% block extra_css %}
    <link rel="stylesheet" href="{% static 'css/blog.css' %}">
{% endblock %}

{% block content %}
    <h1>Latest Blog Posts</h1>
    
    <div class="posts-grid">
        {% for post in posts %}
            <article class="post-card">
                <h2><a href="{% url 'blog:post_detail' post.pk %}">{{ post.title }}</a></h2>
                <p class="post-meta">
                    By {{ post.author.username }} on {{ post.created_at|date:"M d, Y" }}
                </p>
                <p>{{ post.content|truncatewords:30 }}</p>
                <a href="{% url 'blog:post_detail' post.pk %}" class="read-more">Read More</a>
            </article>
        {% empty %}
            <p>No posts available yet.</p>
        {% endfor %}
    </div>
{% endblock %}
```

---

## Part 3: Static Files - Your Restaurant's Ambiance

### Understanding Static Files

Imagine that your restaurant's food is excellent, but you serve it in a plain white room with no decorations, harsh fluorescent lighting, and folding chairs. The food might be great, but the overall experience would be disappointing.

Static files in Django - CSS, JavaScript, and images - are like your restaurant's ambiance. They transform a functional but plain webpage into an engaging, professional experience.

### Configuring Static Files

```python
# settings.py
import os

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# Media files (User uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

### Creating Your Static Files Structure

```
myproject/
├── static/
│   ├── css/
│   │   ├── style.css
│   │   └── blog.css
│   ├── js/
│   │   └── main.js
│   └── images/
│       └── logo.png
├── blog/
│   └── static/
│       └── blog/
│           ├── css/
│           └── images/
└── templates/
```

### Example CSS for Your Blog

```css
/* static/css/style.css */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Georgia', serif;
    line-height: 1.6;
    color: #333;
    background-color: #f9f9f9;
}

header {
    background-color: #2c3e50;
    color: white;
    padding: 1rem 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
}

nav h1 a {
    color: white;
    text-decoration: none;
    font-size: 1.8rem;
}

nav ul {
    list-style: none;
    display: flex;
    gap: 2rem;
}

nav a {
    color: white;
    text-decoration: none;
    transition: color 0.3s ease;
}

nav a:hover {
    color: #3498db;
}

main {
    max-width: 1200px;
    margin: 2rem auto;
    padding: 0 2rem;
    min-height: calc(100vh - 200px);
}

footer {
    background-color: #34495e;
    color: white;
    text-align: center;
    padding: 2rem 0;
    margin-top: 3rem;
}
```

```css
/* static/css/blog.css */
.posts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.post-card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.post-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}

.post-card h2 a {
    color: #2c3e50;
    text-decoration: none;
}

.post-card h2 a:hover {
    color: #3498db;
}

.post-meta {
    color: #7f8c8d;
    font-size: 0.9rem;
    margin: 0.5rem 0 1rem 0;
}

.read-more {
    color: #3498db;
    text-decoration: none;
    font-weight: bold;
}

.read-more:hover {
    text-decoration: underline;
}
```

---

## Part 4: Template Filters and Tags - Your Chef's Special Techniques

### Template Filters - The Garnishes and Seasonings

Think of template filters as your chef's collection of finishing techniques. Just as a chef might add a drizzle of balsamic reduction or a sprinkle of fresh herbs to enhance a dish's presentation, template filters modify how your data appears to users.

### Common Template Filters

```html
<!-- blog/templates/blog/post_detail.html -->
{% extends 'base.html' %}
{% load static %}

{% block title %}{{ post.title }} - My Blog{% endblock %}

{% block content %}
    <article class="post-detail">
        <h1>{{ post.title }}</h1>
        <div class="post-meta">
            <span>By {{ post.author.username|title }}</span>
            <span>{{ post.created_at|date:"F d, Y" }}</span>
            <span>{{ post.content|wordcount }} words</span>
            <span>{{ post.created_at|timesince }} ago</span>
        </div>
        
        <div class="post-content">
            {{ post.content|linebreaks }}
        </div>
        
        {% if post.tags.all %}
            <div class="tags">
                <h3>Tags:</h3>
                {% for tag in post.tags.all %}
                    <span class="tag">{{ tag.name|capfirst }}</span>
                {% endfor %}
            </div>
        {% endif %}
        
        <div class="post-excerpt">
            <h3>Summary:</h3>
            <p>{{ post.content|truncatewords:50 }}</p>
        </div>
    </article>
{% endblock %}
```

### Custom Template Tags - Your Signature Dishes

Just as master chefs create signature dishes that become synonymous with their restaurant, you can create custom template tags for specialized functionality.

```python
# blog/templatetags/blog_extras.py
from django import template
from django.utils.safestring import mark_safe
import markdown

register = template.Library()

@register.filter
def markdown_format(text):
    """Convert markdown text to HTML"""
    return mark_safe(markdown.markdown(text))

@register.simple_tag
def get_popular_posts(limit=5):
    """Get popular posts for sidebar"""
    from blog.models import Post
    return Post.objects.filter(is_published=True).order_by('-views')[:limit]

@register.inclusion_tag('blog/popular_posts.html')
def show_popular_posts(limit=5):
    """Render popular posts widget"""
    posts = Post.objects.filter(is_published=True).order_by('-views')[:limit]
    return {'posts': posts}
```

### Using Custom Tags in Templates

```html
<!-- blog/templates/blog/post_detail.html -->
{% extends 'base.html' %}
{% load static %}
{% load blog_extras %}

{% block content %}
    <div class="row">
        <div class="main-content">
            <article>
                <h1>{{ post.title }}</h1>
                <div class="content">
                    {{ post.content|markdown_format }}
                </div>
            </article>
        </div>
        
        <aside class="sidebar">
            <h3>Popular Posts</h3>
            {% show_popular_posts 3 %}
        </aside>
    </div>
{% endblock %}
```

---

## Part 5: Building Your Dynamic Blog Layout

### The Complete Views

```python
# blog/views.py
from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from .models import Post, Comment

def post_list(request):
    """Display paginated list of published posts"""
    posts = Post.objects.filter(is_published=True).order_by('-created_at')
    paginator = Paginator(posts, 6)  # Show 6 posts per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'blog/post_list.html', {
        'posts': page_obj,
        'page_obj': page_obj,
    })

def post_detail(request, pk):
    """Display individual post with comments"""
    post = get_object_or_404(Post, pk=pk, is_published=True)
    comments = post.comments.filter(is_approved=True).order_by('-created_at')
    
    # Increment view count
    post.views += 1
    post.save()
    
    return render(request, 'blog/post_detail.html', {
        'post': post,
        'comments': comments,
    })

def about(request):
    """About page"""
    return render(request, 'blog/about.html')
```

### The Complete URLs

```python
# blog/urls.py
from django.urls import path
from . import views

app_name = 'blog'

urlpatterns = [
    path('', views.post_list, name='index'),
    path('post/<int:pk>/', views.post_detail, name='post_detail'),
    path('about/', views.about, name='about'),
]
```

### Enhanced Post List with Pagination

```html
<!-- blog/templates/blog/post_list.html -->
{% extends 'base.html' %}
{% load static %}

{% block title %}Latest Posts - My Blog{% endblock %}

{% block extra_css %}
    <link rel="stylesheet" href="{% static 'css/blog.css' %}">
{% endblock %}

{% block content %}
    <div class="blog-header">
        <h1>Latest Blog Posts</h1>
        <p>Sharing thoughts, ideas, and experiences</p>
    </div>
    
    <div class="posts-grid">
        {% for post in posts %}
            <article class="post-card">
                {% if post.featured_image %}
                    <img src="{{ post.featured_image.url }}" alt="{{ post.title }}" class="post-image">
                {% endif %}
                
                <div class="post-card-content">
                    <h2><a href="{% url 'blog:post_detail' post.pk %}">{{ post.title }}</a></h2>
                    <p class="post-meta">
                        By {{ post.author.username|title }} on {{ post.created_at|date:"M d, Y" }}
                        • {{ post.views }} views
                    </p>
                    <p class="post-excerpt">{{ post.content|truncatewords:30 }}</p>
                    
                    {% if post.tags.all %}
                        <div class="post-tags">
                            {% for tag in post.tags.all %}
                                <span class="tag">{{ tag.name }}</span>
                            {% endfor %}
                        </div>
                    {% endif %}
                    
                    <a href="{% url 'blog:post_detail' post.pk %}" class="read-more">Read More →</a>
                </div>
            </article>
        {% empty %}
            <div class="no-posts">
                <h2>No posts available yet</h2>
                <p>Check back soon for new content!</p>
            </div>
        {% endfor %}
    </div>
    
    <!-- Pagination -->
    {% if page_obj.has_other_pages %}
        <div class="pagination">
            {% if page_obj.has_previous %}
                <a href="?page=1" class="page-link">First</a>
                <a href="?page={{ page_obj.previous_page_number }}" class="page-link">Previous</a>
            {% endif %}
            
            <span class="page-current">
                Page {{ page_obj.number }} of {{ page_obj.paginator.num_pages }}
            </span>
            
            {% if page_obj.has_next %}
                <a href="?page={{ page_obj.next_page_number }}" class="page-link">Next</a>
                <a href="?page={{ page_obj.paginator.num_pages }}" class="page-link">Last</a>
            {% endif %}
        </div>
    {% endif %}
{% endblock %}
```

### Complete Post Detail Template

```html
<!-- blog/templates/blog/post_detail.html -->
{% extends 'base.html' %}
{% load static %}
{% load blog_extras %}

{% block title %}{{ post.title }} - My Blog{% endblock %}

{% block extra_css %}
    <link rel="stylesheet" href="{% static 'css/blog.css' %}">
    <link rel="stylesheet" href="{% static 'css/post-detail.css' %}">
{% endblock %}

{% block content %}
    <div class="post-detail-container">
        <article class="post-detail">
            <header class="post-header">
                <h1>{{ post.title }}</h1>
                <div class="post-meta">
                    <span class="author">By {{ post.author.username|title }}</span>
                    <span class="date">{{ post.created_at|date:"F d, Y" }}</span>
                    <span class="views">{{ post.views }} views</span>
                    <span class="reading-time">{{ post.content|wordcount|floatformat:0 }} words</span>
                </div>
            </header>
            
            {% if post.featured_image %}
                <img src="{{ post.featured_image.url }}" alt="{{ post.title }}" class="featured-image">
            {% endif %}
            
            <div class="post-content">
                {{ post.content|markdown_format }}
            </div>
            
            {% if post.tags.all %}
                <div class="post-tags">
                    <h3>Tags:</h3>
                    {% for tag in post.tags.all %}
                        <span class="tag">{{ tag.name|capfirst }}</span>
                    {% endfor %}
                </div>
            {% endif %}
            
            <div class="post-navigation">
                <a href="{% url 'blog:index' %}" class="back-to-posts">← Back to All Posts</a>
            </div>
        </article>
        
        <section class="comments-section">
            <h3>Comments ({{ comments.count }})</h3>
            {% for comment in comments %}
                <div class="comment">
                    <div class="comment-header">
                        <strong>{{ comment.author.username|title }}</strong>
                        <span class="comment-date">{{ comment.created_at|date:"M d, Y" }}</span>
                    </div>
                    <div class="comment-content">
                        {{ comment.content|linebreaks }}
                    </div>
                </div>
            {% empty %}
                <p class="no-comments">No comments yet. Be the first to comment!</p>
            {% endfor %}
        </section>
    </div>
{% endblock %}
```

---

## Assignment: Create a Personal Portfolio Blog

### Your Mission: Build a Professional Portfolio Blog

You are tasked with creating a personal portfolio blog that showcases your projects and thoughts. This blog should demonstrate mastery of Django templates, template inheritance, static files, and custom template features.

### Requirements:

1. **Template Structure:**
   - Create a base template with navigation, header, and footer
   - Implement at least 4 different page templates (home, about, portfolio, contact)
   - Use template inheritance consistently

2. **Static Files:**
   - Create a cohesive design with custom CSS
   - Include at least one JavaScript interaction
   - Use proper static file organization

3. **Content Features:**
   - Display a list of projects with descriptions, technologies used, and links
   - Include an about page with your bio and skills
   - Create a blog section with at least 3 sample posts
   - Implement pagination for blog posts

4. **Template Features:**
   - Use at least 5 different template filters
   - Create one custom template tag or filter
   - Include template conditionals and loops

5. **Styling Requirements:**
   - Responsive design that works on mobile and desktop
   - Consistent color scheme and typography
   - Hover effects and smooth transitions
   - Professional, clean aesthetic

### Deliverables:
- Complete Django project with all templates
- Static files (CSS, JS, images)
- Sample data for projects and blog posts
- Documentation of custom template features used

### Success Criteria:
Your portfolio blog should look professional enough that you'd be comfortable sharing it with potential employers or clients. The code should be clean, well-organized, and demonstrate understanding of Django's template system.

---

## Conclusion: From Raw Ingredients to Michelin-Star Presentation

You've now learned how to transform your Django application from a functional but plain backend into a beautiful, professional web experience. Like a master chef who has perfected both the culinary arts and the art of presentation, you can now create web applications that not only work flawlessly but also provide an engaging, visually appealing experience for your users.

The template system you've mastered today is the foundation of professional web development. You've learned to create reusable, maintainable templates that separate your presentation logic from your business logic, manage static files professionally, and create custom template features that make your applications truly unique.

Remember, just as a restaurant's success depends on both excellent food and exceptional presentation, your web applications will succeed when they combine solid functionality with beautiful, intuitive user interfaces. You now have the tools to create both.