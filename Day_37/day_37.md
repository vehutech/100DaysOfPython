
# **Day 37: HTML Best Practices — Polishing Your Plates** ✨🍽️

Wet your imagination for a sec.....
You're preparing for the most important dinner party of your career. Every plate must be spotless, every garnish perfectly placed, every detail flawless. Your guests will notice everything — from the cleanliness of the silverware to the elegance of the presentation. That's exactly what HTML best practices are like — they're the final polish that transforms good code into exceptional, professional-grade markup that delights both users and search engines.

## 🎯 Objectives
- Master clean, organized HTML code structure and formatting
- Implement comprehensive accessibility features for all users
- Validate HTML code using industry-standard tools
- Create maintainable code that scales with your projects
- Prepare HTML foundation for seamless CSS and JavaScript integration
- Build production-ready, professional-quality web pages

## 🧹 Code Organization: Setting Your Kitchen in Order
Just like a professional kitchen needs everything in its proper place, your HTML needs consistent organization that makes it easy to read, maintain, and debug.

### 🔹 Proper Indentation and Formatting
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grandma's Recipe Collection</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header class="site-header">
        <div class="container">
            <h1 class="site-title">Grandma's Kitchen</h1>
            <nav class="main-navigation" aria-label="Main menu">
                <ul class="nav-list">
                    <li class="nav-item">
                        <a href="/" class="nav-link">Home</a>
                    </li>
                    <li class="nav-item">
                        <a href="/recipes" class="nav-link">Recipes</a>
                    </li>
                    <li class="nav-item">
                        <a href="/about" class="nav-link">About</a>
                    </li>
                </ul>
            </nav>
        </div>
    </header>

    <main class="main-content">
        <section class="hero-section">
            <div class="container">
                <h2 class="hero-title">Welcome to Our Kitchen</h2>
                <p class="hero-description">
                    Discover time-tested recipes passed down through generations
                </p>
            </div>
        </section>
        
        <section class="featured-recipes">
            <div class="container">
                <h2 class="section-title">Featured Recipes</h2>
                <div class="recipe-grid">
                    <article class="recipe-card">
                        <img src="cookies.jpg" alt="Golden chocolate chip cookies" class="recipe-image">
                        <div class="recipe-content">
                            <h3 class="recipe-title">Chocolate Chip Cookies</h3>
                            <p class="recipe-description">Grandma's secret family recipe</p>
                            <div class="recipe-meta">
                                <span class="prep-time">Prep: 15 min</span>
                                <span class="cook-time">Cook: 12 min</span>
                                <span class="difficulty">Easy</span>
                            </div>
                            <a href="/recipes/chocolate-chip-cookies" class="recipe-link">View Recipe</a>
                        </div>
                    </article>
                    <article class="recipe-card">
                        <img src="chocolate-cake.jpg" alt="Three-layer chocolate cake with frosting" class="recipe-image">
                        <div class="recipe-content">
                            <h3 class="recipe-title">Chocolate Cake</h3>
                            <p class="recipe-description">Rich and moist family favorite</p>
                            <div class="recipe-meta">
                                <span class="prep-time">Prep: 20 min</span>
                                <span class="cook-time">Cook: 35 min</span>
                                <span class="difficulty">Medium</span>
                            </div>
                            <a href="/recipes/chocolate-cake" class="recipe-link">View Recipe</a>
                        </div>
                    </article>
                </div>
            </div>
        </section>
    </main>

    <footer class="site-footer">
        <div class="container">
            <p class="copyright">
                © 2024 Grandma's Kitchen. All rights reserved.
            </p>
        </div>
    </footer>
</body>
</html>
```

### 🔸 Consistent Naming Conventions
```html
<!-- Use clear, descriptive class names -->
<div class="recipe-card">
    <img src="cookies.jpg" alt="Golden chocolate chip cookies" class="recipe-image">
    <div class="recipe-content">
        <h3 class="recipe-title">Chocolate Chip Cookies</h3>
        <p class="recipe-description">Grandma's secret family recipe</p>
        <div class="recipe-meta">
            <span class="prep-time">Prep: 15 min</span>
            <span class="cook-time">Cook: 12 min</span>
            <span class="difficulty">Easy</span>
        </div>
        <a href="/recipes/chocolate-chip-cookies" class="recipe-link">
            View Recipe
        </a>
    </div>
</div>

<!-- Group related elements logically -->
<form class="contact-form" action="/submit" method="POST">
    <fieldset class="form-group">
        <legend class="form-legend">Personal Information</legend>
        
        <div class="input-group">
            <label for="full-name" class="input-label">Full Name</label>
            <input type="text" id="full-name" name="fullName" 
                   class="form-input" required>
        </div>
        
        <div class="input-group">
            <label for="email" class="input-label">Email Address</label>
            <input type="email" id="email" name="email" 
                   class="form-input" required>
        </div>
    </fieldset>
    
    <fieldset class="form-group">
        <legend class="form-legend">Your Message</legend>
        
        <div class="input-group">
            <label for="message" class="input-label">Message</label>
            <textarea id="message" name="message" rows="5" 
                      class="form-textarea" required></textarea>
        </div>
    </fieldset>
    
    <button type="submit" class="submit-button">Send Message</button>
</form>
```

## ♿ Accessibility: Making Your Restaurant Welcoming to Everyone
Accessibility isn't just a nice-to-have — it's essential for creating an inclusive web experience. Think of it as making sure every guest can enjoy your restaurant, regardless of their needs.

### 🔹 Comprehensive ARIA Implementation
```html
<!-- Skip links for keyboard users -->
<a href="#main-content" class="skip-link">Skip to main content</a>
<a href="#main-navigation" class="skip-link">Skip to navigation</a>

<header role="banner">
    <nav id="main-navigation" role="navigation" aria-label="Main menu">
        <button class="menu-toggle" 
                aria-expanded="false" 
                aria-controls="menu-list"
                aria-label="Toggle main menu">
            <span class="hamburger-icon"></span>
        </button>
        
        <ul id="menu-list" class="menu-list" role="menubar">
            <li role="none">
                <a href="/" role="menuitem" aria-current="page">Home</a>
            </li>
            <li role="none">
                <a href="/recipes" role="menuitem">Recipes</a>
            </li>
            <li role="none">
                <a href="/about" role="menuitem">About</a>
            </li>
        </ul>
    </nav>
</header>

<main id="main-content" role="main">
    <!-- Live region for dynamic content updates -->
    <div aria-live="polite" aria-atomic="true" class="status-updates">
        <!-- Status messages appear here -->
    </div>
    
    <!-- Search functionality -->
    <section class="search-section" role="search">
        <h2>Find Recipes</h2>
        <form class="search-form">
            <div class="search-input-group">
                <label for="recipe-search" class="search-label">
                    Search recipes by ingredient or name
                </label>
                <input type="search" 
                       id="recipe-search" 
                       name="query"
                       class="search-input"
                       placeholder="Enter ingredients..."
                       aria-describedby="search-help">
                <div id="search-help" class="search-help">
                    Enter ingredients separated by commas
                </div>
            </div>
            
            <button type="submit" class="search-button">
                <span class="button-text">Search</span>
                <span class="sr-only">Search for recipes</span>
            </button>
        </form>
    </section>
</main>

<!-- Accessible modal dialog -->
<div id="recipe-modal" 
     class="modal" 
     role="dialog" 
     aria-labelledby="modal-title"
     aria-describedby="modal-description"
     aria-hidden="true">
    <div class="modal-content">
        <header class="modal-header">
            <h2 id="modal-title">Recipe Details</h2>
            <button class="modal-close" 
                    aria-label="Close recipe details"
                    data-dismiss="modal">
                ×
            </button>
        </header>
        
        <div id="modal-description" class="modal-body">
            <p>Recipe content will be loaded dynamically here.</p>
        </div>
    </div>
</div>
```

### 🔸 Images and Media Accessibility
```html
<!-- Informative images -->
<figure class="recipe-figure">
    <img src="chocolate-cake.jpg" 
         alt="Three-layer chocolate cake with chocolate frosting and fresh strawberries on top, served on a white ceramic plate"
         class="recipe-image">
    <figcaption class="recipe-caption">
        Our signature chocolate cake, perfect for special occasions
    </figcaption>
</figure>

<!-- Decorative images -->
<div class="hero-banner">
    <img src="kitchen-background.jpg" 
         alt="" 
         role="presentation"
         class="hero-bg">
    <div class="hero-content">
        <h1>Welcome to Our Kitchen</h1>
    </div>
</div>

<!-- Complex images with descriptions -->
<figure class="infographic">
    <img src="cooking-times-chart.jpg" 
         alt="Cooking times chart - see detailed description below"
         class="chart-image">
    <figcaption>
        <h3>Cooking Times Reference</h3>
        <p><strong>Detailed description:</strong></p>
        <ul>
            <li>Vegetables: Steam 5-10 minutes, roast 20-30 minutes</li>
            <li>Chicken breast: Bake 20-25 minutes at 375°F</li>
            <li>Fish fillets: Pan-fry 3-4 minutes per side</li>
            <li>Pasta: Boil 8-12 minutes depending on type</li>
        </ul>
    </figcaption>
</figure>

<!-- Video with captions and transcript -->
<div class="video-container">
    <video controls 
           preload="metadata"
           aria-describedby="video-description">
        <source src="cookie-baking.mp4" type="video/mp4">
        <source src="cookie-baking.webm" type="video/webm">
        <track kind="captions" 
               src="cookie-baking-captions.vtt" 
               srclang="en" 
               label="English Captions">
        <p>Your browser doesn't support video. 
           <a href="cookie-baking.mp4">Download the video</a> instead.</p>
    </video>
    
    <div id="video-description" class="video-description">
        <h3>Video: How to Bake Perfect Cookies</h3>
        <p>A step-by-step demonstration of our cookie baking process, 
           from mixing ingredients to removing golden cookies from the oven.</p>
        <details>
            <summary>Full transcript</summary>
            <p>Hi everyone, I'm Chef Maria, and today I'll show you how to bake 
               the perfect chocolate chip cookies...</p>
        </details>
    </div>
</div>
```

### 🔹 Keyboard Navigation Excellence
```html
<!-- Proper focus management -->
<div class="recipe-tabs">
    <div role="tablist" aria-label="Recipe sections">
        <button role="tab" 
                aria-selected="true" 
                aria-controls="ingredients-panel"
                id="ingredients-tab"
                tabindex="0">
            Ingredients
        </button>
        <button role="tab" 
                aria-selected="false" 
                aria-controls="instructions-panel"
                id="instructions-tab"
                tabindex="-1">
            Instructions
        </button>
        <button role="tab" 
                aria-selected="false" 
                aria-controls="notes-panel"
                id="notes-tab"
                tabindex="-1">
            Chef's Notes
        </button>
    </div>
    
    <div id="ingredients-panel" 
         role="tabpanel" 
         aria-labelledby="ingredients-tab">
        <h3>Ingredients</h3>
        <ul class="ingredients-list">
            <li>2 cups all-purpose flour</li>
            <li>1 cup butter, softened</li>
            <li>3/4 cup brown sugar, packed</li>
            <li>3/4 cup granulated sugar</li>
            <li>2 large eggs</li>
            <li>1 tsp vanilla extract</li>
            <li>1 tsp baking soda</li>
            <li>2 cups chocolate chips</li>
        </ul>
    </div>
    
    <div id="instructions-panel" 
         role="tabpanel" 
         aria-labelledby="instructions-tab"
         hidden>
        <h3>Instructions</h3>
        <ol class="instructions-list">
            <li>Preheat oven to 375°F</li>
            <li>Cream butter and sugars until fluffy</li>
            <li>Add eggs and vanilla, mix well</li>
            <li>Combine dry ingredients, add to wet mixture</li>
            <li>Stir in chocolate chips</li>
            <li>Drop by spoonfuls onto baking sheet</li>
            <li>Bake for 10-12 minutes</li>
        </ol>
    </div>
    
    <div id="notes-panel" 
         role="tabpanel" 
         aria-labelledby="notes-tab"
         hidden>
        <h3>Chef's Notes</h3>
        <p>Use high-quality chocolate chips for best flavor.</p>
    </div>
</div>

<!-- Custom interactive elements -->
<div class="rating-widget">
    <span id="rating-label" class="rating-label">Rate this recipe:</span>
    <div role="radiogroup" aria-labelledby="rating-label">
        <input type="radio" id="star1" name="rating" value="1">
        <label for="star1" class="star-label">
            <span class="sr-only">1 star</span>
            ★
        </label>
        
        <input type="radio" id="star2" name="rating" value="2">
        <label for="star2" class="star-label">
            <span class="sr-only">2 stars</span>
            ★
        </label>
        
        <input type="radio" id="star3" name="rating" value="3">
        <label for="star3" class="star-label">
            <span class="sr-only">3 stars</span>
            ★
        </label>
        
        <input type="radio" id="star4" name="rating" value="4">
        <label for="star4" class="star-label">
            <span class="sr-only">4 stars</span>
            ★
        </label>
        
        <input type="radio" id="star5" name="rating" value="5">
        <label for="star5" class="star-label">
            <span class="sr-only">5 stars</span>
            ★
        </label>
    </div>
</div>
```

## ✅ HTML Validation: Quality Assurance for Your Code
Just like a food safety inspector ensures restaurant quality, HTML validation ensures your code meets web standards.

### 🔹 Using the W3C Validator
```html
<!-- Common validation errors to avoid -->

<!-- ❌ WRONG: Missing required attributes -->
<img src="recipe.jpg">
<input type="text">
<label>Name</label>

<!-- ✅ CORRECT: Proper attributes -->
<img src="recipe.jpg" alt="Chocolate chip cookies on a plate">
<input type="text" id="user-name" name="userName">
<label for="user-name">Name</label>

<!-- ❌ WRONG: Incorrect nesting -->
<p>This is a paragraph <div>with a div inside</div></p>
<a href="/recipe"><a href="/ingredients">Ingredients</a></a>

<!-- ✅ CORRECT: Proper nesting -->
<p>This is a paragraph <span>with a span inside</span></p>
<a href="/recipe">Recipe <span class="sub-link">→ Ingredients</span></a>

<!-- ❌ WRONG: Duplicate IDs -->
<input type="text" id="name">
<div id="name">User Name</div>

<!-- ✅ CORRECT: Unique IDs -->
<input type="text" id="user-name">
<div id="name-description">User Name</div>

<!-- ❌ WRONG: Missing doctype and lang -->
<html>
<head>
    <title>Recipe Site</title>
</head>

<!-- ✅ CORRECT: Complete document structure -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recipe Site</title>
</head>
```

### 🔸 Validation Checklist
```html
<!-- Complete head section -->
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Essential meta tags -->
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Traditional family recipes and cooking tips">
    <meta name="keywords" content="recipes, cooking, family, traditional">
    <meta name="author" content="Grandma's Kitchen">
    
    <!-- Page title -->
    <title>Home - Grandma's Kitchen | Traditional Family Recipes</title>
    
    <!-- Favicon -->
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <link rel="apple-touch-icon" href="/apple-touch-icon.png">
    
    <!-- Stylesheets -->
    <link rel="stylesheet" href="styles/main.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap">
    
    <!-- Open Graph meta tags for social sharing -->
    <meta property="og:title" content="Grandma's Kitchen - Traditional Family Recipes">
    <meta property="og:description" content="Discover time-tested recipes passed down through generations">
    <meta property="og:image" content="/images/og-image.jpg">
    <meta property="og:url" content="https://grandmaskitchen.com">
    <meta property="og:type" content="website">
</head>
```

## 💬 Strategic Commenting: Documentation for Future You
Comments are like recipe notes — they help you (and others) understand the purpose and context of your code.

### 🔹 Meaningful Comments
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Recipe Collection</title>
    <!-- Critical CSS loaded inline for above-the-fold content -->
    <style>
        /* Above-the-fold styles */
        .site-header { background: #f8f8f8; padding: 1rem; }
        .main-content { max-width: 1200px; margin: 0 auto; }
    </style>
</head>
<body>
    <!-- Skip navigation for accessibility -->
    <a href="#main-content" class="skip-link">Skip to main content</a>
    
    <!-- === SITE HEADER === -->
    <header class="site-header">
        <div class="container">
            <!-- Logo and site title -->
            <div class="brand">
                <img src="logo.svg" alt="Grandma's Kitchen" class="logo">
                <h1 class="site-title">Grandma's Kitchen</h1>
            </div>
            
            <!-- Primary navigation -->
            <nav class="main-nav" aria-label="Main menu">
                <!-- Navigation items dynamically generated by CMS -->
                <ul class="nav-list">
                    <li><a href="/">Home</a></li>
                    <li><a href="/recipes">Recipes</a></li>
                    <li><a href="/about">About</a></li>
                </ul>
            </nav>
        </div>
    </header>
    
    <!-- === MAIN CONTENT AREA === -->
    <main id="main-content" class="main-content">
        <!-- Hero section - featured recipe or promotion -->
        <section class="hero-section">
            <div class="container">
                <!-- Hero content managed through CMS -->
                <div class="hero-content">
                    <h2 class="hero-title">Today's Featured Recipe</h2>
                    <p class="hero-description">
                        Discover the secret to perfect chocolate chip cookies
                    </p>
                </div>
            </div>
        </section>
        
        <!-- Recipe grid - latest recipes -->
        <section class="recipe-grid-section">
            <div class="container">
                <h2 class="section-title">Latest Recipes</h2>
                
                <!-- Recipe cards container -->
                <div class="recipe-grid">
                    <!-- Individual recipe card -->
                    <article class="recipe-card">
                        <!-- Recipe image with lazy loading -->
                        <div class="recipe-image-container">
                            <img src="placeholder.jpg" 
                                 data-src="cookies.jpg" 
                                 alt="Golden chocolate chip cookies"
                                 class="recipe-image lazy-load">
                        </div>
                        
                        <!-- Recipe meta information -->
                        <div class="recipe-content">
                            <h3 class="recipe-title">
                                <a href="/recipes/chocolate-chip-cookies">
                                    Chocolate Chip Cookies
                                </a>
                            </h3>
                            
                            <!-- Recipe stats -->
                            <div class="recipe-meta">
                                <span class="prep-time" title="Preparation time">
                                    <span aria-label="Prep time">⏱️</span> 15 min
                                </span>
                                <span class="difficulty" title="Difficulty level">
                                    <span aria-label="Difficulty">📊</span> Easy
                                </span>
                            </div>
                        </div>
                    </article>
                    
                    <!-- Another recipe card -->
                    <article class="recipe-card">
                        <div class="recipe-image-container">
                            <img src="placeholder.jpg" 
                                 data-src="chocolate-cake.jpg" 
                                 alt="Three-layer chocolate cake"
                                 class="recipe-image lazy-load">
                        </div>
                        <div class="recipe-content">
                            <h3 class="recipe-title">
                                <a href="/recipes/chocolate-cake">
                                    Chocolate Cake
                                </a>
                            </h3>
                            <div class="recipe-meta">
                                <span class="prep-time" title="Preparation time">
                                    <span aria-label="Prep time">⏱️</span> 20 min
                                </span>
                                <span class="difficulty" title="Difficulty level">
                                    <span aria-label="Difficulty">📊</span> Medium
                                </span>
                            </div>
                        </div>
                    </article>
                </div>
            </div>
        </section>
    </main>
    
    <!-- === SITE FOOTER === -->
    <footer class="site-footer">
        <div class="container">
            <!-- Footer navigation -->
            <nav class="footer-nav" aria-label="Footer menu">
                <ul class="footer-links">
                    <li><a href="/privacy">Privacy Policy</a></li>
                    <li><a href="/terms">Terms of Service</a></li>
                    <li><a href="/contact">Contact</a></li>
                </ul>
            </nav>
            
            <!-- Copyright and legal -->
            <div class="footer-bottom">
                <p class="copyright">
                    © 2024 Grandma's Kitchen. All rights reserved.
                </p>
            </div>
        </div>
    </footer>
    
    <!-- JavaScript files loaded at end for performance-->
    <script src="js/main.js"></script>
    <!-- Analytics script -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'GA_MEASUREMENT_ID');
    </script>
</body>
</html>
```

### 🔸 Section Organization Comments
```html
<!-- ============================================
     RECIPE DETAIL PAGE
     ============================================ -->

<!-- Page header with breadcrumbs -->
<header class="page-header">
    <!-- Breadcrumb navigation for SEO and UX -->
    <nav aria-label="Breadcrumb">
        <ol class="breadcrumb">
            <li><a href="/">Home</a></li>
            <li><a href="/recipes">Recipes</a></li>
            <li aria-current="page">Chocolate Chip Cookies</li>
        </ol>
    </nav>
</header>

<!-- ============================================
     RECIPE CONTENT
     ============================================ -->

<main class="recipe-main">
    <article class="recipe-article">
        <!-- Recipe header with title and meta -->
        <header class="recipe-header">
            <!-- SEO: H1 tag for page title -->
            <h1 class="recipe-title">Grandma's Famous Chocolate Chip Cookies</h1>
            
            <!-- Recipe metadata for rich snippets -->
            <div class="recipe-meta" itemscope itemtype="https://schema.org/Recipe">
                <meta itemprop="name" content="Grandma's Famous Chocolate Chip Cookies">
                <meta itemprop="description" content="Traditional family recipe for perfect chocolate chip cookies">
                
                <!-- Visible metadata -->
                <div class="meta-grid">
                    <span class="prep-time">
                        <strong>Prep time:</strong> <time itemprop="prepTime" datetime="PT15M">15 minutes</time>
                    </span>
                    <span class="cook-time">
                        <strong>Cook:</strong> <time itemprop="cookTime" datetime="PT12M">12 minutes</time>
                    </span>
                    <span class="servings">
                        <strong>Serves:</strong> <span itemprop="recipeYield">24 cookies</span>
                    </span>
                </div>
            </div>
        </header>
        
        <!-- Recipe image -->
        <figure class="recipe-image-figure">
            <img src="cookies-hero.jpg" 
                 alt="Golden brown chocolate chip cookies cooling on a wire rack" 
                 class="recipe-hero-image"
                 itemprop="image">
            <figcaption class="recipe-image-caption">
                Fresh from the oven and perfectly golden cookies
            </figcaption>
        </figure>
        
        <!-- Recipe content sections -->
        <div class="recipe-content">
            <!-- Ingredients section -->
            <section class="ingredients-section">
                <h2 class="section-title">Ingredients</h2>
                <!-- Ingredients list with structured data -->
                <ul class="ingredients-list" itemprop="recipeIngredient">
                    <li>2 cups all-purpose flour</li>
                    <li>1 cup butter, softened</li>
                    <li>3/4 cup brown sugar, packed</li>
                    <li>3/4 cup granulated sugar</li>
                    <li>2 large eggs</li>
                    <li>1 tsp vanilla extract</li>
                    <li>1 tsp baking soda</li>
                    <li>1/2 tsp salt</li>
                    <li>2 cups chocolate chips</li>
                </ul>
            </section>
            
            <!-- Instructions section -->
            <section class="instructions-section">
                <h2 class="section-title">Instructions</h2>
                <!-- Step-by-step instructions -->
                <ol class="instructions-list" itemprop="recipeInstructions">
                    <li itemprop="itemListElement">Preheat oven to 375°F (190°C)</li>
                    <li itemprop="itemListElement">Cream butter and sugars until fluffy</li>
                    <li itemprop="itemListElement">Add eggs and vanilla, mix well</li>
                    <li itemprop="itemListElement">Combine dry ingredients, add to wet mixture</li>
                    <li itemprop="itemListElement">Stir in chocolate chips</li>
                    <li itemprop="itemListElement">Drop by spoonfuls onto baking sheet</li>
                    <li itemprop="itemListElement">Bake for 10-12 minutes until golden</li>
                </ol>
            </section>
        </div>
    </article>
</main>

<!-- ============================================
     SUPPORTING CONTENT
     ============================================ -->

<!-- Chef's tips sidebar -->
<aside class="recipe-sidebar">
    <section class="tips-section">
        <h2 class="sidebar-title">Chef's Tips</h2>
        <!-- Helpful cooking tips -->
        <ul>
            <li>Use room-temperature butter for better creaming.</li>
            <li>Chill dough for 30 minutes to prevent spreading.</li>
            <li>Use parchment paper for easy cleanup.</li>
        </ul>
    </section>
</aside>
```

## 🚀 Preparing for CSS and JavaScript Integration
Your HTML is the foundation — it needs to be ready to support styling and interactivity.

### 🔹 CSS-Ready Structure
```html
<!-- Semantic structure with styling hooks -->
<article class="recipe-card recipe-card--featured">
    <div class="recipe-card__image-container">
        <img src="recipe.jpg" alt="Delicious homemade cookies" class="recipe-card__image">
        <div class="recipe-card__overlay">
            <span class="recipe-card__category">Desserts</span>
        </div>
    </div>
    
    <div class="recipe-card__content">
        <header class="recipe-card__header">
            <h3 class="recipe-card__title">
                <a href="/recipe/cookies" class="recipe-card__link">
                    Chocolate Chip Cookies
                </a>
            </h3>
            <div class="recipe-card__rating">
                <span class="rating rating--4-5">★★★★☆</span>
            </div>
        </header>
        
        <p class="recipe-card__description">
            Traditional family recipe with a secret ingredient
        </p>
        
        <div class="recipe-card__meta">
            <span class="recipe-meta recipe-meta--time">
                <span class="recipe-meta__icon">⏱️</span>
                <span class="recipe-meta__text">30 min</span>
            </span>
            <span class="recipe-meta recipe-meta--difficulty">
                <span class="recipe-meta__icon">📊</span>
                <span class="recipe-meta__text">Easy</span>
            </span>
        </div>
        
        <footer class="recipe-card__footer">
            <button class="btn btn--primary btn--save" 
                    data-recipe-id="123"
                    aria-label="Save recipe to favorites">
                Save Recipe
            </button>
        </footer>
    </div>
</article>
```

### 🔸 JavaScript-Ready Attributes
```html
<!-- Interactive elements with data attributes -->
<div class="recipe-calculator" 
     data-recipe-id="123"
     data-original-servings="4">
    
    <div class="serving-adjuster">
        <label for="servings-input" class="serving-label">Servings:</label>
        <div class="serving-controls">
            <button class="btn-serving" 
                    data-action="decrease"
                    aria-label="Decrease servings">-</button>
            
            <input type="number" 
                   id="servings-input"
                   class="serving-input"
                   value="4" 
                   min="1" 
                   max="99"
                   data-original-value="4">
            
            <button class="btn-serving" 
                    data-action="increase"
                    aria-label="Increase servings">+</button>
        </div>
    </div>
    
    <!-- Ingredients that will be dynamically updated -->
    <ul class="ingredients-list" id="adjustable-ingredients">
        <li class="ingredient" data-amount="2" data-unit="cups">
            <span class="ingredient-amount">2</span>
            <span class="ingredient-unit">cups</span>
            <span class="ingredient-name">all-purpose flour</span>
        </li>
        <li class="ingredient" data-amount="1" data-unit="cup">
            <span class="ingredient-amount">1</span>
            <span class="ingredient-unit">cup</span>
            <span class="ingredient-name">butter, softened</span>
        </li>
        <li class="ingredient" data-amount="0.75" data-unit="cup">
            <span class="ingredient-amount">3/4</span>
            <span class="ingredient-unit">cup</span>
            <span class="ingredient-name">brown sugar, packed</span>
        </li>
        <li class="ingredient" data-amount="0.75" data-unit="cup">
            <span class="ingredient-amount">3/4</span>
            <span class="ingredient-unit">cup</span>
            <span class="ingredient-name">granulated sugar</span>
        </li>
        <li class="ingredient" data-amount="2" data-unit="">
            <span class="ingredient-amount">2</span>
            <span class="ingredient-unit"></span>
            <span class="ingredient-name">large eggs</span>
        </li>
        <li class="ingredient" data-amount="1" data-unit="tsp">
            <span class="ingredient-amount">1</span>
            <span class="ingredient-unit">tsp</span>
            <span class="ingredient-name">vanilla extract</span>
        </li>
        <li class="ingredient" data-amount="1" data-unit="tsp">
            <span class="ingredient-amount">1</span>
            <span class="ingredient-unit">tsp</span>
            <span class="ingredient-name">baking soda</span>
        </li>
        <li class="ingredient" data-amount="0.5" data-unit="tsp">
            <span class="ingredient-amount">1/2</span>
            <span class="ingredient-unit">tsp</span>
            <span class="ingredient-name">salt</span>
        </li>
        <li class="ingredient" data-amount="2" data-unit="cups">
            <span class="ingredient-amount">2</span>
            <span class="ingredient-unit">cups</span>
            <span class="ingredient-name">chocolate chips</span>
        </li>
    </ul>
</div>

<!-- Modal trigger and content -->
<button class="btn btn--secondary"
        data-toggle="modal"
        data-target="#nutrition-modal"
        aria-haspopup="dialog">
    View Nutrition Facts
</button>

<div id="nutrition-modal" 
     class="modal"
     role="dialog"
     aria-labelledby="nutrition-title"
     aria-hidden="true"
     data-backdrop="true"
     data-keyboard="true">
    <div class="modal-dialog">
        <div class="modal-content">
            <header class="modal-header">
                <h2 id="nutrition-title" class="modal-title">Nutrition Facts</h2>
                <button class="modal-close" 
                        data-dismiss="modal"
                        aria-label="Close nutrition facts">×</button>
            </header>
            <div class="modal-body">
                <ul>
                    <li>Calories: 150 per cookie</li>
                    <li>Fat: 8g</li>
                    <li>Carbohydrates: 20g</li>
                    <li>Protein: 2g</li>
                </ul>
            </div>
        </div>
    </div>
</div>

<!-- Form with validation hooks -->
<form class="contact-form" 
      id="contact-form"
      novalidate
      data-validate="true">
    
    <div class="form-group">
        <label for="email" class="form-label">Email Address</label>
        <input type="email" 
               id="email" 
               name="email"
               class="form-control"
               required
               data-validation="email"
               data-error-message="Please enter a valid email address">
    </div>
    
    <div class="form-group">
        <label for="name" class="form-label">Full Name</label>
        <input type="text" 
               id="name" 
               name="name"
               class="form-control"
               required
               data-error-message="Please enter your full name">
    </div>
    
    <div class="form-group">
        <label for="message" class="form-label">Message</label>
        <textarea id="message" 
                  name="message"
                  class="form-control"
                  rows="5"
                  required
                  data-error-message="Please enter your message"></textarea>
    </div>
    
    <button type="submit" class="btn btn--primary">Send Message</button>
</form>
```

## 📝 Assignment: Build a Polished, Accessible Recipe Website
1. Create a complete recipe website homepage using the HTML best practices covered.
2. Include at least:
   - A header with navigation and logo
   - A hero section with a background image
   - A recipe grid with 2-3 recipe cards
   - A search form
   - A contact form in footer
   - A modal for recipe details
   - A video section with captions
3. Ensure full accessibility with ARIA roles, alt text, and keyboard navigation.
4. Add structured data for SEO using schema.org.
5. Validate your HTML using the W3C Validator (validator.w3c.org).
6. Include meaningful comments for each major section.
7. Prepare the structure for CSS styling and JavaScript interactivity.

## ✅ Submission Checklist
- [ ] HTML is properly indented and organized
- [ ] All images have descriptive alt text
- [ ] ARIA roles are correctly implemented
- [ ] Forms are accessible with proper labels
- [ ] Navigation is keyboard-friendly
- [ ] Code passes W3C validation
- [ ] Comments clearly document sections
- [ ] Structured data enhances SEO
- [ ] Site is ready for CSS and JS integration

## 🌟 Bonus Challenge
- Add a recipe calculator that adjusts ingredient quantities based on servings.
- Implement a tabbed interface for recipe details (Ingredients, Instructions, Notes).
- Include Open Graph meta tags for social media sharing.

By mastering these HTML best practices, you're not just coding — you're crafting a delightful, inclusive, and professional web experience that stands out like a perfectly plated dish! 🍽️
