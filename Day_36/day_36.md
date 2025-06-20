---
# **Day 36: Linking and Navigation — Guiding Customers** 🧭🍽️
---
Picture this: you walk into an upscale restaurant and see clear, elegant signage everywhere — "Dining Room," "Private Events," "Wine Cellar," "Restrooms." A friendly host greets you with a well-designed menu that shows exactly how to navigate from appetizers to desserts. Compare this to a confusing maze where guests wander aimlessly, bumping into walls and getting lost. That's the difference between thoughtful navigation and poor linking — good navigation guides your visitors exactly where they want to go, creating a smooth, delightful journey through your website.
---
**## 🎯 Objectives**
* Master advanced `<a>` tag usage for different link types
* Build intuitive navigation systems with `<nav>` and semantic structure
* Implement in-page navigation with anchor links
* Apply user-friendly navigation best practices
* Create accessible, mobile-friendly navigation menus
* Connect multiple pages with consistent navigation systems
* Optimize link behavior for better user experience
---
**## 🔗 The Anatomy of Links: Your Restaurant's Pathways**
Links are the pathways that connect every corner of your digital restaurant. Just like physical signage guides customers to their tables, links guide users through your content with purpose and clarity.

**### 🔹 The `<a>` Tag: Your Digital Host**
The anchor (`<a>`) tag is like your restaurant's host — it welcomes visitors and guides them to exactly where they need to go:

```html
<!-- Basic link structure -->
<a href="destination">Link text that users see</a>

<!-- The host saying: "Right this way to your table!" -->
<a href="/reservations.html">Make a Reservation</a>
```

**### 🔸 Link Attributes: Adding Instructions**
Just like giving your host specific instructions, link attributes control exactly how navigation works:

```html
<!-- Different ways to guide your customers -->

<!-- Internal navigation - staying within your restaurant -->
<a href="/menu.html">View Our Menu</a>
<a href="desserts.html">Dessert Selection</a>
<a href="#appetizers">Jump to Appetizers</a>

<!-- External links - recommending other establishments -->
<a href="https://www.foodnetwork.com" target="_blank" rel="noopener">
    Featured on Food Network
</a>

<!-- Communication pathways -->
<a href="mailto:reservations@restaurant.com">Email Us</a>
<a href="tel:+1-555-DINE">Call for Reservations</a>

<!-- Downloads - like take-home menus -->
<a href="/menu.pdf" download="restaurant-menu.pdf">
    Download Full Menu (PDF)
</a>
```
---
**## 🧭 URL Types: Absolute vs. Relative Paths**
Understanding URL types is like knowing the difference between giving someone your full address versus simple directions within your building:

**### 🔹 Absolute URLs: The Full Address**
```html
<!-- Complete address - like giving full street address -->
<a href="https://www.grandmasrestaurant.com/menu.html">Our Menu</a>
<a href="https://www.instagram.com/grandmasrestaurant">Follow Us</a>
<a href="mailto:chef@grandmasrestaurant.com">Contact Chef</a>

<!-- When to use absolute URLs: -->
<!-- ✅ External websites -->
<!-- ✅ Email addresses -->
<!-- ✅ Phone numbers -->
<!-- ✅ When you need the complete URL -->
```

**### 🔸 Relative URLs: Directions Within Your Space**
```html
<!-- Relative to current location - like "down the hall, second door" -->

<!-- Same directory (same folder) -->
<a href="desserts.html">Desserts</a>
<a href="appetizers.html">Appetizers</a>

<!-- Subdirectory (going into a folder) -->
<a href="recipes/pasta-recipes.html">Pasta Recipes</a>
<a href="images/chef-photo.jpg">Meet Our Chef</a>

<!-- Parent directory (going up a folder) -->
<a href="../index.html">Back to Home</a>
<a href="../../contact.html">Contact Us</a>

<!-- Root relative (from the top of your site) -->
<a href="/menu.html">Menu</a>
<a href="/about/history.html">Our History</a>
```

**### 🔹 URL Best Practices**
```html
<!-- ✅ Good: Clear, descriptive paths -->
<a href="/menu/appetizers">Appetizers</a>
<a href="/about/our-story">Our Story</a>
<a href="/contact">Contact Us</a>

<!-- ❌ Avoid: Confusing, non-descriptive paths -->
<a href="/page1.html">Page 1</a>
<a href="/content/item/12345">Item</a>
<a href="/p/a/g/e">Link</a>
```
---
**## 🏗️ Building Navigation Menus: Your Restaurant's Map**
Navigation menus are like the layout of your restaurant — they should be intuitive, accessible, and guide customers naturally through their journey.

**### 🔹 Basic Navigation Structure**
```html
<!-- The main reception area with clear directions -->
<header>
    <h1>Grandma's Restaurant</h1>
    
    <nav aria-label="Main navigation">
        <ul>
            <li><a href="/" aria-current="page">Home</a></li>
            <li><a href="/menu">Menu</a></li>
            <li><a href="/about">About Us</a></li>
            <li><a href="/reservations">Reservations</a></li>
            <li><a href="/contact">Contact</a></li>
        </ul>
    </nav>
</header>
```

**### 🔸 Multi-Level Navigation: Restaurant Sections**
```html
<!-- Like having different dining areas with their own specialties -->
<nav aria-label="Main navigation">
    <ul>
        <li>
            <a href="/menu">Menu</a>
            <ul>
                <li><a href="/menu/appetizers">Appetizers</a></li>
                <li><a href="/menu/entrees">Entrees</a></li>
                <li><a href="/menu/desserts">Desserts</a></li>
                <li><a href="/menu/beverages">Beverages</a></li>
            </ul>
        </li>
        <li>
            <a href="/events">Events</a>
            <ul>
                <li><a href="/events/private-dining">Private Dining</a></li>
                <li><a href="/events/catering">Catering</a></li>
                <li><a href="/events/wine-tastings">Wine Tastings</a></li>
            </ul>
        </li>
        <li>
            <a href="/about">About</a>
            <ul>
                <li><a href="/about/our-story">Our Story</a></li>
                <li><a href="/about/chef">Meet the Chef</a></li>
                <li><a href="/about/awards">Awards</a></li>
            </ul>
        </li>
    </ul>
</nav>
```

**### 🔹 Breadcrumb Navigation: Showing the Path**
```html
<!-- Like showing customers: "You are here: Lobby → Dining Room → Table 12" -->
<nav aria-label="Breadcrumb">
    <ol>
        <li><a href="/">Home</a></li>
        <li><a href="/menu">Menu</a></li>
        <li><a href="/menu/entrees">Entrees</a></li>
        <li aria-current="page">Grilled Salmon</li>
    </ol>
</nav>
```
---
**## ⚓ Anchor Links: In-Page Navigation**
Anchor links are like having a detailed map of your dining room, allowing customers to jump directly to specific sections without wandering around.

**### 🔹 Creating Anchor Points**
```html
<!-- Setting up landmarks within your page -->
<main>
    <!-- Navigation menu for the page content -->
    <nav aria-label="Page contents">
        <h2>On This Page</h2>
        <ul>
            <li><a href="#appetizers">Appetizers</a></li>
            <li><a href="#entrees">Entrees</a></li>
            <li><a href="#desserts">Desserts</a></li>
            <li><a href="#beverages">Beverages</a></li>
        </ul>
    </nav>

    <!-- The landmarks themselves -->
    <section id="appetizers">
        <h2>Appetizers</h2>
        <p>Start your meal with our carefully crafted appetizers...</p>
        <!-- Appetizer content -->
    </section>

    <section id="entrees">
        <h2>Entrees</h2>
        <p>Our main courses feature seasonal ingredients...</p>
        <!-- Entree content -->
    </section>

    <section id="desserts">
        <h2>Desserts</h2>
        <p>Finish with our house-made desserts...</p>
        <!-- Dessert content -->
    </section>

    <section id="beverages">
        <h2>Beverages</h2>
        <p>Complement your meal with our beverage selection...</p>
        <!-- Beverage content -->
    </section>
</main>
```

**### 🔸 Advanced Anchor Usage**
```html
<!-- Linking to specific content within sections -->
<article>
    <h2>Today's Special: Seafood Platter</h2>
    
    <!-- Quick navigation within the article -->
    <nav aria-label="Article sections">
        <p><strong>Jump to:</strong></p>
        <ul>
            <li><a href="#ingredients">Ingredients</a></li>
            <li><a href="#preparation">Preparation</a></li>
            <li><a href="#wine-pairing">Wine Pairing</a></li>
            <li><a href="#nutritional-info">Nutritional Information</a></li>
        </ul>
    </nav>

    <section id="ingredients">
        <h3>Fresh Ingredients</h3>
        <p>We source our seafood daily from local fishermen...</p>
    </section>

    <section id="preparation">
        <h3>Preparation Method</h3>
        <p>Our chef prepares each platter using traditional techniques...</p>
    </section>

    <section id="wine-pairing">
        <h3>Recommended Wine Pairing</h3>
        <p>This dish pairs beautifully with our house Chardonnay...</p>
    </section>

    <section id="nutritional-info">
        <h3>Nutritional Information</h3>
        <p>Calories: 450 | Protein: 35g | Carbs: 15g...</p>
    </section>

    <!-- Back to top link - like an elevator to the lobby -->
    <p><a href="#top">↑ Back to top</a></p>
</article>
```
---
**## 🎯 Link Behavior and User Experience**
Controlling how links behave is like training your restaurant staff to provide consistent, helpful service to every customer.

**### 🔹 Target Attribute: Controlling Windows**
```html
<!-- Opening external links in new tabs - like giving directions to another restaurant -->
<a href="https://www.yelp.com/biz/grandmas-restaurant" 
   target="_blank" 
   rel="noopener noreferrer">
    Read Our Yelp Reviews
</a>

<!-- Keeping users on your site for internal links -->
<a href="/menu.html">View Our Menu</a>

<!-- Opening email client -->
<a href="mailto:info@restaurant.com?subject=Reservation Request">
    Email for Reservations
</a>

<!-- Opening phone dialer on mobile -->
<a href="tel:+15551234567">Call Us: (555) 123-4567</a>
```

**### 🔸 Download Links: Take-Home Materials**
```html
<!-- Providing downloadable resources -->
<a href="/files/dinner-menu.pdf" download="grandmas-dinner-menu">
    Download Dinner Menu (PDF)
</a>

<a href="/files/wine-list.pdf" download="wine-list-2024">
    Download Wine List
</a>

<a href="/files/catering-package.pdf" download="catering-options">
    Catering Package Information
</a>
```

**### 🔹 Accessibility in Links**
```html
<!-- Clear, descriptive link text -->
<!-- ✅ Good: Tells users exactly what to expect -->
<a href="/menu/vegetarian-options">View Vegetarian Menu Options</a>
<a href="/reservations" aria-describedby="reservation-note">
    Make a Reservation
</a>
<span id="reservation-note">Opens reservation system</span>

<!-- ❌ Avoid: Vague or confusing link text -->
<a href="/menu">Click here</a>
<a href="/info">Read more</a>
<a href="/page">Link</a>

<!-- Links for screen readers -->
<a href="https://instagram.com/grandmasrestaurant" 
   aria-label="Follow Grandma's Restaurant on Instagram (opens in new tab)"
   target="_blank" 
   rel="noopener">
    <img src="instagram-icon.svg" alt="">
    Instagram
</a>
```
---
**## 📱 Mobile-Friendly Navigation**
Creating navigation that works on mobile is like designing a restaurant that's comfortable whether you're dining alone or with a large group.

**### 🔹 Responsive Navigation Menu**
```html
<!-- Mobile-first navigation structure -->
<header>
    <div class="navbar">
        <h1>Grandma's Restaurant</h1>
        
        <!-- Mobile menu toggle button -->
        <button class="menu-toggle" 
                aria-label="Toggle navigation menu"
                aria-expanded="false"
                aria-controls="main-nav">
            <span class="hamburger-line"></span>
            <span class="hamburger-line"></span>
            <span class="hamburger-line"></span>
        </button>
    </div>

    <!-- Main navigation -->
    <nav id="main-nav" class="main-navigation" aria-label="Main navigation">
        <ul>
            <li><a href="/" aria-current="page">Home</a></li>
            <li><a href="/menu">Menu</a></li>
            <li><a href="/about">About</a></li>
            <li><a href="/reservations">Reservations</a></li>
            <li><a href="/contact">Contact</a></li>
        </ul>
    </nav>
</header>

<!-- Basic CSS for mobile menu (inline for demonstration) -->
<style>
/* Mobile-first approach */
.main-navigation {
    display: none; /* Hidden by default on mobile */
}

.main-navigation.active {
    display: block; /* Shown when menu is toggled */
}

.menu-toggle {
    display: block;
    background: none;
    border: none;
    cursor: pointer;
}

/* Desktop styles */
@media (min-width: 768px) {
    .menu-toggle {
        display: none; /* Hide hamburger on desktop */
    }
    
    .main-navigation {
        display: block; /* Always show navigation on desktop */
    }
    
    .main-navigation ul {
        display: flex;
        list-style: none;
        gap: 2rem;
    }
}
</style>

<!-- JavaScript for mobile menu toggle -->
<script>
document.querySelector('.menu-toggle').addEventListener('click', function() {
    const nav = document.querySelector('.main-navigation');
    const button = this;
    
    nav.classList.toggle('active');
    
    // Update ARIA state
    const isExpanded = nav.classList.contains('active');
    button.setAttribute('aria-expanded', isExpanded);
});
</script>
```
---
**## 🎨 Complete Navigation System Example**
Here's a complete multi-page restaurant site with comprehensive navigation:

**### 🔹 Main Layout with Navigation**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grandma's Restaurant - Authentic Italian Cuisine</title>
    <style>
        /* Basic styling for navigation */
        body {
            font-family: 'Georgia', serif;
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }
        
        .header {
            background-color: #8B4513;
            color: white;
            padding: 1rem 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .logo h1 {
            margin: 0;
            font-size: 2rem;
        }
        
        .main-nav ul {
            list-style: none;
            display: flex;
            gap: 2rem;
            margin: 0;
            padding: 0;
        }
        
        .main-nav a {
            color: white;
            text-decoration: none;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        
        .main-nav a:hover,
        .main-nav a:focus {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        .main-nav a[aria-current="page"] {
            background-color: #654321;
        }
        
        .breadcrumb {
            background-color: #f4f4f4;
            padding: 0.5rem 0;
        }
        
        .breadcrumb ol {
            list-style: none;
            display: flex;
            gap: 0.5rem;
            margin: 0;
            padding: 0;
        }
        
        .breadcrumb li:not(:last-child)::after {
            content: " → ";
            margin-left: 0.5rem;
            color: #666;
        }
        
        .breadcrumb a {
            color: #8B4513;
            text-decoration: none;
        }
        
        .breadcrumb a:hover {
            text-decoration: underline;
        }
        
        .main-content {
            padding: 2rem 0;
        }
        
        .page-nav {
            background-color: #f9f9f9;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }
        
        .page-nav h2 {
            margin-top: 0;
            color: #8B4513;
        }
        
        .page-nav ul {
            list-style: none;
            padding: 0;
        }
        
        .page-nav li {
            margin-bottom: 0.5rem;
        }
        
        .page-nav a {
            color: #8B4513;
            text-decoration: none;
        }
        
        .page-nav a:hover {
            text-decoration: underline;
        }
        
        .section {
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid #eee;
        }
        
        .section:last-child {
            border-bottom: none;
        }
        
        .back-to-top {
            text-align: center;
            margin: 2rem 0;
        }
        
        .back-to-top a {
            color: #8B4513;
            text-decoration: none;
            font-weight: bold;
        }
        
        .footer {
            background-color: #333;
            color: white;
            padding: 2rem 0;
            margin-top: 3rem;
        }
        
        .footer-nav ul {
            list-style: none;
            display: flex;
            gap: 2rem;
            margin: 0;
            padding: 0;
            flex-wrap: wrap;
        }
        
        .footer-nav a {
            color: white;
            text-decoration: none;
        }
        
        .footer-nav a:hover {
            text-decoration: underline;
        }
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            .main-nav ul {
                flex-direction: column;
                gap: 1rem;
            }
            
            .breadcrumb ol {
                flex-direction: column;
                gap: 0.25rem;
            }
            
            .breadcrumb li:not(:last-child)::after {
                content: none;
            }
            
            .footer-nav ul {
                flex-direction: column;
                gap: 1rem;
            }
            
            .menu-toggle {
                display: block;
                background: none;
                border: none;
                cursor: pointer;
            }
            
            .main-nav {
                display: none;
            }
            
            .main-nav.active {
                display: block;
            }
        }
        
        @media (min-width: 769px) {
            .menu-toggle {
                display: none;
            }
            
            .main-nav {
                display: block;
            }
        }
        
        .menu-toggle {
            background: none;
            border: none;
            cursor: pointer;
        }
        
        .hamburger-line {
            display: block;
            width: 25px;
            height: 3px;
            background-color: white;
            margin: 5px 0;
        }
    </style>
</head>
<body id="top">
    <!-- Main Header with Navigation -->
    <header class="header">
        <div class="container">
            <div class="navbar">
                <div class="logo">
                    <h1>Grandma's Restaurant</h1>
                </div>
                
                <button class="menu-toggle" 
                        aria-label="Toggle navigation menu"
                        aria-expanded="false"
                        aria-controls="main-nav">
                    <span class="hamburger-line"></span>
                    <span class="hamburger-line"></span>
                    <span class="hamburger-line"></span>
                </button>
                
                <nav class="main-nav" id="main-nav" aria-label="Main navigation">
                    <ul>
                        <li><a href="index.html" aria-current="page">Home</a></li>
                        <li><a href="menu.html">Menu</a></li>
                        <li><a href="about.html">About Us</a></li>
                        <li><a href="reservations.html">Reservations</a></li>
                        <li><a href="events.html">Events</a></li>
                        <li><a href="contact.html">Contact</a></li>
                    </ul>
                </nav>
            </div>
        </div>
    </header>

    <!-- Breadcrumb Navigation -->
    <nav class="breadcrumb" aria-label="Breadcrumb">
        <div class="container">
            <ol>
                <li><a href="index.html">Home</a></li>
                <li><a href="menu.html">Menu</a></li>
                <li aria-current="page">Dinner Menu</li>
            </ol>
        </div>
    </nav>

    <!-- Main Content -->
    <main class="main-content">
        <div class="container">
            <h1>Dinner Menu</h1>
            <p>Discover our authentic Italian dinner selections, prepared with traditional recipes passed down through generations.</p>

            <!-- Page Contents Navigation -->
            <nav class="page-nav" aria-label="Page contents">
                <h2>Menu Sections</h2>
                <ul>
                    <li><a href="#antipasti">Antipasti</a></li>
                    <li><a href="#primi-piatti">Primi Piatti (First Courses)</a></li>
                    <li><a href="#secondi-piatti">Secondi Piatti (Main Courses)</a></li>
                    <li><a href="#contorni">Contorni (Side Dishes)</a></li>
                    <li><a href="#dolci">Dolci (Desserts)</a></li>
                    <li><a href="#beverages">Beverages</a></li>
                </ul>
            </nav>

            <!-- Menu Sections -->
            <section id="antipasti" class="section">
                <h2>Antipasti</h2>
                <p>Begin your culinary journey with our traditional Italian appetizers.</p>
                
                <article class="menu-item">
                    <h3>Bruschetta Trio</h3>
                    <p>Three varieties of our house-made bruschetta: classic tomato basil, roasted pepper and goat cheese, and olive tapenade. <strong>$14</strong></p>
                </article>
                
                <article class="menu-item">
                    <h3>Antipasto della Casa</h3>
                    <p>Chef's selection of cured meats, artisanal cheeses, marinated vegetables, and olives. <strong>$18</strong></p>
                </article>
                
                <article class="menu-item">
                    <h3>Calamari Fritti</h3>
                    <p>Fresh squid rings lightly battered and fried, served with spicy marinara sauce. <strong>$16</strong></p>
                </article>
            </section>

            <section id="primi-piatti" class="section">
                <h2>Primi Piatti (First Courses)</h2>
                <p>Our handmade pasta dishes and risottos, perfect as a first course or light meal.</p>
                
                <article class="menu-item">
                    <h3>Spaghetti Carbonara</h3>
                    <p>Traditional Roman pasta with eggs, pancetta, pecorino Romano, and black pepper. <strong>$22</strong></p>
                </article>
                
                <article class="menu-item">
                    <h3>Risotto ai Funghi Porcini</h3>
                    <p>Creamy Arborio rice with wild porcini mushrooms and Parmesan cheese. <strong>$24</strong></p>
                </article>
                
                <article class="menu-item">
                    <h3>Penne all'Arrabbiata</h3>
                    <p>Penne pasta in a spicy tomato sauce with garlic and red chilies. <strong>$20</strong></p>
                </article>
            </section>

            <section id="secondi-piatti" class="section">
                <h2>Secondi Piatti (Main Courses)</h2>
                <p>Our signature main dishes featuring the finest meats, seafood, and seasonal ingredients.</p>
                
                <article class="menu-item">
                    <h3>Osso Buco alla Milanese</h3>
                    <p>Braised veal shanks with vegetables, white wine, and broth, served with saffron risotto. <strong>$38</strong></p>
                </article>
                
                <article class="menu-item">
                    <h3>Branzino in Crosta di Sale</h3>
                    <p>Mediterranean sea bass baked in a salt crust, served with roasted vegetables and lemon oil. <strong>$32</strong></p>
                </article>
                
                <article class="menu-item">
                    <h3>Pollo alla Parmigiana</h3>
                    <p>Breaded chicken breast topped with tomato sauce and mozzarella, served with pasta. <strong>$28</strong></p>
                </article>
            </section>

            <section id="contorni" class="section">
                <h2>Contorni (Side Dishes)</h2>
                <p>Perfect accompaniments to complement your main course.</p>
                
                <article class="menu-item">
                    <h3>Verdure Grigliate</h3>
                    <p>Seasonal grilled vegetables with herb oil. <strong>$12</strong></p>
                </article>
                
                <article class="menu-item">
                    <h3>Patate al Rosmarino</h3>
                    <p>Roasted potatoes with rosemary and garlic. <strong>$10</strong></p>
                </article>
                
                <article class="menu-item">
                    <h3>Spinaci all'Aglio</h3>
                    <p>Fresh spinach sautéed with garlic and olive oil. <strong>$9</strong></p>
                </article>
            </section>

            <section id="dolci" class="section">
                <h2>Dolci (Desserts)</h2>
                <p>End your meal with our traditional Italian desserts, made fresh daily.</p>
                
                <article class="menu-item">
                    <h3>Tiramisu della Casa</h3>
                    <p>Classic layered dessert with espresso-soaked ladyfingers and mascarpone cream. <strong>$12</strong></p>
                </article>
                
                <article class="menu-item">
                    <h3>Panna Cotta ai Frutti di Bosco</h3>
                    <p>Silky vanilla panna cotta topped with mixed berry compote. <strong>$10</strong></p>
                </article>
                
                <article class="menu-item">
                    <h3>Gelato e Sorbetto</h3>
                    <p>Selection of house-made gelato and sorbet. Ask your server for today's flavors. <strong>$8</strong></p>
                </article>
            </section>

            <section id="beverages" class="section">
                <h2>Beverages</h2>
                <p>Complement your meal with our carefully curated wine list and specialty drinks.</p>
                
                <article class="menu-item">
                    <h3>Wine Selection</h3>
                    <p>Extensive collection of Italian and international wines. <a href="wine-list.html">View full wine list</a></p>
                </article>
                
                <article class="menu-item">
                    <h3>Espresso & Coffee</h3>
                    <p>Traditional Italian espresso, cappuccino, and specialty coffee drinks. <strong>$4-8</strong></p>
                </article>
                
                <article class="menu-item">
                    <h3>Digestivi</h3>
                    <p>Grappa, Limoncello, Sambuca, and other traditional Italian digestifs. <strong>$8-12</strong></p>
                </article>
            </section>

            <!-- Back to top navigation -->
            <div class="back-to-top">
                <a href="#top">↑ Back to Top</a>
            </div>

            <!-- Related pages navigation -->
            <nav aria-label="Related pages">
                <h2>Explore More</h2>
                <ul>
                    <li><a href="lunch-menu.html">Lunch Menu</a></li>
                    <li><a href="wine-list.html">Wine List</a></li>
                    <li><a href="reservations.html">Make a Reservation</a></li>
                    <li><a href="events.html">Private Events & Catering</a></li>
                </ul>
            </nav>
        </div>
    </main>

    <!-- Footer with additional navigation -->
    <footer class="footer">
        <div class="container">
            <nav class="footer-nav" aria-label="Footer navigation">
                <ul>
                    <li><a href="index.html">Home</a></li>
                    <li><a href="menu.html">Menu</a></li>
                    <li><a href="about.html">About</a></li>
                    <li><a href="reservations.html">Reservations</a></li>
                    <li><a href="contact.html">Contact</a></li>
                    <li><a href="privacy.html">Privacy Policy</a></li>
                </ul>
            </nav>
            
            <div class="footer-content">
                <p>© 2024 Grandma's Restaurant. All rights reserved.</p>
                <address>
                    <p>123 Little Italy Street, Food City, FC 12345</p>
                    <p>Phone: <a href="tel:+15551234567">(555) 123-4567</a></p>
                    <p>Email: <a href="mailto:info@grandmasrestaurant.com">info@grandmasrestaurant.com</a></p>
                    <p>Follow us:
                        <a href="https://www.instagram.com/grandmasrestaurant" 
                           target="_blank" 
                           rel="noopener noreferrer"
                           aria-label="Follow Grandma's Restaurant on Instagram (opens in new tab)">
                            Instagram
                        </a> | 
                        <a href="https://www.facebook.com/grandmasrestaurant" 
                           target="_blank" 
                           rel="noopener noreferrer"
                           aria-label="Follow Grandma's Restaurant on Facebook (opens in new tab)">
                            Facebook
                        </a> | 
                        <a href="https://www.twitter.com/grandmasrestaurant" 
                           target="_blank" 
                           rel="noopener noreferrer"
                           aria-label="Follow Grandma's Restaurant on Twitter (opens in new tab)">
                            Twitter
                        </a>
                    </p>
                </address>
            </div>
        </div>
    </footer>

    <!-- JavaScript for mobile menu toggle -->
    <script>
        document.querySelector('.menu-toggle').addEventListener('click', function() {
            const nav = document.querySelector('.main-nav');
            const button = this;
            
            nav.classList.toggle('active');
            
            // Update ARIA state
            const isExpanded = nav.classList.contains('active');
            button.setAttribute('aria-expanded', isExpanded);
        });
    </script>
</body>
</html>
```