
### **Day 31: HTML Basics — Setting the Table**

Welcome to Day 31, chefs! Today, we’re stepping into the front of the house—your web restaurant’s dining area—where HTML is the art of plating. Forget Django for now; we’re focusing purely on crafting the perfect plate to present your content to browsers. HTML is like the plate, cutlery, and tablecloth that give structure to your dish, making it ready for customers (browsers) to enjoy. By the end of today, you’ll plate a simple webpage with headings, paragraphs, and links, ready to impress. Let’s set the table!

---

#### **Objectives**
* Understand HTML’s role as the structure of the web.
* Master the anatomy of an HTML document.
* Use core elements: headings (`<h1>`–`<h6>`), paragraphs (`<p>`), and links (`<a>`).
* Learn tags, attributes, and nesting.
* Explore semantic vs. non-semantic elements.

---

#### 🍴 **The Big Picture: HTML as Your Plating Art**

Imagine you’re a chef presenting a gourmet dish. Without a plate, your food is just a pile—tasty but messy. HTML is that plate, giving structure to your content (text, images, etc.) so browsers can serve it neatly to users. It’s not about making things pretty (that’s CSS, the garnish) or adding interactivity (that’s JavaScript, the cooking process). HTML is about organizing the meal so it’s clear and functional.

Here’s a taste of HTML plating:

```html
<h1>My Signature Pasta</h1>
<p>A creamy, dreamy dish to delight your taste buds.</p>
<a href="more-recipes.html">Explore More Dishes</a>
```

This tells the browser: “Plate a heading, a paragraph, and a link.” The browser serves it as a tidy webpage.

---

#### 🥄 **Step 1: The Anatomy of an HTML Document**

Every HTML document is like a perfectly set dining table, with a standard structure to hold your content.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>My Web Kitchen</title>
</head>
<body>
    <h1>Welcome to My Web Kitchen!</h1>
    <p>Freshly plated web pages, served daily.</p>
</body>
</html>
```

Let’s break it down like a chef checking their mise en place:
- **`<!DOCTYPE html>`**: The tablecloth, telling browsers this is an HTML5 document.
- **`<html lang="en">`**: The dining table, holding all content and specifying the language (English).
- **`<head>`**: The hidden prep area, storing metadata like the page title and character encoding.
- **`<meta charset="UTF-8">`**: Ensures browsers can read special characters (like emojis or accents).
- **`<title>`**: The menu card, shown in the browser’s tab.
- **`<body>`**: The plate itself, where your visible content (headings, paragraphs, etc.) goes.

Save this as `basic.html` on your computer (e.g., in a folder called `web_kitchen`). Double-click it to open in your browser, and you’ll see a simple webpage. No server needed—this is pure HTML plating!

---

#### 🍲 **Step 2: Core HTML Elements — Your Plating Tools**

Let’s plate with three key HTML elements: headings, paragraphs, and links.

##### **Headings (`<h1>`–`<h6>`): The Dish’s Title**
Headings are like the bold titles on your menu, grabbing attention and organizing your content.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Heading Demo</title>
</head>
<body>
    <h1>Main Course: Lasagna</h1>
    <h2>Ingredients</h2>
    <h3>Pasta Layers</h3>
    <h4>Sauce Prep</h4>
    <h5>Cheese Blend</h5>
    <h6>Baking Tips</h6>
</body>
</html>
```

- **`<h1>`**: The main dish title, used once per page for the top heading.
- **`<h2>`–`<h6>`**: Subheadings, getting smaller, like sections on a menu.

Save this as `headings.html` and open it in your browser to see the hierarchy.

##### **Paragraphs (`<p>`): The Dish Description**
Paragraphs hold the main content, like the description of a dish on your menu.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Paragraph Demo</title>
</head>
<body>
    <h1>Lasagna Love</h1>
    <p>Our classic lasagna is layered with rich tomato sauce, creamy ricotta, and melted mozzarella.</p>
    <p>Bake until golden and bubbly for a comforting meal.</p>
</body>
</html>
```

Save as `paragraphs.html` and check it out in your browser.

##### **Links (`<a>`): Menu Navigation**
Links connect pages, like guiding customers to other tables in your restaurant.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Links Demo</title>
</head>
<body>
    <h1>My Kitchen Menu</h1>
    <p>Explore our offerings:</p>
    <a href="https://example.com/recipes">All Recipes</a>
    <a href="desserts.html">Desserts</a>
    <a href="#contact">Contact Us</a>
</body>
</html>
```

- **`<a href="...">`**: The `href` attribute sets the destination.
- **Absolute URLs**: Full addresses like `https://example.com`.
- **Relative URLs**: Local files like `desserts.html` (create an empty `desserts.html` to test).
- **Anchor links**: `#contact` jumps to a section on the same page (we’ll explore later).

Save as `links.html` and test the links in your browser.

---

#### 🍽️ **Step 3: Tags, Attributes, and Nesting — Plating Techniques**

HTML elements are built with **tags** and **attributes**, like the tools and garnishes you use to plate a dish.

- **Tags**: Containers like `<p>Content</p>`. Opening (`<p>`) and closing (`</p>`) tags wrap content.
- **Attributes**: Extra details in the opening tag, like `href` in `<a href="...">`.
- **Nesting**: Tags inside tags, like stacking plates for a multi-course meal.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Nesting Demo</title>
</head>
<body>
    <h1>Signature Lasagna</h1>
    <p>This is a <strong>hearty</strong> dish with <em>fresh</em> ingredients.</p>
    <p>Check out our <a href="recipes.html">other recipes</a>!</p>
</body>
</html>
```

- **`<strong>`**: Bold text, like highlighting a key ingredient.
- **`<em>`**: Italic text, like a subtle garnish.
- **Nesting rules**: Close tags in reverse order (like stacking plates neatly).

Save as `nesting.html` and open it to see the nested structure.

---

#### 🎨 **Step 4: Semantic vs. Non-Semantic Elements**

HTML elements come in two flavors:
- **Semantic**: Tags with meaning, like `<header>` (menu header) or `<article>` (dish description).
- **Non-Semantic**: Generic tags like `<div>` (plain plate) or `<span>` (small garnish).

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Semantic Demo</title>
</head>
<body>
    <header>
        <h1>My Web Kitchen</h1>
    </header>
    <main>
        <article>
            <h2>Lasagna Delight</h2>
            <p>A comforting dish for all seasons.</p>
        </article>
    </main>
    <footer>
        <p>Contact us at chef@webkitchen.com</p>
    </footer>
</body>
</html>
```

- **Semantic benefits**: Improves accessibility, SEO, and clarity for other chefs (developers).
- **Non-semantic example**: `<div>` is a plain plate—useful but not descriptive.

Save as `semantic.html` and view it in your browser.

---

#### 🔧 **Step 5: Testing Your Plates**

Create a folder called `web_kitchen` on your computer. Save all the HTML files (`basic.html`, `headings.html`, `paragraphs.html`, `links.html`, `nesting.html`, `semantic.html`) in this folder. Open each in your browser by double-clicking to see how they render. No server needed—HTML works standalone!

To make testing easier, create an `index.html` to link all your pages:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Web Kitchen Menu</title>
</head>
<body>
    <h1>My Web Kitchen</h1>
    <p>Welcome to my HTML experiments!</p>
    <p><a href="basic.html">Basic Demo</a></p>
    <li><a href="headings.html">Headings Demo</a></li>
        <li><a href="paragraphs.html">Paragraphs Demo</a></li>
        <li><a href="links.html">Links Demo</a></li>
        <li><a href="nesting.html">Nesting Practice</a></li>
        <li><a href="semantic.html">Semantic Elements</a></li>
    </ul>
</body>
</html>

Save this as index.html in your web_kitchen folder. Now you have a central "menu" to navigate all your HTML practice files!
🧑‍🍳 Step 6: Your First Full Web Page

Let’s combine everything into a complete webpage for your "Web Kitchen":

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Chef's Web Kitchen</title>
</head>
<body>
    <header>
        <h1>👨‍🍳 Chef's Web Kitchen</h1>
        <nav>
            <a href="#specials">Today's Specials</a> |
            <a href="#recipes">Recipes</a> |
            <a href="#contact">Contact</a>
        </nav>
    </header>

    <main>
        <section id="specials">
            <h2>🍽️ Today's Special</h2>
            <p>Our <strong>HTML Lasagna</strong> is layered with semantic tags, topped with CSS cheese, and baked with JavaScript interactivity!</p>
            <img src="lasagna.jpg" alt="HTML Lasagna" width="300">
        </section>

        <section id="recipes">
            <h2>📜 Recipe Book</h2>
            <article>
                <h3>Basic HTML Structure</h3>
                <p>Every dish starts with a solid foundation:</p>
                <pre><code>&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;
    &lt;title&gt;My Recipe&lt;/title&gt;
&lt;/head&gt;
&lt;body&gt;
    &lt;h1&gt;Welcome&lt;/h1&gt;
&lt;/body&gt;
&lt;/html&gt;</code></pre>
            </article>
        </section>
    </main>

    <footer id="contact">
        <h2>📞 Contact the Chef</h2>
        <p>Email: <a href="mailto:chef@webkitchen.com">chef@webkitchen.com</a></p>
        <p>© 2025 Web Kitchen. All rights reserved.</p>
    </footer>
</body>
</html>

Key Features:

    Semantic structure (<header>, <nav>, <main>, <section>, <footer>)

    Navigation links with anchor IDs (#specials)

    Image placeholder (save a lasagna.jpg or use a real image)

    Code display with <pre> and <code> tags

    Copyright symbol using &copy;

    🎉 Day 31 Homework: Plate Your Masterpiece

🍳 Assignment:

    Create a my_restaurant.html file with:

        A restaurant name in <h1>

        3 menu sections (<h2> with <ul> lists)

        Contact info in <footer>

        At least one image and one external link

    Bonus Challenge:

        Add a "Reservations" form with <form>, <input>, and <button>

        Use <table> for a pricing grid

            📚 Resources:

    MDN HTML Basics

    HTML Cheat Sheet

💡 Reflection:

    How does HTML compare to setting a table?

    Why is semantic HTML important for accessibility?