### **Day 32: Lists and Tables — Organizing Your Pantry** 🗃️  

Welcome back, chefs! Yesterday, you learned how to **plate** your content with HTML’s basic structure. Today, we’re stepping into the **pantry**—where lists and tables help you organize ingredients, recipes, and menus efficiently. Just like a well-organized kitchen makes cooking smoother, proper HTML structuring makes your web content clear and easy to navigate.  

By the end of today, you’ll:  
✔ **Create bulleted and numbered lists** (like recipe steps).  
✔ **Build structured tables** (for pricing or nutrition info).  
✔ **Nest lists** for complex structures (multi-level menus).  

Let’s tidy up that pantry!  

---

## **🍽️ Today’s Menu**  
1. **Lists: Your Kitchen Shelves**  
   - Unordered Lists (`<ul>`)  
   - Ordered Lists (`<ol>`)  
   - Nested Lists  

2. **Tables: Your Recipe Charts**  
   - Basic Table Structure (`<table>`, `<tr>`, `<td>`)  
   - Table Headers (`<th>`)  
   - Advanced Attributes (`colspan`, `rowspan`)  

3. **Assignment: Build a Recipe Card**  

---

## **1. Lists: Your Kitchen Shelves**  

### **📜 Unordered Lists (`<ul>`) – Bulleted Ingredients**  
Use `<ul>` for **bulleted lists**, like an ingredient list:  

```html
<h3>Lasagna Ingredients</h3>
<ul>
    <li>Lasagna noodles</li>
    <li>Tomato sauce</li>
    <li>Ricotta cheese</li>
    <li>Ground beef</li>
</ul>
```
**Output:**  
- Lasagna noodles  
- Tomato sauce  
- Ricotta cheese  
- Ground beef  

### **🔢 Ordered Lists (`<ol>`) – Numbered Steps**  
Use `<ol>` for **numbered steps**, like a recipe:  

```html
<h3>Cooking Steps</h3>
<ol>
    <li>Boil the noodles</li>
    <li>Layer with sauce and cheese</li>
    <li>Bake at 375°F for 25 minutes</li>
</ol>
```
**Output:**  
1. Boil the noodles  
2. Layer with sauce and cheese  
3. Bake at 375°F for 25 minutes  

### **📂 Nested Lists – Multi-Level Menus**  
Lists can **nest** for complex structures, like a restaurant menu:  

```html
<ul>
    <li>Appetizers
        <ul>
            <li>Bruschetta</li>
            <li>Garlic Bread</li>
        </ul>
    </li>
    <li>Main Courses
        <ul>
            <li>Lasagna</li>
            <li>Pizza</li>
        </ul>
    </li>
</ul>
```
**Output:**  
- Appetizers  
  - Bruschetta  
  - Garlic Bread  
- Main Courses  
  - Lasagna  
  - Pizza  

---

## **2. Tables: Your Recipe Charts**  

### **📊 Basic Table Structure**  
Tables (`<table>`) organize data into **rows (`<tr>`)** and **cells (`<td>`)**. Use `<th>` for headers.  

```html
<table border="1">
    <tr>
        <th>Ingredient</th>
        <th>Quantity</th>
    </tr>
    <tr>
        <td>Flour</td>
        <td>2 cups</td>
    </tr>
    <tr>
        <td>Sugar</td>
        <td>1 cup</td>
    </tr>
</table>
```
**Output:**  

| Ingredient | Quantity |  
|------------|----------|  
| Flour      | 2 cups   |  
| Sugar      | 1 cup    |  

### **🔲 Advanced Tables: `colspan` & `rowspan`**  
Merge cells for complex layouts, like a pricing menu:  

```html
<table border="1">
    <tr>
        <th colspan="2">Pasta Menu</th>
    </tr>
    <tr>
        <td>Spaghetti</td>
        <td>$12</td>
    </tr>
    <tr>
        <td>Lasagna</td>
        <td>$15</td>
    </tr>
</table>
```
**Output:**  

| **Pasta Menu**  |  
|----------------|  
| Spaghetti | $12 |  
| Lasagna   | $15 |  

---

## **🍳 Assignment: Build a Recipe Card**  

**📝 Task:**  
Create an `recipe.html` file with:  
1. A **recipe name** (`<h1>`).  
2. An **unordered list** of ingredients.  
3. An **ordered list** of steps.  
4. A **table** for nutritional info (e.g., calories, protein).  

**Example:**  
```html
<!DOCTYPE html>
<html>
<head>
    <title>My Recipe</title>
</head>
<body>
    <h1>🍕 Homemade Pizza</h1>
    <h2>Ingredients</h2>
    <ul>
        <li>Pizza dough</li>
        <li>Tomato sauce</li>
        <li>Mozzarella cheese</li>
    </ul>
    <h2>Steps</h2>
    <ol>
        <li>Preheat oven to 450°F</li>
        <li>Spread sauce on dough</li>
        <li>Add cheese and toppings</li>
    </ol>
    <h2>Nutrition Facts</h2>
    <table border="1">
        <tr>
            <th>Calories</th>
            <th>Protein</th>
        </tr>
        <tr>
            <td>285</td>
            <td>12g</td>
        </tr>
    </table>
</body>
</html>
```

**🎯 Bonus Challenge:**  
- Add a **nested list** for toppings (e.g., "Vegetables: Mushrooms, Peppers").  
- Use `colspan` in your table for a "Serving Size" header.  

---

## **📚 Resources**  
- [MDN Lists Guide](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/ul)  
- [HTML Tables Tutorial](https://www.w3schools.com/html/html_tables.asp)  

---

## **🔍 Reflection**  
- How do lists improve readability compared to plain paragraphs?  
- When would you use a table instead of a list?  

**Tomorrow:** We’ll garnish our plates with **images and multimedia**! 🎥