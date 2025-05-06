# Day 24: Cooking with Classes — Deep dive into OOP 👨‍🍳🚗

---

In a kitchen, you don’t create every dish from scratch every time — you use **recipes** (blueprints) to make consistent dishes. In programming, that **recipe is a `class`**, and each **dish is an `object`**.

Today, we start learning **Object-Oriented Programming (OOP)** — a powerful way to organize code using **classes** and **objects**.

---

* **Class** = a recipe / blueprint (e.g. "Spaghetti Bolognese")
* **Object** = a specific dish made from the recipe (e.g. "Mama’s spaghetti on May 6")
* **Attributes** = ingredients (e.g. color, year)
* **Methods** = cooking steps (e.g. start\_engine())

---

## 🎯 Objectives

* Understand what classes and objects are.
* Create your own class with attributes.
* Use the `__init__()` constructor to initialize objects.
* Access object attributes using dot notation.

---

## 🧪 Ingredient 1: The Class Blueprint

```python
class Car:
    def __init__(self, model, year, color, for_sale):
        self.model = model
        self.year = year
        self.color = color
        self.for_sale = for_sale
```

This `Car` class is a blueprint for making many types of cars.
The `__init__()` method is called automatically when you create a new object — it’s like the prep station that sets up each dish.

---

## 🧪 Ingredient 2: Creating an Object (a Car Dish)

```python
car1 = Car("Mustang", 2024, "red", False)
```

Here we make one **object** (`car1`) using the `Car` class.

---

## 🧪 Ingredient 3: Accessing Object Data

```python
print(car1.model)  # Output: Mustang
```

Use **dot notation** to access an object’s attributes.

---

## 🍱 Kitchen Table Example

```python
car2 = Car("Corolla", 2020, "white", True)

print(f"{car2.model} ({car2.year}) - Color: {car2.color}")
if car2.for_sale:
    print("🚗 This car is available for sale!")
else:
    print("❌ This car is not for sale.")
```

---

## 🧁 Mini Project: Chef’s Recipe Book

Create a class called `Recipe` with the following:

* `name` (string)
* `ingredients` (list)
* `cook_time` (int, in minutes)
* `vegetarian` (bool)

### Example:

```python
class Recipe:
    def __init__(self, name, ingredients, cook_time, vegetarian):
        self.name = name
        self.ingredients = ingredients
        self.cook_time = cook_time
        self.vegetarian = vegetarian

    def display(self):
        print(f"🍽️ {self.name}")
        print(f"Ingredients: {', '.join(self.ingredients)}")
        print(f"Cook time: {self.cook_time} mins")
        print("Vegetarian: Yes" if self.vegetarian else "Vegetarian: No")

recipe1 = Recipe("Pasta", ["noodles", "tomato sauce", "cheese"], 20, True)
recipe1.display()
```

---

## 📝 Assignment

1. Create a `Book` class with attributes: `title`, `author`, `year`, `available`.
2. Add a method `summary()` that prints the book details.
3. Create 2 objects and call `summary()` on each.
4. Bonus: Add a `borrow()` method that marks the book as unavailable if borrowed.

---