# **Day 7: Welcome to the OOP Kitchen — Organizing Your Culinary Code**

In the past few days, you’ve been cooking individual dishes (functions), prepping ingredients (variables), and sometimes repeating recipes (code). But now, it’s time to open your own **kitchen**. Not just any kitchen—one where you can hire **chefs**, reuse **recipes**, and run multiple **branches** without going crazy.

That’s where **Object-Oriented Programming (OOP)** comes in.

---

## THE BIG IDEA: What is OOP?

Imagine you're running a restaurant. You don't just throw ingredients on a stove randomly. You have:

- **Chefs** (objects) who know how to cook
- **Recipes** (methods) they follow
- **Kitchen stations** (classes) where specific tasks are performed
- **Hierarchy** of chefs (inheritance)
- **Rules and uniforms** (encapsulation)

OOP helps you **organize code like you organize a kitchen**: clean, reusable, and scalable.

---

## 1. **Class** — The Kitchen Blueprint

A `class` is like a blueprint for a station in your kitchen.

```python
class Chef:
    def __init__(self, name, specialty):
        self.name = name
        self.specialty = specialty

    def cook(self, dish):
        print(f"{self.name} is cooking {dish}, a specialty in {self.specialty}.")
```

The `__init__` function is the kitchen **constructor**. It's like training a chef when they join your restaurant.

---

## 2. **Object** — A Working Chef

An `object` is a real chef based on the `Chef` class:

```python
chef_anna = Chef("Anna", "Italian cuisine")
chef_anna.cook("Pasta Carbonara")
```

**Output:**
```
Anna is cooking Pasta Carbonara, a specialty in Italian cuisine.
```

Each object has its own **name** and **specialty** — just like how every chef has unique traits.

---

## 3. **Encapsulation** — Keeping Your Ingredients Sealed

Encapsulation means **hiding the messy stuff** and giving only what’s necessary.

```python
class Pantry:
    def __init__(self):
        self.__secret_ingredient = "Truffle Oil"  # Private variable

    def get_ingredient(self):
        return f"You got a dash of {self.__secret_ingredient}!"
```

```python
kitchen_pantry = Pantry()
print(kitchen_pantry.get_ingredient())
```

**Output:**
```
You got a dash of Truffle Oil!
```

The double underscore `__` keeps the ingredient safe from accidental misuse.

---

##  4. **Inheritance** — Chef Hierarchy

You can have **MasterChefs** who know everything, and **PastryChefs** who specialize.

```python
class PastryChef(Chef):
    def bake(self, dessert):
        print(f"{self.name} is baking {dessert}!")
```

```python
pastry = PastryChef("Léa", "Desserts")
pastry.cook("Croissant")
pastry.bake("Macaron")
```

**Output:**
```
Léa is cooking Croissant, a specialty in Desserts.
Léa is baking Macaron!
```

Inheritance lets `PastryChef` reuse and expand on the `Chef` class.

---

## 5. **Polymorphism** — Same Utensil, Many Recipes

You can use the same `cook` method in different ways:

```python
class SushiChef(Chef):
    def cook(self, dish):
        print(f"{self.name} prepares {dish} with sushi precision.")
```

```python
chefs = [Chef("Mario", "Pizza"), SushiChef("Hiro", "Sushi")]
for c in chefs:
    c.cook("Signature Dish")
```

**Output:**
```
Mario is cooking Signature Dish, a specialty in Pizza.
Hiro prepares Signature Dish with sushi precision.
```

**Polymorphism** means one interface, multiple methods — like one stove, many recipes.

---

## 6. **Why Use OOP in the Kitchen?**

| Problem                  | OOP Solution                            |
|--------------------------|------------------------------------------|
| Repeating code           | Create reusable classes                 |
| Managing many variables  | Bundle them into objects                |
| Changing structure later | Use inheritance and polymorphism        |
| Keeping things secure    | Use encapsulation (private variables)   |

---

## Real-World Kitchen: A Restaurant Management Example

```python
class Menu:
    def __init__(self):
        self.dishes = []

    def add_dish(self, dish):
        self.dishes.append(dish)

    def show_menu(self):
        for dish in self.dishes:
            print(f"- {dish}")

class Restaurant:
    def __init__(self, name):
        self.name = name
        self.menu = Menu()

    def open_doors(self):
        print(f"{self.name} is now open!")
        self.menu.show_menu()
```

```python
bistro = Restaurant("Le Gourmet")
bistro.menu.add_dish("Steak au Poivre")
bistro.menu.add_dish("Ratatouille")
bistro.open_doors()
```

**Output:**
```
Le Gourmet is now open!
- Steak au Poivre
- Ratatouille
```

---

## TL;DR OOP Concepts (Chef Style)

| Concept        | Kitchen Metaphor                   | Code Element        |
|----------------|------------------------------------|---------------------|
| Class          | Kitchen blueprint                  | `class Chef`        |
| Object         | Real chef working in kitchen       | `chef_anna`         |
| Inheritance    | Sous chefs learning from head chef | `class PastryChef(Chef)` |
| Encapsulation  | Secret sauce locked in fridge      | `__secret_ingredient` |
| Polymorphism   | Chefs customizing recipes          | Override `cook()`   |

---

## Challenge for Day 7

Create a **Restaurant Management System** with these features:

- `Chef` class and subclasses like `GrillChef`, `PastryChef`
- `Menu` class to add/show dishes
- `Restaurant` class that hires chefs and opens for service
- Bonus: Use `@staticmethod` or `@classmethod` to define utility functions (like kitchen opening times)

---