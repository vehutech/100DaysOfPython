Day 4 - The Python Pantry — Built-in Functions, Modules, and Your Own Recipes

Helloooooo pythonistaaaaas🚀🚀🚀🚀

You got it, Chef! 👨🏽‍🍳👩🏽‍🍳  
Welcome to **Day 4** of your 100 Days of Python — this one's a _hearty stew_, rich with built-in flavors, imported spices, and custom recipes. We’re taking you from _default tools_ to _signature dishes_, with depth, clarity, and flavor.

---

So far, you’ve been cooking up Python dishes with tools that seemed like magic — `print()`, `input()`, and `type()` — but guess what? They’re **built-in functions**. Today, we crack open the whole pantry:

- 🔧 Python built-in functions (your ever-ready utensils)
- 🧂 Built-in modules (the spice rack you import)
- 🔤 String operations (chopping, dicing, mixing words)
- 🔢 Math operations (stirring numbers)
- 📦 Writing your own functions (your custom recipe book)
- 🧪 Understanding parameters and arguments (ingredients vs. labels)

---

## 🔧 Built-in Functions — Your Default Kitchen Utensils

Python comes with over **60 built-in functions**. These are the non-stick pans, ladles, and cutting boards of coding — always within reach.

### Examples:

```python
print("Dinner is served!")  # Print a message

name = input("What’s your chef name? ")  # Take user input
print("Welcome, Chef", name)

length = len("Spaghetti")  # Get length of a string
print("This word has", length, "characters.")

number = float("3.14")  # Convert string to float
print(type(number))  # Outputs: <class 'float'>
```

Want to see all of them?

```python
print(dir(__builtins__))  # Shows all built-in functions and constants
```

Some key built-in functions to memorize:

| Function  | What It Does                  |
| --------- | ----------------------------- |
| `print()` | Displays output               |
| `input()` | Takes user input              |
| `len()`   | Returns length                |
| `int()`   | Converts to integer           |
| `float()` | Converts to float             |
| `str()`   | Converts to string            |
| `type()`  | Tells the data type           |
| `range()` | Creates a sequence of numbers |
| `sum()`   | Adds up values in an iterable |
| `max()`   | Returns largest number        |
| `min()`   | Returns smallest number       |

---

## 🧂 Python Modules — The Spices You Import

Some tools aren’t out on the counter — they’re tucked in your spice rack. You need to **import** them.

### What’s a Module?

A module is like a mini-library or toolkit. Python has a shelf full of them.

### How to Import:

```python
import random
```

Now you can use tools inside the `random` module:

```python
print(random.randint(1, 5))  # Get a random number between 1 and 5
```

You can also import specific tools only:

```python
from math import sqrt

print(sqrt(25))  # Outputs: 5.0
```

Or give a nickname:

```python
import datetime as dt

print(dt.datetime.now())
```

---

## 🔤 String Operations — Mixing Words Like a Master Chef

Strings in Python are like recipe instructions — ready to be chopped, flipped, and seasoned.

### Common String Tools:

```python
dish = "Chicken Parmesan"

print(dish.upper())      # All uppercase
print(dish.lower())      # All lowercase
print(dish.title())      # Capitalize Each Word
print(dish.replace("Chicken", "Tofu"))  # Swap out ingredients

print("  messy plate   ".strip())  # Clean up whitespace
print("Cake" in dish)     # Check if "Cake" is in the dish -> False
```

### String Formatting — Putting Together a Fancy Menu:

```python
chef = "Tunde"
dish = "jollof rice"

# Old-school
print("Chef %s is making %s" % (chef, dish))

# New-school
print("Chef {} is making {}".format(chef, dish))

# Best-school (f-strings)
print(f"Chef {chef} is making {dish}")
```

---

## 🔢 Arithmetic Operators — Stirring, Mixing, Chopping Numbers

Imagine you’re prepping portions in the kitchen.

```python
plates = 0
plates += 1   # Add one
plates -= 1   # Subtract one
plates *= 2   # Double it
plates /= 2   # Halve it
```

### More Math:

```python
apples = 9
print(apples // 2)  # Floor division = 4
print(apples % 2)   # Modulo = 1
print(apples ** 2)  # Exponent = 81
```

### Using `math` Module:

```python
import math

print(math.sqrt(49))       # Square root
print(math.pi)             # π value
print(math.ceil(2.1))      # Round up
print(math.floor(2.9))     # Round down
```

---

## 📦 Writing Your Own Function — Your Signature Dish

Enough using everyone else's recipes. Time to write your own.

### Function Syntax:

```python
def cook_dish(ingredient):
    print(f"Cooking a dish with {ingredient}")
```

### Call the Function:

```python
cook_dish("tomatoes")
```

### Multiple Parameters:

```python
def bake(cake, temperature):
    print(f"Baking {cake} at {temperature} degrees")

bake("chocolate cake", 180)
```

---

## 🧪 Parameters vs Arguments — Label vs Ingredient

- **Parameter** = placeholder label (in recipe)
- **Argument** = actual ingredient (you pass in)

```python
def blend(fruit):  # "fruit" is a parameter
    print(f"Blending {fruit}...")

blend("mango")     # "mango" is an argument
```

You can even use default values:

```python
def boil(item="eggs"):
    print(f"Boiling {item}...")

boil()           # Boils eggs
boil("yam")      # Boils yam
```

---

## 🧑🏾‍🍳 Practical Kitchen Time

Let’s combine built-ins, imports, math, strings, and your own functions.

### Example:

```python
import random

def make_smoothie():
    fruits = ["banana", "mango", "strawberry", "pineapple"]
    chosen = random.choice(fruits)
    print(f"Blending your {chosen} smoothie... Done!")

make_smoothie()
```

---

## 📘 Assignment: Kitchen Practice (Write These)

1. Write a function called `make_soup()` that takes a vegetable as input and prints: `"Your <vegetable> soup is ready!"`.
2. Write a function `add_ingredients(a, b)` that takes two numbers and prints their sum.
3. Use `math` module to:

   - Calculate the square root of 144
   - Find the ceiling of 5.2
   - Get the value of π

4. Use string methods to:

   - Convert `"hello world"` to uppercase
   - Replace `"bad"` with `"good"` in `"bad chef"`
   - Check if `"soup"` exists in `"chicken soup"`

---

## 🧑🏽‍🏫 Coding Challenge: Build a Recipe Generator

### Goal:

Build a random recipe name generator using your new skills.

```python
import random

def recipe_generator():
    chefs = ["Tunde", "Chioma", "Ngozi"]
    dishes = ["stew", "pasta", "yam porridge", "cake"]
    styles = ["Nigerian style", "spicy twist", "chef's special"]

    chef = random.choice(chefs)
    dish = random.choice(dishes)
    style = random.choice(styles)

    print(f"{chef} is making {dish} with a {style}!")

recipe_generator()
```

---

## ✅ Recap: What You Learned Today

- ✅ **Built-in functions** are always available tools.
- ✅ **Modules** expand your kitchen with imports.
- ✅ **Math + string operations** = mixing your ingredients.
- ✅ **Functions** are how you write reusable recipes.
- ✅ **Parameters** are placeholders; **arguments** are ingredients.

---
