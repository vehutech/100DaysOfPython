**Day 5: The Kitchen Fires – Error Handling, Lists & Tuples*

Welcome back to your Python kitchen!

In cooking, even the best chefs burn the stew, spill the flour, or drop a plate. But the difference between a rookie and a master chef is knowing **how to recover from those mistakes gracefully**.

Today, we prepare you to become that kind of chef in Python. Let’s explore what can go wrong—and how to fix it. Along the way, you’ll learn about two new essential tools: **lists** and **tuples**.

---

## **Today’s Menu**

- Understanding **errors** vs. **exceptions**
- Using `try`, `except`, `else`, and `finally` like a kitchen safety kit
- Working with **lists**: editable ingredient trays
- Working with **tuples**: sealed recipe containers
- Chef's Challenge

---

## PART 1: Kitchen Disasters – Errors and Exceptions

### Syntax Errors: Mistakes in the Recipe Book

These are like typos in your cookbook—your oven won’t even turn on if the instructions aren’t clear.

```python
print("Let’s cook breakfast"  # Missing a closing parenthesis
```

That’s like forgetting to close the lid of the blender. The machine (Python) won’t run.

**Output:**
```
SyntaxError: unexpected EOF while parsing
```

---

### Runtime Errors (Exceptions): Mid-Cooking Mishaps

Even if your recipe is written correctly, things can still go wrong **while cooking**:

```python
water = 0
print(5 / water)  # Trying to divide by zero
```

This is like trying to boil water with an empty pot—nothing will happen, and it might break the pot!

**Output:**
```
ZeroDivisionError: division by zero
```

Other common exceptions:

| Error Type           | What It Means                                      |
|----------------------|----------------------------------------------------|
| `ValueError`         | Bad ingredient type (e.g., converting "eggs" to int) |
| `IndexError`         | Grabbing an ingredient that doesn’t exist          |
| `TypeError`          | Mixing ingredients in the wrong way (e.g., adding an int to a string) |
| `KeyError`           | Looking for a label that’s missing in a recipe book (dictionary) |

---

## PART 2: Try-Except – Fire Extinguishers in the Kitchen

Every professional kitchen has **a plan for accidents**—like you should in Python.

Here’s your safety toolkit:

```python
try:
    eggs = int(input("How many eggs? "))
    print("Cracking", eggs, "eggs.")
except ValueError:
    print("That’s not a number, chef!")
```

Let’s break it down:

- `try:` — Light the stove and get cooking.
- `except:` — Catches the fire if something burns.
- `else:` — Runs only if no fire (optional).
- `finally:` — Clean-up crew, always runs.

---

### Full Try-Except Dish

```python
try:
    milk = int(input("How many cups of milk? "))
except ValueError:
    print("That wasn’t a number!")
else:
    print("You added", milk, "cups.")
finally:
    print("Cleaning up spills...")
```

Output (if the user types "two"):
```
That wasn’t a number!
Cleaning up spills...
```

Output (if the user types 2):
```
You added 2 cups.
Cleaning up spills...
```

Use this pattern when your recipe might go wrong **but you still want to finish serving!**

---

## PART 3: Lists – The Reusable Ingredient Tray

Imagine a tray in your kitchen where you lay out items for a dish. Need salt? Check the tray. Need garlic? Add it there.

That’s a **list** in Python.

```python
ingredients = ["salt", "pepper", "chicken", "rice"]
```

### 🛠️ List Operations

```python
# Accessing ingredients
print(ingredients[2])  # chicken

# Adding a new ingredient
ingredients.append("onions")

# Removing a specific item
ingredients.remove("pepper")

# Replacing an ingredient
ingredients[0] = "sugar"

# Slicing the tray
print(ingredients[1:3])  # ['chicken', 'rice']
```

### Why Lists Are Great

- You can **add, remove, or change** items.
- They hold any data type—even a mix of ingredients!
- They’re **ordered**, like recipes.

```python
kitchen_tray = ["knife", 4, True, 3.5]
```

Just like a real tray, you can reorganize it or clean it entirely:

```python
kitchen_tray.clear()  # Empty the tray
```

---

## PART 4: Tuples – Sealed Recipe Trays

Now, imagine you **seal a recipe** inside a container, never to be changed. That’s a **tuple** in Python.

```python
recipe = ("flour", "sugar", "eggs")
```

### Tuples Can’t Be Modified

```python
recipe[0] = "butter"  # ❌ This will raise an error
```

**Output:**
```
TypeError: 'tuple' object does not support item assignment
```

### Tuples Are Useful When:

- You want **fixed data** (like a standard recipe).
- You need better **performance**.
- You're working with **coordinates, constants, days of the week**, etc.

```python
oven_temperature = (180, "Celsius")
```

Tuples are more memory-efficient and safer from accidental changes.

---

## PART 5: Combined Example – Lists, Tuples, and Exceptions

Let’s simulate a full cooking session:

```python
menu = ["Pizza", "Burger", "Sushi"]
prices = (10.99, 8.99, 13.49)  # Using tuple to ensure prices don't change

try:
    choice = int(input("Pick your meal (0-2): "))
    print("You ordered:", menu[choice])
    print("Price: $", prices[choice])
except IndexError:
    print("That's not on the menu, Chef!")
except ValueError:
    print("Please enter a valid number.")
finally:
    print("Thank you for visiting Python Kitchen!")
```

Real-World Scenario:
- **List**: Editable menu
- **Tuple**: Fixed price sheet
- **Exception handling**: Catch invalid orders

---

## 🧠 Real-Life Analogies

| Python Concept     | Real-World Analogy                             |
|--------------------|------------------------------------------------|
| `list`             | Open spice tray (can add/remove items)         |
| `tuple`            | Prepackaged spice mix (sealed and fixed)       |
| `try-except`       | Kitchen safety protocol                        |
| `finally`          | Always clean up after cooking                  |
| `ValueError`       | Asking for sugar but handing you a shoe        |
| `IndexError`       | Trying to get the 4th item from a 3-item tray  |

---

## Chef’s Challenge (Practice Time)

**Task:**  
1. Create a list of 5 favorite ingredients.
2. Ask the user to pick one by index.
3. Show them what they picked.
4. Catch errors:
   - If they enter something that's not a number
   - If they pick an index out of range

Use `try`, `except`, `else`, and `finally`

```python
ingredients = ["Tomato", "Onion", "Pepper", "Garlic", "Thyme"]

try:
    index = int(input("Pick an ingredient (0-4): "))
    print("You chose:", ingredients[index])
except ValueError:
    print("Please enter a valid number, Chef!")
except IndexError:
    print("That's not on the tray!")
finally:
    print("Ingredient selection complete. ✅")
```

---

## ✅ Recap Table

| Concept | Kitchen Analogy | Python Feature |
|--------|------------------|----------------|
| Errors | Fires or spills | SyntaxError, ZeroDivisionError |
| `try-except` | Fire extinguisher | Prevents crash |
| `list` | Tray you can edit | Mutable, `[]` |
| `tuple` | Sealed spice container | Immutable, `()` |
| `finally` | Always clean the kitchen | Runs always |

---

# 📘 **Assignments & Project: Mastering the Kitchen**

### Covered So Far:
- **Day 1** – Variables, Data Types, Input/Output
- **Day 2** – Operators, Expressions
- **Day 3** – Control Flow (if/else, comparisons)
- **Day 4** – Built-in Functions, Modules, String & Math Ops
- **Day 5** – Error Handling, Lists, Tuples

Now let’s apply all of that to real-world scenarios to become a true Python sous-chef!

---

## **Assignments – Cooking Practice Tasks**

Each of these tasks is designed to drill in today's concepts *plus* reinforce prior lessons.

---

### **Assignment 1: The Ingredient Inspector**

**Goal:** Use lists and exception handling.

**Instructions:**
- Create a list of 6 ingredients.
- Ask the user to enter a number between 0 and 5.
- Display the ingredient at that index.
- Catch both `ValueError` (non-number input) and `IndexError` (invalid index).
- Always thank the user in a `finally` block.

**Extra Challenge:** Use `else` to show a message only if the try block succeeded.

---

### **Assignment 2: The Recipe Tuple Logger**

**Goal:** Practice using tuples and string formatting.

**Instructions:**
- Create a tuple called `recipe` with 4 items: flour, sugar, eggs, butter.
- Print each item in the tuple with a message like: `You need 2 cups of sugar.`
- Try to modify the first item. Handle the resulting `TypeError` with a friendly message.

---

### **Assignment 3: The Budget Planner**

**Goal:** Combine math, lists, input, and exception handling.

**Instructions:**
- Create a list of 3 meal names and a tuple of their prices.
- Ask the user how many units of each meal they want.
- Multiply quantity × price for each and show the total cost.
- If the user enters invalid numbers or characters, handle with `ValueError`.

---

### **Assignment 4: Weekly Menu Organizer (Strings + Lists)**

**Goal:** Use string manipulation and list operations.

**Instructions:**
- Create an empty list called `weekly_menu`.
- Ask the user to input meals for 7 days of the week (using `.append()`).
- After filling, print each day's meal with proper formatting:
  ```
  Monday: Spaghetti
  Tuesday: Fried Rice
  ...
  ```
- Ask which day they’d like to change. Update the meal using list indexing.
- Use exception handling to catch any bad input.

---

## **Mini Project: Python Kitchen Order Management System**

🎯 **Objective:** Build a basic order-taking and recipe management system using everything you’ve learned so far.

---

### 🧾 **Project Requirements**

#### 🍽️ 1. Menu System
- Create a list of meals (e.g., `["Pizza", "Burger", "Salad"]`)
- Create a tuple of prices (e.g., `(8.99, 6.49, 5.99)`)
- Display menu using string formatting:
  ```
  0 - Pizza - $8.99
  1 - Burger - $6.49
  2 - Salad - $5.99
  ```

#### 2. Taking Orders
- Ask the user to select a meal by index.
- Ask how many they want.
- Multiply quantity × price.
- Add order to a list of orders.
- Handle `ValueError` and `IndexError` to avoid crashes.

#### 🧾 3. Order Summary
- Print a receipt with all selected items and total amount.
- Use `for` loops, indexing, and formatting:
  ```
  You ordered:
  - 2x Pizza = $17.98
  - 1x Burger = $6.49
  ---------------------
  Total = $24.47
  ```

#### 4. Kitchen-Safe Mode
- If the user enters text instead of a number, show:  
  `"Oops, that's not a valid number. Please try again."`
- Use `finally` to always print:  
  `"Thanks for visiting Python Kitchen!"`

---

### Optional Add-ons (If You’re Feeling Spicy)
- Add a “chef’s special” randomly selected using the `random` module (from Day 4).
- Let the user repeat the order loop for multiple items.
- Use a tuple to store an immutable tax rate or currency code.

---

## How This Ties It All Together

| Lesson | Project Application |
|--------|----------------------|
| Day 1 – Variables, I/O | Taking user input, storing menu and prices |
| Day 2 – Math | Calculating prices, totals |
| Day 3 – Control Flow | If invalid input, show error |
| Day 4 – Functions & Modules | Use `round()`, `len()`, maybe `random.choice()` |
| Day 5 – Lists, Tuples, Exceptions | Store menu, prices, catch bad input gracefully |

---

## Deliverables

To complete Day 5:
- Submit the 4 assignments as individual `.py` files.
- Submit your project as a single `.py` file titled `kitchen_order_manager.py`.

>  Bonus if you use clear variable names, comments, and clean formatting.

---