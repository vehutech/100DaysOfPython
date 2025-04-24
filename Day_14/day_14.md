# **Day 14: The Art of Function Arguments in Python – The Kitchen Blueprint**

In professional kitchens, how ingredients are prepared, portioned, and ordered defines the success of a recipe. Likewise, in Python, how you pass arguments into functions influences readability, flexibility, and robustness of your code.

Today we’ll break down the **four main types of function arguments** and then refactor a real-world script, `timeUp.py`, to implement each of them using industry-standard practices.

---

## **Understanding Function Arguments**

### 1. **Positional Arguments** – The Assembly Line

**Definition**: Arguments that are assigned to parameters based strictly on their position in the function call.

In a restaurant kitchen, if you're making a sandwich and you pass the ingredients like: `("bread", "lettuce", "cheese")`, the chef assumes:

- First item → base (bread)
- Second item → filling (lettuce)
- Third item → topping (cheese)

In Python:

```python
def make_sandwich(base, filling, topping):
    print(f"Sandwich with {base}, {filling}, and {topping}")

make_sandwich("bread", "lettuce", "cheese")
```

If you swap `filling` and `topping`, you ruin the sandwich.

---

### 2. **Default Arguments** – The House Standard

**Definition**: Parameters that assume a default value if no argument is provided.

Imagine your kitchen has a rule: if no spice is specified, use salt.

```python
def boil(water_ml, spice="salt"):
    print(f"Boiling {water_ml}ml water with {spice}")

boil(500)           # Uses default "salt"
boil(500, "pepper") # Overrides the default
```

Benefits:
- Reduces the number of required arguments
- Makes the function easier to call
- Establishes predictable defaults

---

### 3. **Keyword Arguments** – Labeling Ingredients Clearly

**Definition**: Arguments passed by explicitly specifying which parameter they correspond to, using `name=value`.

This is like labeling containers in your pantry—there’s no confusion even if you pick items in any order.

```python
def bake(temp, duration):
    print(f"Baking at {temp}°C for {duration} minutes")

bake(duration=45, temp=180)  # Readable, order doesn't matter
```

Benefits:
- Improves code readability
- Prevents errors with long parameter lists

---

### 4. **Arbitrary Arguments (`*args`)** – Unlimited Add-ons

**Definition**: Allows a function to accept any number of positional arguments, collected into a tuple.

Like building a custom pizza—you can add as many toppings as you want.

```python
def build_pizza(*toppings):
    print("Pizza with:", ", ".join(toppings))

build_pizza("cheese", "tomatoes", "olives", "mushrooms")
```

Benefits:
- Adds flexibility to your function
- Useful for wrapper functions, loggers, or custom builders

---

## Final Project: `timeUp.py` – Kitchen Timer Script Using All Four Argument Types

Let's rewrite the `timeUp.py` script professionally using **positional**, **default**, **keyword**, and **arbitrary** arguments.

---

### **Objective**: Create a kitchen timer that supports multiple timers, customizable intervals, and user-friendly labels.

```python
# timeUp.py

import time

def count_timer(end, start=0, *, label="Timer", interval=1, **metadata):
    """
    Counts from start to end (inclusive), pausing 'interval' seconds between numbers.

    Parameters:
    - end (int): The ending count (required, positional)
    - start (int): The starting count (default=0)
    - label (str): A label for the timer (keyword-only)
    - interval (float): Time in seconds between each count (keyword-only)
    - metadata (dict): Arbitrary keyword arguments, for additional context (not used in counting logic)
    """

    print(f"\nStarting '{label}' from {start} to {end}...")
    for i in range(start, end + 1):
        print(f"{label}: {i}")
        time.sleep(interval)
    print(f"'{label}' complete!\n")

    if metadata:
        print("Additional Info:")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
    print("-" * 30)
```

### **Explanation of Argument Types Used**:

| Parameter      | Type                  | Purpose |
|----------------|-----------------------|---------|
| `end`          | Positional            | Must be provided; defines the count target. |
| `start=0`      | Default               | Defaults to 0 if not specified. |
| `label=...`    | Keyword-only (`*`)    | Clearly distinguishes optional config settings. |
| `interval=1`   | Keyword-only          | Controls the time delay between counts. |
| `**metadata`   | Arbitrary Keyword     | Allows extensibility; metadata for logs or reports. |

### **Usage Examples**:

```python
# Basic usage (positional + default)
count_timer(5)

# With keyword arguments
count_timer(3, label="Egg Timer", interval=2)

# Mixed usage with metadata
count_timer(4, 2, label="Coffee Brew", interval=0.5, description="Drip method", method="V60")
```

---

## Summary: Think Like a Chef

When building Python functions, think like you're setting up a kitchen station:

- **Positional arguments** are the core ingredients—must be in order.
- **Default arguments** are your pantry staples—used when you don’t specify.
- **Keyword arguments** are your labeled jars—order doesn’t matter.
- **Arbitrary arguments** are the buffet table—flexible and open-ended.

---

## Challenge

1. Extend `count_timer` to log to a file.
2. Add a `callback` keyword argument that runs a function after the timer ends.
3. Refactor the print logic into a helper function for better modularity.