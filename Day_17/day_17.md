---

# Day 17: Match-Case Statement (Python’s Switch Alternative)

---

Chef Lina runs a busy kitchen. Every morning, her staff come to her for instructions:  
*“Chef, what's the plan for today?”*  

If it's Monday, she orders pastries.  
If it's Tuesday, she calls for soups.  
Wednesday is pasta day.  
And so on.

At first, Chef Lina used a simple list stuck on the fridge:
- **If** today is Monday, **then** pastries.
- **If** today is Tuesday, **then** soups.
- **If** today is Wednesday, **then** pasta.
- ...and so on.

But as more dishes and days piled up, reading the fridge list felt clumsy and slow.  
She needed a cleaner, faster system — something like a neat kitchen chalkboard with clear categories.

That’s exactly what Python's `match-case` statement offers:  
a **more organized kitchen chalkboard**, replacing the old messy fridge list of many if-elif-else steps.

---

## Definition

**Match-case** in Python is a control flow structure introduced in **Python 3.10**.  
It allows a program to **compare a value against multiple patterns** and **execute code based on which pattern matches**.  
It is often called Python's version of the **switch-case** statement found in other programming languages like C or JavaScript.  

Where `if-elif-else` handles one condition at a time, `match-case` groups conditions neatly and makes complex decisions **clearer** and **more readable**.

---

## Traditional Kitchen List: If-Elif-Else

Here’s how Chef Lina's original messy fridge list would look in Python:

```python
day = input("Enter the day of the week: ").lower()

if day == "monday":
    print("Prepare pastries")
elif day == "tuesday":
    print("Prepare soups")
elif day == "wednesday":
    print("Prepare pasta")
elif day == "thursday":
    print("Prepare salads")
elif day == "friday":
    print("Prepare seafood")
elif day == "saturday":
    print("Prepare roasts")
elif day == "sunday":
    print("Prepare desserts")
else:
    print("Unknown day — relax.")
```

**What’s happening?**  
Every single new condition needs a new `elif`.  
If the kitchen adds "smoothie day" or "grill day", Chef Lina has to squeeze in even more `elif`s, and the code stretches longer than a noodle.

---

## A Neater Kitchen Chalkboard: Match-Case

Now, here’s how the same kitchen instructions would look on Lina’s neat chalkboard, using `match-case`:

```python
day = input("Enter the day of the week: ").lower()

match day:
    case "monday":
        print("Prepare pastries")
    case "tuesday":
        print("Prepare soups")
    case "wednesday":
        print("Prepare pasta")
    case "thursday":
        print("Prepare salads")
    case "friday":
        print("Prepare seafood")
    case "saturday":
        print("Prepare roasts")
    case "sunday":
        print("Prepare desserts")
    case _:
        print("Unknown day — relax.")
```

**What's happening now?**  
Chef Lina can quickly match the day and act without scanning a hundred lines of if-elif logic.  
The wildcard `_` at the end catches anything unexpected — like a surprise delivery on a holiday.

---

## Deeper into the Kitchen: Other Powerful Uses of Match-Case

Just like a kitchen needs more than one type of pot or pan, `match-case` can handle more than simple direct matches.  
It can handle grouped items, shapes, and even the structure of ingredients.

---

### 1. Grouping Multiple Items

Suppose Chef Lina’s fridge has many similar ingredients.  
She wants to handle "apples", "bananas", and "grapes" the same way: **fruit prep**.

```python
item = input("Enter the ingredient: ").lower()

match item:
    case "apple" | "banana" | "grape":
        print("Start fruit prep")
    case "beef" | "chicken" | "pork":
        print("Start meat prep")
    case "carrot" | "broccoli" | "spinach":
        print("Start vegetable prep")
    case _:
        print("Unknown ingredient")
```

Instead of repeating herself for each fruit, Lina groups them into one case.  
The kitchen runs smoother when similar ingredients are prepared together.

---

### 2. Matching Data Structures (Like Lists)

Sometimes, Chef Lina receives boxes of supplies.  
If a box contains `[flour, sugar]`, she knows it’s baking time.

```python
box = ["flour", "sugar"]

match box:
    case ["flour", "sugar"]:
        print("Start baking preparation")
    case [item1, item2]:
        print(f"Received a box with {item1} and {item2}")
    case _:
        print("Unknown box contents")
```

Here, the pattern not only checks the **contents** but can even **unpack** them like ingredients from a delivery box.

---

### 3. Matching Objects

Suppose every new kitchen order is wrapped as an object:

```python
class Order:
    def __init__(self, dish, quantity):
        self.dish = dish
        self.quantity = quantity

order = Order("pasta", 3)

match order:
    case Order(dish="pasta", quantity=3):
        print("Prepare 3 plates of pasta")
    case Order(dish=dish, quantity=quantity):
        print(f"Prepare {quantity} plates of {dish}")
    case _:
        print("Unknown order")
```

Now Lina can prepare dishes by matching **inside** the objects, like checking a written ticket from the waiter.

---

## Final Thoughts: Which to Use, and When

Chef Lina doesn't throw away the fridge notes completely.  
If she has only one or two small decisions to make, it’s quicker to use a simple `if-else`.

But when the kitchen grows, and the menu expands, she hangs her chalkboard proudly and switches to `match-case`.

**In technical terms:**  
- **Use `if-else`** for a small number of simple, direct checks.  
- **Use `match-case`** when you have many distinct cases, complex structures, or when you want clarity and maintainability.

Good code, like a good kitchen, should always aim for **clarity**, **organization**, and **ease of movement**.

---