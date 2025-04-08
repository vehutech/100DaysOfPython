## **Day 6: The Chef’s Hidden Shelf – Unlocking Python’s Unsung Data Types**

Our chef walks into the kitchen early today, apron on, notebook in hand. He's already mastered the basics: slicing (`lists`), recipe instructions (`tuples`), naming (`strings`), and decision-making (`booleans`). But the head chef calls him over and says, “You’ve done well, but now it’s time to meet the *silent operators*. These are the tools that make large, complex meals possible.”

Let’s follow along.

---

### **1. The Complex Mixer: `complex`**

In a special section of the kitchen sits a machine that handles more than one thing at a time. It can mix flour in the real world *and* add a dash of imagination.

```python
c = 3 + 4j
```

Here, `3` is the real part, and `4j` is the imaginary—just like juggling two values at once.

#### Use Case:
**Electrical Engineering: Impedance calculation in AC circuits.**

```python
# Two circuit components with impedance
resistor = complex(100, 0)
capacitor = complex(0, -50)

total_impedance = resistor + capacitor
print(f"Total impedance: {total_impedance} ohms")
```

#### 🔧 Attributes:
```python
print(c.real)  # 3.0
print(c.imag)  # 4.0
print(abs(c))  # 5.0 – Pythagorean magnitude
```

> **When to use**: Scientific and engineering problems involving imaginary numbers. Not for everyday apps.

---

### **2. The Kitchen Timer: `range`**

The oven timer is your `range`—a way to repeat steps.

```python
for i in range(3):
    print(f"Round {i+1}: Stir the pot.")
```

#### 🍳 Use Case:
**Generating meal IDs automatically**

```python
# Let's say we want to assign table numbers from 1 to 10
table_ids = list(range(1, 11))
print("Tables ready:", table_ids)
```

#### 🔧 Variants:
```python
range(5)         # 0,1,2,3,4
range(2, 10)     # 2 to 9
range(2, 10, 2)  # 2, 4, 6, 8
```

> Clean, memory-efficient, and doesn’t waste ingredients (RAM).

---

### **3. The Recipe Book: `dict`**

The chef’s personal journal. Each dish (key) has detailed instructions (value).

```python
recipes = {
    "omelette": "beat eggs, cook in pan",
    "salad": "chop, mix, dress"
}
```

#### 🍳 Use Case:
**Storing user data in a web app.**

```python
user_profile = {
    "name": "Ada",
    "email": "ada@chefmail.com",
    "is_admin": True,
    "orders": [101, 102, 103]
}

# Accessing info
print(user_profile["name"])
```

#### 🔧 Operations:
```python
# Safe access
print(user_profile.get("location", "Not provided"))

# Update value
user_profile["email"] = "ada@newmail.com"

# Loop
for key, value in user_profile.items():
    print(f"{key.title()}: {value}")
```

> Ideal for structuring config files, user data, and JSON-like structures.

---

### **4. The Ingredient Checklist: `set`**

Only unique items make it in.

```python
ingredients = {"salt", "pepper", "garlic", "salt"}  # Only one "salt"
```

#### Use Case:
**Avoiding duplicate orders in a shopping cart**

```python
orders = ["bread", "milk", "milk", "eggs", "bread"]
unique_orders = set(orders)

print("Send to kitchen:", unique_orders)
```

#### Set Operations:
```python
pantry = {"flour", "sugar", "eggs"}
needed = {"sugar", "eggs", "milk"}

missing = needed - pantry
print("Missing ingredients:", missing)
```

> Fast membership checks, great for filters and categories.

---

### **5. The Laminated Menu: `frozenset`**

Some ingredient combinations are sacred. You don’t change them.

```python
essentials = frozenset(["salt", "sugar", "flour"])
```

#### Use Case:
**Using immutable keys in a recipe database**

```python
# You can't use a list or set as a key, but frozenset works
ingredient_combos = {
    frozenset(["flour", "milk", "egg"]): "Pancake",
    frozenset(["rice", "beans"]): "Jollof Mix"
}

print(ingredient_combos[frozenset(["milk", "flour", "egg"])])
```

> Perfect when you want uniqueness **and** immutability.

---

### **6. The Vacuum-Sealed Ingredients: `bytes`, `bytearray`, `memoryview`**

---

#### a) `bytes` – Immutable Package

```python
message = b"Order Ready"
print(message[0])       # 79 ('O')
print(message.decode()) # "Order Ready"
```

#### Use Case:
**Reading image file data**

```python
with open("logo.png", "rb") as img:
    image_data = img.read()
print(type(image_data))  # <class 'bytes'>
```

---

#### b) `bytearray` – Mutable Package

```python
ba = bytearray(b"milk")
ba[0] = ord('M')
print(ba.decode())  # "Milk"
```

#### Use Case:
**Modifying binary data before saving or sending**

```python
# Simulating an encrypted message
msg = bytearray(b"chef")
for i in range(len(msg)):
    msg[i] += 1
print(msg.decode())  # "digi"
```

---

#### c) `memoryview` – The Window

```python
data = bytearray(b"butter")
view = memoryview(data)

print(view[1:4].tobytes())  # b'utt'
```

#### 🍳 Use Case:
**Efficient slicing without copying**

```python
# Great for performance: no new object is created
raw = bytearray(b"image_data_stream")
header_view = memoryview(raw)[:5]
print(header_view.tobytes())
```

> Ideal in data science, image processing, and networking.

---

### **7. The Empty Bowl: `NoneType`**

Sometimes the chef leaves a bowl empty, with the intention to use it later.

```python
batter = None
```

#### Use Case:
**Setting default behavior or flags**

```python
def serve_dish(garnish=None):
    if garnish is None:
        garnish = "parsley"
    print(f"Served with {garnish}.")

serve_dish()          # Served with parsley
serve_dish("lemon")   # Served with lemon
```

#### Always use:
```python
if batter is None:
```

> Use `is` for identity comparisons. Think of `None` as *not yet used*.

---

## **End of Day Summary: The Hidden Shelf**

| Data Type     | Purpose                                    | Mutable |
|---------------|--------------------------------------------|---------|
| `complex`     | Real + imaginary numbers                   | No      |
| `range`       | Memory-efficient looping                   | No      |
| `dict`        | Key-value mappings                         | Yes     |
| `set`         | Unique items only                          | Yes     |
| `frozenset`   | Immutable set                              | No      |
| `bytes`       | Immutable binary data                      | No      |
| `bytearray`   | Mutable binary data                        | Yes     |
| `memoryview`  | View over bytes without copying            | N/A     |
| `NoneType`    | Represents 'no value' or 'not yet defined' | No      |

---

As the chef wipes down his workstation and sharpens his tools, he now understands that while flour, eggs, and sugar make the dish, it’s the timers, locked ingredients, secret spices, and clean bowls that make the kitchen *professional*.

He closes his journal and nods to the head chef. Tomorrow brings something new.

---

## **Day 6: Assignments – The Chef's Trials**

### **Assignment 1: Kitchen Inventory Tracker**

You're managing a restaurant with limited storage. Design a program to:

- Use a `set` to track unique ingredients in stock.
- Use a `frozenset` to store non-editable combinations for signature dishes.
- Use a `dict` to map ingredients to their quantities.
- Allow the chef to check if ingredients for a dish are available.
- Print a message showing **missing ingredients** (use set operations).

**Bonus**: Add a `range` to simulate 7 days of stock checking.

---

### **Assignment 2: Order Stream Monitor**

Simulate receiving binary data for online orders:

- Store a string message as `bytes`, like `b"Order: 001"`.
- Convert it into a `bytearray` and simulate changing part of the message.
- Use `memoryview` to inspect a slice of the data without copying it.

You must:
- Print the original and modified messages.
- Extract only the customer ID portion using slicing.

---

### **Assignment 3: Circuit Cost Estimator**

You're helping an electrical shop:

- Represent resistors and capacitors using `complex` numbers.
- Write a function to add them and calculate **total impedance**.
- Display both the real and imaginary parts clearly.
- If total impedance magnitude is greater than a certain value, warn the chef that the design may “overcook” the dish.

---

### **Assignment 4: Garnish Handler with `NoneType`**

Create a `serve_dish()` function:

- Accept a `garnish` parameter that defaults to `None`.
- If `None`, print `"Default garnish used: mint"`.
- Otherwise, print `"Custom garnish used: [name]"`.
- Ensure that your check uses the identity operator (`is`).

---

## **Capstone Project: ChefBot – A Restaurant Assistant System**

Design a CLI or basic GUI application that simulates a restaurant assistant. It should leverage **every** data type we’ve studied so far (including from previous days like `int`, `float`, `list`, `tuple`, etc.).

### **Features to Include:**

1. **Menu System (`dict`, `tuple`)**
   - Dishes as keys, with tuple values: (price, estimated time).

2. **Order Management (`list`, `set`, `frozenset`)**
   - Track ongoing orders in a list.
   - Use `set` to avoid duplicate ingredients per dish.
   - Use `frozenset` for fixed meal combos.

3. **Kitchen Inventory (`dict`, `set`)**
   - Ingredient-to-stock mapping with `dict`.
   - Notify when ingredients are missing using set difference.

4. **Order Stream (`bytes`, `bytearray`, `memoryview`)**
   - Simulate receiving orders via binary data.
   - Modify and view parts of the data efficiently.

5. **Range-Based Scheduling (`range`)**
   - Generate schedules for preparing dishes in time slots.

6. **Sensor System (`complex`)**
   - Simulate sensors using `complex` numbers (heat + pressure).
   - Display magnitude and warning if overload is detected.

7. **Garnish Handling (`NoneType`)**
   - Use `None` as default for optional garnishes in the order system.

---

### Sample User Story:
> *Ada logs in to the CLI. She selects a “Combo Meal.” The system checks if all ingredients are available (using `set` comparison). It schedules her dish in the 12:00-12:30 slot (`range`). The sensors detect a complex impedance, and ChefBot confirms optimal cooking conditions. Ada skips the garnish, so the default is applied (`NoneType`). The order data is streamed and stored in binary (`bytes`). It’s later edited (`bytearray`) and logged with `memoryview`.*

---

## Outcome

By the end of this day and project:

- You’ll **think like a Pythonic engineer**.
- You’ll understand when and *why* to use every core data type.
- You’ll see how deep tech meets delightful design—like ingredients in a great dish.