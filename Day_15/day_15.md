# **100 Days of Python – Day 15: Iterables & List Comprehension**
**Theme**: _"Mastering the Pantry – Clean, Sort, and Serve Data Like a Chef."_

---

## 🔹 **Overview: Iterables**

Imagine you're managing a high-end kitchen pantry. You need a way to loop through every ingredient, check labels, sort what's fresh, and prep only what you need.  
In Python, that’s what **iterables** are for.

### **Definition:**
An **iterable** is any Python object capable of returning its members one at a time, allowing it to be looped over. It implements the `__iter__()` method, either directly or via `__getitem__()`.

### **Common Iterable Types:**

| Type         | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `String`     | A sequence of characters. Each character can be accessed individually.     |
| `List`       | An ordered, mutable collection of elements.                                 |
| `Tuple`      | Like a list, but immutable.                                                 |
| `Set`        | An unordered collection of unique items.                                    |
| `Dictionary` | A key-value mapping, where keys are iterable by default.                    |

---

## 🔹 **Using `in` and `not in` for Membership Testing**

These operators are used to check for **presence** or **absence** of an item in an iterable.

### **Example 1: Validate Email Structure**
```python
email = "vehutech@gmail.com"

if "@" in email and "." in email:
    print("Valid email")
```

### **Example 2: Detect Forbidden Words**
```python
forbidden = ["spam", "scam", "phish"]
message = "This is not a scam."

if any(word in message for word in forbidden):
    print("Warning: Suspicious content!")
else:
    print("Message is clean.")
```

### **Example 3: Check Key in Dictionary**
```python
menu = {"pasta": 1200, "steak": 3500}

if "steak" in menu:
    print("Steak is available.")
```

---

## 🔹 **Capabilities of Iterables**

Here are some common tasks you can perform with iterable objects:

### 1. **Looping with `for`**
```python
for char in "recipe":
    print(char)
```

### 2. **Converting Between Types**
```python
list("cook")          # ['c', 'o', 'o', 'k']
tuple([1, 2, 3])      # (1, 2, 3)
set([1, 1, 2])        # {1, 2}
```

### 3. **Measuring Length**
```python
ingredients = ["flour", "sugar", "eggs"]
print(len(ingredients))  # 3
```

### 4. **Enumerating**
```python
steps = ["prep", "cook", "plate"]

for i, step in enumerate(steps, start=1):
    print(f"Step {i}: {step}")
```

### 5. **Sorting**
```python
ratings = [5, 3, 4, 2]
sorted_ratings = sorted(ratings)
print(sorted_ratings)  # [2, 3, 4, 5]
```

---

## 🔹 **List Comprehension – Clean and Elegant List Creation**

### **Professional Definition:**
A list comprehension is a syntactically elegant construct that provides a **concise way to generate lists** from existing iterables using a single line of code. It combines looping and conditional logic.

### **General Syntax:**
```python
[expression for item in iterable if condition]
```

### **Examples:**

#### 1. **Generate Squares**
```python
squares = [x**2 for x in range(10)]
```

#### 2. **Filter Even Numbers**
```python
evens = [x for x in range(20) if x % 2 == 0]
```

#### 3. **Uppercase Fruits**
```python
fruits = ["apple", "banana", "cherry"]
capitalized = [fruit.upper() for fruit in fruits]
```

#### 4. **Extract Gmail Users**
```python
emails = ["a@gmail.com", "b@yahoo.com", "c@gmail.com"]
gmail_users = [e for e in emails if "@gmail.com" in e]
```

#### 5. **Nested Comprehension (Multiplication Table)**
```python
table = [[i * j for j in range(1, 6)] for i in range(1, 6)]
```

---

## 🔹 **Mini-Project of the Day: Email Verifier and Collector**

**Project Title:** _“Clean Contact Sheet”_

### **Objective:**
Build a program that:
- Accepts a list of email addresses.
- Filters out invalid ones (must contain `@` and `.`).
- Stores only valid emails in a cleaned list.
- Categorizes emails by provider (e.g., Gmail, Yahoo, others).

### **Sample Code Template:**
```python
raw_emails = [
    "hello@gmail.com",
    "invalidemail.com",
    "user@yahoo.com",
    "support@vehutech"
]

valid_emails = [email for email in raw_emails if "@" in email and "." in email]

gmail = [e for e in valid_emails if "gmail.com" in e]
yahoo = [e for e in valid_emails if "yahoo.com" in e]
others = [e for e in valid_emails if e not in gmail + yahoo]

print("Valid Emails:", valid_emails)
print("Gmail Users:", gmail)
print("Yahoo Users:", yahoo)
print("Others:", others)
```

---

## 🔹 **Assignment**

### **Part A: Membership Practice**
Write a function that takes a sentence and returns all the vowels present in the sentence using a set and `"in"`.

### **Part B: Custom List Comprehension**
Write a one-liner using list comprehension to:
1. Get the cube of all odd numbers from 1 to 30.
2. Extract usernames (before `@`) from a list of emails.

### **Part C: Iteration Drill**
Using a dictionary of food items and prices, create:
- A list of items costing more than 1000.
- A new dictionary with a 10% discount applied to each item.

---

## Summary

| Concept              | Description                                                                          |
|----------------------|--------------------------------------------------------------------------------------|
| Iterable             | An object capable of returning items one at a time, usable in a loop.               |
| Membership Operator  | `"in"` / `"not in"` checks whether a value exists in a sequence.                    |
| List Comprehension   | A compact syntax for building new lists from iterable sequences.                    |
| Capabilities         | Looping, filtering, transforming, sorting, enumerating, type conversion, etc.       |

---