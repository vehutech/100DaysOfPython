# 🧠 **100 Days of Python – Day 16: Generators & Lazy Evaluation**  
**Theme**: _“Slow Cooking: Serving Dishes as They’re Ready (Not All at Once)”_

---

## 🔹 **Professional Overview: What Are Generators?**

Imagine you're running a buffet. Instead of preparing **every dish in advance** (which consumes time, space, and energy), you decide to cook and serve **one plate at a time** as customers arrive. This is more efficient, especially when you don’t know how many guests you’ll get.

That’s the **generator philosophy** in Python.

### **Definition:**
A **generator** is a type of iterable that **generates values lazily**—one at a time—**only when requested**. Instead of storing the entire sequence in memory, it **yields** values as needed.

> Generators are ideal when working with **large data**, **infinite sequences**, or **resource-intensive computations**.

---

## 🔹 **Difference Between Generators and Lists**

| Feature            | List                              | Generator                         |
|--------------------|------------------------------------|-----------------------------------|
| Memory             | Stores **all elements** in memory  | Yields **one value at a time**    |
| Performance        | Fast access but uses more memory   | Slower access, memory efficient   |
| Syntax             | Uses `[]` or `list()`              | Uses `yield`, `()` or generator functions |
| Example            | `[x**2 for x in range(1000)]`      | `(x**2 for x in range(1000))`     |

---

## 🔹 **How to Create a Generator**

### 1. **Using `yield` in a Function**
```python
def cook_dishes():
    for i in range(1, 4):
        yield f"Dish {i} ready"

orders = cook_dishes()

for order in orders:
    print(order)
```

### 2. **Using Generator Expressions**
```python
squares = (x**2 for x in range(10))  # Note the ()
```

You can loop over `squares` just like any iterable:
```python
for s in squares:
    print(s)
```

---

## 🔹 **The `yield` Keyword**

Using `yield` inside a function **turns it into a generator**. Unlike `return`, it **pauses** the function, remembers its state, and resumes from where it left off.

### Example:
```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1
```

Calling:
```python
for number in countdown(5):
    print(number)
```

Will output:
```
5
4
3
2
1
```

---

## 🔹 **When to Use Generators**

### ✅ Use Generators When:
- You're working with **large datasets** (e.g., reading files, logs).
- You only need to **iterate once**.
- You want to **reduce memory footprint**.
- You're dealing with **streams** or **live data feeds**.

---

## 🔹 **Real-Life Examples**

### **1. Read Large Files Line-by-Line**
```python
def read_large_file(file_path):
    with open(file_path) as f:
        for line in f:
            yield line.strip()
```

### **2. Simulate an Infinite Data Stream**
```python
def infinite_numbers():
    n = 1
    while True:
        yield n
        n += 1
```

Use with caution:
```python
stream = infinite_numbers()
for _ in range(5):
    print(next(stream))
```

---

## 🔹 **How to Use `next()`**

The `next()` function manually pulls the **next value** from a generator.

```python
gen = (x**2 for x in range(3))

print(next(gen))  # 0
print(next(gen))  # 1
print(next(gen))  # 4
```

Calling `next()` again will raise a `StopIteration` exception.

---

## 🛠️ **Mini-Project: Menu on Demand**

**Title:** _“Lazy Kitchen”_

Build a system that:
- Accepts a list of dishes (strings).
- Creates a generator that yields one dish at a time.
- Only prints dishes as customers “ask” for them.

### Template:
```python
def serve_dishes(dishes):
    for dish in dishes:
        yield f"Now serving: {dish}"

orders = ["pasta", "burger", "noodles", "rice"]

diner = serve_dishes(orders)

print(next(diner))  # On-demand serving
print(next(diner))  # Another customer arrives
```

Add your twist: maybe track which customer ordered which dish.

---

## 📝 Assignment

### Part A – Function-Based Generator
Write a generator that yields even numbers up to 100 without using a list.

### Part B – Generator Expression
Create a generator expression that gives the cube of odd numbers between 1 and 20. Use a loop to print them.

### Part C – Simulated File Reader
Write a generator that simulates reading 5 lines of a file by yielding:
```
Line 1: <content>
Line 2: <content>
...
```
Use `yield` in a loop to simulate this.

---

## Summary

| Concept         | Description                                                |
|----------------|------------------------------------------------------------|
| Generator       | A memory-efficient iterable that yields values one at a time |
| `yield`         | Pauses function execution and resumes later                 |
| Generator Expr  | Compact syntax like list comprehensions but with `()`       |
| `next()`        | Pulls the next value from a generator manually              |
| Use Cases       | Big data, file reading, streaming, infinite sequences       |

---