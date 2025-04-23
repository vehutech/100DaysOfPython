---

# Day 13: Functions – Your Personal Chef’s Secret Weapon

## Introduction – Welcome to the Kitchen

Imagine walking into a professional kitchen. Everything is organized: knives are in place, spices are labeled, ingredients are chopped and stored. You, the chef, aren’t guessing or repeating yourself—you have **recipes**. These recipes are your instructions. Once you’ve perfected them, you don’t rewrite them every day. You just reuse them when needed.

In Python, these recipes are called **functions**.

A function is a reusable block of code that performs a specific task. Rather than repeating the same code over and over, you write it once and call it whenever it’s needed. This is how great chefs (and great coders) stay efficient.

---

## What is a Function?

Let’s start with the basics.

### Syntax

```python
def function_name():
    # code to execute
    print("This is a function")
```

The `def` keyword means you’re **defining a function**. Think of it as writing a new recipe in your kitchen notebook.

### Example:

```python
def greet():
    print("Hello there!")
```

To use (or invoke) this recipe in your code, you simply call it:

```python
greet()
```

Just like saying: “Chef, make me a greeting,” and it gets done exactly how you wrote it.

---

## Adding Ingredients: Parameters

A recipe isn’t always the same. Sometimes, you need to adjust for different ingredients or guests. That’s what **parameters** are—**placeholders for data** that your function needs to run.

```python
def happy_birthday(name, age):
    print(f"Happy birthday, {name}")
    print(f"You are {age} years old")
```

Here, `name` and `age` are like the ingredients passed into the recipe.

Call it like this:

```python
happy_birthday("Bro", 20)
happy_birthday("Steve", 30)
happy_birthday("Joe", 40)
```

Each time, the same recipe (function) produces a different meal (output) depending on the ingredients (arguments).

---

## Displaying an Invoice – Another Recipe

Let’s say you run a kitchen that bills customers. Here’s how you could write a function to display an invoice:

```python
def display_invoice(username, amount, due_date):
    print(f"Hello {username}")
    print(f"Your bill of ${amount:.2f} is due: {due_date}")
```

You can call it like this:

```python
display_invoice("Tolu", 249.99, "2025-05-01")
```

It’s just like printing a customer receipt based on the current order.

---

## The Return Statement – Packaging the Meal

Sometimes, you don’t want the function to print something on the spot. Instead, you want it to **prepare a result** and send it back to the caller. This is where the `return` keyword comes in. It’s like plating your dish and handing it over to the waiter.

```python
def add(x, y):
    return x + y
```

This function doesn’t print. It **returns** the result.

```python
result = add(10, 5)
print(result)
```

Or directly:

```python
print(add(10, 5))
```

Let’s define more operations:

```python
def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return "Cannot divide by zero"
    return x / y
```

Each function prepares a dish (a result) and returns it to you.

---

## Small Project: Build a Simple Calculator

Let’s apply everything by writing a calculator.

### Calculator Code:

```python
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return "Cannot divide by zero"
    return x / y

def calculator():
    print("Welcome to the Python Calculator")
    x = float(input("Enter first number: "))
    y = float(input("Enter second number: "))
    operation = input("Choose operation (+, -, *, /): ")

    if operation == '+':
        result = add(x, y)
    elif operation == '-':
        result = subtract(x, y)
    elif operation == '*':
        result = multiply(x, y)
    elif operation == '/':
        result = divide(x, y)
    else:
        result = "Invalid operation"

    print(f"Result: {result}")

calculator()
```

### What’s happening?

This is like an open kitchen. The user brings two ingredients (numbers), chooses a tool (operation), and your kitchen staff (functions) prepare the correct dish (result).

---

## A Recap From the Chef’s Table

- **Functions** are reusable recipes.
- **Parameters** are ingredients.
- **Calling a function** is like giving an order.
- **Return values** are the prepared dish handed back.
- **Printing inside a function** is like serving the meal immediately.
- **Returning a value** is like prepping it for future use.

---

## Assignment

Start simple. Practice makes perfect.

1. Write a function `greet_user(name)` that prints a personalized greeting.
2. Write a function `calculate_discount(price, discount_percent)` that returns the final price after the discount.
3. Write a function `generate_receipt(item, price)` that prints the item and price, and returns the total.