---

# Day 18: Understanding Variable Scope in Python

---

In every good kitchen, not every chef can touch every ingredient at will.  
Some ingredients are kept right next to the working chef. Some are stored in a big pantry that everyone in the restaurant can access.  
Others are available in a manager's private store, and some supplies come from a universal vendor that delivers salt, sugar, and water to all restaurants by default.

Similarly, in Python programming, not every variable is available everywhere in your code.  
This brings us to the concept of **Variable Scope**.

---

## What is Variable Scope?

**Variable scope** refers to **the region of a program** where a particular variable is recognized and accessible.  
Outside of its scope, the variable simply doesn't exist — just like an ingredient that's not in your kitchen when you need it.

Knowing **where** your variables are alive and where they are invisible is a critical part of writing correct, efficient, and understandable code.

---

## Scope Resolution in Python: The LEGB Rule

Python follows a strict method to decide where to look for a variable whenever it’s called.  
This method is known as the **LEGB** rule, an acronym standing for:

- **L** — Local
- **E** — Enclosed
- **G** — Global
- **B** — Built-in

When Python encounters a variable, it searches for it in the following order:

1. **Local** scope: Inside the current function.
2. **Enclosing** scope: Inside any functions that enclose the current function.
3. **Global** scope: At the top level of the current module (the script).
4. **Built-in** scope: Names preassigned by Python itself, such as `len`, `sum`, `print`.

Python moves from innermost (Local) to outermost (Built-in), stopping at the first match it finds.

---

## Local Scope (L)

A **local scope** refers to variables declared **inside a function**.  
They exist **only while** the function is running.
  
Imagine a chopping board where the chef puts a **private stash of garlic** while preparing a sauce.  
Other chefs don't know about this garlic. It's personal and temporary.

**Example:**

```python
def make_sauce():
    garlic = "2 cloves"
    print(garlic)

make_sauce()
```

Here, `garlic` is created and used inside `make_sauce()`.  
Outside the function, it does not exist.

Trying this:

```python
print(garlic)
```

would crash the program because `garlic` has no meaning outside `make_sauce()`.

---

## Enclosed Scope (E)

An **enclosed scope** happens when you have **nested functions** — one function inside another.  
The inner function can access variables from the outer (enclosing) function.


A **sous-chef** can access ingredients set aside by the **head chef** for a special dish.

**Example:**

```python
def kitchen():
    salt = "1 teaspoon"

    def cook():
        print(salt)

    cook()

kitchen()
```

The inner function `cook()` cannot find `salt` locally, so it looks in the enclosing function `kitchen()` and finds it.

---

## Global Scope (G)

A **global scope** refers to variables created at the **top level** of the program — not inside any function.


Think of ingredients kept in a **central pantry** that any chef in the restaurant can walk into and grab.

**Example:**

```python
pantry = "olive oil"

def fry():
    print(pantry)

fry()
print(pantry)
```

Here, `pantry` is available both inside and outside the function.

However, if you want to **modify** a global variable inside a function, you must explicitly tell Python that you're working with the global version by using the `global` keyword.

**Example:**

```python
stock = 5

def use_stock():
    global stock
    stock -= 1

use_stock()
print(stock)  # Output: 4
```

Without the `global` keyword, Python would think you're trying to create a **new local** `stock` inside `use_stock()`, which would lead to errors.

---

## Built-in Scope (B)

Finally, there are **built-in scopes**: variables and functions that Python provides automatically.


Think of **universal ingredients** like air, water, and salt delivered automatically to every kitchen without needing to order them.

Python has built-in functions like:

- `len()`
- `sum()`
- `str()`
- `print()`

**Example:**

```python
print(len("kitchen"))
```

However, **beware**:  
If you name your own variable with a built-in name, you will overwrite it.

**Bad example:**

```python
list = [1, 2, 3]  # Bad: overwrites the 'list' built-in function
print(list)
```

Now, `list()` — the function to create new lists — is broken until you restart your interpreter.

**Best practice:**  
Always avoid naming your variables after built-in Python functions or classes.

---

## An overview: How Python Looks for a Variable

Imagine the following kitchen hierarchy:

- First, the chef looks on their **own table** (Local).
- If it's not there, they ask the **head chef** (Enclosing).
- If still not found, they go to the **main pantry** (Global).
- As a last resort, they contact the **universal supplier** (Built-in).

If the ingredient is still not found after all that searching, the chef gives up and throws an error:

```python
NameError: name 'ingredient' is not defined
```

---

## Mini Project for the Day: Building a Kitchen Management System

Build a **simple kitchen app** that demonstrates different scopes.

**Requirements:**

- Create a global pantry list.
- Create a kitchen() function where ingredients are added to a local working table.
- Create a nested chef_station() function that uses the local ingredients.
- Use a built-in function like `len()` correctly.


```python
# Global pantry
pantry = ["salt", "pepper", "oil"]

def kitchen():
    table = ["onion", "garlic"]  # Local ingredients

    def chef_station():
        print("Working with:", table)
        print("Pantry has:", pantry)
        print("Total pantry items:", len(pantry))

    chef_station()

kitchen()

print("Global pantry is still:", pantry)
```

**Expand it by:**
- Adding functions to add/remove items from pantry.
- Handling an attempt to modify pantry without global keyword, then fixing it properly.

---

## Assignment

Write a program simulating a small restaurant kitchen:

- Define a **global** stock of ingredients.
- Inside a `kitchen()` function:
  - Create a **local** list of today's menu ingredients.
  - Nest a `chef()` function inside `kitchen()` that:
    - Prints the menu ingredients (local).
    - Prints the available stock (global).
    - Uses a built-in function like `sorted()`, `len()`, or `max()`.
- Ensure proper use of **global keyword** if modifying the global stock.

**Bonus:**
- Add an error handler that catches a `NameError` if an undefined variable is used.

---

## Best Practices to Follow

- Prefer **local variables** as much as possible to avoid accidental changes across your code.
- Use the **global** keyword **only when absolutely necessary**.
- Never name variables after **built-in names** like `list`, `sum`, or `input`.
- Understand **nested functions** and the **enclosing scopes** properly for complex applications.
- Keep your code **modular** and **organized**, just like a real kitchen station.

---

If you follow today's lesson properly, you will not only understand variable visibility deeply but also start writing more professional, bug-free Python programs. See ya tomorrow.
