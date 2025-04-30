---

# Day 19: The Power of `if __name__ == "__main__"` in Python

---

In every well-run restaurant, there are two ways to use a recipe:  
1. **Cook the meal yourself directly from the recipe.**  
2. **Lend the recipe to another chef** who wants to reuse just the sauce part.

Python offers this same flexibility through a special kitchen switch:  
`if __name__ == "__main__":`

This line is a **guard clause** that decides:  
> Should we run the main course now, or just share the ingredients and instructions with someone else?

---

## What is `__name__`?

In Python, every file (module) has a built-in variable called `__name__`.

- If the script is being run **directly**, `__name__` is set to `"__main__"`.
- If the script is **imported** into another file, `__name__` is set to the **module’s name** (i.e., the file name without `.py`).

---

## Why `if __name__ == "__main__"` is Important

It gives us **control** over what code runs **when**:

- Run the script directly: execute main logic (like cooking a meal yourself).
- Import the script elsewhere: **reuse its functions or classes** without triggering its main logic.

This helps in:
- **Modularity**: Keep code reusable and testable.
- **Separation of concerns**: Logic vs. execution.
- **Preventing accidental behavior** when importing modules.

---

## Example 1: Cooking Directly

```python
# sauce.py

def make_sauce():
    print("Making a creamy garlic sauce")

if __name__ == "__main__":
    make_sauce()
```

Run this file directly:
```bash
python sauce.py
```

Output:
```
Making a creamy garlic sauce
```

---

## Example 2: Reusing the Sauce in Another Dish

```python
# main_dish.py

import sauce

print("Cooking pasta with reused sauce")
```

Output:
```
Cooking pasta with reused sauce
```

Here, the `sauce.py` functions are available, but its `make_sauce()` call inside the `if __name__ == "__main__"` block is **not executed**.

---

## Real-World Analogy (Kitchen Style 🍝)

- The `__main__` block is like a **demo recipe card** inside a cookbook.
- If you open the book directly, you can follow the whole recipe.
- If another chef borrows only one part of your recipe (e.g., just your sauce-making function), they **don’t get the demo** — just the part they need.

---

## When Should You Use It?

Use `if __name__ == "__main__":` when you want to:

- Run **test code or demos**.
- Add a **main function** to be run only when the script is executed directly.
- Keep your script **safe for imports** in other modules or test files.

---

## Example 3: With a Main Function

```python
def greet():
    print("Hello from the kitchen!")

def main():
    print("Running kitchen setup...")
    greet()

if __name__ == "__main__":
    main()
```

This pattern (`main()` function + name guard) is a best practice in Python, just like **mise en place** (prepping all ingredients) before cooking.

---

## Mini Project for the Day: Chef's Toolkit

Create a Python file `toolkit.py` with:

- A function to list available kitchen tools.
- A function to recommend a tool based on a task.
- A `main()` function to demo both functions.

Wrap the demo logic inside the `if __name__ == "__main__"` block.

Then create a second file `main_kitchen.py` that imports and uses one function from `toolkit.py` — without triggering the main demo.

---

## toolkit.py

```python
def list_tools():
    return ["Knife", "Whisk", "Spatula", "Tongs"]

def recommend_tool(task):
    if task == "flip":
        return "Use a Spatula!"
    elif task == "stir":
        return "Use a Whisk!"
    else:
        return "Try using a Knife or Tongs."

def main():
    print("Welcome to the Chef's Toolkit!")
    print("Tools available:", list_tools())
    print(recommend_tool("stir"))

if __name__ == "__main__":
    main()
```

---

## main_kitchen.py

```python
import toolkit

print("Today's tool suggestion:")
print(toolkit.recommend_tool("flip"))
```

Output when running `main_kitchen.py`:
```
Today's tool suggestion:
Use a Spatula!
```

The demo code in `toolkit.py` does not run. Clean. Modular. Professional.

---

## Assignment

Create a script called `calculator.py` with:

- Functions: `add()`, `subtract()`, `multiply()`, `divide()`.
- A `main()` function that asks the user for two numbers and an operation.
- Put the logic under `if __name__ == "__main__":` so it only runs when executed directly.

Then create `math_app.py` that:

- Imports `calculator`.
- Calls only the `multiply()` function with two numbers.

Make sure the `main()` block in `calculator.py` does not run when importing.

---

## Note:

- Always use `if __name__ == "__main__":` when writing reusable Python scripts.
- Define a `main()` function instead of writing raw code in the `if` block.
- Keep demo/testing logic separated from core logic.
- Write import-safe code to support testing, modular design, and reuse.
- Think like a chef: don’t start cooking a full meal when someone just wants your sauce recipe.

---