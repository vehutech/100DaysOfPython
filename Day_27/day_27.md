---

# **Day 27: Chef Super's Secret Sauce — The `super()` Function in Python** 👨‍🍳🧪

---

Every great kitchen has a **Master Chef** who sets the basic recipe — but sometimes the Sous Chef needs to **enhance** that recipe with a twist. That’s exactly what Python’s `super()` lets you do!

---

## 🎯 Objectives

* Understand what `super()` does in Python
* Learn how to use `super()` in constructors and methods
* Understand method overriding and extension
* Fix a small bug and polish the output formatting

---

## 🧂 What's `super()`?

Think of `super()` as **calling the original version of a recipe** from your head chef. It lets a subclass use or enhance the parent class's methods.

```python
super().method()
```

Or in constructors:

```python
super().__init__(...)
```

---

## 🍳 Kitchen Blueprint: Shapes & Their Recipes

### 🔹 Base Class: `Shape`

```python
class Shape:
    def __init__(self, color, is_filled):
        self.color = color
        self.is_filled = is_filled

    def describe(self):
        fill_status = "filled" if self.is_filled else "not filled"
        print(f"It is {self.color} and {fill_status}.")
```

A generic recipe describing **any shape**.

---

### 🔸 `Circle`, `Square`, `Triangle`: Custom Recipes

```python
class Circle(Shape):
    def __init__(self, color, is_filled, radius):
        super().__init__(color, is_filled)  # Call parent’s __init__
        self.radius = radius

    def describe(self):
        super().describe()  # Extend parent's method
        area = 3.14 * self.radius ** 2
        print(f"It is a circle with an area of {area:.2f} cm².")
```

```python
class Square(Shape):
    def __init__(self, color, is_filled, width):
        super().__init__(color, is_filled)
        self.width = width

    def describe(self):
        super().describe()
        area = self.width ** 2
        print(f"It is a square with an area of {area:.2f} cm².")
```

```python
class Triangle(Shape):
    def __init__(self, color, is_filled, width, height):
        super().__init__(color, is_filled)
        self.width = width
        self.height = height

    def describe(self):
        super().describe()
        area = 0.5 * self.width * self.height
        print(f"It is a triangle with an area of {area:.2f} cm².")
```

---

## 🧪 Taste Testing the Recipes

```python
circle = Circle("blue", True, 5)
square = Square("red", False, 6)
triangle = Triangle("yellow", True, 7, 8)

print("\n🔵 Circle:")
circle.describe()

print("\n🟥 Square:")
square.describe()

print("\n🔺 Triangle:")
triangle.describe()
```

---

## 🍱 Output

```
🔵 Circle:
It is blue and filled.
It is a circle with an area of 78.50 cm².

🟥 Square:
It is red and not filled.
It is a square with an area of 36.00 cm².

🔺 Triangle:
It is yellow and filled.
It is a triangle with an area of 28.00 cm².
```

---

## 📝 Assignment

### Task:

Create a `Vehicle` base class and extend it with:

* `Car` (has `number_of_doors`)
* `Motorcycle` (has `engine_cc`)
* `describe()` should use `super()` and add specific details.

---