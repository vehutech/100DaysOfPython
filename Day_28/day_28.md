---
# **Day 28: The Many Faces of Code — Polymorphism in Python** 🎭🐍
---
Picture this: you're at a theater where actors switch between multiple roles in the same play — sometimes they're the hero, sometimes the villain, but they all respond when the director yells "Action!" That's polymorphism in a nutshell — objects taking many forms while responding to the same command.
---
**## 🎯 Objectives**
* Understand the concept of polymorphism in Python
* Master two approaches to polymorphism: inheritance and duck typing
* Learn to write flexible code that works with different object types
* Implement polymorphic behavior in practical examples
---
**## 🎭 What is Polymorphism?**
The word "polymorphism" comes from Greek:
* "Poly" = Many
* "Morphe" = Form

In programming, it allows objects of different classes to be treated as objects of a common superclass. It's the ability to present the same interface for different underlying forms.
---
**## 🧬 Approach #1: Polymorphism through Inheritance**
With inheritance-based polymorphism, we can process objects of different subclasses through their common parent type.

**### 🔹 The Shape Hierarchy**
```python
from abc import ABC, abstractmethod

class Shape:
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.142 * self.radius ** 2

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return self.base * self.height * 0.5

class Pizza(Circle):
    def __init__(self, topping, radius):
        super().__init__(radius)
        self.topping = topping
```

**### 🔸 The Magic of Polymorphism**
```python
shapes = [
    Circle(4), Square(5), Triangle(6, 7), Pizza("pepperoni", 15)
]

for shape in shapes:
    print(f"{shape.area()}cm²")
```

Even though each shape calculates its area differently, we can treat them all as "shapes" and call the same method on each. The right implementation gets called based on the actual object type!
---
**## 🦆 Approach #2: Duck Typing — "If it quacks like a duck..."**

Duck typing is a more Python-specific approach to polymorphism. The philosophy is simple:

> "If it walks like a duck and quacks like a duck, then it probably is a duck."

With duck typing, we don't care about the object's type or inheritance — we only care that it has the methods and attributes we need.

```python
class Animal:
    alive = True

class Dog(Animal):
    def speak(self):
        print("WOOF!")

class Cat(Animal):
    def speak(self):
        print("MEOW!")

class Car:
    alive = False
    
    def speak(self):
        print("Hoooooorn!!")

animals = [
    Dog(), Cat(), Car()
]

for animal in animals:
    animal.speak()
    print(f"Alive: {animal.alive}")
```

Notice how `Car` isn't even an `Animal`, but it still works in our loop because it has the `speak()` method and `alive` attribute we're using. That's the power of duck typing!
---
**## 🚀 Why Polymorphism Matters**

Polymorphism makes your code:
1. **More flexible** — Write one function that works with many types
2. **More modular** — Extend your system without changing existing code
3. **More readable** — Focus on what objects do, not what they are

Think of it as coding to an interface rather than to an implementation. Your code becomes more about behavior than types.
---
**## 💡 Real-World Applications**

* **File Systems**: Different file types can all implement an "open" method
* **Payment Processing**: Credit cards, PayPal, and cryptocurrencies can all implement a "process_payment" method
* **UI Elements**: Buttons, sliders, and checkboxes can all implement "render" and "handle_click" methods
* **Game Characters**: Different character classes can all implement "attack", "defend", and "move" methods
---
**## 📝 Assignment**
**### Task:**
Create a `MediaPlayer` system with:
* A base `MediaFile` class (or interface)
* Different file types: `MP3File`, `WAVFile`, and `VideoFile`
* Each should implement `play()`, `stop()`, and `get_duration()`
* Create a `Playlist` class that can work with any media file type

**Bonus challenge:** Add a completely different class like `StreamingContent` that isn't a `MediaFile` but still works in your `Playlist` through duck typing.
---
**## 🤔 Food for Thought**

"Programming to an interface, not an implementation" is a fundamental design principle. How does polymorphism help you achieve this? Can you think of cases where strict type checking might be better than duck typing?
---