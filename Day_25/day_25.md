---

# Day 25: Cooking with Inheritance — The Family Recipe 📜🐶🐱🐭

---

In every great kitchen, some recipes are handed down from generation to generation. Instead of rewriting them from scratch, new chefs build on the foundation. That’s **Inheritance** in Object-Oriented Programming!

You create a **base recipe** (parent class) and let other **derived recipes** (child classes) inherit the ingredients and steps — customizing only what's needed.

---

## 🍳 What is Inheritance?

* **Parent Class (Base Recipe)** = A general blueprint (e.g., “Animal”)
* **Child Class (Derived Recipe)** = A specific version (e.g., “Dog”, “Cat”) that inherits and can override parts of the base
* **Inheritance** = Reusing code to reduce duplication

---

## 🎯 Objectives

* Understand how classes inherit from one another.
* Use the `super()` function to access parent class behavior.
* Override methods in child classes.
* Create multiple subclasses from a common base.

---

## 🧪 Ingredient 1: Base Recipe — The Animal Class

```python
class Animal:
    def __init__(self, name):
        self.name = name
        self.is_alive = True

    def eat(self):
        print(f"{self.name} is eating.")

    def sleep(self):
        print(f"{self.name} is sleeping.")
```

This is the **parent class** — a generic animal with basic behaviors.

---

## 🧪 Ingredient 2: Derived Recipes — Dog, Cat, Mouse

```python
class Dog(Animal):
    def bark(self):
        print(f"{self.name} says Woof!")

class Cat(Animal):
    def meow(self):
        print(f"{self.name} says Meow!")

class Mouse(Animal):
    def squeak(self):
        print(f"{self.name} says Squeak!")
```

Each **child class** inherits everything from `Animal` but adds its own flavor.

---

## 🧪 Ingredient 3: Cooking Up Objects

```python
dog = Dog("Scooby")
cat = Cat("Garfield")
mouse = Mouse("Mickey")

print(dog.name)           # Scooby
print(cat.is_alive)       # True

cat.eat()                 # Garfield is eating.
cat.sleep()               # Garfield is sleeping.
cat.meow()                # Garfield says Meow!
```

Notice how `eat()` and `sleep()` are **inherited**, while `meow()` is **specific** to cats.

---

## 🍱 Kitchen Table Example: Animal Parade

```python
animals = [Dog("Buddy"), Cat("Whiskers"), Mouse("Jerry")]

for animal in animals:
    animal.eat()
    animal.sleep()
    if isinstance(animal, Dog):
        animal.bark()
    elif isinstance(animal, Cat):
        animal.meow()
    elif isinstance(animal, Mouse):
        animal.squeak()
```

---

## 🧁 Mini Project: Vehicle Lineup

Create a `Vehicle` parent class with:

* `brand` (string)
* `speed` (int)
* Method: `start()`

Then, create `Car`, `Bike`, and `Truck` child classes that override `start()`.

### Example:

```python
class Vehicle:
    def __init__(self, brand, speed):
        self.brand = brand
        self.speed = speed

    def start(self):
        print(f"{self.brand} is starting at {self.speed} km/h.")

class Car(Vehicle):
    def start(self):
        print(f"🚗 {self.brand} car revs up and goes!")

class Bike(Vehicle):
    def start(self):
        print(f"🏍️ {self.brand} bike zooms off!")

class Truck(Vehicle):
    def start(self):
        print(f"🚚 {self.brand} truck rumbles down the road!")

v1 = Car("Toyota", 120)
v2 = Bike("Yamaha", 90)
v3 = Truck("Volvo", 60)

for vehicle in [v1, v2, v3]:
    vehicle.start()
```

---

## 📝 Assignment

1. Create a parent class `Employee` with attributes: `name`, `role`, `salary`.
2. Add a method `work()` that prints the employee’s job description.
3. Create child classes `Manager`, `Developer`, and `Designer`.
4. Override the `work()` method in each child to reflect their specific duties.
5. Create objects of each type and call their methods.

---