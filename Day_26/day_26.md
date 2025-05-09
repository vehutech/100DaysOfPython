---

# **Day 26: The Animal Kingdom Buffet — Mixing Recipes with Multiple & Multilevel Inheritance** 🐠🦅🐇

---

In a kitchen full of diverse creatures, some recipes are passed down in layers (like lasagna 🧀), while others are fused from different origins (like fusion cuisine 🌮🍣). That’s exactly what **Multilevel** and **Multiple Inheritance** are about in Python!

---

## 🍳 What’s on Today’s Menu?

* **Multilevel Inheritance** = A recipe passed down through multiple chefs 👨‍🍳👨‍🍳👨‍🍳
* **Multiple Inheritance** = A fusion dish combining two parent recipes into one plate 🍝🍕

---

## 🎯 Objectives

* Understand **multilevel** vs **multiple inheritance**
* Practice creating child classes that inherit from more than one parent
* Use inherited methods and understand method resolution order (MRO)
* Build a practical class hierarchy around animals

---

## 🧪 Ingredient 1: The Base Recipe — `Animal`

```python
class Animal:
    def eat(self):
        print("This animal is eating")

    def sleep(self):
        print("This animal is sleeping")
```

This is our **foundation** — every animal knows how to eat and sleep.

---

## 🧪 Ingredient 2: The Inherited Layers — `Prey` and `Predator`

```python
class Prey(Animal):
    def flee(self):
        print("This animal is fleeing")

class Predator(Animal):
    def hunt(self):
        print("This animal is hunting")
```

* `Prey` inherits from `Animal` and adds `flee()`
* `Predator` inherits from `Animal` and adds `hunt()`

These are like layered recipes—each new class adds more behavior!

---

## 🧪 Ingredient 3: Specific Dishes — `Rabbit`, `Hawk`, and `Fish`

```python
class Rabbit(Prey):
    pass  # Multilevel Inheritance: Rabbit → Prey → Animal

class Hawk(Predator):
    pass  # Multilevel Inheritance: Hawk → Predator → Animal

class Fish(Prey, Predator):
    pass  # Multiple Inheritance: Fish → Prey + Predator → Animal
```

* `Rabbit` can **eat**, **sleep**, and **flee**
* `Hawk` can **eat**, **sleep**, and **hunt**
* `Fish` can do **both** — it can **flee** and **hunt**!

---

## 🍱 Kitchen Table Example: Wild Buffet

```python
rabbit = Rabbit()
hawk = Hawk()
fish = Fish()

fish.flee()    # Output: This animal is fleeing
fish.hunt()    # Output: This animal is hunting
hawk.hunt()    # Output: This animal is hunting
# hawk.flee()  # ❌ Uncommenting this would raise an error: Hawk can't flee!

rabbit.eat()   # Output: This animal is eating
rabbit.sleep() # Output: This animal is sleeping

fish.eat()     # Output: This animal is eating
fish.sleep()   # Output: This animal is sleeping
```

---

## 🔍 Bonus: What’s the Method Resolution Order (MRO)?

Let’s peek into how Python decides which parent to call first.

```python
print(Fish.__mro__)
```

Outputs:

```python
(<class '__main__.Fish'>, <class '__main__.Prey'>, <class '__main__.Predator'>, <class '__main__.Animal'>, <class 'object'>)
```

This means:

1. Look in `Fish`
2. Then `Prey`
3. Then `Predator`
4. Then `Animal`

---

## 🧁 Mini Project: Amphibious Vehicles

Design a system of land and water vehicles using **multiple inheritance**.

```python
class Vehicle:
    def start(self):
        print("Vehicle is starting...")

class LandVehicle(Vehicle):
    def drive(self):
        print("Driving on land")

class WaterVehicle(Vehicle):
    def sail(self):
        print("Sailing on water")

class AmphibiousVehicle(LandVehicle, WaterVehicle):
    pass

amphi = AmphibiousVehicle()
amphi.start()  # From Vehicle
amphi.drive()  # From LandVehicle
amphi.sail()   # From WaterVehicle
```

---

## 📝 Assignment

1. Create a base class `Appliance` with methods: `turn_on()`, `turn_off()`
2. Create subclasses `Washer`, `Dryer` inheriting from `Appliance`
3. Create `WasherDryerCombo` that inherits from **both** `Washer` and `Dryer`
4. Call methods from the combo and display the MRO

---