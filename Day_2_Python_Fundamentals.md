## **Day 2: Python Fundamentals** 🍽️

Welcome back to the kitchen of programming! Today, we're going to cook up some Python fundamentals that will make your journey smoother. Think of programming like preparing a meal—each ingredient (data type) has a purpose, every tool (loops, conditionals) makes the process efficient, and the final dish (a working program) is a masterpiece of combined efforts.

---

## **🍜 Understanding Data Types - The Ingredients of Python**

Every great meal starts with good ingredients. In programming, our ingredients are **data types**. If you use the wrong ingredient in a dish, things can go very wrong—imagine using salt instead of sugar in a cake! Similarly, using the wrong data type in Python can cause errors.

### **🍎 Common Data Types in Python**

|Data Type|Example|Kitchen Analogy|
|---|---|---|
|**String (`str`)**|`"Hello"` or `'Python'`|The name of the dish (e.g., "Spaghetti")|
|**Integer (`int`)**|`10`, `-5`, `100`|The number of eggs needed (e.g., 2 eggs)|
|**Float (`float`)**|`3.14`, `0.99`, `-2.5`|The weight of flour (e.g., 2.5 kg)|
|**Boolean (`bool`)**|`True`, `False`|The oven is on or off (`True` or `False`)|

---

### **🛠 Checking Data Types - Knowing Your Ingredients**

Before cooking, you check if you have sugar or salt, right? In Python, you can check a variable’s type using `type()`.

```python
name = "Chef John"
age = 25
gpa = 3.2
isCooking = True

print(type(name))  # <class 'str'>
print(type(age))   # <class 'int'>
print(type(gpa))   # <class 'float'>
print(type(isCooking))  # <class 'bool'>
```

---

## **🔄 Type Conversion - Transforming Ingredients**

Sometimes, you need to convert ingredients—like melting chocolate or grinding sugar into powder. In Python, this is called **typecasting**.

```python
# Converting float to int
gpa = 3.2
gpa_int = int(gpa)  # Removes decimal
print(gpa_int)  # Output: 3

# Converting int to string
age = 25
age_str = str(age)  
print("I am " + age_str + " years old.")  # Output: I am 25 years old.
```

---

## **🖥 Taking User Input - Asking the Customer**

In a restaurant, you take orders from customers. Similarly, in Python, we collect data from users using `input()`.

```python
name = input("What is your name? ")
print("Hello, " + name + "! Welcome to Python Kitchen.")
```

By default, `input()` returns a string. But if you need a number, convert it:

```python
age = int(input("Enter your age: "))  
print("Next year, you will be", age + 1)
```

---

## **🔍 Conditional Statements - Making Cooking Decisions**

Cooking is full of decision-making:

- If the food is too salty, add more water.
    
- If the oven is too hot, reduce the temperature.
    

In Python, we use `if`, `else`, and `elif` to make decisions.

```python
temperature = int(input("Enter the temperature: "))

if temperature > 30:
    print("It's hot outside! Stay hydrated. 🌞")
elif temperature >= 20:
    print("The weather is nice. 😊")
else:
    print("It's cold! Wear a jacket. ❄️")
```

---

## **🔁 Loops - Cooking in Batches**

Loops allow us to repeat actions, just like a chef flipping pancakes one by one.

### **✅ `for` Loop - Cooking a Fixed Number of Items**

```python
for i in range(1, 6):
    print(f"Serving pancake {i}")
```

### **✅ `while` Loop - Cooking Until the Task is Done**

```python
count = 0
while count < 3:
    print("Stirring the soup...")
    count += 1
```

---

## **🎮 Mini-Game: Guess the Secret Ingredient!**

This game ties everything together—**strings, integers, floats, booleans, loops, and conditionals.**

### **📝 Game Rules**

1. The computer selects a **secret ingredient** (a number between 1 and 10).
    
2. The user guesses the ingredient.
    
3. The game provides hints like "Too much!" or "Too little!"
    
4. The game continues until the correct ingredient is guessed.
    
5. The game records the number of attempts.
    

---

### **💻 Full Code Implementation**

```python
import random  # Importing the random module

print("🍽️ Welcome to the Secret Ingredient Game!")
print("I've chosen an ingredient amount (1-10). Can you guess it?")

# Generate a random secret ingredient amount
secret_number = random.randint(1, 10)

attempts = 0  # Integer: Tracks number of attempts
guessed_correctly = False  # Boolean: Checks if user guessed correctly

while not guessed_correctly:  # Loop continues until user guesses correctly
    guess = int(input("Enter your guess: "))  # User's guess (converted to int)
    attempts += 1  # Increment attempts

    if guess == secret_number:
        guessed_correctly = True  # Set to True, loop ends
        print(f"🎉 Correct! You guessed in {attempts} attempts.")
    elif guess < secret_number:
        print("Too little! Try again. 🥄")
    else:
        print("Too much! Try again. 🍽️")
```

---

## **🛠 Code Explanation (Step by Step)**

### **1️⃣ Importing the Random Module**

```python
import random
```

- This allows us to randomly choose a secret ingredient.
    

### **2️⃣ Displaying the Welcome Message**

```python
print("🍽️ Welcome to the Secret Ingredient Game!")
```

- Prints an introduction.
    

### **3️⃣ Generating the Secret Ingredient**

```python
secret_number = random.randint(1, 10)
```

- Picks a random number between 1 and 10.
    

### **4️⃣ Initializing Variables**

```python
attempts = 0  
guessed_correctly = False
```

- `attempts` keeps track of how many tries the user has taken.
    
- `guessed_correctly` determines when to stop.
    

### **5️⃣ `while` Loop - Running the Game**

```python
while not guessed_correctly:
```

- Runs **until** the user guesses correctly.
    

### **6️⃣ Taking User Input**

```python
guess = int(input("Enter your guess: "))
```

- Converts the user’s guess to an integer.
    

### **7️⃣ Counting Attempts**

```python
attempts += 1
```

- Increases the attempt count after each guess.
    

### **8️⃣ Checking the User’s Guess**

```python
if guess == secret_number:
    guessed_correctly = True  
    print(f"🎉 Correct! You guessed in {attempts} attempts.")
```

- If the guess is correct, the loop stops.
    

### **9️⃣ Providing Hints**

```python
elif guess < secret_number:
    print("Too little! Try again. 🥄")
else:
    print("Too much! Try again. 🍽️")
```

- If the guess is too low, the game hints to **increase** it.
    
- If the guess is too high, the game hints to **reduce** it.
    

---

## **🎯 What We Cooked Up Today**

✔ **Data Types**: Strings, Integers, Floats, Booleans  
✔ **User Input**: Using `input()` for interaction  
✔ **Conditionals**: Using `if`, `elif`, `else` for decision-making  
✔ **Loops**: Using `while` and `for` for repetition  
✔ **Building a Real-World Mini-Game!**

### **🛠 Challenge:**

Modify the game so that **instead of numbers, the user guesses a secret ingredient (like 'sugar', 'salt', or 'pepper').**

---

🔥 **Tomorrow’s Lesson: Functions & Error Handling!**  
