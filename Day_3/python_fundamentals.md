
Welcome to **Day 3** of our Python challenge! 🚀 Today, we’ll:  
✅ **Deepen our understanding** of Python input handling and calculations  
✅ **Explore functions and error handling** to make our programs robust  
✅ **Build an interactive Madlibs game** that improves user experience

So, **let’s get cooking**! 🍽️

---

## 🏗 **Exercise 1: Rectangle Area Calculation**

### **Problem: Why does the first code fail?**

Imagine you’re a chef 🔪 preparing a dish. You need **exact ingredient measurements**, but your assistant hands you **ingredients in a text format** instead of numbers. That’s exactly what’s happening in this code!

```python
length = input("Enter the length:  ")
width = input("Enter the width: ")
area = length * width  # ❌ ERROR: Strings can't be multiplied this way!

print(area)
```

### **Fixing the Type Issue**

To ensure **proper measurements**, we **convert** input strings to numbers using `float()` before performing calculations:

```python
length = float(input("Enter the length:  "))
width = float(input("Enter the width: "))
area = length * width

print(f"The area is: {area} cm²")
```

### **Test it Yourself!**

Try **entering `5` and `6`**. The output should be:

```
The area is: 30.0 cm²
```

🔹 **Lesson:** Just like in cooking, always ensure **ingredients (inputs) are in the right format** before using them!

---

## 🛒 **Exercise 2: Shopping Cart Program**

### **Scenario: Handling a Shopping Cart**

Let’s simulate **buying items in a store**. A customer enters:  
1️⃣ The item name 🛍️  
2️⃣ The price 💰  
3️⃣ The quantity 🔢

Here’s the **basic** version:

```python
item = input("What item would you like to buy?: ")
price = float(input("What is the price?: "))
qty = int(input("How many would you like?: "))

total = price * qty

print(total)
```

### **Output Example**

If the customer buys **3 apples** at **₦200 each**, they should see:

```
600.0
```

### **Improving the Output**

Let’s make it **user-friendly** by adding **better print formatting**:

```python
print(f"You have bought {qty} {item}/s for ₦{total}")
```

🔹 **Lesson:** Just like a restaurant **prints a bill with item details**, always ensure your program gives **clear outputs!**

---

## 🎭 **Building a Madlibs Game**

Now, let’s **get creative** and build a **Madlibs game**! 🎮

### **📜 What is Madlibs?**

It’s a **fun word game** where players fill in blanks with random words to create hilarious stories!

---

## 🍳 **Step 1: Basic Madlibs Game (No Functions Yet)**

Here’s a simple **version** to get started:

```python
noun = input("Enter a noun: ")
verb = input("Enter a verb: ")
adjective = input("Enter an adjective: ")

story = f"Today, I saw a {adjective} {noun} that decided to {verb} all day long!"

print("\n🎭 YOUR MADLIBS STORY 🎭")
print(story)
```

### **Test it!**

If you enter:

```
noun: cat  
verb: dance  
adjective: funny  
```

The output could be:

```
🎭 YOUR MADLIBS STORY 🎭
Today, I saw a funny cat that decided to dance all day long!
```

✅ **Nice! But we can improve this!**

---

## 🔄 **Step 2: Using Functions for Reusability**


### 🧩 **What is a Function in Python?**

A **function** is like a reusable recipe in your kitchen. 🍲  
Instead of writing the same steps again and again, you **bundle** them into a function and **call it whenever needed**.

🔹 **Syntax Example:**

```python
def greet():
    print("Hello, welcome to the program!")

greet()  # Calling the function
```

def     ----->   means function definition
greet   ----->   is the name of the function


🔸 **Why use functions?**

- To keep your code clean and organized 💅
    
- To avoid repeating yourself (DRY principle) 🔁
    
- To break down big tasks into smaller, manageable parts 🧩
    

🔧 In our **Madlibs game**, we used functions to:

- Ask for user input
    
- Build the story
    
- Display results
    

---

### 🚨 **What is Error Handling in Python?**

Imagine you’re cooking and someone hands you **spoiled ingredients** 🥴 — you'd need a way to **catch that mistake** before using them!

In programming, **error handling** helps you deal with unexpected problems (like empty inputs or wrong data types) **without crashing the program**.

🔹 **Example using a loop to catch empty input:**

```python
def get_word(word_type):
    while True:
        word = input(f"Enter a {word_type}: ").strip()
        if word:
            return word
        print("❌ Error: You must enter a value!")
```

🔸 **Why use error handling?**

- To avoid bugs and crashes 🐛💥
    
- To guide users with helpful messages 📣
    
- To build professional, smooth programs 💼
    

---
lets try to implement our knowledge on functions with our midlibs program

Instead of writing the **same code repeatedly**, let’s **put it in a function**:

```python
def madlibs():
    noun = input("Enter a noun: ")
    verb = input("Enter a verb: ")
    adjective = input("Enter an adjective: ")

    story = f"Today, I saw a {adjective} {noun} that decided to {verb} all day long!"
    
    print("\n🎭 YOUR MADLIBS STORY 🎭")
    print(story)

# Call the function to start the game
madlibs()
```

### **Why use functions?**

✔ Code is **cleaner and reusable**  
✔ We can **call the function multiple times**

---

## 🚀 **Step 3: Adding Error Handling**

Now, let’s **handle user mistakes**. What if they accidentally press **Enter** without typing anything?

```python
def get_word(word_type):
    while True:
        word = input(f"Enter a {word_type}: ").strip()
        if word:
            return word  # Return valid input
        print("❌ Oops! You must enter something!")

def madlibs():
    noun = get_word("noun")
    verb = get_word("verb")
    adjective = get_word("adjective")

    story = f"Today, I saw a {adjective} {noun} that decided to {verb} all day long!"
    
    print("\n🎭 YOUR MADLIBS STORY 🎭")
    print(story)

# Start the game
madlibs()
```

🔹 **Lesson:** **Always validate user input!** Just like a chef checks ingredients before cooking, we should **prevent empty inputs!**

---


## 🎮 **Final Features in Our Game**

✅ **Multiple random stories** each time you play  
✅ **Error handling** to prevent empty inputs  
✅ **Stories saved** to a file for future reference

---

## 🎯 **Day 3 Summary**

✔ **Learned type conversion & why input defaults to strings**  
✔ **Built a shopping cart system & improved output readability**  
✔ **Created a Madlibs game with functions & error handling**  
✔ **Added story storage for better user experience**

---
### 🏆 **Assignment – Python & Madlibs Challenge!** 🎭

Alright, Python warriors! ⚔️ Time to put today's knowledge into practice.

---

## 📝 **Assignment 1: Enhanced Shopping Cart** 🛒

Modify the **Shopping Cart Program** to include:  
✅ A **discount system** (e.g., 10% off if total > ₦5000)  
✅ A **delivery fee** (₦500 if total is below ₦3000)  
✅ A **receipt-like output**

🎯 **Example Output:**

```
You bought 3 Apples for ₦200 each.
Total: ₦600
Delivery Fee: ₦500
Final Amount: ₦1100
```

---

## 🎭 **Assignment 2: Advanced Madlibs Game**

Modify the **Madlibs Game** to:  
✅ **Include 5 different sentence templates** (randomly chosen)  
✅ **Ask for more word categories** (e.g., adverb, place, animal)  
✅ **Give users a chance to replay without restarting the program**  
✅ **Add a fun "score" system** based on the length of words used

🎯 **Example Output:**

```
Today, a curious monkey danced wildly at the beach!
Score: 25 points! 🎉
Do you want to play again? (yes/no)
```

---

## 🔥 **Bonus Challenge (For Extra Reputation!)**

🔹 **Assignment 3: Store & Retrieve User Stories**  
Modify the Madlibs game to **save stories in a file** (`madlibs_stories.txt`).  
Then, add a **"View Past Stories"** option before starting the game.

🎯 **Example:**

```
1. Play Madlibs  
2. View Past Stories  
Enter your choice:  
```

---

### ⏳ **Deadline:** Before Day 4 session begins!

📢 **Submit by sharing your code and test results in the group!** 🚀

Let’s see who can come up with the **funniest Madlibs story & smartest shopping cart system!** 😆💡
