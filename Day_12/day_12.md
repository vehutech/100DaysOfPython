# 🍳 **Day 12: Cooking Up a Python Quiz Game**

Welcome to **Chef Py’s Kitchen of Knowledge**! Today, we're baking a **Quiz Game Soufflé** — light, fun, and full of learning. Let’s examine the ingredients (code) line-by-line, refine our recipe, and plate it professionally.

---

```python
# Python Quiz Game - Day 12
# 🧑‍🍳 Welcome to the Quiz Kitchen!

# Step 1: Our Questions - The core ingredients of our dish
questions = [
    "How many elements are in the periodic table?",
    "Which animal lays the largest eggs?",
    "What is the most abundant gas in Earth's atmosphere?",
    "How many bones are in the human body?",
    "Which planet in the solar system is the hottest?"
]

# Step 2: Our Options - The side dishes for each question
options = [
    ["A. 116", "B. 117", "C. 118", "D. 119"],
    ["A. Whale", "B. Crocodile", "C. Elephant", "D. Ostrich"],
    ["A. Nitrogen", "B. Oxygen", "C. Carbon Dioxide", "D. Hydrogen"],
    ["A. 206", "B. 207", "C. 208", "D. 209"],
    ["A. Mercury", "B. Venus", "C. Earth", "D. Mars"]
]

# Step 3: Our correct answers - The secret spices!
answers = ["C", "D", "A", "A", "B"]

# Step 4: Prepping the mixing bowl
guesses = []
score = 0

# Step 5: Let the cooking begin!
for index, question in enumerate(questions):
    print("\n--------------------------------------------------")
    print(f"Q{index + 1}: {question}")

    # Display options for current question
    for option in options[index]:
        print(option)

    # Get user's guess and store it
    guess = input("Enter your answer (A, B, C, D): ").strip().upper()
    guesses.append(guess)

    # Taste test: Compare guess with correct answer
    if guess == answers[index]:
        print("✅ Correct! Well done, Chef!")
        score += 1
    else:
        print("❌ Incorrect.")
        print(f"👉 The correct answer was: {answers[index]}")

# Step 6: Plating the result (the grand reveal!)
print("\n====================")
print("🍽️     RESULTS     ")
print("====================")

print("✅ Correct Answers: ", " ".join(answers))
print("❓ Your Guesses:     ", " ".join(guesses))

percentage = (score / len(questions)) * 100
print(f"\n🥇 Score: {score} / {len(questions)}  ➡️ {percentage:.1f}%")

# 🍴 Bon Appétit!
```

---

## 🔍 Let’s Break It Down Like a Chef!

### 1. **Data Structures = Kitchen Inventory**
We use lists (`[]`) for storing:
- `questions` - the main dish
- `options` - seasoning choices per dish
- `answers` - what the judge (our logic) expects
- `guesses` - how the sous chef (player) responded

### 2. **Looping Over Questions = Serving Each Dish**
```python
for index, question in enumerate(questions):
```
- `enumerate()` gives us both the **position** and the **question** — just like plating dish #1, #2, etc.

### 3. **Print Options = Displaying the Menu**
```python
for option in options[index]:
    print(option)
```
- We show all options for that question. Think of it like displaying a spice rack before seasoning.

### 4. **User Input = The Chef Makes a Choice**
```python
guess = input("Enter your answer (A, B, C, D): ").strip().upper()
```
- `.strip()` removes any accidental whitespaces
- `.upper()` ensures consistency (caps only) for evaluation

### 5. **Taste Testing = Compare Answers**
```python
if guess == answers[index]:
```
- If the chef’s guess matches the secret spice, score goes up! Yum! 😋

### 6. **Score Calculation = Final Dish Rating**
```python
percentage = (score / len(questions)) * 100
```
- This gives a neat percent, like a Michelin star rating.

---

## 🧼 Cleanup & Best Practices Used

| Original | Improvement | Why? |
|---------|-------------|------|
| `()` for tuples | `[]` for lists | Better when working with mutable collections like options |
| Manual index tracking (`questionNum`) | `enumerate()` | Cleaner and Pythonic |
| `input().upper()` | `strip().upper()` | Handles trailing spaces |
| Code in one block | Modular with comments | Easier to read and debug |

---

## 🍰 Chef’s Bonus Tips

- **Shuffle the questions** using `random.shuffle()` to spice up the quiz.
- **Add difficulty levels** (easy, medium, hard) for meal variation.
- **Add a score summary or feedback message** depending on performance.
- **Use `colorama` or `rich` for colored output** like a MasterChef display.