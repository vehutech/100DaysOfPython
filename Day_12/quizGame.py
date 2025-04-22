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
