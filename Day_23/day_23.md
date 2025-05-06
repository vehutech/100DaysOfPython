---

# Day 23: Building a Python Hangman Game 🎯

---

Just like a chef preparing a mystery soup, today you're guessing **the secret ingredient** — one letter at a time! But be careful... too many wrong guesses and your hangman (kitchen assistant!) gets hung.

We're building a **command-line Hangman game** using core programming concepts: loops, conditions, functions, lists, and external files (`wordlist.py`, `hangman_art.py`).

---

* **Correct guess**: You add the right spice to the pot.
* **Wrong guess**: The soup sours a little, and the assistant’s fate worsens.
* **Win**: You guessed the word before hanging the assistant.
* **Lose**: Game over — the assistant didn’t survive!

Think of this as a **puzzle dish** — revealing just enough to keep the player guessing.

---

## 🎯 Objectives

* Use sets and lists to track guesses and word progress.
* Show visual feedback with ASCII art.
* Use modular functions for clarity.
* Handle replay logic cleanly.

---

## 🧪 Ingredients: Code Functions

### 1. Display the Hangman Figure

```python
def display_man(wrong_guesses):
    for line in hangman_art[wrong_guesses]:
        print(line)
    print("***************")
```

Shows the assistant’s current (unfortunate) state.

---

### 2. Show the Word Hint

```python
def display_hint(hint):
    print(" ".join(hint))
```

Prints the word with underscores and guessed letters revealed.

---

### 3. Show the Full Answer

```python
def display_answer(answer):
    print(" ".join(answer))
```

Used at the end to show the full word, whether you won or lost.

---

### 4. Ask if Player Wants to Play Again

```python
def play_again():
    while True:
        again = input("Do you want to play again? (y/n): ").lower()
        if again in ['y', 'n']:
            return again == 'y'
        else:
            print("Invalid input, please enter 'y' or 'n'.")
```

Keeps your game interactive and polite!

---

### 5. Reset for a New Game

```python
def reset_game():
    answer = random.choice(words)
    hint = ["_"] * len(answer)
    wrong_guesses = 0
    guessed_letters = set()
    return answer, hint, wrong_guesses, guessed_letters
```

Selects a new word and resets everything.

---

### 6. The Main Game Loop

```python
def main():
    answer, hint, wrong_guesses, guessed_letters = reset_game()
    is_running = True

    while is_running:
        display_man(wrong_guesses)
        display_hint(hint)
        guess = input("Enter a letter: ").lower()

        if len(guess) != 1 or not guess.isalpha():
            print("Invalid input.")
            continue

        if guess in guessed_letters:
            print(f"{guess} is already guessed.")
            continue

        guessed_letters.add(guess)

        if guess in answer:
            for i in range(len(answer)):
                if answer[i] == guess:
                    hint[i] = guess
        else:
            wrong_guesses += 1

        if "_" not in hint:
            display_man(wrong_guesses)
            display_answer(answer)
            print("🎉 YOU WIN!!")
            is_running = play_again()
            if is_running:
                answer, hint, wrong_guesses, guessed_letters = reset_game()
            else:
                print("Thanks for playing!")

        elif wrong_guesses >= len(hangman_art) - 1:
            display_man(wrong_guesses)
            display_answer(answer)
            print("💀 YOU LOSE!")
            is_running = play_again()
            if is_running:
                answer, hint, wrong_guesses, guessed_letters = reset_game()
            else:
                print("Thanks for playing!")
```

Runs your soup-guessing challenge from start to finish!

---

## 🧁 Mini Project: Custom Word Packs

Try this extension:

1. Create **difficulty levels** (Easy, Medium, Hard).
2. Store 3 different word lists in your `wordlist.py` file.
3. Ask the player to choose a difficulty before `reset_game()`.

Example:

```python
# wordlist.py
easy_words = ["apple", "hat", "car"]
medium_words = ["banana", "python", "guitar"]
hard_words = ["pneumonia", "dichotomy", "zookeeper"]
```

---

## 📝 Assignment

1. Add a feature to show all guessed letters so far.
2. If a letter is guessed again, show a custom warning.
3. Save win/loss stats in a `stats.txt` file — and print them at the end of each game.
4. Bonus: Allow guessing the full word at once (`input("Guess the word or a letter: ")`).

---