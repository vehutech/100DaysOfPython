---

# Day 21: Building a Python Slot Machine Game 🎰

---

Like a chef spinning a spice rack for random ingredients, today’s challenge is about **luck**, **symbols**, and a sprinkle of **randomness**!

We’ll create a fun **slot machine game** using the command line. It’ll simulate spinning reels, check for matching symbols, and payout accordingly — all from a balance a player manages carefully.

---

* **Spin** = Shuffle the kitchen counter for 3 random dishes.
* **Match** = Get 3 identical ingredients and win kitchen coins!
* **Balance** = Your pantry budget — spend it wisely, win big!

A great slot machine keeps players engaged, checks input, and handles payouts cleanly.

---

## 🎯 Objectives

* Use `random` to simulate spinning symbols.
* Work with `lists`, `dictionaries`, and control flow.
* Practice input validation and loops.
* Simulate wins, losses, and calculate payouts.

---

## 🧪 Ingredients: Code Functions

### 1. Spin the Reels

```python
import random

def spin_row():
    symbols = ["🍒", "🍉", "🍋", "🔔", "⭐"]
    return [random.choice(symbols) for _ in range(3)]
```

This selects 3 random symbols — think of it like blindly grabbing ingredients from a shuffled shelf.

---

### 2. Print the Resulting Row

```python
def print_row(row):
    print("*************")
    print(" | ".join(row))
    print("*************")
```

This cleanly displays your spin result, formatted like a classic slot machine.

---

### 3. Determine the Payout

```python
def get_payout(row, bet):
    if row[0] == row[1] == row[2]:
        symbols_and_values = {
            "🍒": 3,
            "🍉": 4,
            "🍋": 5,
            "🔔": 10,
            "⭐": 20
        }
        multiplier = 5 if row[0] == "⭐" else 1
        return symbols_and_values.get(row[0], 0) * bet * multiplier
    return 0
```

If all three ingredients match, the player earns coins based on the symbol’s value × their bet. ⭐ triggers a **jackpot multiplier**!

---

## 🧠 Why Use `if __name__ == "__main__"`?

```python
if __name__ == "__main__":
    main()
```

Just like a chef only starts cooking when they’re in the kitchen — this ensures the game only runs when the file is executed directly.

---

## 🧁 Mini Project: Fancy Casino Receipt

Extend your slot machine by:

* Writing each round to `slot_history.txt`.
* Formatting the history neatly (e.g. using tabular display).
* Add a timestamp to each spin using `datetime.now()`.

---

## 📝 Assignment

1. Multiply payouts by the bet (already implemented above).

2. Add a **jackpot bonus** for 3 ⭐ (already included with `multiplier`).

3. Save spin history to a file at the end:

   ```python
   with open("slot_history.txt", "w") as f:
       for t in transactions:
           f.write(f"{t['spin']} | Bet: ${t['bet']} | Payout: ${t['payout']}\n")
   ```

4. Bonus: Add a sound-like message on wins like `print("🎵 Ding Ding!")`

---