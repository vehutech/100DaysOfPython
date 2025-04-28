Assignment

---

Chef Lina’s restaurant is getting busier.  
This month, she's planning special theme nights — depending on the day, the kitchen prepares a different **style of cuisine**:

- Monday and Tuesday → Italian Night
- Wednesday and Thursday → Mexican Night
- Friday → Seafood Night
- Saturday → Grill and Barbecue Night
- Sunday → Family Buffet Day

Sometimes, due to unexpected events (like a holiday), there might be **no planned event**.

Chef Lina wants you — her apprentice — to **build a simple program** using `match-case` to **help the kitchen plan the day's menu**.

---

## The Task

**Write a Python program** that:

1. Asks the user to input a day of the week.
2. Matches the day to the correct kitchen plan using `match-case`.
3. If the day is not valid, politely respond with “No events planned.”

---

## Example Interaction

```plaintext
Enter the day of the week: Wednesday
Tonight is Mexican Night! Prepare the tacos and burritos.

Enter the day of the week: Sunday
Tonight is Family Buffet Day! Prepare a variety of dishes.

Enter the day of the week: Holiday
No events planned.
```

---

## Starter Guide

Use `input()`, `match-case`, and group days where necessary using `|` (the "or" symbol).

Hint: Think of Monday and Tuesday belonging together, Wednesday and Thursday together, etc.

---

## Extra Challenge (Optional)

- Make the program **case-insensitive** (so "wednesday", "Wednesday", "WEDNESDAY" all work).
- Add **default dishes** that should always be ready, no matter the day (like water and bread).

---