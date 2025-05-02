---

# Day 20: Building a Python Banking Program 🏦

---

Just like a head chef managing a kitchen's finances, we need a **simple yet reliable system** to keep track of our resources — deposits, withdrawals, and balances.

Today, we’ll design a **command-line banking app** that mimics basic banking operations: showing your balance, depositing money, and withdrawing cash.

---

* **Deposit**: Just like restocking ingredients in the pantry.
* **Withdraw**: Taking ingredients out to cook a dish.
* **Balance**: Your kitchen inventory — it tells you what’s available.

Managing a kitchen well means **not taking more than you have**, **adding only valid items**, and **keeping your operations clean and consistent**.

---

## 🎯 Objectives

* Use functions to organize code cleanly.
* Handle input errors using `try` and `except`.
* Maintain a running balance using a `main()` loop.
* Structure code with `if __name__ == "__main__"`.

---

## 🧪 Ingredients: Code Functions

### 1. Show Balance

```python
def show_balance(balance):
    print(f"\n💰 Your current balance is: ${balance:.2f}\n")
```

Shows the current balance formatted to two decimal places.

---

### 2. Deposit Money

```python
def deposit():
    try:
        amount = float(input("Enter an amount to deposit: $"))
        if amount > 0:
            return amount
        else:
            print("🚫 Deposit must be a positive number.\n")
            return 0
    except ValueError:
        print("🚫 Please enter a valid number.\n")
        return 0
```

Validates user input and ensures only positive values are deposited.

---

### 3. Withdraw Money

```python
def withdraw(balance):
    try:
        amount = float(input("Enter an amount to withdraw: $"))
        if amount <= 0:
            print("🚫 Withdrawal must be greater than zero.\n")
            return 0
        elif amount > balance:
            print(f"🚫 Insufficient funds. Your balance is ${balance:.2f}\n")
            return 0
        else:
            return amount
    except ValueError:
        print("🚫 Please enter a valid number.\n")
        return 0
```

Prevents overdrawing and blocks invalid input.

---

### 4. The Main Kitchen Loop

```python
def main():
    balance = 0.0
    is_running = True

    print("🏦 Welcome to the Python Bank 🏦\n")

    while is_running:
        print("What would you like to do?")
        print("1. Show Balance")
        print("2. Deposit")
        print("3. Withdraw")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            show_balance(balance)
        elif choice == "2":
            balance += deposit()
        elif choice == "3":
            balance -= withdraw(balance)
        elif choice == "4":
            print("\n👋 Thank you for banking with us. Goodbye!")
            is_running = False
        else:
            print("🚫 Invalid choice. Please select a number between 1 and 4.\n")
```

This loop controls the kitchen workflow. All functions connect here.

---

## 🧠 Why Use `if __name__ == "__main__"`?

```python
if __name__ == "__main__":
    main()
```

This makes your banking system safe to **import** elsewhere — say, into a financial dashboard app — without triggering the interactive menu.

Think of it like not turning on the oven unless you’re in the kitchen ready to cook.

---

## 🧁 Mini Project: Transaction History Tracker

Extend your banking system to:

* Store every deposit and withdrawal in a `transactions` list.
* Each entry should include type (`Deposit`/`Withdrawal`) and amount.
* When exiting, print the full transaction history.

---

## 📝 Assignment

1. Add a `transactions = []` list at the start of `main()`.
2. Each time a user deposits or withdraws, append a dictionary:

   ```python
   {"type": "Deposit", "amount": 50.0}
   ```
3. When the user exits, print all transactions like a mini statement.
4. Bonus: Write the statement to a file called `bank_statement.txt`.

---