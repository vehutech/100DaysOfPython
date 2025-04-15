---

## Day 11 – The Order Ledger: Saving the Sauce

---

### Welcome Back, Python Chefs!

Hellooooo Python Chefs! 🚀🚀🚀
Today, we level up from noodle-slinging apprentices to full-blown **restaurant managers**.  
You see, our lovely virtual bistro has been taking orders... but like a forgetful waiter, we’ve been losing them the moment they’re placed.

Imagine telling your chef, “We had a ₦12,000 order today!” and he goes, “Prove it.” 😳  
Now we fix that with **File I/O**, **Folders**, and a tidy touch of **Error Handling**.  
Your kitchen is going paperless and professional.

---

## Your Task Today:
1. Greet the customer.
2. Display the menu.
3. Take their order.
4. Save the order inside a new folder called `orders`.
5. Handle weird input or errors with grace (like a kind chef who doesn't throw pans).

---

## Concepts We’re Cooking With:

| Concept             | Kitchen Analogy                            |
|---------------------|---------------------------------------------|
| `os.makedirs()`     | Build a pantry if it doesn’t exist          |
| `os.path.join()`    | Keep all your ingredients organized         |
| `try/except`        | Fire extinguisher for code errors           |
| `open()`            | Recipe book – write your customer’s order   |
| `encoding='utf-8'`  | Use proper spices so ₦ shows up nicely      |
| `title()`           | Makes user input neat & fancy               |

---

## 🧾 Full Code - with Comments

```python
import os  # Allows us to work with files and folders

# Step 1: Define our menu using a dictionary
menu = {
    'Burger': 2500,
    'Pizza': 4000,
    'Salad': 1500,
    'Fries': 1200,
    'Smoothie': 1800
}

# Step 2: Display the menu nicely
def display_menu():
    print("🍔 Welcome to Python Bistro!")
    print("Here's our menu:")
    for item, price in menu.items():
        print(f"- {item}: ₦{price}")
    print()  # Just a nice space after the menu

# Step 3: Main order-taking function
def take_order():
    display_menu()  # Show menu at the beginning

    try:
        name = input("Please enter your name: ").strip()
        if not name:
            raise ValueError("Name cannot be empty.")  # Don't allow blank names

        # Step 4: Figure out the directory where this script is saved
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Step 5: Create an 'orders' folder if it doesn't already exist
        orders_dir = os.path.join(script_dir, "orders")
        os.makedirs(orders_dir, exist_ok=True)

        # Step 6: Create the full path for saving this order
        order_file_path = os.path.join(orders_dir, f"{name.lower()}.txt")

        total = 0  # To track the total bill
        orders = []  # To store all the items the user orders

        # Step 7: Start taking the order in a loop
        while True:
            item = input("Enter item to order (or 'done' to finish): ").title()

            if item.lower() == 'done':
                break  # Exit the loop if customer is done ordering

            if item not in menu:
                print("❌ Item not on the menu. Try again.")
                continue  # Skip back to the beginning of the loop

            try:
                quantity = int(input(f"How many {item}s do you want? "))
                if quantity <= 0:
                    raise ValueError("Quantity must be greater than 0.")
            except ValueError:
                print("❌ Invalid quantity. Try again.")
                continue

            cost = menu[item] * quantity  # Calculate cost for this item
            total += cost  # Add to total
            orders.append(f"{item} x{quantity} = ₦{cost}")  # Save this line

        # Step 8: If no items were ordered, exit
        if not orders:
            print("No items ordered. Exiting.")
            return

        # Step 9: Write the order to a file inside the 'orders' folder
        with open(order_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Order for {name}\n")
            f.write("\n".join(orders))  # Write all items
            f.write(f"\nTotal: ₦{total}\n")

        # Step 10: Final printout to confirm order
        print(f"\n✅ Order saved in '{order_file_path}'.")
        print("🧾 Summary:")
        for order in orders:
            print(order)
        print(f"Total: ₦{total}")

    except Exception as e:
        # This catches any unexpected errors
        print("⚠️ An error occurred:", e)

    finally:
        # Runs no matter what — like cleaning the kitchen at the end of the night
        print("🧹 Kitchen cleanup complete.")

# Step 11: Fire up the kitchen
take_order()
```

---

## What You’ve Learned

- **File I/O**: You can now create, write, and organize files like a backend chef.
- **Folder Management**: With `os.makedirs()`, you're building a directory structure like a pro.
- **User Input Validation**: Catch silly mistakes like ordering zero pizzas. 🍕🚫
- **Exception Handling**: You’re not scared of runtime errors anymore — you’ve got a fire extinguisher (`try/except`)!
- **Character Encoding**: You’ve solved the “₦ doesn’t work” problem with UTF-8. 🧠✅

---

## Kitchen Challenge (Optional):
Add these toppings if you want extra flavor:
- Timestamp each order in the filename (e.g. `vehu_2025-04-15.txt`)
- Append new orders to existing customer files
- Create a `view_orders.py` script to read all `.txt` files from the `orders/` folder

---

### 🧠 Zen Chef Quote of the Day:
> “A well-organized kitchen is like a well-structured program — every ingredient (or variable) has its place.”

---