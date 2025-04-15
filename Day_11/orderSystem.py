import os

menu = {
    'Burger': 2500,
    'Pizza': 4000,
    'Salad': 1500,
    'Fries': 1200,
    'Smoothie': 1800
}

def display_menu():
    print("🍔 Welcome to Python Bistro!")
    print("Here's our menu:")
    for item, price in menu.items():
        print(f"- {item}: ₦{price}")
    print()

def take_order():
    display_menu()

    try:
        name = input("Please enter your name: ").strip()
        if not name:
            raise ValueError("Name cannot be empty.")
        
        # Get directory where orderSystem.py is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Create 'orders' folder if it doesn't exist
        orders_dir = os.path.join(script_dir, "orders")
        os.makedirs(orders_dir, exist_ok=True)

        # Create full path for the order file inside 'orders/'
        order_file_path = os.path.join(orders_dir, f"{name.lower()}.txt")

        total = 0
        orders = []

        while True:
            item = input("Enter item to order (or 'done' to finish): ").title()
            if item.lower() == 'done':
                break

            if item not in menu:
                print("❌ Item not on the menu. Try again.")
                continue

            try:
                quantity = int(input(f"How many {item}s do you want? "))
                if quantity <= 0:
                    raise ValueError("Quantity must be greater than 0.")
            except ValueError:
                print("❌ Invalid quantity. Try again.")
                continue

            cost = menu[item] * quantity
            total += cost
            orders.append(f"{item} x{quantity} = ₦{cost}")

        if not orders:
            print("No items ordered. Exiting.")
            return

        with open(order_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Order for {name}\n")
            f.write("\n".join(orders))
            f.write(f"\nTotal: ₦{total}\n")

        print(f"\n✅ Order saved in '{order_file_path}'.")
        print("🧾 Summary:")
        for order in orders:
            print(order)
        print(f"Total: ₦{total}")

    except Exception as e:
        print("⚠️ An error occurred:", e)

    finally:
        print("🧹 Kitchen cleanup complete.")

# Run the program
take_order()
