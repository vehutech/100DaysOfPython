"""
Day 10 - Python Chef Challenge
Project: Cinema Snack Bar Order System
Description: A simple order system for a cinema snack bar that allows customers to select items from a menu, add them to their cart, and view their total order cost.
"""

# ------------------------------
# 🧾 Menu Setup
# ------------------------------
menu = {
    "pizza": 3.00,
    "nachos": 4.50,
    "popcorn": 6.00,
    "fries": 2.50,
    "chips": 1.00,
    "pretzel": 3.50,
    "soda": 3.00,
    "lemonade": 4.25 
}

# ------------------------------
# 🛒 Cart & Order Tracking
# ------------------------------
customer_cart = []
order_total = 0.00

# ------------------------------
# 🖨️ Display Menu
# ------------------------------
print("\n🎉 Welcome to the Cinema Snack Bar 🎉")
print("----------- MENU -----------")
for item, price in menu.items():
    print(f"{item.capitalize():10} : ${price:.2f}")
print("----------------------------\n")

# ------------------------------
# 🧍 Customer Order Loop
# ------------------------------
while True:
    selection = input("Please select an item (or type 'q' to checkout): ").lower().strip()
    
    if selection == 'q':
        break
    elif selection in menu:
        customer_cart.append(selection)
        print(f"✅ Added {selection.capitalize()} to your cart.")
    else:
        print("❌ Item not found. Please choose from the menu.")

# ------------------------------
# 🧾 Final Order Summary
# ------------------------------
print("\n-------- YOUR ORDER --------")

if not customer_cart:
    print("🛑 You didn't order anything!")
else:
    for item in customer_cart:
        price = menu[item]
        order_total += price
        print(f"{item.capitalize():10} - ${price:.2f}")
    
    print("----------------------------")
    print(f"💰 Total: ${order_total:.2f}")

print("\nThank you for your order! Enjoy the movie! 🎬🍿")
