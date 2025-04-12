Helllloooooo Python Chefs! 🚀🚀🚀🚀
Welcome to **Day 10** of our 100 Days of Python challenge!  
Today we’re setting the kitchen on fire (safely 😅) with a delicious real-world project — **a digital snack ordering system** — chef-style.

---

## 👨‍🍳 **Day 10: The Hungry Moviegoer's Snack Bar**

> 🎬 Picture this: You're running a mini snack bar in a cozy little cinema.  
> The lights dim, the trailers roll, and your customers start lining up, eyes fixed on the goodies.  
> Your job? Build a slick, intuitive **digital menu system** so they can order popcorn, soda, and all the good stuff.

---


```python
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
```
🔸 We're defining a **dictionary** called `menu`.  
Each **key** is a food item (as a string), and the **value** is its price (as a float).

✅ This allows us to look up the price of any item quickly using its name.

---

```python
# ------------------------------
# 🛒 Cart & Order Tracking
# ------------------------------
customer_cart = []
order_total = 0.00
```
🔸 `customer_cart` is an **empty list** that will store the items the customer selects.  
🔸 `order_total` is a **float** variable that will keep track of the total cost.

Think of `customer_cart` as a tray and `order_total` as the bill.

---

```python
# ------------------------------
# 🖨️ Display Menu
# ------------------------------
print("\n🎉 Welcome to the Cinema Snack Bar 🎉")
print("----------- MENU -----------")
```
🔸 We use `print()` to greet the customer and display a clean, formatted menu.

---

```python
for item, price in menu.items():
    print(f"{item.capitalize():10} : ${price:.2f}")
```
🔸 This **loop** goes through each item in the `menu`.  
- `menu.items()` gives us both the item name (`item`) and its price (`price`).  
- `item.capitalize()` ensures the food name starts with a capital letter.  
- `:10` sets a **minimum width** of 10 characters to align everything nicely.  
- `${price:.2f}` means we show the price with 2 decimal places.

📌 Result: A beautifully formatted, professional menu display.

---

```python
print("----------------------------\n")
```
🔸 Prints a line to **separate the menu** from the next section.

---

```python
# ------------------------------
# 🧍 Customer Order Loop
# ------------------------------
while True:
```
🔸 This is an **infinite loop** that will keep asking the customer for orders until they type `'q'`.

---

```python
    selection = input("Please select an item (or type 'q' to checkout): ").lower().strip()
```
🔸 The user is prompted to type in the name of the food item.  
- `.lower()` converts the input to lowercase so it matches the keys in our menu.
- `.strip()` removes extra spaces before or after the input.

---

```python
    if selection == 'q':
        break
```
🔸 If the customer types `'q'`, we **exit** the loop and move to checkout using `break`.

---

```python
    elif selection in menu:
        customer_cart.append(selection)
        print(f"✅ Added {selection.capitalize()} to your cart.")
```
🔸 If the typed item exists in the `menu`, we:
1. Add it to the `customer_cart` using `.append()`.
2. Display a confirmation message using `print()`.

---

```python
    else:
        print("❌ Item not found. Please choose from the menu.")
```
🔸 If the customer types something not in the menu, we show an error message.

---

```python
# ------------------------------
# 🧾 Final Order Summary
# ------------------------------
print("\n-------- YOUR ORDER --------")
```
🔸 Once the customer exits the loop, we print the header for the final receipt.

---

```python
if not customer_cart:
    print("🛑 You didn't order anything!")
```
🔸 If the cart is still empty (`not customer_cart`), we tell the customer they haven’t selected anything.

---

```python
else:
    for item in customer_cart:
        price = menu[item]
        order_total += price
        print(f"{item.capitalize():10} - ${price:.2f}")
```
🔸 Otherwise, we:
1. Go through every item in `customer_cart`
2. Look up its price from the `menu`
3. Add that price to `order_total`
4. Print the item and its price nicely formatted

---

```python
    print("----------------------------")
    print(f"💰 Total: ${order_total:.2f}")
```
🔸 We print a divider, then display the **final total**, again formatted with two decimal places.

---

```python
print("\nThank you for your order! Enjoy the movie! 🎬🍿")
```
🔸 Finally, we thank the customer and add a fun touch with movie emojis. 🥳

---

## 🔚 Summary

- We used a **dictionary** to store menu items and prices.
- A **list** to hold the selected items.
- A **loop** to take continuous input until the user quits.
- **Formatted strings** to make the output clean and professional.
- Clean **control flow** with `if`, `elif`, and `else` to handle user input properly.

---

## 🧪 Your Turn!

- Try changing the prices!
- Add more items to the menu!

---

## 🧠 Concepts Covered Today

- Dictionary usage (`menu`)
- List operations (`append`, loop through `cart`)
- User input with validation
- String formatting and casing
- Real-world app simulation
- Clean, readable variable names

---

## 🔥 Bonus Ideas for Advanced Chefs:
- Add quantity support for each item
- Support multiple customers (maybe a loop of sessions)
- Save orders to a file using Python's `open()` and `write()`
- Introduce item categories (drinks, snacks, etc.)
- Use `datetime` to timestamp orders

---


Let’s keep this kitchen hot, Chef! 🍳💻  
#pythonchef #100DaysOfPython #Day10 #SnackBarSystem #vehutech