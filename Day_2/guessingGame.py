import random  # Importing the random module

print("🍽️ Welcome to the Secret Ingredient Game!")
print("I've chosen an ingredient amount (1-10). Can you guess it?")

# Generate a random secret ingredient amount
secret_number = random.randint(1, 10)

attempts = 0  # Integer: Tracks number of attempts
guessed_correctly = False  # Boolean: Checks if user guessed correctly

while not guessed_correctly:  # Loop continues until user guesses correctly
    guess = int(input("Enter your guess: "))  # User's guess (converted to int)
    attempts += 1  # Increment attempts

    if guess == secret_number:
        guessed_correctly = True  # Set to True, loop ends
        print(f"🎉 Correct! You guessed in {attempts} attempts.")
    elif guess < secret_number:
        print("Too little! Try again. 🥄")
    else:
        print("Too much! Try again. 🍽️")