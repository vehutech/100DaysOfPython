# Python Banking Program

def show_balance(balance):
    print(f"Your balance is ${balance:.2f}")

def deposit():
    
    amount = input("Enter an amount to be deposited: \n")

    if amount.isdigit():
        amount = float(amount)
        return amount
    else:
        print("Please enter a valid deposit")
        return 0

def withdraw(balance):
    
    amount = input("Enter an amount to be withdrawn: \n")

    if amount.isdigit():

        amount = float(amount)

        if amount > balance:

            print("Insufficient Funds")
            print(f"You can't withdraw {amount}, because your balance is {balance}")
            return 0
        else:
            return amount
    
    else:
        print("Please enter a valid withdrawal amount")
        return 0

def main():
    balance = 0
    is_running = True

    while is_running:
        print("Banking Program")
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
            is_running = False
        else:
            print("That is not a valid choice")

    print("Thank you! have a nice day")

if __name__ == "__main__":
    main()