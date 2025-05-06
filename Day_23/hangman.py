import random
from wordlist import words
from hangman_art import human as hangman_art


def display_man(wrong_guesses):
    for line in hangman_art[wrong_guesses]:
        print(line)
    print("***************")

def display_hint(hint):
    print(" ".join(hint))

def display_answer(answer):
    print(" ".join(answer))

def play_again():
    while True:
        again = input("Do you want to play again? (y/n): ").lower()
        if again in ['y', 'n']:
            return again == 'y'
        else:
            print("Invalid input, please enter 'y' or 'n'.")

def reset_game():
    answer = random.choice(words)
    hint = ["_"] * len(answer)
    wrong_guesses = 0
    guessed_letters = set()
    return answer, hint, wrong_guesses, guessed_letters

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
            print("YOU WIN!!")
            is_running = play_again()
            if is_running:
                answer, hint, wrong_guesses, guessed_letters = reset_game()
            else:
                print("Thanks for playing!")

        elif wrong_guesses >= len(hangman_art) - 1:
            display_man(wrong_guesses)
            display_answer(answer)
            print("YOU LOSE!")
            is_running = play_again()
            if is_running:
                answer, hint, wrong_guesses, guessed_letters = reset_game()
            else:
                print("Thanks for playing!")

if __name__ == "__main__":
    main()
