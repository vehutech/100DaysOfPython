def get_word(word_type):
    while True:
        word = input(f"Enter a {word_type}: ").strip()
        if word:
            return word  # Return valid input
        print("Oops! You must enter something!")

def madlibs():
    noun = get_word("noun")
    verb = get_word("verb")
    adjective = get_word("adjective")

    story = f"Today, I saw a {adjective} {noun} that decided to {verb} all day long!"
    
    print("\n YOUR MADLIBS STORY")
    print(story)

# Start the game
madlibs()