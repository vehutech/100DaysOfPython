---

# Day 22: Secret Message Cipher 🔐

---

Today, you're not just a chef — you're a **culinary spy** in the kitchen of cryptography! Every recipe you send needs to be top secret, and the ingredients? Hidden with a twist of randomness.

We’re building a **substitution cipher**: each character in your message gets swapped for another — like replacing sugar with salt on purpose (but in a good way).

---

* **Encryption** = Scramble your message with a secret spice mix.
* **Decryption** = Use your secret recipe to bring it back to life.
* **Key** = Your secret mapping — guard it like a legendary sauce recipe.

---

## 🎯 Objectives

* Use `random` to shuffle characters.
* Master lists, string operations, and indexing.
* Build a working encoder/decoder tool.
* Add optional key saving/loading for reuse.

---

## 🧪 Ingredients: Code Functions

### 1. Mix the Character Rack

```python
import random
import string

chars = " " + string.punctuation + string.digits + string.ascii_letters
chars = list(chars)
key = chars.copy()

random.shuffle(key)

print(f"chars: {chars}")
print(f"key: {key}")
```

This is your custom **secret ingredient list**. We mix it up to create a 1-to-1 map from original characters to their shuffled counterparts.

---

### 2. Encrypt the Message

```python
plain_text = input("Enter a message to encrypt: ")
cipher_text = ""

for letter in plain_text:
    index = chars.index(letter)
    cipher_text += key[index]

print(f"Original Message: {plain_text}")
print(f"Encrypted Message: {cipher_text}")
```

Each letter gets its secret twin — your message is now unreadable to outsiders! (Unless they’ve got your spice list.)

---

### 3. Decrypt With Confidence

```python
ans = input("Do you wanna decrypt message? (Y/N): ").upper()

if ans == "Y":
    decrypted_text = ""

    for letter in cipher_text:
        index = key.index(letter)
        decrypted_text += chars[index]

    print(f"Ciphered Message: {cipher_text}")
    print(f"Decrypted Message: {decrypted_text}")
```

Like carefully reversing a dish’s secret ingredients, we use the `key` to put everything back just right.

---

## 🧠 Why Is This Cool?

You're learning **symmetric encryption** — where the same key encrypts and decrypts. It’s simple but surprisingly powerful when handled right.

This is how old-school secret agents and wartime cryptographers used to work (before AI came along and spilled the soup).

---

## 🧁 Mini Project: Save Your Cipher Key

Encrypting is fun — but what if you want to decrypt later?

### Save the Key:

```python
import json

with open("cipher_key.json", "w") as f:
    json.dump({"chars": chars, "key": key}, f)
```

### Load the Key for Decryption:

```python
with open("cipher_key.json", "r") as f:
    data = json.load(f)
    chars = data["chars"]
    key = data["key"]
```

Now you can **send encrypted messages** and decrypt them days later — just keep that `cipher_key.json` safe!

---

## 📝 Assignment

1. Save your `chars` and `key` into a `.json` file after generating them.
2. On decryption, load from the saved file instead of reshuffling.
3. Add error handling if a character isn’t found in the cipher (e.g. emojis!).
4. Bonus: Wrap everything into functions like `encrypt(text)` and `decrypt(text)`.
5. Super Bonus: Let users choose between encrypting a **new message** or **decrypting an existing one** from file.

---