import random
import string

chars = " " + string.punctuation + string.digits + string.ascii_letters

chars = list(chars)

key = chars.copy()

random.shuffle(key)

print(f"chars: {chars}")
print(f"key: {key}")

# ENCRYPT
plain_text = input("Enter a message to encrypt: ")
cipher_text = ""

for letter in plain_text:
    index = chars.index(letter)
    cipher_text += key[index]

print(f"Original Message: {plain_text}")
print(f"Encrypted Message: {cipher_text}")

# DECRYPT

ans = input("Do you wanna decrypt message? (Y/N)").upper()

if ans == "Y":
    original_text = plain_text
    decrypted_text = ""

    for letter in cipher_text:
        index = key.index(letter)
        decrypted_text += chars[index]

    print(f"Ciphered Message: {cipher_text}")
    print(f"Decrypted Message: {decrypted_text}")