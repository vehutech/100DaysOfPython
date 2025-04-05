import random


def recipe_generator():
    chefs = ["Tunde", "Chioma", "Ngozi"]
    dishes = ["stew", "pasta", "yam porridge", "cake"]
    styles = ["Nigerian style", "spicy twist", "chef's special"]

    chef = random.choice(chefs)
    dish = random.choice(dishes)
    style = random.choice(styles)

    print(f"{chef} is making {dish} with a {style}!")


recipe_generator()
