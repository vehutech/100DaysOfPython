# Global pantry
# implementing the following ideas:
# - **L** — Local
# - **E** — Enclosed
# - **G** — Global
# - **B** — Built-in

pantry = ["salt", "pepper", "oil"]

def kitchen():
    table = ["onion", "garlic"]  # Local ingredients

    def chef_station():
        print("Working with:", table)
        print("Pantry has:", pantry)
        print("Total pantry items:", len(pantry))

    chef_station()

kitchen()

print("Global pantry is still:", pantry)
