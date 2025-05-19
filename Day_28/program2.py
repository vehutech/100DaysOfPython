# program2.py

class Animal:
    alive = True

class Dog(Animal):
    def speak(self):
        print("WOOF!")

class Cat(Animal):
    def speak(self):
        print("MEOW!")

class Car:

    alive = False

    #This will not work 
    # def horn(self):
        # print("Hooooooorn!")
    
    def speak(self):
        print("Hoooooorn!!")

animals = [
    Dog(), Cat(), Car()
]

for animal in animals:
    animal.speak()
    print(animal.alive)