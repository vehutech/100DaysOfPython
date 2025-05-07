class Animal:

    def __init__(self, name):
        self.name = name
        self.is_alive = True

    def eat(self):
        print("f{self.name} is eating")

    def sleep(self):
        print("f{self.name} is sleeping")

class Dog(Animal):
    def speak(self):
        print("Wof wofff!")

class Cat(Animal):
    def speak(self):
        print("Meeew")

class Mouse(Animal):
    def speak(self):
        print("Squeek")

dog1 = Dog("Hero")
dog2 = Dog("Scooby")
cat = Cat("Garfielld")
mouse = Mouse("Mickey")


print(dog1.name)
dog1.speak()
mouse.sleep
