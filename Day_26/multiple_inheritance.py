class Animal:
    def eat(self):
        print("This animal is eating")
    
    def sleep(self):
        print("This animal is sleeping")

class Prey(Animal):
    def flee(self):
        print("This animal is fleeing")

class Predator(Animal):
    def hunt(self):
        print("This animal is hunting")

class Rabbit(Prey):
    pass

class Hawk(Predator):
    pass

class Fish(Prey, Predator):
    pass

rabbit = Rabbit()
hawk = Hawk()
fish = Fish()

fish.flee()
fish.hunt()
hawk.hunt()
# This will return an error, cos haks are predators, not preys, so they cant "flee"
# hawk.flee()
rabbit.eat()
rabbit.sleep()
fish.eat()
fish.sleep()