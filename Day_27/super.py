# super() = Function used in a child class to call methods from a parent call methods from a parent class (superclass).
# The child class is the sub-class, the parent class is the super class
# It allows you to extend the functionality of the inherited methods

class Shape:
    def __init__(self, color, is_filled):
        self.color = color
        self.is_filled = is_filled

    def describe(self):
        print(f"It is {self.color} and {"filled" if self.is_filled else "not filled"}")

class Circle(Shape):
    def __init__(self, color, is_filled, radius):
        super().__init__(color, is_filled)
        self.radius = radius
    
    # method overiding
    def describe(self):
        # Extending methodology from a parent
        super().describe()
        print(f"It is a circle with an area of {3.14 * self.radius *self.radius}cm^2")

class Square(Shape):
    def __init__(self, color, is_filled, width):
        super().__init__(color, is_filled)
        self.width = width
    
     
    # method overiding
    def describe(self):
        # Extending methodology from a parent
        super().describe()
        print(f"It is a circle with an area of {3.14 * self.radius *self.radius}cm^2")

class Triangle(Shape):
    def __init__(self, color, is_filled, width, height):
        super().__init__(color, is_filled)
        self.width = width
        self.height = height

circle = Circle("blue", True, 5)

print(circle.color)
print(circle.is_filled)
print(circle.radius)
print(circle.describe())

square = Square("red", False, 6)

print(square.color)
print(square.is_filled)
print(square.width)
print(square.describe())

triangle = Triangle("Yellow", True, 7, 8)

print(triangle.color)
print(triangle.is_filled)
print(triangle.width)
print(triangle.height)
print(triangle.describe())