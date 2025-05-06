# Class variables = Shared among all instances of a class
#                   Defined outside the constructor (__init__)
#                   Useful for keeping shared data (e.g., count of students)

# Instance variables = Unique to each instance
#                   Defined inside the constructor (__init__)
#                   Used to store data specific to each object

class Student:
    class_year = 2024       # Class variable
    num_students = 0        # Class variable

    def __init__(self, name, age):
        self.name = name    # Instance variable
        self.age = age      # Instance variable
        Student.num_students += 1

student1 = Student("Spongebob", 30)
student2 = Student("Patrick", 35)
student3 = Student("Squidward", 55)
student4 = Student("Sandie", 35)

print(student1.name)
print(student1.age)
print(Student.class_year)
print(Student.num_students)

# Proper f-string formatting with conditional pluralization
print(f"My graduating class of {Student.class_year} has {Student.num_students} student{'s' if Student.num_students > 1 else ''}.")
print(f"{student1.name} is {student1.age} years old.")  