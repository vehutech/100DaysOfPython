---
# **Day 29: The Method to the Madness — Static, Class, and Magic Methods** ⚡🐍
---
Imagine you're running a company. Some tasks require specific employee information (instance methods), others can be handled by any department member (static methods), some need company-wide data (class methods), and a few are so fundamental they happen automatically (magic methods). Today, we're mastering all these "methods to the madness"!
---
## 🎯 **Objectives**
* Master the differences between instance, static, and class methods
* Understand when and why to use each method type
* Implement magic methods to customize object behavior
* Write cleaner, more organized code with proper method selection
---
## 🏢 **Instance Methods: The Personal Touch**
Instance methods are the workhorses of your classes. They operate on specific instances (objects) and have access to `self` — the individual object's data.

```python
class Employee:
    def __init__(self, name, position, salary):
        self.name = name
        self.position = position
        self.salary = salary
    
    # Instance method - works with specific employee data
    def get_info(self):
        return f"{self.name} works as {self.position} and earns ${self.salary:,}"
    
    def give_raise(self, amount):
        self.salary += amount
        return f"{self.name} received a ${amount:,} raise!"

# Each employee is unique
employee1 = Employee("Alice", "Developer", 75000)
employee2 = Employee("Bob", "Manager", 85000)

print(employee1.get_info())  # Alice's specific info
print(employee2.get_info())  # Bob's specific info
```

**Use instance methods when:** You need to work with specific object data or modify individual object state.
---
## 🔧 **Static Methods: The Independent Workers**
Static methods are like utility functions that live inside your class. They don't need access to instance data (`self`) or class data (`cls`) — they're completely independent.

```python
class Employee:
    def __init__(self, name, position, salary):
        self.name = name
        self.position = position
        self.salary = salary
    
    def get_info(self):
        return f"{self.name} works as {self.position} and earns ${self.salary:,}"
    
    @staticmethod
    def is_valid_position(position):
        """Check if a position exists in our company"""
        valid_positions = [
            "Developer", "Manager", "Designer", "Analyst", 
            "HR Specialist", "Sales Rep", "Accountant"
        ]
        return position in valid_positions
    
    @staticmethod
    def calculate_annual_bonus(salary, performance_rating):
        """Calculate bonus based on salary and performance (1-5 scale)"""
        bonus_multiplier = {
            5: 0.20,  # Exceptional
            4: 0.15,  # Exceeds expectations  
            3: 0.10,  # Meets expectations
            2: 0.05,  # Below expectations
            1: 0.00   # Poor performance
        }
        return salary * bonus_multiplier.get(performance_rating, 0)

# Can be called without creating an Employee object
print(Employee.is_valid_position("Developer"))  # True
print(Employee.is_valid_position("Wizard"))     # False

# Or called from an instance (works either way)
employee = Employee("Charlie", "Designer", 65000)
bonus = Employee.calculate_annual_bonus(65000, 4)
print(f"Bonus calculation: ${bonus:,.2f}")
```

**Use static methods when:** You have utility functions related to the class but don't need access to instance or class data.
---
## 🏛️ **Class Methods: The Company-Wide Operations**
Class methods work with class-level data and receive `cls` (the class itself) as their first parameter. They're perfect for operations that affect or use data shared by all instances.

```python
class Student:
    # Class variables (shared by all instances)
    total_students = 0
    total_gpa_points = 0.0
    school_name = "Python Academy"
    
    def __init__(self, name, gpa):
        self.name = name
        self.gpa = gpa
        # Update class-level statistics
        Student.total_students += 1
        Student.total_gpa_points += gpa
    
    def get_info(self):
        return f"{self.name}: GPA {self.gpa}"
    
    @classmethod
    def get_student_count(cls):
        return f"Total students enrolled: {cls.total_students}"
    
    @classmethod
    def get_average_gpa(cls):
        if cls.total_students == 0:
            return "No students enrolled yet"
        avg_gpa = cls.total_gpa_points / cls.total_students
        return f"School average GPA: {avg_gpa:.2f}"
    
    @classmethod
    def get_school_info(cls):
        return f"Welcome to {cls.school_name}!"
    
    @classmethod
    def create_honor_student(cls, name):
        """Alternative constructor for honor students"""
        return cls(name, 4.0)  # Honor students start with perfect GPA

# Create some students
student1 = Student("Emma", 3.8)
student2 = Student("Liam", 3.2)
student3 = Student("Sophia", 3.9)

# Class methods work on class-level data
print(Student.get_student_count())    # Total students: 3
print(Student.get_average_gpa())      # Average GPA: 3.63
print(Student.get_school_info())      # Welcome to Python Academy!

# Alternative constructor
honor_student = Student.create_honor_student("Noah")
print(honor_student.get_info())       # Noah: GPA 4.0
```

**Use class methods when:** You need to work with class-level data, create alternative constructors, or perform operations that affect all instances.
---
## ✨ **Magic Methods: The Automatic Operators**
Magic methods (also called "dunder methods" for "double underscore") are special methods that Python calls automatically. They let you customize how your objects behave with built-in operations.

```python
class Book:
    def __init__(self, title, author, pages, price=0):
        self.title = title
        self.author = author
        self.pages = pages
        self.price = price
    
    def __str__(self):
        """String representation for users"""
        return f"'{self.title}' by {self.author} ({self.pages} pages)"
    
    def __repr__(self):
        """String representation for developers"""
        return f"Book('{self.title}', '{self.author}', {self.pages}, {self.price})"
    
    def __eq__(self, other):
        """Check if two books are the same"""
        if not isinstance(other, Book):
            return False
        return (self.title == other.title and 
                self.author == other.author)
    
    def __lt__(self, other):
        """Compare books by page count"""
        return self.pages < other.pages
    
    def __add__(self, other):
        """Add page counts of two books"""
        if isinstance(other, Book):
            return self.pages + other.pages
        return NotImplemented
    
    def __contains__(self, keyword):
        """Check if keyword appears in title or author"""
        keyword_lower = keyword.lower()
        return (keyword_lower in self.title.lower() or 
                keyword_lower in self.author.lower())
    
    def __getitem__(self, key):
        """Allow dictionary-style access to book attributes"""
        attributes = {
            'title': self.title,
            'author': self.author,
            'pages': self.pages,
            'price': self.price
        }
        if key in attributes:
            return attributes[key]
        raise KeyError(f"'{key}' is not a valid book attribute")
    
    def __len__(self):
        """Return the number of pages"""
        return self.pages

# Create some books
book1 = Book("The Hobbit", "J.R.R. Tolkien", 310, 12.99)
book2 = Book("The Hobbit", "J.R.R. Tolkien", 310, 15.99)  # Same book, different price
book3 = Book("Dune", "Frank Herbert", 688, 14.99)
book4 = Book("1984", "George Orwell", 328, 13.99)

# Magic methods in action!
print(book1)                    # __str__ called
print(repr(book3))              # __repr__ called
print(book1 == book2)           # __eq__ called - True (same title/author)
print(book1 < book3)            # __lt__ called - True (310 < 688 pages)
print(book1 + book3)            # __add__ called - 998 (total pages)
print("Hobbit" in book1)        # __contains__ called - True
print(book1["title"])           # __getitem__ called - "The Hobbit"
print(len(book4))               # __len__ called - 328

# Sort books by page count (uses __lt__)
books = [book1, book3, book4]
books.sort()
for book in books:
    print(f"{book} - {len(book)} pages")
```
---
## 🎯 **Method Selection Guide**

| Method Type | When to Use | Example Use Case |
|-------------|-------------|------------------|
| **Instance** | Working with specific object data | `student.calculate_grade()` |
| **Static** | Utility functions related to class | `MathUtils.calculate_distance()` |
| **Class** | Working with class-level data | `Student.get_total_enrollment()` |
| **Magic** | Customizing built-in operations | Making objects comparable or printable |
---
## 🚀 **Real-World Applications**

**Static Methods:**
- Data validation functions
- Utility calculations
- Factory methods that don't need class state

**Class Methods:**
- Alternative constructors
- Tracking class-level statistics
- Configuration management

**Magic Methods:**
- Making objects sortable and comparable
- Custom string representations
- Operator overloading for math classes
---
## 📝 **Assignment**
**### Task 1: Library Management System**
Create a `Library` class with:
- Instance methods for checking out/returning books
- Static methods for ISBN validation
- Class methods for tracking total books and library statistics
- Magic methods to make books comparable and searchable

**### Task 2: Bank Account Enhanced**
Enhance a `BankAccount` class with:
- Static method for validating account numbers
- Class method for tracking total bank assets
- Magic methods for comparing accounts and string representation
- Instance methods for deposits/withdrawals

**Bonus Challenge:** Create a `Money` class that uses magic methods to handle currency operations with proper formatting and arithmetic.
---
## 🤔 **Food for Thought**

Why do you think Python uses double underscores for magic methods? How does this naming convention help prevent naming conflicts? When might you prefer a static method over a regular function outside the class?
---