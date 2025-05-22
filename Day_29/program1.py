# Static Methods - program1.py
# Static methods belong to a class rather than any specific instance
# They're perfect for utility functions that don't need access to instance data

class Employee:
    def __init__(self, name, position):
        self.name = name
        self.position = position

    def get_info(self):
        """Instance method - works with specific employee data"""
        return f"{self.name} => {self.position}"
    
    @staticmethod
    def is_valid_position(position):
        """Static method - validates position without needing employee data"""
        valid_positions = [
            "Manager", "Cashier", "Cook", "Janitor", "Developer", "Designer"
        ]
        return position in valid_positions
    
    @staticmethod
    def calculate_hourly_wage(annual_salary):
        """Static method - utility function for wage calculation"""
        # Assuming 40 hours/week, 52 weeks/year
        return round(annual_salary / (40 * 52), 2)

# Create employee instances
employee1 = Employee("Eugene", "Manager")
employee2 = Employee("Squidward", "Cashier")
employee3 = Employee("SpongeBob", "Cook")

# Test static methods - can be called without creating an instance
print("=== Static Method Tests ===")
print(f"Is 'Manager' valid? {Employee.is_valid_position('Manager')}")
print(f"Is 'Rocket Scientist' valid? {Employee.is_valid_position('Rocket Scientist')}")
print(f"Hourly wage for $50,000 salary: ${Employee.calculate_hourly_wage(50000)}")

# Static methods can also be called from instances (but it's not recommended)
print(f"Called from instance: {employee1.is_valid_position('Cook')}")

print("\n=== Instance Method Tests ===")
print(employee1.get_info())
print(employee2.get_info())
print(employee3.get_info())