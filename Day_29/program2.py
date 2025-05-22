# Magic Methods - program3.py
# Magic methods (dunder methods) are automatically called by Python's built-in operations
# They allow developers to define or customize the behavior of objects

class Book:
    def __init__(self, title, author, num_pages, price=0.0):
        self.title = title
        self.author = author
        self.num_pages = num_pages
        self.price = price

    def __str__(self):
        """String representation for end users"""
        return f"'{self.title}' by {self.author} ({self.num_pages} pages)"
    
    def __repr__(self):
        """String representation for developers/debugging"""
        return f"Book('{self.title}', '{self.author}', {self.num_pages}, {self.price})"
    
    def __eq__(self, other):
        """Equality comparison - books are equal if title and author match"""
        if not isinstance(other, Book):
            return False
        return self.title == other.title and self.author == other.author
    
    def __lt__(self, other):
        """Less than comparison - compare by number of pages"""
        if not isinstance(other, Book):
            return NotImplemented
        return self.num_pages < other.num_pages
    
    def __le__(self, other):
        """Less than or equal to"""
        if not isinstance(other, Book):
            return NotImplemented
        return self.num_pages <= other.num_pages
    
    def __gt__(self, other):
        """Greater than comparison"""
        if not isinstance(other, Book):
            return NotImplemented
        return self.num_pages > other.num_pages
    
    def __add__(self, other):
        """Addition - combine page counts"""
        if isinstance(other, Book):
            return self.num_pages + other.num_pages
        elif isinstance(other, (int, float)):
            return self.num_pages + other
        return NotImplemented
    
    def __contains__(self, keyword):
        """Membership test - check if keyword is in title or author"""
        keyword_lower = keyword.lower()
        return (keyword_lower in self.title.lower() or 
                keyword_lower in self.author.lower())
    
    def __getitem__(self, key):
        """Dictionary-style access to book attributes"""
        attributes = {
            'title': self.title,
            'author': self.author,
            'num_pages': self.num_pages,
            'pages': self.num_pages,  # Alternative key
            'price': self.price
        }
        
        if key in attributes:
            return attributes[key]
        else:
            return f"Key '{key}' was not found"
    
    def __len__(self):
        """Return length (number of pages)"""
        return self.num_pages

# Create book instances
book1 = Book("The Hobbit", "J.R.R. Tolkien", 310, 12.99)
book2 = Book("The Hobbit", "J.R.R. Tolkien", 295, 15.99)  # Same book, different edition
book3 = Book("Harry Potter and the Philosopher's Stone", "J.K. Rowling", 223, 14.99)
book4 = Book("The Lion, the Witch and the Wardrobe", "C.S. Lewis", 172, 10.99)

print("=== Magic Methods in Action ===")

print("\n--- String Representation ---")
print(f"__str__: {book1}")           # Calls __str__
print(f"__repr__: {repr(book1)}")    # Calls __repr__

print("\n--- Equality Comparison ---")
print(f"book1 == book2: {book1 == book2}")  # Same title/author, different pages
print(f"book1 == book3: {book1 == book3}")  # Different books

print("\n--- Size Comparison ---")
print(f"book1 > book2: {book1 > book2}")    # 310 > 295 pages
print(f"book1 < book3: {book1 < book3}")    # 310 < 223? False
print(f"book4 < book3: {book4 < book3}")    # 172 < 223? True

print("\n--- Addition ---")
print(f"book1 + book2 pages: {book1 + book2}")  # Total pages
print(f"book1 + 50 pages: {book1 + 50}")        # Add extra pages

print("\n--- Membership Test ---")
print(f"'Hobbit' in book1: {'Hobbit' in book1}")       # True
print(f"'Lion' in book4: {'Lion' in book4}")           # True  
print(f"'Dragon' in book1: {'Dragon' in book1}")       # False

print("\n--- Dictionary-style Access ---")
print(f"book1['title']: {book1['title']}")
print(f"book1['num_pages']: {book1['num_pages']}")
print(f"book1['author']: {book1['author']}")
print(f"book4['invalid_key']: {book4['invalid_key']}")

print("\n--- Length ---")
print(f"len(book1): {len(book1)} pages")
print(f"len(book4): {len(book4)} pages")

print("\n--- Sorting Books by Pages ---")
books = [book1, book2, book3, book4]
books.sort()  # Uses __lt__ for comparison
print("Books sorted by page count:")
for book in books:
    print(f"  {book} - {len(book)} pages")