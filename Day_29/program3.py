# MAGIC METHODS

class Book:

    def __init__(self, title, author, num_pages):
        self.title = title
        self.author = author
        self.num_pages = num_pages

    def __str__(self):
        return f"'{self.title}' by '{self.author}, {self.num_pages} pages'"
    
    def __eq__(self, value):
        return self.title == value.title and self.author == value.author
    
    def __lt__(self, value):
        return self.num_pages < value.num_pages
    
    def __add__(self, value):
        return self.num_pages + value.num_pages
    
    def __contains__(self, value):
        return value in self.author or value in self.title
    
    def __getitem__(self, key):  
        if key == "title":
            return self.title
        if key == "num_pages":
            return self.num_pages
        if key == "author":
            return self.author
        else:
            return f"Key {key} was not found"

book1 = Book("The Hobbit", "J.R.R. Tolkein", 310)
book2 = Book("The Hobbit", "J.R.R. Tolkein", 223)
book3 = Book("Harry Potter and the Philosopher's Stone", "J.R.R. Tolkein", 223)
book4 = Book("The Lion, the Witch and the Wardrobe", "C.S. Lewis", 172)

print(book1)
print(book1 == book2)
print(book1 > book2)
print(book1 + book2)
print("Lion" in book4)
print(book1["num_pages"])
print(book4["sakdfjk"])