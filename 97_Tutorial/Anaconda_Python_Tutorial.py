# single line of comments

"""
multiple line of comments
"""

print("""\
    ==================================================================
    =============================Indexing=============================
    =============================20210312=============================
    ==================================================================
    """
      )

# this is the first comment
spam = 1  # and this is the second comment
# ... and now a third!
text = "# This is not a comment because it's inside quotes."

print(text)

print("# division always returns a floating point number")
print(1 / 3 + 2)

print("# floor division discards the fractional part")
print(17 // 3)

print("# the % operator returns the remainder of the division")
print(17 % 3)

print("equal sign (=) is used to assign a value to a variable")
width = 2
height = 9
area = width * height
print(area)

print("# single quotes")
print('spam eggs')

print("# use \' to escape the single quote")
print('doesn\'t')

print("or use double quotes instead")
print("doesn't")

print(
    "If you don’t want characters prefaced by \ to be interpreted as special characters, you can use raw strings by adding an r before the first quote")
print('C:\some\name')
print(r'C:\some\name')

print("""\
Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
""")

print("Strings can be concatenated (glued together) with the + operator, and repeated with *")
print(3 * 'un' + 'ium')

print('Py' 'thon')

print('Put several strings within parentheses '
      'to have them joined together. '
      'this is the 3rd line. '
      'this is the 4th line.'
      )

print("concatenate variables or a variable and a literal, use +")
prefix = 'Py'
print(prefix + 'thon')

print("Strings can be indexed (subscripted)")
word = 'Python'
print(word[0])
print(word[1])
print(word[2])
print(word[3])
print(word[4])
print(word[5])
print(word[-1])

print("slicing is also supported")
print(word[0:2])
print(word[2:4])
print(word[:2] + word[2:])

print("""\
 +---+---+---+---+---+---+
 | P | y | t | h | o | n |
 +---+---+---+---+---+---+
 0   1   2   3   4   5   6
-6  -5  -4  -3  -2  -1
""")

print(
    "Python strings cannot be changed — they are immutable. Therefore, assigning to an indexed position in the string results in an error:")
# word[0] = 'J'

s = 'supercalifragilisticexpialidocious'
print(len(s))

print("The most versatile is the list")
squares = [1, 4, 9, 16, 25]
squares2 = squares + [36, 49, 64, 81, 100]
print(squares)
print(squares[1:3])
print(squares[:3])
print(squares2)

print("Unlike strings, which are immutable, lists are a mutable type")
squares[2] = 1000
print(squares)
squares.append(2000)
print(squares)

print("It is possible to nest lists (create lists containing other lists),")
a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]
print(x)
print(x[0][2])

print("# Fibonacci series:")
a, b = 0, 1
while a < 10:
    print(a)
    a, b = b, a + b

'''
x = int(input("Please enter an integer: "))
if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')
'''

print("# Measure some strings:")
words = ['cat', 'window', 'defenestrate']
for w in words:
    print(w, len(w))

print("-----------------------")
for i in range(2, 10):
    print(i)

print("-----------------------")
for i in range(0, 10, 3):
    print(i)

print("-----------------------")
for i in range(0, 10, 2):
    print(i)

print("-----------------------")
a = ['Mary', 'had', 'a', 'little', 'lamb']
for i in range(len(a)):
    print(i, a[i])

print("-----------------------")
print("""\
In many ways the object returned by range() behaves as if it is a list, 
but in fact it isn’t. It is an object which returns the successive items 
of the desired sequence when you iterate over it, 
but it doesn’t really make the list, thus saving space.
""")
print(range(10))
print(sum(range(4)))
print(list(range(4)))

print("-----------------------")
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n // x)
            # break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')

print("-----------------------")
for num in range(2, 10):
    if num % 2 == 0:
        print("Found an even number", num)
        continue
    print("Found an odd number", num)

print("-----------------------")
print("We can create a function that writes the Fibonacci series to an arbitrary boundary:")


def fib(n):
    a, b = 0, 1
    while a < n:
        print('a before = ', a)
        print('b before = ', b)
        a, b = b, a + b
        print('a after = ', a)
        print('b after = ', b)
    print()


fib(3)

print("-----------------------")
a, b = 1, 2
print(a)
print(b)

print("-----------------------")
# a, b = b, a + b
a = b
b = a + b
print(a)
print(b)

f = fib
f(3)


def fib2(n):
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a + b
    return result


print(fib2(10))

'''
def ask_ok(prompt, retries=4, reminder='Please try again!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)


ask_ok('Do you really want to quit? y/n: ')
'''


def f(a, L=[]):
    L.append(a)
    return L


print("-----------------------")
print(f(1))
print(f(2))
print(f(3))


def f2(a, L=None):
    if L is None:
        L = []
    L.append(a)
    return L


print(f2(2))

print("-----------------------")


# arguments may be passed by position or keyword
def standard_arg(arg):
    print(arg)


# only use positional parameters
def pos_only_arg(arg, /):
    print(arg)


# only allows keyword arguments
def kwd_only_arg(*, arg):
    print(arg)


# all three calling conventions in the same function definition
def combined_example(pos_only, /, standard, *, kwd_only):
    print(pos_only, standard, kwd_only)


def concat(*args, sep="/"):
    return sep.join(args)


print(concat("earth", "mars", "venus"))
print(concat("earth", "mars", "venus", sep="."))

print(list(range(3, 6)))
args = [3, 6]
print(list(range(*args)))

print("-----------------------")
print("# Lambda Expressions")
add = lambda x, y: x + y
print(add(3, 4))
print((lambda x, y: x + y)(5, 6))

my_list = [3, 5, -4, -1, 0, -2, -6]
print(sorted(my_list, key=lambda x: abs(x)))

print(list(map(lambda x: x * x, range(1, 21))))
print(list(filter(lambda x: x % 2 == 0, range(1, 21))))

from functools import reduce

print(reduce(lambda x, y: x + y, range(1, 101)))


def add(n):
    return lambda x: x + n


add2 = add(5)
print(add2(15))

# more on list
print("-----------------------")
fruits = ['orange', 'apple', 'pear', 'banana', 'kiwi', 'apple', 'banana']
print(fruits.count('apple'))
print(fruits.index('banana'))
print(fruits.index('banana', 4))
fruits.reverse()
print(fruits)
fruits.append('grape')
print(fruits)
fruits.sort()
print(fruits)
print(fruits.pop())
print(fruits.pop())
print(fruits.pop())
print(fruits)

print("-----------------------")
print("List Comprehensions")
squares = []
for x in range(10):
    squares.append(x ** 2)

print(squares)
print(list(map(lambda x: x ** 2, range(10))))
print([x ** 2 for x in range(10)])
print([(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y])

vec = [-4, -2, 0, 2, 4]
print([x * 2 for x in vec])
print([x for x in vec if x >= 0])
print([abs(x) for x in vec])

freshfruit = ['  banana', '  loganberry ', 'passion fruit  ']
print([weapon.strip() for weapon in freshfruit])

print("""\
    ==================================================================
    =============================Indexing=============================
    =============================20210313=============================
    ==================================================================
    """
      )

# sequence data types
# list -> mutable
t = [12345, 54321, 'hello!']
print(t)
t[0] = 100
print(t)

# tuple -> immutable but can include mutable objects
t = (12345, 54321, 'hello!')
print(t)
# t[0]=100 -> TypeError: 'tuple' object does not support item assignment

v = ([1, 2, 3], [3, 2, 1])
print(v)
v[0][0] = 100
print(v)

# set
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)
print('orange' in basket)
# Demonstrate set operations on unique letters from two words
a = set('abracadabra')
b = set('alacazam')
print(a)
print(b)
print(a - b)
print(a | b)
print(a & b)
print(a ^ b)

c = set(['abc', 'bcd'])
d = set(['abc', 'edf'])
print(c)
print(d)
print(c - d)
print(c | d)
print(c & d)
print(c ^ d)

print("-----------------------")
print("Unlike sequences, which are indexed by a range of numbers, dictionaries"
      " are indexed by keys, which can be any immutable type")

tel = {'jack': 4098, 'sape': 4139}
tel['guido'] = 4127
print(tel)
print(list(tel))
print(sorted(tel))
tel2 = sorted(tel)
print(tel2)

print("The dict() constructor builds dictionaries directly from sequences of key-value pairs:")
print(dict([('sape', 4139), ('guido', 4127), ('jack', 4098)]))

print("In addition, dict comprehensions can be used to create dictionaries from arbitrary key and value expressions:")
print({x: x ** 2 for x in (2, 4, 6)})

print("When looping through dictionaries, the key and corresponding value can be retrieved at the same time using the"
      " items() method")
knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for k, v in knights.items():
    print(k, v)

print("When looping through a sequence, the position index and corresponding value can be retrieved at the same time "
      "using the enumerate() function")
for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)

print("To loop over two or more sequences at the same time, the entries can be paired with the zip() function")
questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
for q, a in zip(questions, answers):
    print('What is your {0}?  It is {1}.'.format(q, a))

print("To loop over a sequence in sorted order, use the sorted() function which returns a new sorted list while "
      "leaving the source unaltered")
basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for i in sorted(basket):
    print(i)

print("Using set() on a sequence eliminates duplicate elements")
basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for f in sorted(set(basket)):
    print(f)

print("-----------------------")
print("Standard Modules")
import sys

sys.path.append("C:/Users/jason/PycharmProjects/Python/MyPackage")
for i in sys.path:
    print(i)

print("-----------------------")
print("Packages")
print("Here’s a possible structure for your package")
print("""\
sound/                          Top-level package
      __init__.py               Initialize the sound package
      formats/                  Subpackage for file format conversions
              __init__.py
              wavread.py
              wavwrite.py
              aiffread.py
              aiffwrite.py
              auread.py
              auwrite.py
              ...
      effects/                  Subpackage for sound effects
              __init__.py
              echo.py
              surround.py
              reverse.py
              ...
      filters/                  Subpackage for filters
              __init__.py
              equalizer.py
              vocoder.py
              karaoke.py
              ...
""")

print("""\
    ==================================================================
    =============================Indexing=============================
    =============================20210317=============================
    ==================================================================
    """
      )

import pandas as pd

print("reading file")
with open('C:/Users/jason/Working_Folder/03_Python/testing.txt') as f:
    # read_data = f.read()
    # read_data_l1 = f.readline()
    # read_data_l2 = f.readline()
    for line in f:
        print(line, end='')

# print(read_data)
# print(read_data_l1)
# print(read_data_l2)

print()
print(f.closed)

print("-----------------------")
print("Working With JSON Data in Python")
print("https://realpython.com/python-json/")
import json

print("""\
The process of encoding JSON is usually called serialization. 
This term refers to the transformation of data into a series of 
bytes (hence serial) to be stored or transmitted across a network. 
You may also hear the term marshaling, but that’s a whole other discussion. 
Naturally, deserialization is the reciprocal process of decoding data 
that has been stored or delivered in the JSON standard.
""")

print("""\
Simple Python objects are translated to JSON according to a fairly intuitive conversion.

Python	                JSON
dict	                object
list,tuple	            array
str	                    string
int, long, float	    number
True	                true
False	                false
None	                null
""")

data = {
    "president": {
        "name": "Zaphod Beeblebrox",
        "species": "Betelgeusian"
    }
}

with open("C:/Users/jason/Working_Folder/03_Python/data_file.json", "w") as write_file:
    json.dump(data, write_file)

print(json.dumps(data))
print(json.dumps(data, indent=4))

import requests

response = requests.get("https://jsonplaceholder.typicode.com/todos")
todos = json.loads(response.text)

print(todos == response.json())
print(type(todos))
print(todos[:10])

# Map of userId to number of complete TODOs for that user
todos_by_user = {}

# Increment complete TODOs count for each user.
for todo in todos:
    if todo["completed"]:
        try:
            # Increment the existing user's count.
            todos_by_user[todo["userId"]] += 1
        except KeyError:
            # This user has not been seen. Set their count to 1.
            todos_by_user[todo["userId"]] = 1

print(todos_by_user)

# Create a sorted list of (userId, num_complete) pairs.
top_users = sorted(todos_by_user.items(),
                   key=lambda x: x[1], reverse=True)

print(top_users)

# Get the maximum number of complete TODOs.
max_complete = top_users[0][1]
print(max_complete)

# Create a list of all users who have completed
# the maximum number of TODOs.
users = []
for user, num_complete in top_users:
    if num_complete < max_complete:
        break
    users.append(str(user))

max_users = " and ".join(users)

print(max_users)

print("""\
    ==================================================================
    =============================Indexing=============================
    =============================20210318=============================
    ==================================================================
    """
      )

print("Errors and Exceptions")

try_cnt = 4

"""
while try_cnt > 0:
    try:
        x = int(input("Please enter a number: "))
        print("The number you enter is", x)
        break
    except ValueError:
        if try_cnt - 1 > 0:
            try_cnt = try_cnt - 1
            print("Oops!  That was no valid number.  Try again...", try_cnt, " attempt(s) left")
        else:
            print("Game Over !")
            break
"""

print("""\
    ==================================================================
    =============================Indexing=============================
    =============================20210322=============================
    ==================================================================
    """
      )

print("Python Scopes and Namespaces")
print("""/
At any time during execution, there are 3 or 4 nested scopes whose namespaces are directly accessible:

the innermost scope, which is searched first, contains the local names

the scopes of any enclosing functions, which are searched starting with the nearest enclosing scope, contains non-local, but also non-global names

the next-to-last scope contains the current module’s global names

the outermost scope (searched last) is the namespace containing built-in names
""")

print("A special quirk of Python is that – if no global or nonlocal statement is in effect – "
      "assignments to names always go into the innermost scope.")


def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)


scope_test()
print("In global scope:", spam)


class MyClass:
    """class instantiation automatically invokes"""
    def __init__(self):
        self.data = []

    """A simple example class"""
    i = 12345

    def f(self):
        return 'hello world'


MC_ob_1 = MyClass()

print(MC_ob_1.i)
print(MC_ob_1.f())


class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart


x = Complex(3.0, -4.5)

print(x.r)
print(x.i)


print("Data attributes need not be declared; like local variables, "
      "they spring into existence when they are first assigned to")
x.counter = 1
while x.counter < 10:
    x.counter = x.counter * 2
print(x.counter)
del x.counter

