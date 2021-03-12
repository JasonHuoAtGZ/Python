# single line of comments

"""
multiple line of comments
"""

"""
==================================================================
=============================Indexing=============================
=============================20210312=============================
==================================================================
"""

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
b = a+b
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
add = lambda x,y:x+y
print(add(3,4))
print((lambda x,y:x+y)(5,6))

my_list = [3,5,-4,-1,0,-2,-6]
print(sorted(my_list, key=lambda x: abs(x)))

print(list(map(lambda x:x*x,range(1,21))))
print(list(filter(lambda x:x%2 == 0,range(1,21))))


from functools import reduce
print(reduce(lambda x,y:x+y,range(1,101)))


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
    squares.append(x**2)

print(squares)
print(list(map(lambda x: x**2, range(10))))
print([x**2 for x in range(10)])
print([(x, y) for x in [1,2,3] for y in [3,1,4] if x != y])


vec = [-4, -2, 0, 2, 4]
print([x*2 for x in vec])
print([x for x in vec if x >= 0])
print([abs(x) for x in vec])

freshfruit = ['  banana', '  loganberry ', 'passion fruit  ']
print([weapon.strip() for weapon in freshfruit])

