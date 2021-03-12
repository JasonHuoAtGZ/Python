# single line of comments

"""
multiple line of comments
"""

"""
==================================================================
=============================20210312=============================
==================================================================
"""

# this is the first comment
spam = 1  # and this is the second comment
          # ... and now a third!
text = "# This is not a comment because it's inside quotes."

print(text)

print("# division always returns a floating point number")
print(1/3+2)

print("# floor division discards the fractional part")
print(17//3)

print("# the % operator returns the remainder of the division")
print(17 % 3)

print("equal sign (=) is used to assign a value to a variable")
width=2
height=9
area = width * height
print(area)

print("# single quotes")
print('spam eggs')

print("# use \' to escape the single quote")
print('doesn\'t')

print("or use double quotes instead")
print("doesn't")

print("If you don’t want characters prefaced by \ to be interpreted as special characters, you can use raw strings by adding an r before the first quote")
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
word='Python'
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

print("Python strings cannot be changed — they are immutable. Therefore, assigning to an indexed position in the string results in an error:")
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
