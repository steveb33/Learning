"""Testing new stuff from chapters 5 & 6"""
import math


# Recursion Example
def countdown(n):
    if n <= 0:
        print('Blastoff!')
    else:
        print(n)
        countdown(n-1)
countdown(10)

# Exercise 5.3
def check_fermat(a, b, c, n):
    if n < 2:
        print("n must be larger than 2")
    elif a**n + b**n == c**n:
        print('Holy smokes, Fermat was wrong!')
    else:
        print("No that doesn't work")
check_fermat(2,3,4,2)

# Exercise 5.4
def is_triangle(a, b, c):
    if a+b<c or b+c<a or a+c<b:
        print('No')
    else:
        print('Yes')
is_triangle(10,3,12)


# Area of a circle - can just return the answer without having to assign it first
def area(radius):
    return math.pi*radius**2

# Distance with return
def distance(x1, x2, y1, y2):
    dx = x2 - x1
    dy = y2 - y1
    dsquared = dx**2 + dy**2
    result = math.sqrt(dsquared)
    return result   # Only use print for proving/debugging if it is not integral to the script

# Defining a circle function with composition (calling a function within another)
# You can also shorten the lines of code by combining functions
def circle_area(xc, yc, xp, yp):
    return area(distance(xc, yc, xp, yp))
print(round(circle_area(1, 2, 4, 6),4))

# Recurssion example with factorials - Note, the first two if statements act as 'Guardians'
def factorial(n):
    if not isinstance(n, int):
        print('Factorial is only defined for integers')
        return None
    elif n < 0:
        print('Factorial is not defined for negative integers')
        return None
    elif n == 0:
        return 1
    else:
        return n * factorial(n-1)
print(factorial(8))

# Fibonacci recursion example
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
print(fibonacci(10))

# Fancy Factorial Printing
def factorial(n):
    space = ' ' * (4 * n)
    print(space, 'factorial', n)
    if n == 0:
        print(space, 'returning 1')
        return 1
    else:
        recurse = factorial(n-1)
        result = n * recurse
        print(space, 'returning', result)
        return result
factorial(6)

