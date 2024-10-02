from swampy.TurtleWorld import *
import math

world = TurtleWorld()
bob = Turtle()
print(bob)
bob.delay = 0.01

# for looping example to make a square
for i in range(4):
    fd(bob, 100)
    lt(bob)

# Putting the loop in a function to make a square
def square(t, l):
    for i in range(4):
        fd(t, l)
        lt(t)
square(bob, 50)


# Making a polygon
def polygon(t, n, length):
    angle = 360.0 / n
    for i in range(n):
        fd(t, length)
        rt(t, angle)
polygon(bob, 100, 7)

# Making a circle using the polygon function
def circle(t, r):
    circumference = 2 * math.pi * r
    n = 50
    length = circumference / n
    polygon(t, n, length)
bob.delay = 0.01
circle(bob, 100)

# Make a circle based on arc rather than on polygon
def polyline(t, n, l, angle):
    for i in range(n):
        fd(t, l)
        lt(t, angle)
def polygon(t, n, l):
    angle = 360.0 / n
    polyline(t, n, l, angle)
def arc(t, r, angle):
    arc_length = 2 * math.pi * r * angle / 360
    n = int(arc_length / 3) + 1
    step_length = arc_length / n
    step_angle = float(angle) / n
    polyline(t, n, step_length, step_angle)
def circle(t, r):
    arc(t, r, 360)

bob.delay = 0.01
arc(bob, 100, 330)

# Draw a spiral
def draw_spiral(t, n, length=3, a=0.1, b=0.0002):
    """Draws an Archimedian spiral starting at the origin.

    Args:
      n: how many line segments to draw
      length: how long each segment is
      a: how loose the initial spiral starts out (larger is looser)
      b: how loosly coiled the spiral is (larger is looser)

    http://en.wikipedia.org/wiki/Spiral
    """
    theta = 0.0

    for i in range(n):
        t.fd(length)
        dtheta = 1 / (a + b * theta)

        t.lt(dtheta)
        theta += dtheta

draw_spiral(bob, n=1000)

wait_for_user()
