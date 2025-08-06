import turtle as t
import random

from scipy.stats import randint

doug = t.Turtle()
doug.shape('turtle')
doug.speed('fastest')
t.colormode(255)


def square():
    for i in range(4):
        doug.forward(100)
        doug.right(90)
# square()

def dash_line():
    for i in range(10):
        doug.penup()
        doug.forward(8)
        doug.pendown()
        doug.forward(8)
# dash_line()

def draw_shape(sides):
    angle = 360 / sides
    for _ in range(sides):
        doug.left(angle)
        doug.forward(60)
# draw_shape(9)

def random_colors():
    r = random.randint(1, 255)
    g = random.randint(1, 255)
    b = random.randint(1, 255)
    random_color = (r, g, b)
    return random_color

def random_direction():
    doug.pencolor(random_colors())
    doug.right(random.randint(1, 360))
    doug.forward(25)
    doug.speed(10)


def random_walk(steps):
    for _ in range(steps):
        random_direction()

random_walk(25)


screen = t.Screen()
screen.exitonclick()