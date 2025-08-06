import turtle as t

doug = t.Turtle()
screen = t.Screen()

def move_forwards():
    doug.forward(10)

def move_backwards():
    doug.backward(10)

def turn_left():
    doug.left(10)

def turn_right():
    doug.right(10)

screen.listen()
screen.onkey(key='w', fun=move_forwards)
screen.onkey(key='s', fun=move_backwards)
screen.onkey(key='a', fun=turn_left)
screen.onkey(key='d', fun=turn_right)
screen.onkey(key='c', fun=doug.clear)

screen.exitonclick()
