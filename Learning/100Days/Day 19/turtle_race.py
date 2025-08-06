import turtle
import turtle as t
import random

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
is_race_on = False
screen = t.Screen()
screen.setup(500, 400)
user_bet = screen.textinput(title='Make your bet!', prompt='Which turtle will win the race? Enter a color?')
all_turtles = []

# doug = t.Turtle(shape='turtle')
# doug.penup()
# doug.goto(x=-230, y=-100)
"""Failed idea"""
# turtle_dict = {}
# for i in colors:
#     turtle_dict[i] = t.Turtle(shape='turtle')
#
# for turtle_ in turtle_dict:
#     turtle_dict[turtle].color()
#     turtle_dict[turtle].penup()
#     y_factor =
#     y_coordinate = -100 + y_factor*(200/6)
#     turtle_dict[turtle].goto(x=-230, y=y_coordinate)

"""Answer"""
y_positions = [-70, -40, -10, 20, 50, 80]
for turtle_index in range(0, 6):
    new_turtle = t.Turtle(shape='turtle')
    new_turtle.color(colors[turtle_index])
    new_turtle.penup()
    new_turtle.goto(x=-230, y=y_positions[turtle_index])
    turtle.speed('fastest')
    all_turtles.append(new_turtle)

if user_bet:
    is_race_on = True

while is_race_on:
    for turtle in all_turtles:
        if turtle.xcor() > 230:
            is_race_on = False
            winning_color = turtle.pencolor()
            if winning_color == user_bet:
                print(f"You've won! The {winning_color} turtle has won the race!")
            else:
                print(f"You've lost! The {winning_color} turtle has won the race!")
        rand_distance = random.randint(0,10)
        turtle.forward(rand_distance)



screen.exitonclick()

