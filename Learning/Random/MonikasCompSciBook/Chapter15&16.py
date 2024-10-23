import math
import copy

# 15.1 - Distance between 2 points
def distance_between_points(point1, point2):
    distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    print(distance)

distance_between_points((5.0, 16.0), (3.0, 4.0))

class Point(object):
    """Represents a point in 2D space"""

class Rectange(object):
    """Represents a rectangle

    attributes: width, height, corner
    """
def print_point(p):
    print('(%g, %g)' % (p.x, p.y))

box = Rectange()
box.width = 100.0
box.height = 200.0

box.corner = Point()
box.corner.x = 0.0
box.corner.y = 0.0

def find_center(rect):
    p = Point()
    p.x = rect.corner.x + rect.width/2.0
    p.y = rect.corner.y + rect.height/2.0
    return p

center = find_center(box)
print_point(center)

# Modifying objects - growing a rectangle
def grow_rectangle(rect, dwidth, dheight):
    rect.width += dwidth
    rect.height += dheight
print(f'Original width and height: ({box.width}, {box.height})')
grow_rectangle(box, 50, 100)
print(f'New width and height: ({box.width}, {box.height})')

# 15.2 - Chaninging the location of an object
def move_rectangle(rect, dx, dy):
    rect.corner.x += dx
    rect.corner.y += dy
    return rect
print(box.corner.x, box.corner.y)
move_rectangle(box, 5, 10)
print(box.corner.x, box.corner.y)

# Text notes contd
box2 = copy.copy(box)
print(box2.corner is box.corner)

# 15.3 - Deepcopy version of move rectangle
def move_rectangle(rect, dx, dy):
    new_rect = copy.deepcopy(rect)
    new_rect.corner.x += dx
    new_rect.corner.y += dy
    return new_rect

""" Test the original and new rectangle """
print(f"Original Rectangle Corner: ({box.corner.x}, {box.corner.y})")
moved_box = move_rectangle(box, 5, 10)
print(f"Moved Rectangle Corner: ({moved_box.corner.x}, {moved_box.corner.y})")
print(f"Original Rectangle Corner After Move: ({box.corner.x}, {box.corner.y})")

Chapter 16 Starts Here
class Time(object):
    """Represents the time of day

    attributes: hour, minute, second
    """

time = Time()
time.hour = 11
time.minute = 59
time.second = 30

# 16.1 Print Time
def print_time(time):
    print('%.2d:%.2d:%.2d' % (time.hour, time.minute, time.second))

print_time(time)

# 16.2 - Is time 1 after time 2
time2 = Time()
time2.hour = 6
time2.minute = 47
time2.second = 13
def is_after(t1, t2):
    return (t1.hour, t1.minute, t1.second) > (t2.hour, t2.minute, t2.second)
print(is_after(time, time2))

# Chapter 16 notes
def add_time(t1, t2):
    sum = Time()
    sum.hour = t1.hour + t2.hour
    sum.minute = t1.minute + t2.minute
    sum.second = t1.second + t2.second
    if sum.second >= 60:
        sum.second -= 60
        sum.minute += 1
    if sum.minute >= 60:
        sum.minute -= 60
        sum.hour += 1
    return sum

total_time = add_time(time, time2)
print_time(total_time)

# 16.3 - Incrementing without loops
def increment(time, seconds):
    time.second += seconds

    # Calculate the amount of full minutes in the seconds and the remainder
    time.minute += time.second // 60
    time.second = time.second % 60

    # Calculate the amount of full hours in the minutes and the remainder
    time.hour += time.minute // 60
    time.minute = time.minute % 60
    return time

increment_test = increment(time, 50)
print_time(increment_test)

# 16.4 - Pure version of increment
def pure_increment(time, seconds):
    # Create a new time object
    new_time = Time()
    new_time.hour = time.hour
    new_time.minute = time.minute
    new_time.second = time.second + seconds

    # Calc the amount of full minutes and adjust seconds
    new_time.minute += new_time.second // 60
    new_time.second = new_time.second % 60

    # Calc the amount of full hours and adjust minutes
    new_time.hour += new_time.minute // 60
    new_time.minute = new_time.minute % 60
    return new_time
pure_test = pure_increment(time, 5000)
print_time(pure_test)

# 16.5 Rewrite increment with new functions
def time_to_int(time):
    minutes = time.hour * 60 + time.minute
    seconds = minutes * 60 + time.second
    return seconds

def int_to_time(seconds):
    time = Time()
    minutes, time.second = divmod(seconds, 60)
    time.hour, time.minute = divmod(minutes, 60)
    return time

def new_increment(time, seconds):
    total_seconds = time_to_int(time) + seconds
    return int_to_time(total_seconds)

new_inc_test = new_increment(time, 6000)
print_time(new_inc_test)

