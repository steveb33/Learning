import math

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
print((box.width, box.height))
grow_rectangle(box, 50, 100)
print((box.width, box.height))

# 15.2 - Chaninging the location of an object
def move_rectangle(rect, dx, dy):
    rect.corner.x += dx
    rect.corner.y += dy
    return rect
print(box.corner.x, box.corner.y)
move_rectangle(box, 5, 10)
print(box.corner.x, box.corner.y)

