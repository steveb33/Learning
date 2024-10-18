import math

# 15.1 - Distance between 2 points
def distance_between_points(point1, point2):
    distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    print(distance)

distance_between_points((5.0, 16.0), (3.0, 4.0))

