class Time(object):
    """Represents the time of day"""

    def print_time(self):
        print('%.2d:%.2d:%.2d' % (self.hour, self.minute, self.second))

    def time_to_int(self):
        minutes = self.hour * 60 + self.minute
        seconds = minutes * 60 + self.second
        return seconds

    @classmethod
    def int_to_time(cls, seconds):
        time = cls()
        minutes, time.second = divmod(seconds, 60)
        time.hour, time.minute = divmod(minutes, 60)
        return time

    def increment(self, seconds):
        seconds += self.time_to_int()
        return Time.int_to_time(seconds)

    def add_time(self, other):
        seconds = self.time_to_int() + other.time_to_int()
        return Time.int_to_time(seconds)

    def is_after(self, other):
        return self.time_to_int() > other.time_to_int()

    def __init__(self, hour=0, minute=0, second=0):
        self.hour = hour
        self.minute = minute
        self.second = second

    def __str__(self):
        return '%.2d:%.2d:%.2d' % (self.hour, self.minute, self.second)

    def __add__(self, other):
        if isinstance(other, Time):
            return self.add_time(other)
        else:
            return self.increment(other)

    def __radd__(self, other):
        return self.__add__(other)


start = Time()
start.hour = 9
start.minute = 45
start.second = 00

# # 2 different ways to use a method within an object
# start.print_time()
# Time.print_time(start)
#
# print('Seconds in start time:', start.time_to_int())
#
# end = start.increment(1337)
# end.print_time()
#
# print(end.is_after(start))

# Testing out the new methods
time = Time(9, 45)
duration = Time(1, 35)
print(time + duration)  # Tests out the add_time method
print(time + 1337)      # Tests out the increment method
print(1337 + time)      # Tests out the __radd__ method


# 17.2 through 17.4 - Point with learned methods
class Point(object):

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return '(%g, %g)' % (self.x, self.y)

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        elif isinstance(other, tuple):
            return Point(self.x + other[0], self.y + other[1])
        else:
            raise TypeError("Operand must be Point or tuple types")


point = Point()
print(point)
new_point = Point(10, 20)
print(new_point)
move_point = Point(3, 7)
print(new_point + move_point)   # Tests out adding points
print(new_point + (15, 3))      # Tests out adding tuple to point

# 17.7 - Kangaroo
class Kangaroo(object):

    def __init__(self, name, contents=[]):
        """Initializes an empty pouch"""
        self.name = name
        self.pouch_contents = contents

    def __str__(self):
        """Returns a string representation of this Kangaroo"""
        t = [self.name + ' has pouch contents:']
        for obj in self.pouch_contents:
            s = '    ' + object.__str__(obj)
            t.append(s)
        return '\n'.join(t)
    def put_in_pouch(self, item):
        """Adds a new item to the pouch"""
        self.pouch_contents.append(item)


kanga = Kangaroo('Kanga')   # Create kanga object
roo = Kangaroo('Roo')       # Create roo object
kanga.put_in_pouch('wallet')
kanga.put_in_pouch('car keys')
kanga.put_in_pouch(roo)
print(kanga)
print(roo)