class Time(object):
    """Represents the time of day"""

    def print_time(self):
        print('%.2d:%.2d:%.2d' % (self.hour, self.minute, self.second))

    def time_to_int(self):
        minutes = self.hour * 60 + self.minute
        seconds = minutes * 60 + self.second
        return seconds

start = Time()
start.hour = 9
start.minute = 45
start.second = 00

# 2 different ways to use a method within an object
start.print_time()
Time.print_time(start)

print(start.time_to_int())