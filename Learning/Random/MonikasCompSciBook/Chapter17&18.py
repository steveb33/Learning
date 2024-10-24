import random

# Chapter 17
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

# 2 different ways to use a method within an object
start.print_time()
Time.print_time(start)

print('Seconds in start time:', start.time_to_int())

end = start.increment(1337)
end.print_time()

print(end.is_after(start))

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

# Chapter 18
import random


class Card(object):
    """Represents a standard playing card"""

    def __init__(self, suit=0, rank=2):
        self.suit = suit
        self.rank = rank

    suit_names = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    rank_names = [None, 'Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']

    def __str__(self):
        return '%s of %s' % (Card.rank_names[self.rank], Card.suit_names[self.suit])

    # Old cmp used in the book
    # def __cmp__(self, other):
    #     # Check the suits
    #     if self.suit > other.suit: return 1
    #     if self.suit < other.suit: return -1
    #
    #     # Suits are the same... check ranks
    #     if self.rank > other.rank: return 1
    #     if self.rank < other.rank: return -1
    #
    #     # Ranks are the same... it's a tie
    #     return 0

    # Using __lt__ instead of __cmp__ per python 3
    def __lt__(self, other):
        if self.suit != other.suit:         # Check the suits first
            return self.suit < other.suit
        return self.rank < other.rank       # If suits are the same, check rank

class Deck(object):

    def __init__(self):
        self.cards = []
        for suit in range(4):
            for rank in range(1, 14):
                card = Card(suit, rank)
                self.cards.append(card)

    def __str__(self):
        res = []
        for card in self.cards:
            res.append(str(card))
        return '\n'.join(res)

    def pop_card(self):
        return self.cards.pop()

    def add_card(self, card):
        self.cards.append(card)

    def shuffle(self):
        random.shuffle(self.cards)

    def sort(self):         # Exercise 18.2
        self.cards.sort()   # Python will use the Card's lt method for sorting

# # Proof for Exercise 18.2
# deck = Deck()
# deck.shuffle()
# print("Shuffled deck:")
# print(deck)
# print("\nSorted deck:")
# deck.sort()
# print(deck)

# 18.1 - Time class with cmp
class Time(object):
    """Represents a time of the day"""

    def __init__(self, hour=0, minute=0, second=0):
        self.hour = hour
        self.minute = minute
        self.second = second

# My attempt at exercise 18.1

    # def __cmp__(self, other):
    #     if self.hour > other.hour:
    #         return 1
    #     if self.hour < other.hour:
    #         return -1
    #     if self.minute > other.minute:
    #         return 1
    #     if self.minute < other.minute:
    #         return -1
    #     if self.second > other.second:
    #         return 1
    #     if self.second < other.second:
    #         return -1
    #     return 0

    # More concise answer to exercise 18.1
    def __cmp__(self, other):
        return (self.hour, self.minute, self.second) > (other.hour, other.minute, other.second)
        """If I want to return the -1, 0, and 1 like the above example, I would need to change the '>' to '-' """

