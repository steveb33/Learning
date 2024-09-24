#Lessons 1 though 3 review

#split() can be useful when you cant be bothered for doing quotes for every individual element of a string
the_string = 'Lamar Jackson rushed for 10 yards'
print(the_string.split(' '))
print(' '.join(['Lamar', 'Jackson', 'rushed', 'for', '10', 'yards']))

first_name = 'Lamar'
last_name = 'Jackson'

### Different string methods
# our_string = '{} {} rushed for 10 yards'.format(first_name, last_name)
# print(our_string)
# our_string = '{0} {1} rushed for 10 yards'.format(first_name, last_name)
# print(our_string)
# our_string = '{first_name} {last_name} rushed for 10 yards'.format(first_name=first_name, last_name=last_name)
# print(our_string)
# print(f'{first_name} {last_name} rushed for 10 yards')

#Official stats from 2022

fantasy_data = {
    'player_name': 'Kenneth Walker',
    'rushing_yards': 1050
}

# appending a new key:value pair to our dictionary
fantasy_data['rushing_touchdowns'] = 9

print(fantasy_data)

player = {
    'name': 'Justin Jefferson',
    'catches': 128,
    'RAC': 340
}

yds = player.get('yds', 0)
catches = player.get('catches', 0)

print(yds, catches)

for k, v in player.items():
    print('{0}: {1}'.format(k, v))

print(player.items())

#Advanced Iterable Operations and More List Methods
redzone_yardlines = []

for yardline in range(1, 21):
    redzone_yardlines.append(yardline)
print(redzone_yardlines)

redzone_yardlines = [yardline for yardline in range(1, 21)]
print(redzone_yardlines)

players = [{
    'name': 'Justin Jefferson',
    'catches': 128,
    'yds': 1809,
    'td': 8
}, {
    'name': 'Tyreek Hill',
    'catches': 119,
    'yds': 1553,
    'td': 11
}]

fantasy_points = []
for player in players:
    points_scored = player.get('catches', 0) + player.get('yds')*0.1 + player.get('td')*6
    fantasy_points.append(points_scored)

print(sum(fantasy_points)/len(fantasy_points))

#This is a more pythonic method that also gives me a function I can use later on too
def calc_fantasy_points(player):
    return player.get('catches', 0) + player.get('yds', 0)*0.1 + player.get('td', 0)*6

fantasy_points = [calc_fantasy_points(player) for player in players]
print(sum(fantasy_points)/len(fantasy_points))

catches = [10, 15, 34, 23]

#lambda funtions are anonymous functions
half_ppr_values = map(lambda x: x/2, catches)
half_ppr_values = list(half_ppr_values)

print(half_ppr_values)

#WHILE
i = 0
while i <= 6:
    print(i)
    i = i + 1

#ZIP
player_names = ['Christian McCaffrey', 'Austin Ekeler', 'Rhamondre Stevenson']
receptions = [85, 107, 69]

player_receptions = dict(zip(player_names, receptions))

print(player_receptions)