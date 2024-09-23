#Section: Functions & Conditionals

import pandas as pd

players = [{
   'name': 'A.J. Brown',
   'catches': 88,
   'targets': 145
   },
   {
   'name': 'CeeDee Lamb',
   'catches': 107,
   'targets': 156
   },
   {
   'name': 'Justin Jefferson',
   'catches': 128,
   'targets': 184
   },
]

def get_catch_rate(player):
    if type(player) is not dict:
        print('You need to pass in a dictionary!')
        return
    else:
        pass

    catches = player['catches']
    targets = player['targets']

    if catches > targets:
        print('You can not have more catches than targets')
        return
    else:
        pass

    catch_rate = catches/targets

    return catch_rate

brown = players[0]
browns_catch_rate = get_catch_rate(brown)
print(browns_catch_rate)

def calc_yards_per_carry(attempts, yards):
    yards_per_carry = yards / attempts
    return yards_per_carry

attempts = 20
yards = 100
ypc = calc_yards_per_carry(attempts, yards)
print("Yards per carry:", ypc)

def score_group(total_points):
    if total_points > 200:
        return "You are a top scorer"
    elif total_points >= 100 and total_points >= 200:
        return "You are an average scorer"
    else:
        return "You are a low scorer"

player1_points = 250
player2_points = 150

print(score_group(player1_points))
print(score_group(player2_points))

my_var = 5 > 6
print(my_var, type(my_var))

# Let's define two dictionaries representing the performance stats for two quarterbacks in a game
mahomes_performance_game1 = {"Passing Yards": 340, "Touchdowns": 3, "Interceptions": 0}
mahomes_performance_game2 = {"Passing Yards": 340, "Touchdowns": 3, "Interceptions": 0}

print(mahomes_performance_game1 is mahomes_performance_game2)

