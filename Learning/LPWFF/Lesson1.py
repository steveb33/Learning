import pandas as pd

#Official player stats for 2022

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
brows_catch_rate = get_catch_rate(brown)
print(round(brows_catch_rate,4))

def calculate_yards_per_carry(attempts, yards):
    yards_per_carry = yards / attempts
    return yards_per_carry

attempts = 20
yards = 100
ypc = calculate_yards_per_carry(attempts, yards)
print("Yards per carry:", ypc)

most_single_game_ppr_points = [59.5, 57.9, 57.4 ,56.2]
most_single_game_rushing_yards = ["Rushing", 296, 295, 286, 278]
list_of_lists = [most_single_game_ppr_points, most_single_game_rushing_yards]
lists = pd.DataFrame(list_of_lists)

print(lists)

#Official player stats for 2022

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

for player in players:
    name = player["name"]
    catches = player["catches"]
    targets = player["targets"]
    catch_rate = catches/targets
    print(name + ' has a catch rate of ' + str(round(catch_rate,3)))

