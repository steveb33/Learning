# DATA MUNGING

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/06-Data%20Munging/01-FDP%20Projections%20-%20(2023.03.30).csv')

# Operating DataFrame
# print(df.head())
# print(type(df))
# print(df[:5])
# print(df.iloc[10:15, 0:10])
# print(', '.join(df.columns))

# Adding a column to the DataFrame
scoring_weights = {
    'receptions': 0.5, # half-PPR
    'receiving_yds': 0.1,
    'receiving_td': 6,
    'rushing_yds': 0.1,
    'rushing_td': 6,
    'passing_yds': 0.04,
    'passing_td': 4,
    'int': -2
}

df['FantasyPoints'] = (
    df['Receptions']*scoring_weights['receptions'] + df['ReceivingYds']*scoring_weights['receiving_yds'] + \
    df['ReceivingTD']*scoring_weights['receiving_td'] + \
    df['RushingYds']*scoring_weights['rushing_yds'] + df['RushingTD']*scoring_weights['rushing_td'] + \
    df['PassingYds']*scoring_weights['passing_yds'] + df['PassingTD']*scoring_weights['passing_td'] + \
    df['Int']*scoring_weights['int'])
# Note that the backslash "\" is a line continuation character that allows for easier to read code

# print(df.head())

# How to Calculate VOR
"""
.loc is a way of getting back specified cross sections of your DataFrame.

The syntax is as follows:

new_df = old_df.loc[row_indexer, column_indexer]

Where row_indexer can take the form of a boolean indexer.

For example, df['Pos'] == 'RB'

or, df['RushingAtt'] > 20

or, df['Pos'].isin(['QB', 'WR', 'RB', TE]) # check if a player's position is a skill position

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html # docs on loc

https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html # docs on indexing

"""

# rb_df = df.loc[df['Pos'] == 'RB']
# print(rb_df.head())

base_columns = ['Player', 'Team', 'Pos']
rushing_columns = ['FantasyPoints', 'Receptions', 'ReceivingYds', 'ReceivingTD', 'RushingAtt', 'RushingYds', 'RushingTD']

rb_df = df.loc[(df['Pos'] == 'RB', base_columns + rushing_columns)]
# print(rb_df.sort_values(by='RushingYds', ascending=False).head(15))

# print(rb_df.describe().transpose())

# print(rb_df['RushingAtt'][:10])

rb_df['RushingTDRank'] = rb_df['RushingTD'].rank(ascending=False)
# print(rb_df.sort_values(by='RushingTDRank').head(5))

# sns.set_style('whitegrid')
# sns.displot(rb_df['RushingAtt'], kde=True, stat='density')
# Set the x-axis limit to start at 0
# plt.xlim(0)
# plt.show()

# Grabbing ADP Data
adp_df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/06-Data%20Munging/02-ADP%20Data%20-%20(2023.03.30).csv')

adp_df['ADP RANK'] = adp_df['Current ADP'].rank()
# print(adp_df.head())

adp_df_cutoff = adp_df[:75]
# print(adp_df_cutoff.shape)

replacement_players = {
    'RB': '',
    'QB': '',
    'WR': '',
    'TE': ''
}

for _, row in adp_df_cutoff.iterrows():
    position = row['Pos'] # extract the position and player value from each row as we loop through it
    player = row['Player']

    if position in replacement_players: # if the position is in the dict's keys
        replacement_players[position] = player # set that player as the replacement player

# print(replacement_players)

df = df[['Player', 'Pos', 'Team', 'FantasyPoints']] # filtering out the columns we need
# print(df.head())

replacement_values = {}

for position, player_name in replacement_players.items():
    player = df.loc[df['Player'] == player_name.strip()]
    replacement_values[position] = player['FantasyPoints'].tolist()[0]

# print(replacement_values)

pd.set_option('chained_assignment', None)

df = df.loc[df['Pos'].isin(['QB', 'RB', 'WR', 'TE'])]

df['VOR'] = df.apply(
    lambda row: row['FantasyPoints'] - replacement_values.get(row['Pos']), axis=1
)

# print(df.head())

pd.set_option('display.max_rows', None)

df['VOR Rank'] = df['VOR'].rank(ascending=False)

# print(df.sort_values(by='VOR', ascending=False).head(100))

# print(df.groupby('Pos')['VOR'].describe())

df['VOR'] = df['VOR'].apply(lambda x: (x - df['VOR'].min()) / (df['VOR'].max() - df['VOR'].min()))
df = df.sort_values(by='VOR Rank')
# print(df.head())

num_teams = 12
num_spots = 16
draft_pool = num_teams * num_teams

df_copy = df[:draft_pool]

# sns.set_style('whitegrid')
# sns.boxplot(x=df_copy['Pos'], y=df_copy['VOR'], palette='Set2')
# plt.show()

# Renaming columns before merging the dataframes
df = df.rename({
    'VOR': 'Value',
    'VOR Rank': 'Value Rank'
}, axis=1)

adp_df = adp_df.rename({
    'PLAYER': 'Player',
    'POS': 'Pos',
    'AVG': 'Average ADP',
    'ADP RANK': 'ADP Rank'
}, axis=1)

adp_df = adp_df.drop('Team', axis=1)

df['Player'] = df['Player'].replace({
    'Kenneth Walker III': 'Kenneth Walker',
    'Travis Etienne Jr.': 'Travis Etienne',
    'Brian Robinson Jr.': 'Brian Robinson',
    'Pierre Strong Jr.': 'Pierre Strong',
    'Michael Pittman Jr.': 'Michael Pittman',
    'A.J. Dillon': 'AJ Dillon',
    'D.J. Moore': 'DJ Moore'
})

final_df = df.merge(adp_df, how='left', on=['Player', 'Pos'])

# print(final_df.head(100))

# Calc difference between value rank and ADP rank plus remove outliers
final_df['Diff in ADP and Value'] = final_df['ADP Rank'] - final_df['Value Rank']
final_df = final_df.loc[final_df['ADP Rank'] <= 212]
# print(final_df.head())

draft_pool = final_df.sort_values(by='ADP Rank')[:196]

rb_draft_pool = draft_pool.loc[draft_pool['Pos'] == 'RB']
qb_draft_pool = draft_pool.loc[draft_pool['Pos'] == 'QB']
wr_draft_pool = draft_pool.loc[draft_pool['Pos'] == 'WR']
te_draft_pool = draft_pool.loc[draft_pool['Pos'] == 'TE']

print(wr_draft_pool.sort_values(by='Diff in ADP and Value', ascending=False)[:10])