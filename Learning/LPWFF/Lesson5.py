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

sns.set_style('whitegrid')
sns.displot(rb_df['RushingAtt'], kde=True, stat='density')
# Set the x-axis limit to start at 0
plt.xlim(0)
# plt.show()

# Grabbing ADP Data
adp_df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/06-Data%20Munging/02-ADP%20Data%20-%20(2023.03.30).csv')

print(adp_df.head())