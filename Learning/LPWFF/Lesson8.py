# TD Regression Candidates

import pandas as pd; pd.set_option('display.max_columns', None)
import nfl_data_py as nfl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set_style('whitegrid')
import warnings; warnings.simplefilter('ignore')

seasons = range(2008, 2023)
df = nfl.import_pbp_data(seasons)
print(df.head())

for column in df.columns:
    if 'rush' in column:
        print(column)
    elif 'distance' in column:
        print(column)
    elif 'yardline' in column:
        print(column)

rushing_df = df[['rush_attempt', 'rush_touchdown', 'yardline_100', 'two_point_attempt']]
rushing_df = rushing_df.loc[(rushing_df['two_point_attempt'] == 0) & (rushing_df['rush_attempt'] == 1)]

rushing_df_probs = rushing_df.groupby('yardline_100')['rush_touchdown'].value_counts(normalize=True)
# this give s a series, so it needs to turn back into a dataframe
rushing_df_probs = pd.DataFrame({
    'probability_of_touchdown': rushing_df_probs.values
}, index=rushing_df_probs.index).reset_index()

# only keep rushing_touchdown = 1
rushing_df_probs = rushing_df_probs.loc[rushing_df_probs['rush_touchdown'] == 1]

# drop the rush_touchdown since it is now redundant
rushing_df_probs = rushing_df_probs.drop('rush_touchdown', axis=1)

rushing_df_probs.plot(x='yardline_100', y='probability_of_touchdown');
plt.show()

pbp_2022 = nfl.import_pbp_data([2022])

pbp_2022 = pbp_2022[['rusher_player_name', 'rusher_player_id', 'posteam', 'rush_touchdown', 'yardline_100']].dropna()

exp_df = pbp_2022.merge(rushing_df_probs, how='left', on='yardline_100')

exp_df = exp_df.groupby(['rusher_player_name', 'rusher_player_id', 'posteam'], as_index=False).agg({
    'probability_of_touchdown': np.sum,
    'rush_touchdown': np.sum
}).rename({'probability_of_touchdown': 'Expected Touchdowns'}, axis=1)

exp_df = exp_df.sort_values(by='Expected Touchdowns', ascending=False)

exp_df = exp_df.rename(columns={
    'rusher_player_name': 'Player',
    'posteam': 'Team',
    'rusher_player_id': 'ID',
    'rush_touchdown': 'Actual Touchdowns'
})

exp_df['Expected Touchdowns Rank'] = exp_df['Expected Touchdowns'].rank(ascending=False)

roster = nfl.import_rosters([2022])
roster = roster[['player_id', 'position']].rename(columns = {'player_id': 'ID'})

exp_df = exp_df.merge(roster, on='ID')
exp_df = exp_df[exp_df['position'] == 'RB'].drop('position', axis=1)

exp_df['Actual Touchdowns Rank'] = exp_df['Actual Touchdowns'].rank(ascending=False)

exp_df['Regression Candidate'] = exp_df['Expected Touchdowns'] - exp_df['Actual Touchdowns']
exp_df['Regression Candidate Rank'] = exp_df['Actual Touchdowns Rank'] - exp_df['Expected Touchdowns Rank']

fig, ax = plt.subplots(figsize=(12, 8))

exp_df['Positive Regression Candidate'] = exp_df['Regression Candidate'] > 0

sns.scatterplot(
    x='Expected Touchdowns',
    y='Actual Touchdowns',
    hue='Positive Regression Correlation',
    data=exp_df,
    palette=['r', 'g']
);

max_act_touchdowns = int(exp_df['Actual Touchdowns'].max())
max_exp_touchdowns = int(exp_df['Expected Touchdowns'].max())
max_tds = max(max_act_touchdowns, max_exp_touchdowns)

sns.lineplot(x=range(max_tds), y=range(max_tds))

notable_players = ['T.Etienne', 'Ja.Williams', 'A.Ekeler']

for _, row in exp_df.iterrows():
    if row['Player'] in notable_players:
        ax.test(
            x=row['Expected Touchdowns']+0.1,
            y=row['Actual Touchdowns']+0.05,
            s=row['Player']
        )

plt.show()