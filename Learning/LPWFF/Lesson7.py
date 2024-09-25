# Correlation Matrices

import pandas as pd; pd.set_option('display.max_columns', None)
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/08-Correlation%20Matrices/weekly_df.csv')

# Print out to see all the different positions available for analysis
# print(df['Position'].unique())

skill_positions = ['QB', 'WR', 'TE', 'RB']

df = df.loc[df['Position'].isin(skill_positions)]
# print(df.shape)

columns = ['Player', 'Tm', 'Position', 'Week', 'PPRFantasyPoints']

new_df = df[columns]

new_df = new_df.groupby(['Player', 'Tm', 'Position'], as_index=False).agg({
    'PPRFantasyPoints': np.mean # calculating the mean for FantasyPoints per game values
})
# print(new_df.head())

position_map = {
    'QB': 1,
    'RB': 2,
    'WR': 3,
    'TE': 2
}

def get_top_n_player_at_each_position(df, pos, n):

    df = df.loc[df['Position'] == pos]
    return df.groupby('Tm', as_index=False).apply(
        lambda x: x.nlargest(n, ['PPRFantasyPoints']).min()
    )

corr_df = pd.DataFrame(columns=columns) # initialize an empty DataFrame with out columns we initialized in the cell above

for pos, n_spots in position_map.items():
    for n in range(1, n_spots + 1):
        pos_df = get_top_n_player_at_each_position(new_df, pos, n)
        pos_df = pos_df.rename({'PPRFantasyPoints': f'{pos}{n}'}, axis=1)
        corr_df = pd.concat([corr_df, pos_df], axis=1)

corr_df = corr_df.dropna(axis=1)
corr_df = corr_df.drop(['Position', 'Player', 'Tm'], axis=1)

# print(corr_df.shape)
# print(corr_df.corr())

sns.set_style('whitegrid');
plt.figure(figsize=(10, 7))
sns.heatmap(corr_df.corr(), annot=True, cmap=sns.diverging_palette(0, 250));
plt.show()