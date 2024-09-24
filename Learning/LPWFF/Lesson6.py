import pandas as pd; pd.set_option('display.max_columns', None)
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/07-Data%20Visualizations/03-Yearly%20Fantasy%20Stats%20-%202022.csv').iloc[:, 1:]

# The following line allows me to see the information about the csv
# print(df.info(verbose=True))

pd.set_option('chained_assignment', None)

rb_df = df.loc[df['Pos'] == 'RB'].copy()
rb_df['Usage/G'] = (rb_df['Tgt'] + rb_df['RushingAtt']) / rb_df['G']
rb_df['FantasyPoints/G'] = rb_df['FantasyPoints'] / rb_df['G']

print(rb_df.iloc[:,-1].head())

sns.set_style('whitegrid')
plt.figure(figsize=(8, 8))
sns.regplot(x=rb_df['Usage/G'], y=rb_df['FantasyPoints/G'])
sns.kdeplot(rb_df['RushingAtt'])
sns.displot(rb_df['Tgt'], bins=30)
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))

notable_players = ['Austin Ekeler', 'Aaron Jones', 'Jamaal Williams', 'Christian McCaffrey*']

rb_df_filtered = rb_df.loc[rb_df['RushingAtt'] > 50]

for player_name in notable_players:
    player = rb_df_filtered.loc[rb_df_filtered['Player'] == player_name]

    # if our df we get back is not empty, run the code below
    if not player.empty:
        # grab targets and rushing attempts
        targets = player['Tgt']
        rushes = player['RushingAtt']
        ax.annotate(player_name, xy=(rushes+2, targets+2), color='red', fontsize=12)
        ax.scatter(rushes, targets, color='red')

sns.kdeplot(x=rb_df_filtered['RushingAtt'], y=rb_df_filtered['Tgt'], ax=ax, bw_method=0.7)
sns.jointplot(x=rb_df_filtered['RushingAtt'], y=rb_df_filtered['Tgt'], kind='hex', dropna=True);
sns.jointplot(x=rb_df_filtered['RushingAtt'], y=rb_df_filtered['Tgt'], kind='hex');
plt.show()

sns.set_style('dark')

sns.residplot(x=rb_df['Usage/G'], y=rb_df['FantasyPoints/G'])
plt.title('Residual plot')
plt.xlabel('Usage/G')
plt.ylabel('Residual')
plt.show()

rb_df_copy = rb_df[['RushingAtt', 'RushingTD', 'FantasyPoints/G', 'Tgt']]
sns.pairplot(rb_df_copy, kind='reg')
plt.show()

weekly_df = pd.read_csv("https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/07-Data%20Visualizations/01-Weekly%20Fantasy%20Stats%20-%202022/weekly_df.csv")

allen = weekly_df.loc[weekly_df['Player'] == 'Josh Allen']
mahomes = weekly_df.loc[weekly_df['Player'] == 'Patrick Mahomes']
wilson = weekly_df.loc[weekly_df['Player'] == 'Russell Wilson']

sns.set_style('whitegrid')
plt.subplots(figsize=(10, 8))
plt.plot(wilson['Week'], wilson['StandardFantasyPoints']) # first argument is x, second is y
plt.plot(mahomes['Week'], mahomes['StandardFantasyPoints'])
plt.plot(allen['Week'], allen['StandardFantasyPoints'])
plt.legend(['Wilson', 'Mahomes', 'Allen']) # setting legend in order of how we plotted things
plt.xlabel('Week')
plt.ylabel('Fantasy Points Scored')
plt.title('Wilson vs. Mahomes vs. Lamar - week by week Fantasy Performance', fontsize=16, fontweight='bold') # adjusting font size to 16px
plt.show()

plt.figure(figsize=(15, 10))
sns.heatmap(allen.select_dtypes(include=[float, int]).corr()[['StandardFantasyPoints']], annot=True)

df['Usage/G'] = (df['PassingAtt'] + df['Tgt'] + df['RushingAtt'])/df['G']
df['FantasyPoints/G'] = df['FantasyPoints'] / df['G']
sns.lmplot(data=df, x='Usage/G', y='FantasyPoints/G', hue='Pos', height=7);
plt.show()

combine_df = pd.read_csv("https://raw.githubusercontent.com/fantasydatapros/LearnPythonWithFantasyFootball/master/2023/07-Data%20Visualizations/02-Combine%20Data%202000%20to%202023.csv")

print(combine_df.groupby('Pos')['40YD'].describe())

plt.figure(figsize=(8, 8))
sns.boxplot(x='Pos', y='40YD', data=combine_df.loc[combine_df['Pos'].isin(['RB', 'QB', 'TE', 'WR'])], palette=sns.color_palette('husl', n_colors=4))
plt.show()

