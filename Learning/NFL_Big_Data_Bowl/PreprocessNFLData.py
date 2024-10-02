import pandas as pd; pd.set_option('display.max_columns', None)
from sklearn.preprocessing import StandardScaler, LabelEncoder

"""
The goal of this program is to attempt to accurately predict the yards gained per running play based on the offensive
and defense personnel and packages present.

The dataset can be found at the following: https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/data

This first section is the preprocessing of the data.
"""

df = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/NFL Big Data Bowl/train.csv', low_memory=False)

# Simplify the df so that the only instance of each PlayId is for the rusher
df = df[df['NflId'] == df['NflIdRusher']].copy()

# Remove non-traditional defensive packages
df = df[~df['DefensePersonnel'].str.contains('RB|OL')]

# Remove non-traditional offensive packages
df = df[~df['OffensePersonnel'].str.contains('DB|DL|LB|OL|QB')]

# Helper function to extract the number of players of each position type
def extract_count(position, personnel):
    parts = [part.strip() for part in personnel.split(',')]
    for part in parts:
        if position in part:
            return int(part.split()[0]) # Get the number before the position
    return 0 # return 0 if the position is not found

# Apply the function to create position count columns
df['CountRB'] = df['OffensePersonnel'].apply(lambda x: extract_count('RB', x))
df['CountTE'] = df['OffensePersonnel'].apply(lambda x: extract_count('TE', x))
df['CountWR'] = df['OffensePersonnel'].apply(lambda x: extract_count('WR', x))
df['CountDL'] = df['DefensePersonnel'].apply(lambda x: extract_count('DL', x))
df['CountLB'] = df['DefensePersonnel'].apply(lambda x: extract_count('LB', x))
df['CountDB'] = df['DefensePersonnel'].apply(lambda x: extract_count('DB', x))

# Create a custom yardline function for how far the ball is from opponent's endzone
def change_yardline(row):
    if row['PossessionTeam'] != row['FieldPosition']:
        return row['YardLine']
    else:
        return (50-row['YardLine']) + 50

# Apply the yardline function
df['Ball_X'] = df.apply(change_yardline, axis=1)

# Function to categorize defensive personnel groupings
def categorize_defense(personnel):
    if '7 DB' in personnel or '8 DB' in personnel:
        return 'Quarter' # Typically prevent defense
    elif '6 DB' in personnel:
        return 'Dime'
    elif '5 DB' in personnel:
        return 'Nickel'
    elif '4 DB' in personnel:
        return 'Base'
    elif '3 DB' in personnel:
        return '3 DB' # This would be 6-2, 5-3, or 4-4 formations (heavy run stuff)
    else:
        return 'Heavy' # For the 1 or 2 DB packages (goaline)

# Add the Defense Package to the Dataframe
df['DefensePackage'] = df['DefensePersonnel'].apply(categorize_defense)

# Simplify the Offensive Personnel to common terminology (13, 21, 30, etc. personnel)
df['PersonnelPkg'] = df['CountRB'].astype(str) + df['CountTE'].astype(str)

# Select defensive, offensive, and other relevant variables
defensive_variables = df[['DefensePersonnel', 'DefensePackage', 'DefendersInTheBox', 'CountDB', 'CountDL', 'CountLB']]
offensive_variables = df[['OffenseFormation', 'OffensePersonnel', 'PersonnelPkg', 'CountWR', 'CountTE', 'CountRB']]
target = df['Yards']

# Concatenate the different variables into a single dataframe
df = pd.concat([df[['PlayId']], defensive_variables, offensive_variables, target], axis=1)

# Set 'PlayId' as the index
df.set_index('PlayId', inplace=True)

# Output the dataframe to a CSV file
df.to_csv('/Users/stevenbarnes/Desktop/Resources/Data/NFL Big Data Bowl/cleaned_data.csv', index=True)

"""
The with preprocessing completed for the ML methods, the following section will now prepare the data for the array
of ML methods being tested in this analysis.
"""

# Separate features into continuous and discrete
continuous_vars = ['DefendersInTheBox', 'CountDB', 'CountDL', 'CountLB', 'CountWR',
                   'CountTE', 'CountRB', 'Yards']
discrete_vars = ['DefensePersonnel', 'DefensePackage', 'OffenseFormation', 'OffensePersonnel',
                 'PersonnelPkg']

# 1. Dataset for Linear Regression / Neural Networks (scaling & one-hot encoding)

# One-hot encode categorical variables
df_lr_nn = pd.get_dummies(df, columns=discrete_vars, drop_first=True)

# Scale the continuous variables
scaler = StandardScaler()
df_lr_nn[continuous_vars] = scaler.fit_transform(df_lr_nn[continuous_vars])

# Save df_lr_nn as the dataset for Linear Regression / Neural Networks
df_lr_nn.to_csv('/Users/stevenbarnes/Desktop/Resources/Data/NFL Big Data Bowl/lin_reg_nn_data.csv', index=False)

# 2. Dataset for tree-based models (label encoding for categorical variables, no scaling)

# Make a copy
df_tree = df.copy()

# Label encode categorical variables (instead of one-hot encoding)
label_encoder = LabelEncoder()
for col in discrete_vars:
    df_tree[col] = label_encoder.fit_transform(df_tree[col])

# Save df_tree as the dataset for Tree-Based Models
df_tree.to_csv('/Users/stevenbarnes/Desktop/Resources/Data/NFL Big Data Bowl/tree_data.csv', index=False)

# Display a sample from each dataset
print("Linear Regression / Neural Networks Dataset:")
print(df_lr_nn.head())

print("\nTree-Based Models Dataset:")
print(df_tree.head())