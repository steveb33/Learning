import pandas as pd

# Load the datasets
no_enviro_df = pd.read_csv('/Users/stevenbarnes/Desktop/Dissertation/NO_ENVIRO.csv')
all_data_df = pd.read_csv('/Users/stevenbarnes/Desktop/Dissertation/ALL_DATA.csv')

# Display the first few rows of each dataframe to understand their structure
no_enviro_df.head(), all_data_df.head()
