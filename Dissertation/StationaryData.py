import pandas as pd
import numpy as np

# Load the datasets
base_data_path = '/Users/stevenbarnes/Desktop/Dissertation/BaseData.csv'
summary_data_path = '/Users/stevenbarnes/Desktop/Dissertation/Summary.csv'

base_data = pd.read_csv(base_data_path)
summary_data = pd.read_csv(summary_data_path)

# Function to apply transformations
def transform_series(series, transformation):
    if transformation == 1:
        return series
    elif transformation == 2:
        return series.diff().dropna()
    elif transformation == 3:
        return series.diff().diff().dropna()
    elif transformation == 4:
        return series.pct_change().dropna()
    elif transformation == 5:
        return np.log(series).dropna()
    elif transformation == 6:
        return np.log(series).diff().dropna()
    elif transformation == 7:
        return np.log(series).diff().diff().dropna()

# Create a dictionary to store transformed data
transformed_data = {}

# Apply transformations based on Summary.csv
for index, row in summary_data.iterrows():
    variable = row['Variable']
    transformation = row['Transformation']
    if variable in base_data.columns:
        transformed_data[variable] = transform_series(base_data[variable], transformation)

# Convert the dictionary to a DataFrame
transformed_df = pd.DataFrame(transformed_data)

# Ensure the first date is 2011-11-01
transformed_df['Date'] = base_data['Date']
transformed_df = transformed_df.set_index('Date')
transformed_df = transformed_df.loc['2011-11-01':]

# Ensure all values have 7 decimal points
transformed_df = transformed_df.applymap(lambda x: f"{x:.7f}")

# Save the results to CSV file
output_path = '/Users/stevenbarnes/Desktop/Dissertation/Starting Data/'
transformed_df.to_csv(output_path + 'ALL_DATA.csv')

print("Transformations and outputs are successfully saved.")
