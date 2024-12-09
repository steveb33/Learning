

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Read in CSV and create output path
data = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/LoanDefaultPrediction/Loan_default.csv')
output_path = '/Users/stevenbarnes/Desktop/Resources/Data/LoanDefaultPrediction/'
os.makedirs(output_path, exist_ok=True)

# # Look for Nulls to see if data needs to be amended
# pd.set_option('display.max_columns', None)  # Show all columns
# print(data.info())

# Initialize the Label Encoder and encode the categorical features
le = LabelEncoder()
obj_col = ['HasCoSigner', 'LoanPurpose', 'HasDependents', 'HasMortgage', 'MaritalStatus', 'EmploymentType', 'Education']

for col in obj_col:
    data[col] = le.fit_transform(data[col])

# # Make sure the categorical features were properly encoded
# print(data.info())

# Create a correlation matrix with the encoded data to visualize and play with in Tableau
corr_matrix = data.corr(numeric_only=True)

# Add an index column titled "Features" with feature names
corr_matrix.insert(0, 'Features', corr_matrix.columns)

# Output the correlation matrix
output_file = os.path.join(output_path, 'LoanDefaultCorr.csv')
corr_matrix.to_csv(output_file, index=False)

