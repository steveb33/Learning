import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Load the BaseData dataset
base_data_path = '/Users/stevenbarnes/Desktop/Dissertation/BaseData.csv'
base_data = pd.read_csv(base_data_path)


# Define transformation functions
def first_difference(series):
    return series.diff().dropna()


def second_difference(series):
    return series.diff().diff().dropna()


def return_transformation(series):
    return series.pct_change().dropna()


def log_transformation(series):
    return np.log(series).dropna()


def log_first_difference(series):
    return np.log(series).diff().dropna()


def log_second_difference(series):
    return np.log(series).diff().diff().dropna()


# Function to perform ADF test and check if a series is stationary
def is_stationary(series, significance_level=0.05):
    result = adfuller(series.dropna())
    return result[1] < significance_level, result[
        1]  # Return True if p-value < significance level (stationary), and the p-value


# Function to apply transformations iteratively until the series is stationary
def ensure_stationarity(series):
    # No transformation
    stationary, p_value = is_stationary(series)
    if stationary:
        return series, 1, p_value

    # First difference
    transformed_series = first_difference(series)
    stationary, p_value = is_stationary(transformed_series)
    if stationary:
        return transformed_series, 2, p_value

    # Second difference
    transformed_series = second_difference(series)
    stationary, p_value = is_stationary(transformed_series)
    if stationary:
        return transformed_series, 3, p_value

    # Log transformation
    transformed_series = log_transformation(series)
    stationary, p_value = is_stationary(transformed_series)
    if stationary:
        return transformed_series, 5, p_value

    # Log first difference
    transformed_series = log_first_difference(series)
    stationary, p_value = is_stationary(transformed_series)
    if stationary:
        return transformed_series, 6, p_value

    # Log second difference
    transformed_series = log_second_difference(series)
    stationary, p_value = is_stationary(transformed_series)
    if stationary:
        return transformed_series, 7, p_value

    # If none of the above transformations work, return the series as is (may need custom handling)
    return transformed_series, -1, p_value  # -1 indicates that stationarity was not achieved


# Define target variables and other "_Close" variables that should use the return transformation
close_vars = [col for col in base_data.columns if col.endswith('_Close')]

# Create a dictionary to store transformations and p-values
transformation_dict = {}
p_value_dict = {}
transformed_series_dict = {}

# Determine and apply transformation for each variable
for col in base_data.columns:
    if col == 'Date':
        transformed_series_dict[col] = base_data[col]  # Keep Date column in the transformed data
    elif col in close_vars:
        transformed_series = return_transformation(base_data[col])  # Return transformation for all "_Close" variables
        stationary, p_value = is_stationary(transformed_series)
        transformation = 4
        transformation_dict[col] = transformation  # Store transformation number
        p_value_dict[col] = p_value  # Store p-value
        transformed_series_dict[col] = transformed_series
    else:
        # Start with initial transformation
        transformed_series, transformation, p_value = ensure_stationarity(base_data[col])

        # Store the transformation type, p-value, and add the transformed series to the dictionary
        transformation_dict[col] = transformation
        p_value_dict[col] = p_value
        transformed_series_dict[col] = transformed_series

# Convert the dictionary of transformed series to a DataFrame in one step
transformed_data = pd.DataFrame(transformed_series_dict)

# Final check to ensure all non-"Close" suffix variables are stationary
for col in transformed_data.columns:
    if col not in close_vars and col != 'Date':
        stationary, p_value = is_stationary(transformed_data[col])
        if not stationary:
            # Reapply transformations until stationary
            transformed_series, transformation, p_value = ensure_stationarity(transformed_data[col])
            while not is_stationary(transformed_series)[0]:
                transformed_series, transformation, p_value = ensure_stationarity(transformed_series)
            # Update the transformation table, p-value table, and transformed data
            transformation_dict[col] = transformation
            p_value_dict[col] = p_value
            transformed_data[col] = transformed_series

# Convert the transformation dictionary to a DataFrame for better visualization
transformation_df = pd.DataFrame.from_dict(transformation_dict, orient='index', columns=['Transformation'])

# Add p-values to the transformation table for verification
transformation_df['ADF_p_value'] = pd.Series(p_value_dict)

# Save the transformed data and the transformation table
output_path = '/Users/stevenbarnes/Desktop/Dissertation/Starting Data/'
transformed_data.to_csv(output_path + 'Transformed_Data.csv', index=False)
transformation_df.to_csv(output_path + 'Transformation_Table.csv', index=True)

print("Transformation table with p-values and transformed data have been successfully saved.")
