import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# File path for saving outputs
output_path = "/Users/stevenbarnes/Desktop/Resources/Ideas/Ridge/"
os.makedirs(output_path, exist_ok=True)

def create_dataset(function_type='quadratic', n_samples=1000, noise_level=0.5, random_seed=42):
    """
    Create a synthetic dataset based on a quadratic or cubic function

    Parameters:
        function_type (str): Either quadratic or cubic
        n_samples (int): Number of samples to generate
        noise_level (float): Standard deviation of Gaussian noise to add
        random_seed (int): Seed for reproducibility

    Returns:
        X (numpy.ndarray): Features (single feature for simplicity)
        y (numpy.ndarray): Target values
    """
    np.random.seed(random_seed)
    X = np.linspace(-10, 10, n_samples).reshape(-1, 1) # Generate X values in range [-10, 10] evenly

    if function_type == 'quadratic':
        y = 3 * X**2 - 5 * X + 7    # Quadratic function
    elif function_type == 'cubic':
        y = 0.5 * X**3 - 2 * X**2 + 3 * X +1    # Cubic function
    else:
        raise ValueError('Invalid function_type. Choose quadratic or cubic')

    # Add noise to y
    y += np.random.normal(0, noise_level * np.std(y), size=y.shape)

    return X, y

# Generate datasets
X_quad, y_quad = create_dataset(function_type='quadratic', n_samples=1000)      # big m quad
X_cube, y_cube = create_dataset(function_type='cubic', n_samples=1000)          # big m cube
X_q_small, y_q_small = create_dataset(function_type='quadratic', n_samples=100) # small m quad
X_c_small, y_c_small = create_dataset(function_type='cubic', n_samples=100) # small m cube

# Organize the datasets for looping
datasets = [
    ('Big Quadratic', X_quad, y_quad, 2),   # (Title, X, y, degree)
    ('Big Cubic', X_cube, y_cube, 3),
    ('Small Quadratic', X_q_small, y_q_small, 2),
    ('Small Cubic', X_c_small, y_c_small, 3)
]

# Define alpha values
alphas = [0, 10**-5, 1]

# Prepare 2x2 grid for subplots
fig, axes = plt.subplot(2, 2, figsize=(14, 10))

# Loop through datasets and assign subplots based on position
for (title, X, y, degree), ax in zip(datasets, axes.flat):
    # Loop through the alpha values
    for alpha, linestyle in zip(alphas, ['dotted', 'dashed', 'solid']):
        # Create a Ridge regression pipeline with polynomial features
        model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha))
        model.fit(X, y)
        