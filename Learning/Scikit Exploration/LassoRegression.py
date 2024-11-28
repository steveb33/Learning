import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline


def create_dataset(function_type='quadratic', n_samples=100, noise_level=0.5, random_seed=42):
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
        y = (3 * X**2 - 5 * X + 7)/10    # Quadratic function
    elif function_type == 'cubic':
        y = (0.5 * X**3 - 2 * X**2 + 3 * X +1)/10    # Cubic function
    else:
        raise ValueError('Invalid function_type. Choose quadratic or cubic')

    # Add noise to y
    y += np.random.normal(0, noise_level * np.std(y), size=y.shape)

    return X, y

# Generate datasets
X_quad, y_quad = create_dataset(function_type='quadratic', n_samples=100)      # big m quad
X_cube, y_cube = create_dataset(function_type='cubic', n_samples=100)          # big m cube
X_q_small, y_q_small = create_dataset(function_type='quadratic', n_samples=20) # small m quad
X_c_small, y_c_small = create_dataset(function_type='cubic', n_samples=20) # small m cube

# Organize the datasets for looping
datasets = [
    ('Big Quadratic', X_quad, y_quad),   # (Title, X, y)
    ('Big Cubic', X_cube, y_cube),
    ('Small Quadratic', X_q_small, y_q_small),
    ('Small Cubic', X_c_small, y_c_small)
]

# Define alpha values
alphas = [0, 1, 10, 100]

# Prepare 2x2 grid for subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loop through datasets and assign subplots based on position
for (title, X, y), ax in zip(datasets, axes.flat):
    # Loop through the alpha values
    for alpha, color in zip(alphas, ['orange', 'red', 'green', 'purple']):
        # Create Lasso Regression pipeline
        model = make_pipeline(PolynomialFeatures(degree=10), Lasso(alpha=alpha))
        model.fit(X, y)

        # Model predictions
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(X_plot)

        # Plot the model predictions
        ax.plot(X_plot, y_pred, color=color, linewidth=2, label=f"$alpha={alpha}$")

    # Scatter the original data
    ax.scatter(X, y, color='blue', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()
    ax.grid()

# Set layout and save
plt.tight_layout()
plt.show()

"""
The resulting figure from this code illustrates how as you increase the alpha within lasso,
the bias of the model increases, and thus decreases the variance. This effect is particularly apparent when looking
at the smaller sample(m) plots on the bottom axis which show cases the greater degree of variation of fit regarding the 
different alpha values. These findings are similar to those from the ridge regression. However, it should be noted
that the feature selection factor of the lasso is not being displayed as their is only one feature (X) being modeled
"""