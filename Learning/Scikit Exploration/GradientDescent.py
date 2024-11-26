import numpy as np
import matplotlib.pyplot as plt
import os

# File path for saving outputs
output_path = "/Users/stevenbarnes/Desktop/Resources/Ideas/Gradient Descent/"
os.makedirs(output_path, exist_ok=True)

def create_dataset(function_type='quadratic', n_samples=1000, noise_level=0.5, random_seed=42):
    """
    Create a synthetic dataset based on a quadratic or cubic function

    Parameters:
        function_type (str) = Either quadratic or cubic
        n_samples (int) = Number of samples to generate
        noise_level (float) = Standard deviation of Gaussian noise to add
        random_seed (int) = Seed for reproducibility

    Returns:
        X (numpy.ndarray) = Features (single feature for simplicity)
        y (numpy.ndarray) = Target values
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


# Plot big m quadratic
plt.figure(figsize=(7, 6))
plt.scatter(X_quad, y_quad, alpha=0.5)
plt.title('Quadratic Dataset with Noise (Big m)')
plt.xlabel('X')
plt.ylabel("y")
quad_path = os.path.join(output_path, "QuadraticStartBig.png")
plt.savefig(quad_path)
plt.close()

# Plot small m quadratic
plt.figure(figsize=(7, 6))
plt.scatter(X_q_small, y_q_small, alpha=0.5)
plt.title('Quadratic Dataset with Noise (Small m)')
plt.xlabel('X')
plt.ylabel("y")
quad_path = os.path.join(output_path, "QuadraticStartSmall.png")
plt.savefig(quad_path)
plt.close()

# Plot big m cubic dataset
plt.figure(figsize=(7, 6))
plt.scatter(X_cube, y_cube, alpha=0.5, color="orange")
plt.title("Cubic Dataset with Noise (Big m)")
plt.xlabel("X")
plt.ylabel("y")
cubic_path = os.path.join(output_path, "CubicStartBig.png")
plt.savefig(cubic_path)
plt.close()

# Plot small m cubic dataset
plt.figure(figsize=(7, 6))
plt.scatter(X_c_small, y_c_small, alpha=0.5, color="orange")
plt.title("Cubic Dataset with Noise (Small m)")
plt.xlabel("X")
plt.ylabel("y")
cubic_path = os.path.join(output_path, "CubicStartSmall.png")
plt.savefig(cubic_path)
plt.close()