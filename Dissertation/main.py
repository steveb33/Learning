import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Load and preprocess the datasets
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop rows with invalid date formats
    df = df.dropna(subset=['Date'])

    # Set the 'Date' column as the index
    df.set_index('Date', inplace=True)

    # Drop missing values for simplicity
    df = df.dropna()

    return df


# Check for multicollinearity using VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data


# Remove features with high VIF
def remove_high_vif_features(X, threshold=10):
    vif_data = calculate_vif(X)
    while vif_data["VIF"].max() > threshold:
        feature_to_remove = vif_data.sort_values("VIF", ascending=False).iloc[0]
        print(f'Removing feature {feature_to_remove["feature"]} with VIF {feature_to_remove["VIF"]}')
        X = X.drop(columns=[feature_to_remove["feature"]])
        vif_data = calculate_vif(X)
    return X, vif_data


# Load and preprocess NO_ENVIRO dataset
no_enviro_path = '/Users/stevenbarnes/Desktop/Dissertation/NO_ENVIRO.csv'
df_no_enviro = load_and_preprocess_data(no_enviro_path)

# Load and preprocess ALL_DATA dataset
all_data_path = '/Users/stevenbarnes/Desktop/Dissertation/ALL_DATA.csv'
df_all_data = load_and_preprocess_data(all_data_path)


# Define functions for training models and evaluating performance
def train_lasso(X_train, y_train):
    lasso = Lasso(max_iter=10000)  # Increase the number of iterations
    params = {'alpha': np.logspace(-4, 2, 20)}  # Increase regularization range
    lasso_cv = GridSearchCV(lasso, params, cv=KFold(n_splits=10, shuffle=True, random_state=42))
    lasso_cv.fit(X_train, y_train)
    return lasso_cv


def train_elastic_net(X_train, y_train):
    enet = ElasticNet(max_iter=10000)  # Increase the number of iterations
    params = {'alpha': np.logspace(-4, 2, 20), 'l1_ratio': np.linspace(0, 1, 10)}  # Increase regularization range
    enet_cv = GridSearchCV(enet, params, cv=KFold(n_splits=10, shuffle=True, random_state=42))
    enet_cv.fit(X_train, y_train)
    return enet_cv


def train_neural_network(X_train, y_train):
    nn = MLPRegressor(max_iter=2000, early_stopping=True)
    params = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': np.logspace(-4, 4, 20),
        'learning_rate': ['constant', 'adaptive']
    }
    nn_cv = RandomizedSearchCV(nn, params, n_iter=20, cv=KFold(n_splits=10, shuffle=True, random_state=42),
                               random_state=42)
    nn_cv.fit(X_train, y_train)
    return nn_cv


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2


# Suppress convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings('ignore', category=ConvergenceWarning, append=True)

# List of target variables (soft commodity tracking assets)
targets = ['COCO_Close', 'COTN_Close', 'CANE_Close', 'WEAT_Close', 'CORN_Close', 'SOYB_Close']

# Dictionary to store results
results = {'Model': [], 'Asset': [], 'Dataset': [], 'MSE': [], 'R2': []}

# Dictionary to store feature importances
feature_importances = {'Model': [], 'Asset': [], 'Feature': [], 'Importance': []}


# Function to process dataset
def process_dataset(df, dataset_name):
    for target in targets:
        # Split data into features (X) and target (y)
        X = df.drop(columns=targets)
        y = df[target]

        # Check for multicollinearity and remove high VIF features
        X, vif_data = remove_high_vif_features(X)
        print(f"VIF for {target} in {dataset_name} dataset after removing high VIF features:\n", vif_data)

        # Save the remaining variables after VIF removals to a CSV file
        remaining_vars_path = f'/Users/stevenbarnes/Desktop/Dissertation/Code Outputs/{target}_{dataset_name}_remaining_vars.csv'
        vif_data.to_csv(remaining_vars_path, index=False)

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train and evaluate models
        lasso_model = train_lasso(X_train, y_train)
        enet_model = train_elastic_net(X_train, y_train)
        nn_model = train_neural_network(X_train, y_train)

        mse_lasso, r2_lasso = evaluate_model(lasso_model, X_test, y_test)
        mse_enet, r2_enet = evaluate_model(enet_model, X_test, y_test)
        mse_nn, r2_nn = evaluate_model(nn_model, X_test, y_test)

        # Append results to dictionary
        results['Model'].extend(['Lasso', 'Elastic Net', 'Neural Network'])
        results['Asset'].extend([target] * 3)
        results['Dataset'].extend([dataset_name] * 3)
        results['MSE'].extend([mse_lasso, mse_enet, mse_nn])
        results['R2'].extend([r2_lasso, r2_enet, r2_nn])

        # Get feature importances
        lasso_coefs = pd.Series(lasso_model.best_estimator_.coef_, index=X.columns)
        enet_coefs = pd.Series(enet_model.best_estimator_.coef_, index=X.columns)

        for feature, importance in lasso_coefs.sort_values(ascending=False).head(30).items():
            feature_importances['Model'].append('Lasso')
            feature_importances['Asset'].append(target)
            feature_importances['Feature'].append(feature)
            feature_importances['Importance'].append(importance)

        for feature, importance in enet_coefs.sort_values(ascending=False).head(30).items():
            feature_importances['Model'].append('Elastic Net')
            feature_importances['Asset'].append(target)
            feature_importances['Feature'].append(feature)
            feature_importances['Importance'].append(importance)

        # For neural networks, we can get the feature importances through permutation importance or other methods
        # Here we use the first layer weights as a proxy for feature importance
        nn_importances = pd.Series(np.abs(nn_model.best_estimator_.coefs_[0]).sum(axis=1), index=X.columns)
        for feature, importance in nn_importances.sort_values(ascending=False).head(30).items():
            feature_importances['Model'].append('Neural Network')
            feature_importances['Asset'].append(target)
            feature_importances['Feature'].append(feature)
            feature_importances['Importance'].append(importance)


# Process both datasets
process_dataset(df_no_enviro, 'NO_ENVIRO')
process_dataset(df_all_data, 'ALL_DATA')

# Convert results dictionary to DataFrame
results_df = pd.DataFrame(results)
feature_importances_df = pd.DataFrame(feature_importances)

# Display results
print(results_df)
print(feature_importances_df)

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

# Plot MSE
sns.barplot(ax=axes[0], data=results_df, x='Asset', y='MSE', hue='Model')
axes[0].set_title('MSE Comparison')
axes[0].set_ylabel('Mean Squared Error')

# Plot R2
sns.barplot(ax=axes[1], data=results_df, x='Asset', y='R2', hue='Model')
axes[1].set_title('R2 Comparison')
axes[1].set_ylabel('R-squared')

# Summary tables
mse_summary = results_df.pivot_table(values='MSE', index='Model', columns='Asset')
r2_summary = results_df.pivot_table(values='R2', index='Model', columns='Asset')

# Table for MSE
axes[2].axis('off')
mse_table = axes[2].table(cellText=mse_summary.values, colLabels=mse_summary.columns, rowLabels=mse_summary.index,
                          cellLoc='center', loc='center')
mse_table.auto_set_font_size(False)
mse_table.set_fontsize(10)
mse_table.scale(1.2, 1.2)
axes[2].set_title('MSE Summary Table', pad=20)

# Create a new figure for R2 summary table
fig_r2, ax_r2 = plt.subplots(figsize=(14, 5))
ax_r2.axis('off')
r2_table = ax_r2.table(cellText=r2_summary.values, colLabels=r2_summary.columns, rowLabels=r2_summary.index,
                       cellLoc='center', loc='center')
r2_table.auto_set_font_size(False)
r2_table.set_fontsize(10)
r2_table.scale(1.2, 1.2)
ax_r2.set_title('R2 Summary Table', pad=20)

# Save top 30 features by model and asset to CSV
output_path = '/Users/stevenbarnes/Desktop/Dissertation/Code Outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)
results_df.to_csv(os.path.join(output_path, 'model_results.csv'), index=False)
feature_importances_df.to_csv(os.path.join(output_path, 'top_30_feature_importances.csv'), index=False)

# Save the plots
fig.savefig(os.path.join(output_path, 'mse_r2_comparison.png'))
fig_r2.savefig(os.path.join(output_path, 'r2_summary_table.png'))

# Show the plots
plt.show()
