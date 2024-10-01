import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load and preprocess the datasets
def load_and_preprocess_data(file_path):
    print(f"Beginning the loading and preprocessing of data from {file_path}")
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

# Load and preprocess NO_ENVIRO dataset
no_enviro_path = '/Users/stevenbarnes/Desktop/Dissertation/NO_ENVIRO.csv'
df_no_enviro = load_and_preprocess_data(no_enviro_path)

# Load and preprocess ALL_DATA dataset
all_data_path = '/Users/stevenbarnes/Desktop/Dissertation/ALL_DATA.csv'
df_all_data = load_and_preprocess_data(all_data_path)

# Define functions for training models and evaluating performance
def train_baseline(X_train, y_train):
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    return baseline

def train_lasso(X_train, y_train):
    lasso = Lasso(max_iter=10000)  # Increase the number of iterations
    params = {'alpha': np.logspace(-4, 2, 20)}  # Increase regularization range
    lasso_cv = GridSearchCV(lasso, params, cv=KFold(n_splits=10, shuffle=True, random_state=42))
    lasso_cv.fit(X_train, y_train)
    # Print the number of iterations used
    lasso_model = lasso_cv.best_estimator_
    print(f"Lasso model iterations: {lasso_model.n_iter_}")
    return lasso_cv

def train_elastic_net(X_train, y_train):
    enet = ElasticNet(max_iter=10000)  # Increase the number of iterations
    params = {'alpha': np.logspace(-4, 2, 20), 'l1_ratio': np.linspace(0, 1, 10)}  # Increase regularization range
    enet_cv = GridSearchCV(enet, params, cv=KFold(n_splits=10, shuffle=True, random_state=42))
    enet_cv.fit(X_train, y_train)
    # Print the number of iterations used
    enet_model = enet_cv.best_estimator_
    print(f"ElasticNet model iterations: {enet_model.n_iter_}")
    return enet_cv

def train_neural_network(X_train, y_train):
    nn = MLPRegressor(max_iter=2000, early_stopping=True)
    params = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': np.logspace(-4, 4, 20),
        'learning_rate': ['constant', 'adaptive']
    }
    nn_cv = RandomizedSearchCV(nn, params, n_iter=20, cv=KFold(n_splits=10, shuffle=True, random_state=42), random_state=42)
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
results_no_enviro = {'Model': [], 'Asset': [], 'Dataset': [], 'MSE': [], 'R2': []}
results_all_data = {'Model': [], 'Asset': [], 'Dataset': [], 'MSE': [], 'R2': []}

# Dictionary to store feature importances
feature_importances_no_enviro = {'Model': [], 'Asset': [], 'Feature': [], 'Importance': []}
feature_importances_all_data = {'Model': [], 'Asset': [], 'Feature': [], 'Importance': []}

# Function to process dataset
def process_dataset(df, dataset_name, results, feature_importances):
    for target in targets:
        print(f"Beginning the processing of dataset for target: {target} in dataset: {dataset_name}")
        # Split data into features (X) and target (y)
        X = df.drop(columns=targets)
        y = df[target]

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        print("Optimizing models")
        # Train and evaluate models
        baseline_model = train_baseline(X_train, y_train)
        lasso_model = train_lasso(X_train, y_train)
        enet_model = train_elastic_net(X_train, y_train)
        nn_model = train_neural_network(X_train, y_train)

        mse_baseline, r2_baseline = evaluate_model(baseline_model, X_test, y_test)
        mse_lasso, r2_lasso = evaluate_model(lasso_model, X_test, y_test)
        mse_enet, r2_enet = evaluate_model(enet_model, X_test, y_test)
        mse_nn, r2_nn = evaluate_model(nn_model, X_test, y_test)

        # Append results to dictionary
        results['Model'].extend(['Baseline', 'Lasso', 'Elastic Net', 'Neural Network'])
        results['Asset'].extend([target] * 4)
        results['Dataset'].extend([dataset_name] * 4)
        results['MSE'].extend([round(mse_baseline, 4), round(mse_lasso, 4), round(mse_enet, 4), round(mse_nn, 4)])
        results['R2'].extend([round(r2_baseline, 4), round(r2_lasso, 4), round(r2_enet, 4), round(r2_nn, 4)])

        # Get feature importances
        baseline_coefs = pd.Series(baseline_model.coef_, index=X.columns)
        lasso_coefs = pd.Series(lasso_model.best_estimator_.coef_, index=X.columns)
        enet_coefs = pd.Series(enet_model.best_estimator_.coef_, index=X.columns)

        for feature, importance in baseline_coefs.sort_values(ascending=False).head(30).items():
            feature_importances['Model'].append('Baseline')
            feature_importances['Asset'].append(target)
            feature_importances['Feature'].append(feature)
            feature_importances['Importance'].append(round(importance, 4))

        for feature, importance in lasso_coefs.sort_values(ascending=False).head(30).items():
            feature_importances['Model'].append('Lasso')
            feature_importances['Asset'].append(target)
            feature_importances['Feature'].append(feature)
            feature_importances['Importance'].append(round(importance, 4))

        for feature, importance in enet_coefs.sort_values(ascending=False).head(30).items():
            feature_importances['Model'].append('Elastic Net')
            feature_importances['Asset'].append(target)
            feature_importances['Feature'].append(feature)
            feature_importances['Importance'].append(round(importance, 4))

        # For neural networks, we can get the feature importances through permutation importance or other methods
        # Here we use the first layer weights as a proxy for feature importance
        nn_importances = pd.Series(np.abs(nn_model.best_estimator_.coefs_[0]).sum(axis=1), index=X.columns)
        for feature, importance in nn_importances.sort_values(ascending=False).head(30).items():
            feature_importances['Model'].append('Neural Network')
            feature_importances['Asset'].append(target)
            feature_importances['Feature'].append(feature)
            feature_importances['Importance'].append(round(importance, 4))

# Process both datasets
process_dataset(df_no_enviro, 'NO_ENVIRO', results_no_enviro, feature_importances_no_enviro)
process_dataset(df_all_data, 'ALL_DATA', results_all_data, feature_importances_all_data)

# Convert results dictionary to DataFrame
results_df_no_enviro = pd.DataFrame(results_no_enviro)
feature_importances_df_no_enviro = pd.DataFrame(feature_importances_no_enviro)

results_df_all_data = pd.DataFrame(results_all_data)
feature_importances_df_all_data = pd.DataFrame(feature_importances_all_data)

# Define the output path
output_path = '/Users/stevenbarnes/Desktop/Dissertation/Code Outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save results and feature importances to CSV
print("Saving results and feature importances to CSV")
results_df_no_enviro.to_csv(os.path.join(output_path, 'results_NO_ENVIRO.csv'), index=False)
feature_importances_df_no_enviro.to_csv(os.path.join(output_path, 'feature_importances_NO_ENVIRO.csv'), index=False)

results_df_all_data.to_csv(os.path.join(output_path, 'results_ALL_DATA.csv'), index=False)
feature_importances_df_all_data.to_csv(os.path.join(output_path, 'feature_importances_ALL_DATA.csv'), index=False)

# Visualization for NO_ENVIRO
print("Creating visualization for NO_ENVIRO")
fig_no_enviro, axes_no_enviro = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Plot MSE for NO_ENVIRO
sns.barplot(ax=axes_no_enviro[0], data=results_df_no_enviro, x='Asset', y='MSE', hue='Model')
axes_no_enviro[0].set_title('MSE Comparison (NO_ENVIRO)')
axes_no_enviro[0].set_ylabel('Mean Squared Error')

# Plot R2 for NO_ENVIRO
sns.barplot(ax=axes_no_enviro[1], data=results_df_no_enviro, x='Asset', y='R2', hue='Model')
axes_no_enviro[1].set_title('R2 Comparison (NO_ENVIRO)')
axes_no_enviro[1].set_ylabel('R-squared')

plt.tight_layout()
fig_no_enviro.savefig(os.path.join(output_path, 'visualization_NO_ENVIRO.png'))
plt.show()

# Visualization for ALL_DATA
print("Creating visualization for ALL_DATA")
fig_all_data, axes_all_data = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Plot MSE for ALL_DATA
sns.barplot(ax=axes_all_data[0], data=results_df_all_data, x='Asset', y='MSE', hue='Model')
axes_all_data[0].set_title('MSE Comparison (ALL_DATA)')
axes_all_data[0].set_ylabel('Mean Squared Error')

# Plot R2 for ALL_DATA
sns.barplot(ax=axes_all_data[1], data=results_df_all_data, x='Asset', y='R2', hue='Model')
axes_all_data[1].set_title('R2 Comparison (ALL_DATA)')
axes_all_data[1].set_ylabel('R-squared')

plt.tight_layout()
fig_all_data.savefig(os.path.join(output_path, 'visualization_ALL_DATA.png'))
plt.show()

# Summary tables for NO_ENVIRO
print("Creating summary tables for NO_ENVIRO")
mse_summary_no_enviro = results_df_no_enviro.pivot_table(values='MSE', index='Model', columns='Asset')
r2_summary_no_enviro = results_df_no_enviro.pivot_table(values='R2', index='Model', columns='Asset')

# Save the summary tables for NO_ENVIRO as CSV
mse_summary_no_enviro.to_csv(os.path.join(output_path, 'mse_summary_NO_ENVIRO.csv'))
r2_summary_no_enviro.to_csv(os.path.join(output_path, 'r2_summary_NO_ENVIRO.csv'))

# Summary tables for ALL_DATA
print("Creating summary tables for ALL_DATA")
mse_summary_all_data = results_df_all_data.pivot_table(values='MSE', index='Model', columns='Asset')
r2_summary_all_data = results_df_all_data.pivot_table(values='R2', index='Model', columns='Asset')

# Save the summary tables for ALL_DATA as CSV
mse_summary_all_data.to_csv(os.path.join(output_path, 'mse_summary_ALL_DATA.csv'))
r2_summary_all_data.to_csv(os.path.join(output_path, 'r2_summary_ALL_DATA.csv'))

# Print "Done" after all outputs have been generated
print("Done")
