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
import time

# Load and preprocess the datasets
def load_and_preprocess_data(file_path):
    print(f"Loading and preprocessing data from {file_path}")
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
    print(f"Lasso model iterations: {lasso_cv.best_estimator_.n_iter_}")
    return lasso_cv

def train_elastic_net(X_train, y_train):
    enet = ElasticNet(max_iter=10000)  # Increase the number of iterations
    params = {'alpha': np.logspace(-4, 2, 20), 'l1_ratio': np.linspace(0, 1, 10)}  # Increase regularization range
    enet_cv = GridSearchCV(enet, params, cv=KFold(n_splits=10, shuffle=True, random_state=42))
    enet_cv.fit(X_train, y_train)
    print(f"ElasticNet model iterations: {enet_cv.best_estimator_.n_iter_}")
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
results = {'Model': [], 'Asset': [], 'Dataset': [], 'MSE': [], 'R2': [], 'InSample': []}

# Dictionary to store feature importances
feature_importances = {'Model': [], 'Asset': [], 'Feature': [], 'Importance': []}

# Function to process dataset
def process_dataset(df, dataset_name):
    for target in targets:
        print(f"Processing target: {target}, dataset: {dataset_name}")

        # Split data into features (X) and target (y)
        X = df.drop(columns=targets)
        y = df[target]

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Outer loop for cross-validation
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train_val_idx, test_idx) in enumerate(outer_cv.split(X_scaled)):
            X_train_val, X_test = X_scaled[train_val_idx], X_scaled[test_idx]
            y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

            # Inner loop for cross-validation
            inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)

            for train_idx, val_idx in inner_cv.split(X_train_val):
                X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
                y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

                # Train and evaluate models for in-sample
                print(f"Optimizing in-sample model for target: {target}, dataset: {dataset_name}")
                baseline_model = train_baseline(X_train, y_train)
                lasso_model = train_lasso(X_train, y_train)
                enet_model = train_elastic_net(X_train, y_train)
                nn_model = train_neural_network(X_train, y_train)

                mse_baseline, r2_baseline = evaluate_model(baseline_model, X_val, y_val)
                mse_lasso, r2_lasso = evaluate_model(lasso_model, X_val, y_val)
                mse_enet, r2_enet = evaluate_model(enet_model, X_val, y_val)
                mse_nn, r2_nn = evaluate_model(nn_model, X_val, y_val)

                # Append results to dictionary for in-sample
                results['Model'].extend(['Baseline', 'Lasso', 'Elastic Net', 'Neural Network'])
                results['Asset'].extend([target] * 4)
                results['Dataset'].extend([dataset_name] * 4)
                results['MSE'].extend([round(mse_baseline, 4), round(mse_lasso, 4), round(mse_enet, 4), round(mse_nn, 4)])
                results['R2'].extend([round(r2_baseline, 4), round(r2_lasso, 4), round(r2_enet, 4), round(r2_nn, 4)])
                results['InSample'].extend(['in-sample'] * 4)

                # Train and evaluate models for out-of-sample
                print(f"Optimizing out-of-sample model for target: {target}, dataset: {dataset_name}")
                mse_baseline, r2_baseline = evaluate_model(baseline_model, X_test, y_test)
                mse_lasso, r2_lasso = evaluate_model(lasso_model, X_test, y_test)
                mse_enet, r2_enet = evaluate_model(enet_model, X_test, y_test)
                mse_nn, r2_nn = evaluate_model(nn_model, X_test, y_test)

                # Append results to dictionary for out-of-sample
                results['Model'].extend(['Baseline', 'Lasso', 'Elastic Net', 'Neural Network'])
                results['Asset'].extend([target] * 4)
                results['Dataset'].extend([dataset_name] * 4)
                results['MSE'].extend([round(mse_baseline, 4), round(mse_lasso, 4), round(mse_enet, 4), round(mse_nn, 4)])
                results['R2'].extend([round(r2_baseline, 4), round(r2_lasso, 4), round(r2_enet, 4), round(r2_nn, 4)])
                results['InSample'].extend(['out-of-sample'] * 4)

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
start_time = time.time()
process_dataset(df_no_enviro, 'NO_ENVIRO')
process_dataset(df_all_data, 'ALL_DATA')

# Convert results dictionary to DataFrame
results_df = pd.DataFrame(results)
feature_importances_df = pd.DataFrame(feature_importances)

# Display results
print(results_df)
print(feature_importances_df)

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Plot MSE
sns.barplot(ax=axes[0], data=results_df, x='Asset', y='MSE', hue='Model')
axes[0].set_title('MSE Comparison')
axes[0].set_ylabel('Mean Squared Error')

# Plot R2
sns.barplot(ax=axes[1], data=results_df, x='Asset', y='R2', hue='Model')
axes[1].set_title('R2 Comparison')
axes[1].set_ylabel('R-squared')

plt.tight_layout()
plt.show()

# Summary tables
mse_in_sample_summary = results_df[results_df['InSample'] == 'in-sample'].pivot_table(values='MSE', index='Model', columns='Asset')
r2_in_sample_summary = results_df[results_df['InSample'] == 'in-sample'].pivot_table(values='R2', index='Model', columns='Asset')
mse_out_of_sample_summary = results_df[results_df['InSample'] == 'out-of-sample'].pivot_table(values='MSE', index='Model', columns='Asset')
r2_out_of_sample_summary = results_df[results_df['InSample'] == 'out-of-sample'].pivot_table(values='R2', index='Model', columns='Asset')

# Create a new figure for MSE summary table
fig_mse_in_sample, ax_mse_in_sample = plt.subplots(figsize=(14, 5))
ax_mse_in_sample.axis('off')
mse_in_sample_table = ax_mse_in_sample.table(cellText=mse_in_sample_summary.values, colLabels=mse_in_sample_summary.columns, rowLabels=mse_in_sample_summary.index, cellLoc='center', loc='center')
mse_in_sample_table.auto_set_font_size(False)
mse_in_sample_table.set_fontsize(10)
mse_in_sample_table.scale(1.2, 1.2)
ax_mse_in_sample.set_title('MSE In-Sample Summary Table', pad=20)

fig_mse_out_of_sample, ax_mse_out_of_sample = plt.subplots(figsize=(14, 5))
ax_mse_out_of_sample.axis('off')
mse_out_of_sample_table = ax_mse_out_of_sample.table(cellText=mse_out_of_sample_summary.values, colLabels=mse_out_of_sample_summary.columns, rowLabels=mse_out_of_sample_summary.index, cellLoc='center', loc='center')
mse_out_of_sample_table.auto_set_font_size(False)
mse_out_of_sample_table.set_fontsize(10)
mse_out_of_sample_table.scale(1.2, 1.2)
ax_mse_out_of_sample.set_title('MSE Out-of-Sample Summary Table', pad=20)

# Create a new figure for R2 summary table
fig_r2_in_sample, ax_r2_in_sample = plt.subplots(figsize=(14, 5))
ax_r2_in_sample.axis('off')
r2_in_sample_table = ax_r2_in_sample.table(cellText=r2_in_sample_summary.values, colLabels=r2_in_sample_summary.columns, rowLabels=r2_in_sample_summary.index, cellLoc='center', loc='center')
r2_in_sample_table.auto_set_font_size(False)
r2_in_sample_table.set_fontsize(10)
r2_in_sample_table.scale(1.2, 1.2)
ax_r2_in_sample.set_title('R2 In-Sample Summary Table', pad=20)

fig_r2_out_of_sample, ax_r2_out_of_sample = plt.subplots(figsize=(14, 5))
ax_r2_out_of_sample.axis('off')
r2_out_of_sample_table = ax_r2_out_of_sample.table(cellText=r2_out_of_sample_summary.values, colLabels=r2_out_of_sample_summary.columns, rowLabels=r2_out_of_sample_summary.index, cellLoc='center', loc='center')
r2_out_of_sample_table.auto_set_font_size(False)
r2_out_of_sample_table.set_fontsize(10)
r2_out_of_sample_table.scale(1.2, 1.2)
ax_r2_out_of_sample.set_title('R2 Out-of-Sample Summary Table', pad=20)

# Save results to CSV
output_path = '/Users/stevenbarnes/Desktop/Dissertation/Code Outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)
results_df.to_csv(os.path.join(output_path, 'model_results.csv'), index=False)
feature_importances_df.to_csv(os.path.join(output_path, 'top_30_feature_importances.csv'), index=False)

# Save the summary tables as CSV files
mse_in_sample_summary.to_csv(os.path.join(output_path, 'mse_in_sample_summary.csv'))
r2_in_sample_summary.to_csv(os.path.join(output_path, 'r2_in_sample_summary.csv'))
mse_out_of_sample_summary.to_csv(os.path.join(output_path, 'mse_out_of_sample_summary.csv'))
r2_out_of_sample_summary.to_csv(os.path.join(output_path, 'r2_out_of_sample_summary.csv'))

end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Done. The execution of this code took {int(hours)} hour(s) and {int(minutes)} minute(s).")
