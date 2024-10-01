import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Start timing the script execution
start_time = time.time()

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

# Define functions for training models and evaluating performance
def train_baseline(X_train, y_train):
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    return baseline

def train_lasso(X_train, y_train):
    lasso = Lasso(max_iter=10000)
    params = {'alpha': np.logspace(-4, 2, 20)}
    lasso_cv = GridSearchCV(lasso, params, cv=KFold(n_splits=10, shuffle=True, random_state=42))
    lasso_cv.fit(X_train, y_train)
    print(f"Lasso model iterations: {lasso_cv.best_estimator_.n_iter_}")
    return lasso_cv

def train_elastic_net(X_train, y_train):
    enet = ElasticNet(max_iter=10000)
    params = {'alpha': np.logspace(-4, 2, 20), 'l1_ratio': np.linspace(0, 1, 10)}
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

# Load and preprocess datasets
no_enviro_path = '/Users/stevenbarnes/Desktop/Dissertation/NO_ENVIRO.csv'
all_data_path = '/Users/stevenbarnes/Desktop/Dissertation/ALL_DATA.csv'
df_no_enviro = load_and_preprocess_data(no_enviro_path)
df_all_data = load_and_preprocess_data(all_data_path)

# List of target variables (soft commodity tracking assets)
targets = ['COCO_Close', 'COTN_Close', 'CANE_Close', 'WEAT_Close', 'CORN_Close', 'SOYB_Close']

# Dictionary to store results
results = {'Model': [], 'Asset': [], 'Dataset': [], 'InSample': [], 'MSE': [], 'R2': []}
# Dictionary to store feature importances
feature_importances = {'Model': [], 'Asset': [], 'Dataset': [], 'InSample': [], 'Feature': [], 'Importance': []}

# Function to process dataset
def process_dataset(df, dataset_name):
    for target in targets:
        for in_sample in [True, False]:
            sample_type = 'in-sample' if in_sample else 'out-of-sample'
            print(f"Processing target: {target}, dataset: {dataset_name}, sample type: {sample_type}")

            # Split data into features (X) and target (y)
            X = df.drop(columns=targets)
            y = df[target]

            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if in_sample:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            else:
                X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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
            results['InSample'].extend([sample_type] * 4)
            results['MSE'].extend([round(mse_baseline, 4), round(mse_lasso, 4), round(mse_enet, 4), round(mse_nn, 4)])
            results['R2'].extend([round(r2_baseline, 4), round(r2_lasso, 4), round(r2_enet, 4), round(r2_nn, 4)])

            # Get feature importances
            baseline_coefs = pd.Series(baseline_model.coef_, index=X.columns)
            lasso_coefs = pd.Series(lasso_model.best_estimator_.coef_, index=X.columns)
            enet_coefs = pd.Series(enet_model.best_estimator_.coef_, index=X.columns)

            for feature, importance in baseline_coefs.items():
                feature_importances['Model'].append('Baseline')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['InSample'].append(sample_type)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))

            for feature, importance in lasso_coefs.items():
                feature_importances['Model'].append('Lasso')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['InSample'].append(sample_type)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))

            for feature, importance in enet_coefs.items():
                feature_importances['Model'].append('Elastic Net')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['InSample'].append(sample_type)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))

            # For neural networks, we can get the feature importances through permutation importance or other methods
            nn_importances = pd.Series(np.abs(nn_model.best_estimator_.coefs_[0]).sum(axis=1), index=X.columns)
            for feature, importance in nn_importances.items():
                feature_importances['Model'].append('Neural Network')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['InSample'].append(sample_type)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))

# Process both datasets
print("Processing NO_ENVIRO dataset")
process_dataset(df_no_enviro, 'NO_ENVIRO')
print("Processing ALL_DATA dataset")
process_dataset(df_all_data, 'ALL_DATA')

# Convert results dictionary to DataFrame
results_df = pd.DataFrame(results)
feature_importances_df = pd.DataFrame(feature_importances)

# Display results
print(results_df)
print(feature_importances_df)

output_path = '/Users/stevenbarnes/Desktop/Dissertation/Code Outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)

results_df.to_csv(os.path.join(output_path, 'model_results.csv'), index=False)
feature_importances_df.to_csv(os.path.join(output_path, 'feature_importances.csv'), index=False)

# Visualization for in-sample and out-of-sample results
for sample_type in ['in-sample', 'out-of-sample']:
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    subset = results_df[results_df['InSample'] == sample_type]

    # Plot MSE
    sns.barplot(ax=axes[0], data=subset, x='Asset', y='MSE', hue='Model')
    axes[0].set_title(f'MSE Comparison ({sample_type})')
    axes[0].set_ylabel('Mean Squared Error')

    # Plot R2
    sns.barplot(ax=axes[1], data=subset, x='Asset', y='R2', hue='Model')
    axes[1].set_title(f'R2 Comparison ({sample_type})')
    axes[1].set_ylabel('R-squared')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'visualization_{sample_type}.png'))
    plt.show()

# Summary tables for in-sample and out-of-sample results
for sample_type in ['in-sample', 'out-of-sample']:
    subset = results_df[results_df['InSample'] == sample_type]

    mse_summary = subset.pivot_table(values='MSE', index='Model', columns='Asset')
    r2_summary = subset.pivot_table(values='R2', index='Model', columns='Asset')

    mse_summary.to_csv(os.path.join(output_path, f'mse_summary_{sample_type}.csv'))
    r2_summary.to_csv(os.path.join(output_path, f'r2_summary_{sample_type}.csv'))

# End timing the script execution
end_time = time.time()
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Done. The execution of this code took {int(hours)} hour(s) and {int(minutes)} minute(s).")

# Importance in this context means the coefficient values for linear models (Baseline, Lasso, Elastic Net)
# and the absolute sum of the first layer weights for the neural network, indicating the impact of each feature on the target variable.
