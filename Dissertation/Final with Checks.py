import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, TimeSeriesSplit, \
    cross_val_score
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Start timing the code execution
start_time = time.time()


# Load and preprocess the datasets
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)
    df = df.dropna()
    return df


# Load datasets
no_enviro_path = '/Users/stevenbarnes/Desktop/Dissertation/NO_ENVIRO.csv'
all_data_path = '/Users/stevenbarnes/Desktop/Dissertation/ALL_DATA.csv'
df_no_enviro = load_and_preprocess_data(no_enviro_path)
df_all_data = load_and_preprocess_data(all_data_path)


# Functions for training models and evaluating performance
def train_baseline(X_train, y_train):
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    return baseline


def train_lasso(X_train, y_train):
    lasso = Lasso(max_iter=10000)
    params = {'alpha': np.logspace(-6, 2, 100)}  # Adjusted the range to include smaller values
    lasso_cv = GridSearchCV(lasso, params, cv=KFold(n_splits=10, shuffle=True, random_state=42))
    lasso_cv.fit(X_train, y_train)
    return lasso_cv


def train_elastic_net(X_train, y_train):
    enet = ElasticNet(max_iter=10000)
    params = {'alpha': np.logspace(-4, 2, 20), 'l1_ratio': np.linspace(0, 1, 10)}
    enet_cv = GridSearchCV(enet, params, cv=KFold(n_splits=10, shuffle=True, random_state=42))
    enet_cv.fit(X_train, y_train)
    return enet_cv


def train_neural_network(X_train, y_train):
    nn = MLPRegressor(max_iter=2000, early_stopping=True)
    params = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu'],
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

# List of target variables
targets = ['COCO_Close', 'COTN_Close', 'CANE_Close', 'WEAT_Close', 'CORN_Close', 'SOYB_Close']

# Dictionary to store results
results = {'Model': [], 'Asset': [], 'Dataset': [], 'MSE': [], 'R2': [], 'Robustness_Metrics': []}
variable_importances = {'Model': [], 'Asset': [], 'Dataset': [], 'Variable': [], 'Importance': [], 'Fold': []}


# Function for robustness and sensitivity checks
def robustness_sensitivity_checks(model, X_train, y_train, X_test, y_test):
    # Bootstrapping
    bootstrap_scores = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42))

    # Perturbation with noise
    noise = np.random.normal(0, 0.01, X_train.shape)
    X_train_noisy = X_train + noise
    model.fit(X_train_noisy, y_train)
    mse_noisy, r2_noisy = evaluate_model(model, X_test, y_test)

    # Collect metrics
    return {
        'Bootstrap_Score_Mean': np.mean(bootstrap_scores),
        'Bootstrap_Score_Std': np.std(bootstrap_scores),
        'MSE_Noise': mse_noisy,
        'R2_Noise': r2_noisy
    }


# Process dataset with cross-validation
def process_dataset_with_cross_validation(df, dataset_name):
    tscv = TimeSeriesSplit(n_splits=5)
    for target in targets:
        print(f"Processing target: {target} in dataset: {dataset_name}")
        X = df.drop(columns=targets)
        y = df[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled)):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            print("Optimizing in-sample model for target: {}, dataset: {}".format(target, dataset_name))
            baseline_model_in = train_baseline(X_train, y_train)
            lasso_model_in = train_lasso(X_train, y_train)
            enet_model_in = train_elastic_net(X_train, y_train)
            nn_model_in = train_neural_network(X_train, y_train)

            mse_baseline_in, r2_baseline_in = evaluate_model(baseline_model_in, X_test, y_test)
            mse_lasso_in, r2_lasso_in = evaluate_model(lasso_model_in, X_test, y_test)
            mse_enet_in, r2_enet_in = evaluate_model(enet_model_in, X_test, y_test)
            mse_nn_in, r2_nn_in = evaluate_model(nn_model_in, X_test, y_test)

            # Robustness and sensitivity checks
            robustness_baseline = robustness_sensitivity_checks(baseline_model_in, X_train, y_train, X_test, y_test)
            robustness_lasso = robustness_sensitivity_checks(lasso_model_in.best_estimator_, X_train, y_train, X_test,
                                                             y_test)
            robustness_enet = robustness_sensitivity_checks(enet_model_in.best_estimator_, X_train, y_train, X_test,
                                                            y_test)
            robustness_nn = robustness_sensitivity_checks(nn_model_in.best_estimator_, X_train, y_train, X_test, y_test)

            results['Model'].extend(['Baseline', 'Lasso', 'Elastic Net', 'Neural Network'])
            results['Asset'].extend([target] * 4)
            results['Dataset'].extend([dataset_name] * 4)
            results['MSE'].extend(
                [round(mse_baseline_in, 4), round(mse_lasso_in, 4), round(mse_enet_in, 4), round(mse_nn_in, 4)])
            results['R2'].extend(
                [round(r2_baseline_in, 4), round(r2_lasso_in, 4), round(r2_enet_in, 4), round(r2_nn_in, 4)])
            results['Robustness_Metrics'].extend(
                [robustness_baseline, robustness_lasso, robustness_enet, robustness_nn])

            # Handle variable importance
            baseline_coefs = pd.Series(baseline_model_in.coef_, index=X.columns)
            lasso_coefs = pd.Series(lasso_model_in.best_estimator_.coef_, index=X.columns)
            enet_coefs = pd.Series(enet_model_in.best_estimator_.coef_, index=X.columns)

            for variable in X.columns:
                if variable in baseline_coefs.index:
                    importance = baseline_coefs[variable]
                else:
                    importance = 0.0
                variable_importances['Model'].append('Baseline')
                variable_importances['Asset'].append(target)
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Fold'].append(fold)

            for variable in X.columns:
                if variable in lasso_coefs.index:
                    importance = lasso_coefs[variable]
                else:
                    importance = 0.0
                variable_importances['Model'].append('Lasso')
                variable_importances['Asset'].append(target)
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Fold'].append(fold)

            for variable in X.columns:
                if variable in enet_coefs.index:
                    importance = enet_coefs[variable]
                else:
                    importance = 0.0
                variable_importances['Model'].append('Elastic Net')
                variable_importances['Asset'].append(target)
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Fold'].append(fold)

            nn_importances = pd.Series(np.abs(nn_model_in.best_estimator_.coefs_[0]).sum(axis=1), index=X.columns)
            for variable in X.columns:
                if variable in nn_importances.index:
                    importance = nn_importances[variable]
                else:
                    importance = 0.0
                variable_importances['Model'].append('Neural Network')
                variable_importances['Asset'].append(target)
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Fold'].append(fold)

# Process both datasets
process_dataset_with_cross_validation(df_no_enviro, 'NO_ENVIRO')
process_dataset_with_cross_validation(df_all_data, 'ALL_DATA')

# Convert results dictionary to DataFrame
results_df = pd.DataFrame(results)
variable_importances_df = pd.DataFrame(variable_importances)

# Display results
print(results_df)
print(variable_importances_df)

# Visualization
def create_visualization(results_df, dataset_name):
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    sns.barplot(ax=axes[0], data=results_df[results_df['Dataset'] == dataset_name], x='Asset', y='MSE', hue='Model')
    axes[0].set_title(f'MSE Comparison - {dataset_name}')
    axes[0].set_ylabel('Mean Squared Error')

    sns.barplot(ax=axes[1], data=results_df[results_df['Dataset'] == dataset_name], x='Asset', y='R2', hue='Model')
    axes[1].set_title(f'R2 Comparison - {dataset_name}')
    axes[1].set_ylabel('R-squared')

    plt.tight_layout()
    output_path = '/Users/stevenbarnes/Desktop/Dissertation/Code Outputs/New Output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, f'visualization_{dataset_name}.png'))
    plt.show()

# Create visualizations for both datasets
create_visualization(results_df, 'NO_ENVIRO')
create_visualization(results_df, 'ALL_DATA')

# Save results to CSV
output_path = '/Users/stevenbarnes/Desktop/Dissertation/Code Outputs/New Output'
results_df.to_csv(os.path.join(output_path, 'model_results.csv'), index=False)
variable_importances_df.to_csv(os.path.join(output_path, 'variable_importances.csv'), index=False)

# Summary tables
mse_summary = results_df.pivot_table(values='MSE', index=['Model'], columns=['Dataset', 'Asset'])
r2_summary = results_df.pivot_table(values='R2', index=['Model'], columns=['Dataset', 'Asset'])

# Save the summary tables as CSV files
mse_summary.to_csv(os.path.join(output_path, 'mse_summary.csv'))
r2_summary.to_csv(os.path.join(output_path, 'r2_summary.csv'))

# End timer
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print("The execution of this code took {:0>2}:{:0>2}:{:05.2f} (HH:MM:SS)".format(int(hours), int(minutes), seconds))

# Importance explanation
print("# Importance in this context means the magnitude of the model coefficients for linear models (Lasso, Elastic Net, Baseline), and the sum of absolute weights from the input layer for neural networks.")

print("Done")
