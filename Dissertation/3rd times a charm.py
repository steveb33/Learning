import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, TimeSeriesSplit
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Start timer
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

# Dictionary to store variable importances
variable_importances = {'Model': [], 'Asset': [], 'Variable': [], 'Importance': [], 'Dataset': [], 'InSample': []}


# Function to process dataset with cross-validation
def process_dataset_with_cross_validation(df, dataset_name):
    for target in targets:
        print(f"Processing target: {target}")
        X = df.drop(columns=targets)
        y = df[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=10)

        for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled)):
            print(f"Processing fold {fold + 1} for target: {target}, dataset: {dataset_name}")

            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # In-sample
            print(f"Optimizing in-sample model for target: {target}, dataset: {dataset_name}")
            baseline_model = train_baseline(X_train, y_train)
            lasso_model = train_lasso(X_train, y_train)
            enet_model = train_elastic_net(X_train, y_train)
            nn_model = train_neural_network(X_train, y_train)

            mse_baseline, r2_baseline = evaluate_model(baseline_model, X_train, y_train)
            mse_lasso, r2_lasso = evaluate_model(lasso_model, X_train, y_train)
            mse_enet, r2_enet = evaluate_model(enet_model, X_train, y_train)
            mse_nn, r2_nn = evaluate_model(nn_model, X_train, y_train)

            results['Model'].extend(['Baseline', 'Lasso', 'Elastic Net', 'Neural Network'])
            results['Asset'].extend([target] * 4)
            results['Dataset'].extend([dataset_name] * 4)
            results['MSE'].extend([mse_baseline, mse_lasso, mse_enet, mse_nn])
            results['R2'].extend([r2_baseline, r2_lasso, r2_enet, r2_nn])
            results['InSample'].extend([True] * 4)

            for variable, importance in zip(X.columns, baseline_model.coef_):
                variable_importances['Model'].append('Baseline')
                variable_importances['Asset'].append(target)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['InSample'].append(True)

            for variable, importance in zip(X.columns, lasso_model.best_estimator_.coef_):
                variable_importances['Model'].append('Lasso')
                variable_importances['Asset'].append(target)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['InSample'].append(True)

            for variable, importance in zip(X.columns, enet_model.best_estimator_.coef_):
                variable_importances['Model'].append('Elastic Net')
                variable_importances['Asset'].append(target)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['InSample'].append(True)

            nn_importances = pd.Series(np.abs(nn_model.best_estimator_.coefs_[0]).sum(axis=1), index=X.columns)
            for variable, importance in nn_importances.items():
                variable_importances['Model'].append('Neural Network')
                variable_importances['Asset'].append(target)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['InSample'].append(True)

            # Out-of-sample
            print(f"Optimizing out-of-sample model for target: {target}, dataset: {dataset_name}")
            mse_baseline, r2_baseline = evaluate_model(baseline_model, X_test, y_test)
            mse_lasso, r2_lasso = evaluate_model(lasso_model, X_test, y_test)
            mse_enet, r2_enet = evaluate_model(enet_model, X_test, y_test)
            mse_nn, r2_nn = evaluate_model(nn_model, X_test, y_test)

            results['Model'].extend(['Baseline', 'Lasso', 'Elastic Net', 'Neural Network'])
            results['Asset'].extend([target] * 4)
            results['Dataset'].extend([dataset_name] * 4)
            results['MSE'].extend([mse_baseline, mse_lasso, mse_enet, mse_nn])
            results['R2'].extend([r2_baseline, r2_lasso, r2_enet, r2_nn])
            results['InSample'].extend([False] * 4)

            for variable, importance in zip(X.columns, baseline_model.coef_):
                variable_importances['Model'].append('Baseline')
                variable_importances['Asset'].append(target)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['InSample'].append(False)

            for variable, importance in zip(X.columns, lasso_model.best_estimator_.coef_):
                variable_importances['Model'].append('Lasso')
                variable_importances['Asset'].append(target)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['InSample'].append(False)

            for variable, importance in zip(X.columns, enet_model.best_estimator_.coef_):
                variable_importances['Model'].append('Elastic Net')
                variable_importances['Asset'].append(target)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['InSample'].append(False)

            nn_importances = pd.Series(np.abs(nn_model.best_estimator_.coefs_[0]).sum(axis=1), index=X.columns)
            for variable, importance in nn_importances.items():
                variable_importances['Model'].append('Neural Network')
                variable_importances['Asset'].append(target)
                variable_importances['Variable'].append(variable)
                variable_importances['Importance'].append(round(importance, 4))
                variable_importances['Dataset'].append(dataset_name)
                variable_importances['InSample'].append(False)

# Process datasets
process_dataset_with_cross_validation(df_no_enviro, 'NO_ENVIRO')
process_dataset_with_cross_validation(df_all_data, 'ALL_DATA')

# Convert results dictionary to DataFrame
results_df = pd.DataFrame(results)
variable_importances_df = pd.DataFrame(variable_importances)

# Output path
output_path = '/Users/stevenbarnes/Desktop/Dissertation/Code Outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save results to CSV
results_df.to_csv(os.path.join(output_path, 'model_results.csv'), index=False)
variable_importances_df.to_csv(os.path.join(output_path, 'variable_importances.csv'), index=False)

# Visualization and summary tables split by target and dataset
for target in targets:
    for dataset in ['NO_ENVIRO', 'ALL_DATA']:
        subset_results = results_df[(results_df['Asset'] == target) & (results_df['Dataset'] == dataset)]
        subset_importances = variable_importances_df[(variable_importances_df['Asset'] == target) & (variable_importances_df['Dataset'] == dataset)]

        mse_in_sample_summary = subset_results[subset_results['InSample'] == True].pivot_table(values='MSE', index='Model', columns='Asset')
        r2_in_sample_summary = subset_results[subset_results['InSample'] == True].pivot_table(values='R2', index='Model', columns='Asset')
        mse_out_of_sample_summary = subset_results[subset_results['InSample'] == False].pivot_table(values='MSE', index='Model', columns='Asset')
        r2_out_of_sample_summary = subset_results[subset_results['InSample'] == False].pivot_table(values='R2', index='Model', columns='Asset')

        # Save results to CSV
        mse_in_sample_summary.to_csv(os.path.join(output_path, f'mse_in_sample_summary_{target}_{dataset}.csv'))
        r2_in_sample_summary.to_csv(os.path.join(output_path, f'r2_in_sample_summary_{target}_{dataset}.csv'))
        mse_out_of_sample_summary.to_csv(os.path.join(output_path, f'mse_out_of_sample_summary_{target}_{dataset}.csv'))
        r2_out_of_sample_summary.to_csv(os.path.join(output_path, f'r2_out_of_sample_summary_{target}_{dataset}.csv'))
        subset_importances.to_csv(os.path.join(output_path, f'variable_importance_{target}_{dataset}.csv'), index=False)

        # Visualization
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        sns.barplot(ax=axes[0], data=subset_results, x='InSample', y='MSE', hue='Model')
        axes[0].set_title(f'MSE Comparison for {target} - {dataset}')
        axes[0].set_ylabel('Mean Squared Error')

        sns.barplot(ax=axes[1], data=subset_results, x='InSample', y='R2', hue='Model')
        axes[1].set_title(f'R2 Comparison for {target} - {dataset}')
        axes[1].set_ylabel('R-squared')

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'visualization_{target}_{dataset}.png'))
        plt.close()

# End timer
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Done. The execution of this code took {int(hours)} hour(s) and {int(minutes)} minute(s).")

# Importance explanation
print("# Importance in this context means the magnitude of the coefficient for linear models and the sum of the absolute weights for the first layer in the neural network model, indicating the variable's impact on the prediction.")
