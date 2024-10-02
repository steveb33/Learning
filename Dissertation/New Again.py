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
    params = {'alpha': np.logspace(-4, 2, 20)}
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

# List of target variables
targets = ['COCO_Close', 'COTN_Close', 'CANE_Close', 'WEAT_Close', 'CORN_Close', 'SOYB_Close']

# Dictionary to store results
results = {'Model': [], 'Asset': [], 'Dataset': [], 'InSample': [], 'MSE': [], 'R2': []}
feature_importances = {'Model': [], 'Asset': [], 'Dataset': [], 'Feature': [], 'Importance': [], 'InSample': []}

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

            results['Model'].extend(['Baseline', 'Lasso', 'Elastic Net', 'Neural Network'])
            results['Asset'].extend([target] * 4)
            results['Dataset'].extend([dataset_name] * 4)
            results['InSample'].extend([True] * 4)
            results['MSE'].extend([round(mse_baseline_in, 4), round(mse_lasso_in, 4), round(mse_enet_in, 4), round(mse_nn_in, 4)])
            results['R2'].extend([round(r2_baseline_in, 4), round(r2_lasso_in, 4), round(r2_enet_in, 4), round(r2_nn_in, 4)])

            baseline_coefs = pd.Series(baseline_model_in.coef_, index=X.columns)
            lasso_coefs = pd.Series(lasso_model_in.best_estimator_.coef_, index=X.columns)
            enet_coefs = pd.Series(enet_model_in.best_estimator_.coef_, index=X.columns)

            for feature, importance in baseline_coefs.sort_values(ascending=False).items():
                feature_importances['Model'].append('Baseline')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))
                feature_importances['InSample'].append(True)

            for feature, importance in lasso_coefs.sort_values(ascending=False).items():
                feature_importances['Model'].append('Lasso')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))
                feature_importances['InSample'].append(True)

            for feature, importance in enet_coefs.sort_values(ascending=False).items():
                feature_importances['Model'].append('Elastic Net')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))
                feature_importances['InSample'].append(True)

            nn_importances = pd.Series(np.abs(nn_model_in.best_estimator_.coefs_[0]).sum(axis=1), index=X.columns)
            for feature, importance in nn_importances.sort_values(ascending=False).items():
                feature_importances['Model'].append('Neural Network')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))
                feature_importances['InSample'].append(True)

            print("Optimizing out-of-sample model for target: {}, dataset: {}".format(target, dataset_name))
            baseline_model_out = train_baseline(X_train, y_train)
            lasso_model_out = train_lasso(X_train, y_train)
            enet_model_out = train_elastic_net(X_train, y_train)
            nn_model_out = train_neural_network(X_train, y_train)

            mse_baseline_out, r2_baseline_out = evaluate_model(baseline_model_out, X_test, y_test)
            mse_lasso_out, r2_lasso_out = evaluate_model(lasso_model_out, X_test, y_test)
            mse_enet_out, r2_enet_out = evaluate_model(enet_model_out, X_test, y_test)
            mse_nn_out, r2_nn_out = evaluate_model(nn_model_out, X_test, y_test)

            results['Model'].extend(['Baseline', 'Lasso', 'Elastic Net', 'Neural Network'])
            results['Asset'].extend([target] * 4)
            results['Dataset'].extend([dataset_name] * 4)
            results['InSample'].extend([False] * 4)
            results['MSE'].extend([round(mse_baseline_out, 4), round(mse_lasso_out, 4), round(mse_enet_out, 4), round(mse_nn_out, 4)])
            results['R2'].extend([round(r2_baseline_out, 4), round(r2_lasso_out, 4), round(r2_enet_out, 4), round(r2_nn_out, 4)])

            baseline_coefs_out = pd.Series(baseline_model_out.coef_, index=X.columns)
            lasso_coefs_out = pd.Series(lasso_model_out.best_estimator_.coef_, index=X.columns)
            enet_coefs_out = pd.Series(enet_model_out.best_estimator_.coef_, index=X.columns)

            for feature, importance in baseline_coefs_out.sort_values(ascending=False).items():
                feature_importances['Model'].append('Baseline')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))
                feature_importances['InSample'].append(False)

            for feature, importance in lasso_coefs_out.sort_values(ascending=False).items():
                feature_importances['Model'].append('Lasso')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))
                feature_importances['InSample'].append(False)

            for feature, importance in enet_coefs_out.sort_values(ascending=False).items():
                feature_importances['Model'].append('Elastic Net')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))
                feature_importances['InSample'].append(False)

            nn_importances_out = pd.Series(np.abs(nn_model_out.best_estimator_.coefs_[0]).sum(axis=1), index=X.columns)
            for feature, importance in nn_importances_out.sort_values(ascending=False).items():
                feature_importances['Model'].append('Neural Network')
                feature_importances['Asset'].append(target)
                feature_importances['Dataset'].append(dataset_name)
                feature_importances['Feature'].append(feature)
                feature_importances['Importance'].append(round(importance, 4))
                feature_importances['InSample'].append(False)

# Process both datasets
process_dataset_with_cross_validation(df_no_enviro, 'NO_ENVIRO')
process_dataset_with_cross_validation(df_all_data, 'ALL_DATA')

# Convert results dictionary to DataFrame
results_df = pd.DataFrame(results)
feature_importances_df = pd.DataFrame(feature_importances)

# Display results
print(results_df)
print(feature_importances_df)

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
    output_path = '/Users/stevenbarnes/Desktop/Dissertation/Code Outputs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, f'visualization_{dataset_name}.png'))
    plt.show()

# Create visualizations for both datasets
create_visualization(results_df, 'NO_ENVIRO')
create_visualization(results_df, 'ALL_DATA')

# Save results to CSV
output_path = '/Users/stevenbarnes/Desktop/Dissertation/Code Outputs'
results_df.to_csv(os.path.join(output_path, 'model_results.csv'), index=False)
feature_importances_df.to_csv(os.path.join(output_path, 'top_30_feature_importances.csv'), index=False)

# Summary tables
mse_summary = results_df.pivot_table(values='MSE', index=['Model', 'InSample'], columns=['Dataset', 'Asset'])
r2_summary = results_df.pivot_table(values='R2', index=['Model', 'InSample'], columns=['Dataset', 'Asset'])

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