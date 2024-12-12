import numpy as np
import pandas as pd
import os
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.inspection import permutation_importance

# Start script timer
start_time = time.time()

# Read in CSV and create output path
print("Loading dataset...")
data = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/LoanDefaultPrediction/Loan_default.csv')
output_path = '/Users/stevenbarnes/Desktop/Resources/Data/LoanDefaultPrediction/'
os.makedirs(output_path, exist_ok=True)
print("Dataset loaded successfully. Starting preprocessing...")

# Encode categorical features
le = LabelEncoder()
obj_col = ['HasCoSigner', 'LoanPurpose', 'HasDependents', 'HasMortgage', 'MaritalStatus', 'EmploymentType', 'Education']
for col in obj_col:
    data[col] = le.fit_transform(data[col])

# Define predictor variables (X) and target variables (y)
X = data.drop(columns=['Default', 'LoanID'])    # Excludes the target and identifier features
y = data['Default']

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]} samples, Testing set size: {X_test.shape[0]} samples")

# Scale the numeric features
scaler = StandardScaler()
num_col = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'DTIRatio']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_col] = scaler.fit_transform(X_train[num_col])
X_test_scaled[num_col] = scaler.transform(X_test[num_col])

# Create the combined correlation matrix, including the 'Default' feature
correlation_output = []

# Combine X_train and y to handle correlations with 'Default'
X_train_scaled_with_default = X_train_scaled.copy()
X_train_scaled_with_default['Default'] = y.loc[X_train.index]

# Generate correlations
for feature in X_train_scaled_with_default.columns:
    for correlated_feature in X_train_scaled_with_default.columns:
        correlation = X_train_scaled_with_default[feature].corr(X_train_scaled_with_default[correlated_feature])
        correlation_output.append({
            'Feature': feature,
            'Corr Ft': correlated_feature,
            'Correlation': correlation
        })

# Convert to DataFrame and save to CSV
correlation_df = pd.DataFrame(correlation_output)
correlation_df.to_csv(os.path.join(output_path, 'LoanDefaultCorr.csv'), index=False)

# Models and their hyperparameter grids
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000), {'C': [0.01, 0.1, 1, 10]}),
    ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
    ('Gradient Boosting', GradientBoostingClassifier(), {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]}),
    ('K-Nearest Neighbors', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9]})
]

# Function to evaluate models, tune thresholds, and record feature importance
def evaluate_tune_and_record(models, X_train, X_test, y_train, y_test, thresholds, output_path):
    comparison_results = []
    threshold_results = []
    feature_importances = []
    best_params = []

    for model_name, model, param_grid in models:
        # Train and predict
        print(f"Starting hyperparameter tuning for {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(X_train, y_train)
        print(f"Completed hyperparameter tuning for {model_name}. Best parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
        best_params.append({'Model': model_name, 'Best Params': grid_search.best_params_})  # Records the best params
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Evaluate base metrics
        accuracy = model.score(X_test, y_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        # Add to comparison results
        comparison_results.append({'Model': model_name, 'Metric': 'Accuracy', 'Values': accuracy})
        comparison_results.append({'Model': model_name, 'Metric': 'Precision', 'Values': precision})
        comparison_results.append({'Model': model_name, 'Metric': 'Recall', 'Values': recall})
        comparison_results.append({'Model': model_name, 'Metric': 'F1-Score', 'Values': f1})
        if auc_roc is not None:
            comparison_results.append({'Model': model_name, 'Metric': 'AUC-ROC', 'Values': auc_roc})

        # Threshold tuning
        if y_proba is not None:
            for threshold in thresholds:
                y_pred_thresh = (y_proba >= threshold).astype(int)

                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()

                precision_thresh, recall_thresh, f1_thresh, _ = precision_recall_fscore_support(
                    y_test, y_pred_thresh, average='binary'
                )

                false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
                false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

                threshold_results.append({
                    'Model': model_name,
                    'Threshold': threshold,
                    'Metric': 'Precision', 'Values': precision_thresh
                })
                threshold_results.append({
                    'Model': model_name,
                    'Threshold': threshold,
                    'Metric': 'Recall', 'Values': recall_thresh
                })
                threshold_results.append({
                    'Model': model_name,
                    'Threshold': threshold,
                    'Metric': 'F1-Score', 'Values': f1_thresh
                })
                threshold_results.append({
                    'Model': model_name,
                    'Threshold': threshold,
                    'Metric': 'False Negative Rate', 'Values': false_negative_rate
                })
                threshold_results.append({
                    'Model': model_name,
                    'Threshold': threshold,
                    'Metric': 'False Positive Rate', 'Values': false_positive_rate
                })
                threshold_results.append({
                    'Model': model_name,
                    'Threshold': threshold,
                    'Metric': 'Accuracy', 'Values': accuracy  # Accuracy does not change with thresholds
                })
                if auc_roc is not None:
                    threshold_results.append({
                        'Model': model_name,
                        'Threshold': threshold,
                        'Metric': 'AUC-ROC', 'Values': auc_roc  # AUC-ROC remains constant for the model
                    })
            print(f"Threshold tuning completed for {model_name}.")

        # Feature importance with ranking recording
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (Random Forest, Gradient Boosting)
            print(f"Extracting feature importances for {model_name}...")
            importances = model.feature_importances_
            for feature, importance in zip(X_train.columns, importances):
                feature_importances.append({
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': importance
                })
        elif isinstance(model, LogisticRegression):
            # Logistic Regression (coefficients)
            print(f"Calculating feature importances for {model_name} using coefficients...")
            coefficients = model.coef_[0]
            for feature, coef in zip(X_train.columns, coefficients):
                feature_importances.append({
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': coef,
                })
        elif model_name == 'K-Nearest Neighbors':
            # K-Nearest Neighbors using permutation importance
            print(f"Estimating permutation importances for {model_name} (this may take time)...")
            perm_importance = permutation_importance(model, X_test, y_test, scoring='accuracy', n_repeats=5, random_state=42)
            for feature, importance in zip(X_train.columns, perm_importance.importances_mean):
                feature_importances.append({
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': importance
                })

    # Save results to CSV
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(f"{output_path}/ModelComparisonResults.csv", index=False)

    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv(f"{output_path}/ThresholdTuningResults.csv", index=False)

        # Saves feature importance and creates a rank for feature by model to make comparison easier
    feature_importances_df = pd.DataFrame(feature_importances)
    feature_importances_df['Rank'] = feature_importances_df.groupby('Model')['Importance'].rank(ascending=False).astype(int)
    feature_importances_df.to_csv(f"{output_path}/FeatureImportance.csv", index=False)

    best_params_df = pd.DataFrame(best_params)
    best_params_df.to_csv(os.path.join(output_path, 'BestHyperparameters.csv'), index=False)

    print('All files have been saved')

# Call all the functions
thresholds = np.arange(0.1, 1.0, 0.1)
evaluate_tune_and_record(models, X_train_scaled, X_test_scaled, y_train, y_test, thresholds, output_path)

# End timer
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print("The execution of this code took {:0>2}:{:0>2}:{:05.2f} (HH:MM:SS)".format(int(hours), int(minutes), seconds))