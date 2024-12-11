"""
This code used the Loan Default Prediction Challenge from Kaggle at
https://www.kaggle.com/datasets/nikhil1e9/loan-default/data

This code outputs csvs formatted in a way that allows for easy Tableau visual creations
"""

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Read in CSV and create output path
data = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/LoanDefaultPrediction/Loan_default.csv')
output_path = '/Users/stevenbarnes/Desktop/Resources/Data/LoanDefaultPrediction/'
os.makedirs(output_path, exist_ok=True)

# Encode categorical features
le = LabelEncoder()
obj_col = ['HasCoSigner', 'LoanPurpose', 'HasDependents', 'HasMortgage', 'MaritalStatus', 'EmploymentType', 'Education']
for col in obj_col:
    data[col] = le.fit_transform(data[col])

# Define predictor varaibles (X) and target variables (y)
X = data.drop(columns=['Default', 'LoanID'])    # Excludes the target and identifier features
y = data['Default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the numeric features
scaler = StandardScaler()
num_col = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'DTIRatio']

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_col] = scaler.fit_transform(X_train[num_col])
X_test_scaled[num_col] = scaler.transform(X_test[num_col])

"""
The following creates a correlation matrix, but the output is structured so that it will be easier to use in Tableau
"""
# Create the combined correlation matrix, including the 'Default' feature
correlation_output = []

# Combine X_train and y to handle correlations with 'Default'
X_train_with_default = X_train.copy()
X_train_with_default['Default'] = y.loc[X_train.index]

X_train_scaled_with_default = X_train_scaled.copy()
X_train_scaled_with_default['Default'] = y.loc[X_train.index]

# Generate correlations
for feature in X_train_with_default.columns:
    for correlated_feature in X_train_with_default.columns:
        unscaled_corr = X_train_with_default[feature].corr(X_train_with_default[correlated_feature])
        scaled_corr = X_train_scaled_with_default[feature].corr(X_train_scaled_with_default[correlated_feature])
        correlation_output.append({
            'Feature': feature,
            'Corr Ft': correlated_feature,
            'Unscaled Corr': unscaled_corr,
            'Scaled Corr': scaled_corr
        })

# Convert to DataFrame and save to CSV
correlation_df = pd.DataFrame(correlation_output)
correlation_df.to_csv(os.path.join(output_path, 'LoanDefaultCorr.csv'), index=False)

"""
Now we will start creating the metrics for the different models and data types
"""


# Metric Function
def evaluate_model(model, X_train, X_test, y_train, y_test, data_type, model_name, results):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Collect metrics
    metrics = {
        'Accuracy': model.score(X_test, y_test),
        'Precision': classification_report(y_test, y_pred, output_dict=True)['1']['precision'],
        'Recall': classification_report(y_test, y_pred, output_dict=True)['1']['recall'],
        'F1-Score': classification_report(y_test, y_pred, output_dict=True)['1']['f1-score'],
        'AUC-ROC': roc_auc_score(y_test, y_prob)
    }

    # The following formats the data created above into a structure that makes it easier to use in Tableau
    for metric, value in metrics.items():
        results.append({
            'Model': model_name,
            'Data': data_type,
            'Metric': metric,
            'Values': value
        })

# Initialize models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=41)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier())
]

# Evaluate the models for scaled and scaled data
results = []
for model_name, model in models:
    evaluate_model(model, X_train, X_test, y_train, y_test, 'Unscaled', model_name, results)
    evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, 'Scaled', model_name, results)

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_path, 'ModelComparisonResults.csv'), index=False)

# Function that adds in the feature importance to add some depth to my write-up
def generate_feature_importance_csv(X_train, y_train, output_path):
    feature_importance_results = []

    # Random Forest Feature Importance
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_importances = rf_model.feature_importances_
    for feature, importance in zip(X_train.columns, rf_importances):
        feature_importance_results.append({
            'Model': 'Random Forest',
            'Feature': feature,
            'Importance': importance
        })

    # Gradient Boosting Feature Importance
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    gb_importances = gb_model.feature_importances_
    for feature, importance in zip(X_train.columns, gb_importances):
        feature_importance_results.append({
            'Model': 'Gradient Boosting',
            'Feature': feature,
            'Importance': importance
        })

    # Logistic Regression Coefficients (optional for scaled data)
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    logreg_model.fit(X_train, y_train)
    logreg_coefficients = logreg_model.coef_[0]
    for feature, coefficient in zip(X_train.columns, logreg_coefficients):
        feature_importance_results.append({
            'Model': 'Logistic Regression',
            'Feature': feature,
            'Importance': coefficient
        })

    # Convert to DataFrame and save to CSV
    feature_importance_df = pd.DataFrame(feature_importance_results)
    feature_importance_df.to_csv(os.path.join(output_path, 'FeatureImportance.csv'), index=False)

# Call the function
generate_feature_importance_csv(X_train_scaled, y_train, output_path)

# Function to generate confusion matrix metrics and output as CSV formatted for Tableau
def generate_confusion_metrics_single(models, X_train, X_test, y_train, y_test, output_path):
    confusion_results = []

    for model_name, model in models:
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Calculate derived metrix
        confusion_results.append([
            {'Model': model_name, 'Metric': 'True Positive', 'Values': tp},
            {'Model': model_name, 'Metric': 'True Negatives', 'Values': tn},
            {'Model': model_name, 'Metric': 'False Positives', 'Values': fp},
            {'Model': model_name, 'Metric': 'False Negatives', 'Values': fn},
            {'Model': model_name, 'Metric': 'False Positive Rate', 'Values': fp / (fp + tn)},
            {'Model': model_name, 'Metric': 'False Negative Rate', 'Values': fn / (fn + tp)}
        ])

    # Convert to DataFrame and save to CSV
    confusion_df = pd.DataFrame(confusion_results)
    confusion_df.to_csv(os.path.join(output_path, 'ConfusionMetrics.csv'), index=False)

# Call confusion metrics function
generate_confusion_metrics_single(models, X_train_scaled, X_test_scaled, y_train, y_test, output_path)
