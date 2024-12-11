import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support, confusion_matrix

# Read in CSV and create output path
data = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/LoanDefaultPrediction/Loan_default.csv')
output_path = '/Users/stevenbarnes/Desktop/Resources/Data/LoanDefaultPrediction/'
os.makedirs(output_path, exist_ok=True)

# Encode categorical features
le = LabelEncoder()
obj_col = ['HasCoSigner', 'LoanPurpose', 'HasDependents', 'HasMortgage', 'MaritalStatus', 'EmploymentType', 'Education']
for col in obj_col:
    data[col] = le.fit_transform(data[col])

# Define predictor variables (X) and target variables (y)
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

# Initialize models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=41)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier())
]

# Function to evaluate models and tune thresholds
def evaluate_and_tune(models, X_train, X_test, y_train, y_test, thresholds, output_path):
    comparison_results = []
    threshold_results = []

    for model_name, model in models:
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate base metrics
        accuracy = model.score(X_test, y_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        auc_roc = roc_auc_score(y_test, y_proba)

        # Add to comparison results
        comparison_results.append({'Model': model_name, 'Metric': 'Accuracy', 'Values': accuracy})
        comparison_results.append({'Model': model_name, 'Metric': 'Precision', 'Values': precision})
        comparison_results.append({'Model': model_name, 'Metric': 'Recall', 'Values': recall})
        comparison_results.append({'Model': model_name, 'Metric': 'F1-Score', 'Values': f1})
        comparison_results.append({'Model': model_name, 'Metric': 'AUC-ROC', 'Values': auc_roc})

        # Threshold tuning
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
            threshold_results.append({
                'Model': model_name,
                'Threshold': threshold,
                'Metric': 'AUC-ROC', 'Values': auc_roc  # AUC-ROC remains constant for the model
            })

    # Save results to CSV
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(f"{output_path}/ModelComparisonResults.csv", index=False)

    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv(f"{output_path}/ThresholdTuningResults.csv", index=False)

# Call the evaluation and threshold tuning function
thresholds = np.arange(0.1, 1.0, 0.1)
evaluate_and_tune(models, X_train_scaled, X_test_scaled, y_train, y_test, thresholds, output_path)