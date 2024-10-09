"""
With my dissertation focusing on Lasso and Elastic net, I would like to revisit these
methods with a dataset that means more to me

A baseline linear regression will be constructed to compare and contrast the effects of the variables used
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Read in data
df = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/NFL Big Data Bowl/lin_reg_nn_data.csv')

# Define the output path
output_path = '/Users/stevenbarnes/Desktop/Resources/Data/NFL Big Data Bowl/Lasso & Enet/'

# Separate the features (X) from the target (Y)
X = df.drop('Yards', axis=1)
y = df['Yards']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in the feature sets
imputer = SimpleImputer(strategy='most_frequent')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Function to save model performance metrics to a csv
def save_performance(model_name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    performance_results = pd.DataFrame({
        'Model': [model_name],
        'MSE': [mse],
        'R-Squared': [r2]
    })
    performance_results.to_csv(f'{output_path}{model_name}_performance.csv', index=False)

# Function to save coefficients to a csv
def save_coefficients(model_name, model, feature_names):
    coeffs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    coeffs.to_csv(f'{output_path}{model_name}_coefficients.csv', index=False)

# Model Functions
def train_linreg(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def train_lasso(X_train, y_train):
    lasso = Lasso(max_iter=10000)
    params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    lasso_cv = GridSearchCV(lasso, params, cv=5)
    lasso_cv.fit(X_train, y_train)
    return lasso_cv

def train_enet(X_train, y_train):
    enet = ElasticNet(max_iter=10000)
    params = {'alpha': [0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.5, 0.9]}
    enet_cv = GridSearchCV(enet, params, cv=5)
    enet_cv.fit(X_train, y_train)
    return enet_cv

# Train the models
linreg = train_linreg(X_train, y_train)
lasso_cv = train_lasso(X_train, y_train)
enet_cv = train_enet(X_train, y_train)

# Make predictions using the best models found
y_pred_lr = linreg.predict(X_test)
y_pred_lasso = lasso_cv.predict(X_test)
y_pred_enet = enet_cv.predict(X_test)

# Save model performance metrics
save_performance('Lasso', y_test, y_pred_lasso)
save_performance('ElasticNet', y_test, y_pred_enet)
save_performance('LinReg', y_test, y_pred_lr)

# Get the feature names from the training data

feature_names = X.columns

# Save the coefficients for Lasso, ElasticNet, and Linear Regression
save_coefficients('Lasso', lasso_cv.best_estimator_, feature_names)
save_coefficients('ElasticNet', enet_cv.best_estimator_, feature_names)
save_coefficients('LinearRegression', linreg, feature_names)

print(f"Results saved to {output_path}")