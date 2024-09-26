# Last Lesson - Machine Learning: Classification

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import pyplot as plt
import warnings; warnings.simplefilter('ignore')

weekly_df = nfl.import_weekly_data(years=range(2000, 2022))

# filter out the positions that don't matter
eligible_positions = ['RB', 'WR', 'QB']
grouping_columns = ['player_id', 'season']
features = ['targets', 'receptions', 'rushing_yards', 'receiving_yards', 'passing_yards']
target = ['position']

# filtering the dataset
train_df = weekly_df.loc[weekly_df['position'].isin(eligible_positions), grouping_columns + features + target]

# group by week season, sum up feature columns, get back the player's position
groupby_funcs = {
    'position': 'first'
}

for feature in features:
    groupby_funcs[feature] = np.sum

train_df = train_df.groupby(grouping_columns, as_index=False).agg(groupby_funcs)

# turn the categorical targets to numerical values
train_df['position'] = train_df['position'].replace({
    'RB': 0,
    'WR': 1,
    'QB': 2
})

# dropna values in the dataset and filter out players who did not play
train_df = train_df.dropna()
train_df = train_df.loc[(train_df['rushing_yards'] > 200) | (train_df['passing_yards'] > 300) | (train_df['receiving_yards'] > 150)]

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    train_df[features],
    train_df[target],
    test_size=0.2,
    random_state=123
)

# create decision tree classifier, fit the model to training data, predict class labels, and evaluate performance
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

# Visualize the Decision Tree
fig, ax = plt.subplots(figsize=(10, 6))
class_names = ['RB', 'WR', 'QB']
plot_tree(clf, ax=ax, feature_names=features, class_names=class_names)
# plt.show()

# Decision Tree with hyperparameter tuning
params = {'max_depth': range(1, 10), 'min_samples_split': range(2, 6)}
clf2 = DecisionTreeClassifier()
grid_search = GridSearchCV(clf2, params, cv=10)
grid_search.fit(train_df[features], train_df[target])

best_clf2 = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(best_params)
print(best_score)
plot_tree(best_clf2, ax=ax, feature_names=features, class_names=class_names)

# Model Evaluation
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)
plt.show()

