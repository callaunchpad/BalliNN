'''Model baselines on series data.'''

import feather
import numpy as np
import pandas as pd
import warnings
import pickle

from pipeline import Model
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
from utils import normalize_by_year, season_to_playoff_year, tune_classifier
from sklearn import preprocessing
from sklearn.utils import multiclass

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

# All of our available data
joined_data = feather.read_dataframe('joined.data')

# Data from 2019 for final testing
hold_out_data = joined_data[joined_data['Year'] == '2019']
hold_out_data.reset_index(drop=True).to_feather('2019_playoffs.feather')

# Set aside some data for final playoff prediction
joined_data = joined_data[joined_data['Year'] != '2019']
joined_data.to_feather('all_training_data.feather')

# drop unnecessary data
total_drop_names = ['TEAM_ID_winner', 'GP_winner', 'WINS_winner', 'LOSSES_winner',  'CONF_RANK_winner', 'DIV_RANK_winner',  'PO_WINS_winner', 'PO_LOSSES_winner', 'PTS_RANK_winner', 'CONF_COUNT_winner', 'Series', 'Winner', 'Loser', 'home_team', 'DIV_COUNT_winner', 'Year']
data = joined_data.drop(total_drop_names, axis=1)

# extract features and labels and store data for later
X = data.drop(['Normalized_margin'], axis=1)
y = data['Normalized_margin']


# Function for calculating win/loss accuracy from regression results
def accuracy_score(y_pred, y_true):
    return sum(np.sign(y_pred - 0.5) == np.sign(y_true - 0.5)) / len(y_pred)



################ Hyperparameter tuning and model selection ################

# Dictionary to store our trained models
models = {}

### Regressors ###

# split data
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25)


# Linear Regression
print("Linear Regression")
params_to_tune = {
    'normalize' : [True, False],
    'alpha' : [0.0, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0]
}

# Cross validation with grid search 
linear_model = Ridge()
tuned_params = tune_classifier(linear_model, params_to_tune, Xtr, ytr)

# Fit with our tuned parameters
linear_model.set_params(**tuned_params)
linear_model.fit(Xtr, ytr)
y_pred = linear_model.predict(Xte)
score = linear_model.score(Xte, yte)
accuracy = accuracy_score(y_pred, yte)
print("MAE:", metrics.mean_absolute_error(yte, y_pred)) 
print('R2:', score)
print("Accuracy:", accuracy)
models[linear_model] = accuracy

# MLP Regresion
print("MLP Regression")
# These params (taken from above) made it worse (0.21 w params vs 0.17 wo params)
params = {
    'hidden_layer_sizes' : (100,),
    'alpha' : 0.5e-4,
    'max_iter' : 3000,
    'verbose' : False
}
mlp_model = MLPRegressor(**params)
mlp_model.fit(Xtr, ytr)
y_pred = mlp_model.predict(Xte)
score = mlp_model.score(Xte, yte)
accuracy = accuracy_score(y_pred, yte)
print("MAE:", metrics.mean_absolute_error(yte, y_pred))
print('R2:', score)
print("Accuracy:", accuracy)
models[mlp_model] = accuracy

# DT Regression
print("DT Regression")
params_to_tune = {
    'max_depth' : [2, 5, 10, 20, 50, 100],
    'n_estimators' : [5, 10, 20, 50, 100]
}

# Cross validation and grid search
dt_model = RandomForestRegressor()
tuned_params = tune_classifier(dt_model, params_to_tune, Xtr, ytr)

# Fit with final hyperparameters
dt_model.set_params(**tuned_params)
dt_model.fit(Xtr, ytr)
y_pred = dt_model.predict(Xte)
score = dt_model.score(Xte, yte)
accuracy = accuracy_score(y_pred, yte)
print("MAE:", metrics.mean_absolute_error(yte, y_pred))
print('R2', score)
print("Accuracy:", accuracy)
models[dt_model] = accuracy




### Classifiers ###

# Discretize our labels into binary win-loss labels
y_cls = (y > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.3)

# Logistic Regression
print('Training Logistic Regression')
m = LogisticRegression(solver='lbfgs', multi_class='multinomial')
m.fit(X_train, y_train)
score = m.score(X_test, y_test)
print('LR accuracy score: %f' % score)
models[m] = score

# Decision Tree
print('training decision tree...')
params_to_tune = {
    'max_depth' : [2, 5, 10, 20, 50, 100],
    'n_estimators' : [5, 10, 20, 50, 100]
}
tree = RandomForestClassifier()
best_params = tune_classifier(tree, params_to_tune, X_train, y_train)

tree.set_params(**best_params)
tree.fit(X_train, y_train)
test_score = tree.score(X_test, y_test)
print('DT accuracy score: %f' % test_score)
models[tree] = test_score

print(X_train.columns)
feature_importances = pd.Series(tree.feature_importances_, index=X_train.columns)
print(feature_importances)

# SVM
print('Training SVM...')
params_to_tune = {
    'kernel' : ['linear', 'rbf'],
    'C' : [0.01, 0.05, 0.1, 0.5]
}
svm = SVC()
best_params = tune_classifier(svm, params_to_tune, X_train, y_train)

svm.set_params(**best_params)
svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print('SVM accuracy score: %f' % score)
models[svm] = score



### Saving our models ###

# Select and save our best performing model
best_model = max(models, key=lambda k : models[k])
with open('model.p', 'wb') as f:
    pickle.dump(linear_model, f)
