'''Model baselines on series data.'''

import feather
import numpy as np
import pandas as pd
import warnings

from pipeline import Model
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
from utils import normalize_by_year, season_to_playoff_year, tune_classifier
from sklearn import preprocessing
from sklearn.utils import multiclass
from data_preprocessing import joined_data 

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


# drop unnecessary data
#TODO
total_drop_names = ['TEAM_ID_winner', 'GP_winner', 'WINS_winner', 'LOSSES_winner',  'CONF_RANK_winner', 'DIV_RANK_winner',  'PO_WINS_winner', 'PO_LOSSES_winner', 'PTS_RANK_winner']
data = joined_data.drop(total_drop_names, axis=1)

# extract features and labels
# CHANGE THIS: Dropped columns with nan values because the models didn't like them
X = data.drop(['Normalized_Margin'], axis=1).dropna(axis=1, how='any')
y = data['Normalized_Margin'].values

# models
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25)

# MLP
# print('training MLP...')
# params = {
#     'hidden_layer_sizes' : (8,),
#     'alpha' : 1e-2,
#     'max_iter' : 500,
#     'verbose' : False
# }
# clf = MLP(**params)
# #params_to_tune = {'hidden_layer_sizes': [2, 4, 8, 16, 32],
# #                  'alpha': [1e-4, 1e-3, 1e-2]}
# #best_params = tune_classifier(clf, params_to_tune, Xtr, ytr)
# #clf.set_params(best_params)
# y_pred = clf.fit(Xtr, ytr).predict(Xte)
# test_score = clf.score(Xte, yte)
# print('MLP accuracy score: %f' % test_score)

# Linear Regression
print("Training Linear Regression")
linear_model = LinearRegression()
linear_model.fit(Xtr, ytr)
y_pred = linear_model.predict(Xte)
print("MAE:", metrics.mean_absolute_error(yte, y_pred))  

# MLP Regresion
print("Training MLP Regression")
# These params (taken from above) made it worse (0.21 w params vs 0.17 wo params)
params = {
    'hidden_layer_sizes' : (8,),
    'alpha' : 1e-2,
    'max_iter' : 500,
    'verbose' : False
}
mlp_model = MLPRegressor(**params)
mlp_model.fit(Xtr, ytr)
print("MAE:", metrics.mean_absolute_error(yte, mlp_model.predict(Xte)))

# DT Regression
print("Training DT Regression")
dt_model = DecisionTreeRegressor()
dt_model.fit(Xtr, ytr)
print("MAE:", metrics.mean_absolute_error(yte, dt_model.predict(Xte)))


# Weird Stuff that some Kaggle guy said to try
# lab_enc = preprocessing.LabelEncoder()
# training_scores_encoded = lab_enc.fit_transform(ytr)
# print(training_scores_encoded)
# print(multiclass.type_of_target(ytr))
# print(multiclass.type_of_target(ytr.astype('int')))
# print(multiclass.type_of_target(training_scores_encoded))
# print('training linear regression...')

# Logistic Regression
# m = Model(LogisticRegression(solver='lbfgs', multi_class='multinomial'))
# m.train(Xtr, training_scores_encoded)
# preds = m.predict(Xte)
# accuracy = 1 - np.count_nonzero(yte-preds)/len(yte)
# print('LR accuracy score: %f' % accuracy)

# # Decision Tree
# print('training decision tree...')
# tree = DecisionTreeClassifier()
# tree.fit(Xtr, ytr)
# test_score = tree.score(Xte, yte)
# print('DT accuracy score: %f' % test_score)
