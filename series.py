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
from sklearn.tree import DecisionTreeClassifier
from utils import normalize_by_year, season_to_playoff_year, tune_classifier

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# read datasets
series_data = feather.read_dataframe('series.data')
year_data = feather.read_dataframe('year.data')

# normalize year-by-year data
drop_names = ['CONF_COUNT', 'DIV_COUNT', 'TEAM_ID', 'TEAM_CITY', 'TEAM_NAME', 'NBA_FINALS_APPEARANCE']
year_data = normalize_by_year(year_data, not_considering=drop_names)

# join series and year-by-year datasets
year_data['TEAM_FULLNAME'] = year_data['TEAM_CITY'] + ' ' + year_data['TEAM_NAME']
year_data['PLAYOFF_YEAR'] = year_data['YEAR'].map(season_to_playoff_year)

winner_join = series_data.merge(year_data, left_on=['Winner', 'Year'], right_on=['TEAM_FULLNAME', 'PLAYOFF_YEAR'])
loser_join = series_data.merge(year_data, left_on=['Loser', 'Year'], right_on=['TEAM_FULLNAME', 'PLAYOFF_YEAR'])
data_join = winner_join.merge(loser_join, on=['Winner', 'Loser', 'Margin', 'Year', 'PLAYOFF_YEAR'])

# drop unnecessary data
drop_names = ['CONF_COUNT', 'DIV_COUNT', 'TEAM_ID', 'TEAM_CITY', 'TEAM_NAME', 'YEAR', 'TEAM_FULLNAME', 'NBA_FINALS_APPEARANCE']
total_drop_names = []
for name in drop_names:
    total_drop_names.extend([name+'_x',name+'_y'])
total_drop_names.extend(['Winner', 'Loser', 'Year', 'PLAYOFF_YEAR'])
data = data_join.drop(total_drop_names, axis=1)

# extract features and labels
X = data.drop(['Margin'], axis=1)
y = data['Margin']

# models
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25)

# MLP
print('training MLP...')
params = {
    'hidden_layer_sizes' : (8,),
    'alpha' : 1e-2,
    'max_iter' : 500,
    'verbose' : False
}
clf = MLP(**params)
#params_to_tune = {'hidden_layer_sizes': [2, 4, 8, 16, 32],
#                  'alpha': [1e-4, 1e-3, 1e-2]}
#best_params = tune_classifier(clf, params_to_tune, Xtr, ytr)
#clf.set_params(best_params)
y_pred = clf.fit(Xtr, ytr).predict(Xte)
test_score = clf.score(Xte, yte)
print('MLP accuracy score: %f' % test_score)

# Linear
print('training linear regression...')
m = Model(LogisticRegression(solver='lbfgs', multi_class='multinomial'))
m.train(Xtr, ytr)
preds = m.predict(Xte)
accuracy = 1 - np.count_nonzero(yte-preds)/len(yte)
print('LR accuracy score: %f' % accuracy)

# Decision Tree
print('training decision tree...')
tree = DecisionTreeClassifier()
tree.fit(Xtr, ytr)
test_score = tree.score(Xte, yte)
print('DT accuracy score: %f' % test_score)
