import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP
from utils import plot_confusion_matrix, tune_classifier, normalize_by_year
import feather, os

# Used to specify a list for each hyperparam to tune in grid search
# params_to_tune = {

# }

# Hard-coded hyperparams
params = {
    "hidden_layer_sizes" : (25,),
    'alpha' : 1e-2,
    'max_iter' : 500,
    'verbose' : True
}

# Read in our static data stored in our cwd
cwd = os.path.abspath('.')
PATH_TO_DATA = os.path.join(cwd, 'year_by_year.feather')
data = feather.read_dataframe(PATH_TO_DATA)

# Preprocess data
data = data.drop(['CONF_COUNT', 'DIV_COUNT', 'TEAM_ID', 'TEAM_CITY', 'TEAM_NAME'], axis=1)
norm_data = normalize_by_year(data)
y = norm_data['NBA_FINALS_APPEARANCE']
X = norm_data.drop(['NBA_FINALS_APPEARANCE', 'YEAR'], axis=1)



# Get our splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Instantiate our classifier
clf = MLP(**params)

# Use cross validation to find best parameters using grid search
# best_params = tune_classifier(clf, params_to_tune, X_train, y_train)
# clf.set_params(best_params)

# Fit our final classifier
y_pred = clf.fit(X_train, y_train).predict(X_test)
test_score = clf.score(X_test, y_test)

# Score reporting
print('Accuracy score on testing set')
print(test_score)

plot_confusion_matrix(y_test, y_pred, title='Confusion Matrix for MLP Classifier')
plt.show()
