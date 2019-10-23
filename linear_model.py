import feather
import numpy as np

from pipeline import Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# pull data
data = feather.read_dataframe('./datafile')

# extract + format features/labels
feature_names = data.columns.tolist()
X = data[feature_names[4:]]
y = data['NBA_FINALS_APPEARANCE']
X = X.drop(['NBA_FINALS_APPEARANCE', 'CONF_COUNT', 'DIV_COUNT'], axis=1)

y[y == 'N/A'] = 0
y[y == 'FINALS APPEARANCE'] = 1
y[y == 'LEAGUE CHAMPION'] = 2
y = y.astype('int')

X = np.array(X)
y = np.array(y)

# extract training/testing data
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25)

# model
m = Model(LogisticRegression(solver='lbfgs', multi_class='multinomial'))
m.train(Xtr, ytr)
preds = m.predict(Xte)

# accuracy measurements
accuracy = 1 - np.count_nonzero(yte-preds)/len(yte)
dist_matrix = np.zeros((3,3))
for i in range(len(yte)):
    dist_matrix[yte[i], preds[i]] += 1
print(dist_matrix)
print(accuracy)
