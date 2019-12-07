import feather, pickle
import pandas as pd

# Load in all our data
training_data = feather.read_dataframe('all_training_data.feather')
testing_data = feather.read_dataframe('2019_playoffs.feather')

# Drop unecessary data
total_drop_names = ['TEAM_ID_winner', 'GP_winner', 'WINS_winner', 'LOSSES_winner',  'CONF_RANK_winner', 'DIV_RANK_winner',  'PO_WINS_winner', 'PO_LOSSES_winner', 'PTS_RANK_winner', 'CONF_COUNT_winner', 'Series', 'Winner', 'Loser', 'home_team', 'DIV_COUNT_winner', 'Year']
training_data = training_data.drop(total_drop_names, axis=1)

# This will be used to identify our final predictions
predictions = testing_data[['Series', 'Winner', 'Loser', 'home_team']]
testing_data = testing_data.drop(total_drop_names, axis=1)

# Train/test split
X_train = training_data.drop(['Normalized_margin'], axis=1)
y_train = training_data['Normalized_margin']
X_test = testing_data.drop(['Normalized_margin'], axis=1)
y_test = testing_data['Normalized_margin']

# Load our model
with open('model.p', 'rb') as f:
    model = pickle.load(f)

# Get model predictions
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions['Predicted Margin'] = y_pred * 4


print(predictions)
