import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

melbourne_data = pd.read_csv('melb_data.csv')

melbourne_data.describe()

melbourne_data.columns

melbourne_data.dropna()

# set the prediction target (label)
y = melbourne_data.Price

# set features, X
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
X.describe()
X.head()

# -----------------------BEFORE SPLITTING THE DATA-----------------------------#
# Define model. Specify a number for random_state to ensure same results each run
# melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
# melbourne_model.fit(X, y)

# print('Making predictions for the following 5 houses:')
# print(X.head())
# print('The predictions are')
# print(melbourne_model.predict(X.head()))

# MAE = 1125.1804614629357


# -----------------------AFTER SPLITTING THE DATA-----------------------------#
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
melbourne_model = DecisionTreeRegressor()

# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# MAE = 242567.53078055964
