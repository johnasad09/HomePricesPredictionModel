import pandas as pd
from sklearn.tree import DecisionTreeRegressor

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

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

print('Making predictions for the following 5 houses:')
print(X.head())
print('The predictions are')
print(melbourne_model.predict(X.head()))
