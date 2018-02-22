
# importing libraries
import numpy as np
import pandas as pd

# importing dataset
# create matrix of independent variables(features)
data_X = pd.read_csv('secom.data.txt', sep = ' ')
X = data_X.values

#create dependent variable vector
data_y = pd.read_csv('secom_labels.data.txt', sep = ' ')
y = data_y.iloc[:, 0].values

# handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

#use LassoCV as base estimator
from sklearn.linear_model import LassoCV
clf = LassoCV()

# Set a minimum threshold of 0.25
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(clf, threshold=0.9)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]
