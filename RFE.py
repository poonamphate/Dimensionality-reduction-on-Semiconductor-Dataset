# Recurssive feature elimination method

#importing libraries
import pandas as pd

#importing dataset
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

# feature selection using Recurssive feature eleimination
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) 
rfe = RFE(estimator = classifier)
rfe.fit(X, y)
ranking = rfe.ranking_


