
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

#splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Applying Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

#print feature importances in descending order
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot bar graph of feature importances
import matplotlib.pyplot as plt
plt.figure()
plt.title("Feature importances")
plt.xlabel('Number of features')
plt.ylabel('Importances')
plt.bar(range(X_train.shape[1]), importances[indices],
       color="b", align="center")

plt.show()
 
