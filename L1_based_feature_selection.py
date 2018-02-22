# importing libraries
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

# L1 based feature selection
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

# SelectFromModel and Logistic Regression
logreg = LogisticRegression(random_state = 0) 
logreg.fit(X, y)
model = SelectFromModel(logreg)
X_logreg = model.fit(X, y)
#X_logreg.shape

# Get indices of selected features
X_logreg.get_support(indices=True)

# SelectFromModel and LinearSVC
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_lsvc = model.transform(X)
X_lsvc.shape

# Get indices of selected features
#X_logreg.get_support(indices=True)

#splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_lsvc, y, test_size = 0.2, random_state = 0)

# fitting logistic regression to Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train)

#predicting Test set results
y_pred = classifier.predict(X_test)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('Training accuracy:', classifier.score(X_train, y_train))
print('Test accuracy:', classifier.score(X_test, y_test))
