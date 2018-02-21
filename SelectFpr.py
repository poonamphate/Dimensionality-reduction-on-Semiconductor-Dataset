# Univariate feature selection using false positive rate

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

#feature scaling
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_norm = mms.fit_transform(X)

# Univariate feature selection using false positive rate
from sklearn.feature_selection import SelectFpr, f_classif
X_fpr = SelectFpr(f_classif, alpha = 0.05).fit(X, y)

# Get indices of selected features
X_fpr.get_support(indices=True)

# select features using false positive rate method
X_fpr = SelectFpr(f_classif, alpha = 0.05).fit_transform(X, y)
print(X_fpr.shape)

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_fpr, y, test_size = 0.2, random_state = 0)

# fitting logistic regression to Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train)

# Confusion matrix for training set
from sklearn.metrics import confusion_matrix
X_pred = classifier.predict(X_train)
cm_training = confusion_matrix(y_train, X_pred)

#predicting Test set results
y_pred = classifier.predict(X_test)

# Confusion matrix for test set
cm_test = confusion_matrix(y_test, y_pred)

print('Training accuracy:', classifier.score(X_train, y_train))
print('Test accuracy:', classifier.score(X_test, y_test))


