# Dimensionality-reduction-on-Semiconductor-Dataset
### Introduction

In this project, semiconductor manufacturing dataset has been studied and analyzed. 
In a modern semiconductor industry, manufacturing is a complex and evolved process. Semiconductor wafers undergo various processing tools where its electric and physical characteristics are altered. At each stage, vast number of signals are collected and analyzed. However, not all the signals are equally valuable. The information obtained from these collected signals contain useful information as well as noise. Thus, dimensionality reduction techniques such as feature selection and feature extraction can be applied to obtain most relevant features. 
In this project we will explore and compare following techniques.
1.	Removing features with low variance
2.	Univariate feature selection
3.	L1 based feature selection
4.	Feature selection using Random Forest
5.	Principal Component Analysis

### Dataset

Dataset used for this project is secom semiconductor manufacturing dataset from UCI machine learning repository. (http://archive.ics.uci.edu/ml/datasets/secom)
secom dataset is a classification dataset with 1567 samples and 591 attributes (590 variables and 1 class variable).

### Technical requirement

This project is developed in Python using scikit learn machine learning library. In order to run this code onto your system, you need to have python and scikit learn package installed onto your system. You can use any python IDE. I found Spyder (python IDE) really helpful for visualizing data frames and variables. 

### Feature selection techniques

1.	Removing features with low variance
In this approach features whose variance is below certain predefined threshold are removed. The variance of variables can be calculated by the formula
		Var[x] = p (1 - p)

2.	Univariate feature selection
In this approach features are selected based on univariate statistical test. It is considered as a processing step to an estimator. 

3.	L1 based feature selection
L1 based feature selection is useful for sparse dataset. This approach is used as a sparse estimator to select non-zero coefficients.

4.	Feature selection using Random Forest
This approach can be used to compute feature importances and then we can decide which features to keep and which to discard for designing our classification model.

### Feature extraction technique
Principal Component Analysis
PCA is a feature extraction technique that uses orthogonal transformation to convert set of corelated features into linearly unrelated features called principle components.



### Comparative analysis

1.	Base case accuracy for secom dataset using logistic regression classifier is found as
Training - 99.12%
Testing - 92.67%
2.	It can be seen clearly that the accuracy of training set is higher than test set. This indicates that our model is overfitting the training data and it fails to generalize well on unseen data. Thus, applying dimensionality reduction techniques would help to reduce the noise from training data and thus improving model performance on test set.
3.	Following table shows performance of different dimensionally reduction techniques on secom datasets.

Dimensionality reduction technique	Features	Training set accuracy(%)	Test set accuracy(%)
1.Logistic regression	590	99.12	92.67
2.Low variance	266	95.76	93.31
3.Univariate	50	92.89	95.85
4.L1 based feature selection	
103	
94.00	
92.99
5.PCA	95	94.03	93.87  

4.	From the above table, it can be concluded that, after applying feature selection techniques like low variance, univariate feature selection and L1 based feature selection as a preprocessing step, performance of classifier has improved. However, the best performance is achieved by applying PCA with 95 transformed features.
 

