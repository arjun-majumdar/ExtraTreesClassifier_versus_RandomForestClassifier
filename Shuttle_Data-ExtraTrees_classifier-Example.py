

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
import time


'''
This is a multi-class classification. The target attribute is 'Class'

Statlog (Shuttle) Data Set
Abstract: The shuttle dataset contains 9 attributes all of which are numerical.
Approximately 80% of the data belongs to class 1

To download and read more about dataset, refer-
https://archive.ics.uci.edu/ml/datasets/Statlog+%28Shuttle%29
'''


# Column names to be used for training and testing sets-
col_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'Class']

# Read in training and testing datasets-
training_data = pd.read_csv("shuttle_training.csv", delimiter = ' ', names = col_names)
testing_data = pd.read_csv("shuttle_test.csv", delimiter = ' ', names = col_names)

# Training data dimension-
training_data.shape
# (43499, 10)

# Testing data dimension-
testing_data.shape
# (14499, 10)

# Check training data for missing values-
training_data.isnull().values.any()
# False

# Check testing data for missing values-
testing_data.isnull().values.any()
# False


# Get distribution of target attribute-
training_data['Class'].value_counts()
'''
1    34108
4     6748
5     2458
3      132
2       37
7       11
6        6
Name: Class, dtype: int64
'''


# DROPPING 'Class' == 6 due to extremely small proportion of values as compared
# to entire dataset!

# To get indices of all rows in DataFrame where 'Class' == 6-
# training_data[training_data['Class'] == 6].index

# To store the indices in a list-
# td_class_6 = list(training_data[training_data['Class'] == 6].index)


# To REMOVE 'Class' == 6 from 'training_data'-
td_class_6 = training_data.Class == 6

type(td_class_6)
# pandas.core.series.Series

training_data_modified = training_data.loc[~td_class_6]


# Sanity check-
training_data_modified['Class'].value_counts()
'''
1    34108
4     6748
5     2458
3      132
2       37
7       11
Name: Class, dtype: int64
'''


'''
# To get distribution of unique values for attribute 'Class'-
training_data["Class"].value_counts()

# To visualize a categorical attribute 'Class' in training data-
training_data["Class"].value_counts().plot(kind = 'bar')

plt.xlabel("Class")
plt.ylabel("Class - Value Counts")

plt.show()


# OR
# Using 'seaborn'-
sns.countplot(training_data["Class"])

plt.xlabel("Class")
plt.ylabel("Class - Value Counts")

plt.show()
'''




# To divide the data into attributes and labels, execute the following code:

# 'X' contains attributes
# X = training_data.drop('Class', axis = 1)
X = training_data_modified.drop('Class', axis = 1)

# Convert 'X' to float-
X = X.values.astype("float")

# 'y' contains labels
y = training_data_modified['Class']


# Standardize Features-
# scaler = StandardScaler()
# mm_scaler = MinMaxScaler()
rb_scaler = RobustScaler()

# Standardize features (X)-
# X_std = scaler.fit_transform(X)
# X_std = mm_scaler.fit_transform(X)
X_std = rb_scaler.fit_transform(X)


# Divide attributes & labels into training & testing sets-
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.30, stratify = y)

print("\nDimensions of training and testing sets are:")
print("X_train = {0}, y_train = {1}, X_test = {2} and y_test = {3}\n\n".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
# X_train = (30450, 9), y_train = (30450,), X_test = (13050, 9) and y_test = (13050,)




# Using 'ExtraTreesClassifier' model-

# Initialize 'ExtraTreesClassifier' model-
et_clf = ExtraTreesClassifier(n_estimators=500)

# rf_clf = RandomForestClassifier(n_estimators=500)

# Train model on training data-
et_clf.fit(X_train, y_train)

# Make predictions using trained model-
y_pred = et_clf.predict(X_test)


# Get model metrics to gauge model performance-
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print("\nExtraTreesClassifier (n_estimators = 500) model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} & Recall = {2:.4f}\n".format(accuracy, precision, recall))
# ExtraTreesClassifier (n_estimators = 500) model metrics are:
# Accuracy = 0.9994, Precision = 0.8838 & Recall = 0.8846




# Using 'RandomForestClassifier' model-

# Initialize classifier-
rf_clf = RandomForestClassifier(n_estimators=500)

# Train model on training data-
rf_clf.fit(X_train, y_train)

# Make predictions using trained model-
y_pred_rf = rf_clf.predict(X_test)


# Get model metrics to gauge model performance-
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='macro')
recall_rf = recall_score(y_test, y_pred_rf, average='macro')

print("\nRandomForestClassifier (n_estimators = 500) model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} & Recall = {2:.4f}\n".format(accuracy_rf, precision_rf, recall_rf))
# RandomForestClassifier (n_estimators = 500) model metrics are:
# Accuracy = 0.9997, Precision = 0.9881 & Recall = 0.8847




# Use 5-fold Cross-Validation for ExtraTreesClassifier-
et_cvs = cross_val_score(et_clf, X_train, y_train, cv = 5)

print("\n'accuracy' metrics for 5-fold CV for 'ExtraTreesClassifier' are:")
print("Mean = {0:.4f} and Standard Deviation = {1:.4f}\n".format(et_cvs.mean(), et_cvs.std()))
# 'accuracy' metrics for 5-fold CV for 'ExtraTreesClassifier' are:
# Mean = 0.9995 and Standard Deviation = 0.0003


# Use 5-fold Cross-Validation for RandomForestClassifier-
rf_cvs = cross_val_score(rf_clf, X_train, y_train, cv = 5)

print("\n'accuracy' metrics for 5-fold CV for 'RandomForestClassifier' are:")
print("Mean = {0:.4f} and Standard Deviation = {1:.4f}\n".format(rf_cvs.mean(), rf_cvs.std()))
# 'accuracy' metrics for 5-fold CV for 'RandomForestClassifier' are:
# Mean = 0.9998 and Standard Deviation = 0.0001


# Observation: Since 'RandomForestClassifier' gives us better accuracy scores when
# used with 5-fold CV, hyper parameter tuning is done for it


# Get feature importance using 'ExtraTreesClassifier'-
fi_etc = pd.DataFrame({'feature_importance_scores': et_clf.feature_importances_, 'attributes': col_names[:-1]})

# Sort in descending order-
fi_etc.sort_values('feature_importance_scores', ascending=False, inplace=True)


# Get feature importance using 'RandomForestClassifier'-
fi_rfc = pd.DataFrame({'feature_importance_scores': rf_clf.feature_importances_, 'attributes': col_names[:-1]})

# Sort in descending order-
fi_rfc.sort_values('feature_importance_scores', ascending=False, inplace=True)




# Using hyper parameter optimization-

# Performing hyperparameter optimization using 'hyperopt'-

start_time = time.time()

def accuracy_model(params):
	# Train a RF classifier using 'params'-
	params = {
			'n_estimators': int(params['n_estimators']),
			'max_depth': int(params['max_depth']),
			'min_samples_split': int(params['min_samples_split']),
			'min_samples_leaf': int(params['min_samples_leaf']),

		}

	# clf = RandomForestClassifier(n_jobs=4, class_weight='balanced', **params)
	rf_clf = RandomForestClassifier(**params)

	# Use 5-fold CV to get mean of cross validated score(s)-
	return cross_val_score(rf_clf, X_train, y_train, cv = 5).mean()


# Parameters to be used by 'hyperopt'-
param_space = {

	'n_estimators': hp.uniform('n_estimators', 200, 400),
	'max_features': hp.choice('max_features', ('auto', 'sqrt', 'log2')),
	'max_depth': hp.uniform( 'max_depth', 2, 10),
	'min_samples_split': hp.uniform('min_samples_split', 2, 10),
	'min_samples_leaf': hp.uniform('min_samplesleaf', 1, 5),
	'criterion': hp.choice('criterion', ['gini', 'entropy']),
	'bootstrap': hp.choice('bootstrap', [True, False])
}


# Keep track of 'best' accuracy score found so far-
best = 0


def get_best_accuracy(params):
	global best
	accuracy = accuracy_model(params)

	if accuracy > best:
		best = accuracy

	print("\nNew best accuracy = {0:.4f} & parameters: {1}".format(best, params))
	
	return {'loss': (1 - accuracy), 'status': STATUS_OK}


trials = Trials()

best = fmin(get_best_accuracy, param_space, algo = tpe.suggest, max_evals=100, trials = trials)

print("\n\nBest parameters obtained using 'hyperopt' for RandomForest classifier are:\n{0}\n\n".format(best))
'''
Best parameters obtained using 'hyperopt' for RandomForest classifier are:
{'bootstrap': 0, 'criterion': 1, 'max_depth': 9.091719812358047, 'max_features': 0,
'min_samples_split': 5.0415612221264, 'min_samplesleaf': 1.4195401649972466,
'n_estimators': 271.6604782077072}
'''

end_time = time.time()
print("\n\nTotal time taken by hyperopt for RF classifier = {0:.4f}\n\n\n".format(end_time - start_time))
# Total time taken by hyperopt for RF classifier = 1815.2220




# Using RandomizedSearchCV-
'''
# Random Search is a similar approach to Grid Search however now instead of testing
# all possible combinations, 'n_iter' sets of parameters are randomly selected.

# According to RandomSearchCV documentation, it is highly recommended to draw from
# continuous distributions for continuous parameters, thus we will use uniform
# distribution U(0.6, 1.0) for subsample and column sampling by tree parameters.

# Additionally, the number of estimators will be drawn from discrete uniform distribution
'''

start_time = time.time()

# Parameters to be used for RandomizedSearchCV-
rs_params = {
	'n_estimators': sp_randint(250, 350),
	'max_features': ['auto'],
	'bootstrap': [True],
	'criterion': ['gini', 'entropy'],
	'max_depth': sp_randint(7, 10),
	'min_samples_split': sp_randint(4, 7),
	'min_samples_leaf': sp_randint(1, 3)

}

# Using 5-fold CV using RandomizedSearchCV-
rs_cv = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=rs_params, cv = 5, n_iter=100)

# Train on training data-
rs_cv.fit(X_train, y_train) 

print("\n\nBest parameters found using RandomizedSearchCV are:\n{0}".format(rs_cv.best_params_))
'''
Best parameters found using RandomizedSearchCV are:
{'bootstrap': True, 'criterion': 'gini', 'max_depth': 9, 'max_features': 'auto',
'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 334}
'''

print("Best score achieved using RandomizedSearchCV = {0:.4f}\n\n".format(rs_cv.best_score_))
# Best score achieved using RandomizedSearchCV = 0.9999


end_time = time.time()
print("\n\nTraining time of RandomizedSearchCV for Random Forest classifier = {0:.4f}\n\n".format(end_time - start_time))
# Training time of RandomizedSearchCV for Random Forest classifier = 1913.9328




# Fine tune parameters from above using 'GridSearchCV'-
start_time = time.time()

# Parameters to be used for GridSearchCV-
gs_params = {
	'n_estimators': [320, 334, 345],
	'max_features': ['auto'],
	'bootstrap': [True],
	'criterion': ['gini'],
	'max_depth': [8, 9, 10],
	'min_samples_split': [5, 6, 7],
	'min_samples_leaf': [1, 2, 3]

}

# Using 5-fold CV using GridSearchCV-
gs_cv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=gs_params, cv = 5)

# Train on training data-
gs_cv.fit(X_train, y_train)

print("\n\nBest parameters found using GridSearchCV are:\n{0}".format(gs_cv.best_params_))
'''
Best parameters found using GridSearchCV are:
{'bootstrap': True, 'criterion': 'gini', 'max_depth': 8, 'max_features': 'auto',
'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 320}
'''

print("Best score achieved using GridSearchCV = {0:.4f}\n\n".format(gs_cv.best_score_))
# Best score achieved using GridSearchCV = 0.9999


end_time = time.time()
print("\n\nTraining time of GridSearchCV for Random Forest classifier = {0:.4f}\n\n".format(end_time - start_time))
# Training time of GridSearchCV for Random Forest classifier = 1806.8843




# Train a 'best' RF classifier using parameters from above-
best_rf_clf = RandomForestClassifier(
	n_estimators=320, min_samples_split=5, min_samples_leaf=1,
	max_features='auto', max_depth=8, criterion='gini', bootstrap=True
	)

# Train best model on training data-
best_rf_clf.fit(X_train, y_train)

# Make predictions using best model-
y_pred_best = best_rf_clf.predict(X_test)


# Get model metrics-
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best, average = 'macro')
recall = recall_score(y_test, y_pred_best, average = 'macro')
f1score = f1_score(y_test, y_pred_best, average = 'macro')

print("\nRandom Forest best model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f}, Recall = {2:.4f} & F1-Score = {3:.4f}\n\n".format(accuracy, precision, recall, f1score))
# Random Forest best model metrics are:
# Accuracy = 0.9998, Precision = 0.9919, Recall = 0.8847 & F1-Score = 0.9105


# Use 5-fold CV-
best_rf_cvs = cross_val_score(best_rf_clf, X_train, y_train, cv = 5)

print("\n'accuracy' metrics for 5-fold CV for 'best' model of 'RandomForestClassifier' are:")
print("Mean = {0:.4f} and Standard Deviation = {1:.4f}\n".format(best_rf_cvs.mean(), best_rf_cvs.std()))
# 'accuracy' metrics for 5-fold CV for 'best' model of 'RandomForestClassifier' are:
# Mean = 0.9998 and Standard Deviation = 0.0001


print("\nConfusion Matrix using 'best' model is:")
print(confusion_matrix(y_test, y_pred_best))
'''
Confusion Matrix using 'best' model is:
[[10233     0     0     0     0     0]
 [    0    11     0     0     0     0]
 [    1     0    39     0     0     0]
 [    0     0     0  2025     0     0]
 [    0     0     0     0   737     0]
 [    0     0     2     0     0     1]]
'''


