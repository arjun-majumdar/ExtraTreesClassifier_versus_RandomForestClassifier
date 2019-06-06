# ExtraTreesClassifier_versus_RandomForestClassifier
A comparison between ExtraTreesClassifier and RandomForestClassifier for Statlog Shuttle data set from UCI archive


The dataset consists of 2 CSV files viz., 'shuttle_training.csv' and 'shuttle_test.csv'.

The accompanying Python code starts off with using base models of ExtraTreesClassifier and RandomForestClassifier,
and measures model metrics such as accuracy, precision and recall, followed with using 5-fold Cross-Validation.

Finally, hyper parameter tuning is done for the best performing model using-
1.) 'hyperopt' using Bayesian Optimization
2.) 'RandomizedSearchCV'
3.) 'GridSearchCV'

Each of the parameter tuning, builds on top of the previous model.
