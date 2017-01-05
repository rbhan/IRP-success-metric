# Rebecca Han
# 6/14/16 updated
#####################################################
    # Classifies UCI Machine Learning: Contraceptive Methods (1473 instances, 10 attributes inc class)
    # uses dataframes!
    # creates graphical decision tree as png

import os
import subprocess
import math
import pandas as pd
import numpy as np
import pydot
#import matplotlib.pyplot as plt
from sklearn import ensemble, linear_model, preprocessing, cross_validation, metrics
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, log_loss, mean_absolute_error, mean_squared_error

# Load data as DataFrame ########################################################################################################################################################
def read_data(filename):
    df = pd.read_table(filename, sep=",", header=None, names=("wife_age", "wife_edu", "hus_edu", "num_chd", "wife_rlgn", "wife_emp", "hus_occ", "sol_index", "media", "bc_method"))
    return df

# Calculate misclassification error
def misclassification_error(confusion_matrix):
    cm = confusion_matrix
    m = len(cm)
    classified = 0
    misclassified = 0
    total = 0
    for i in range(0,m-1):
        for j in range(0,m-1):
            if i == j:
                total = total + cm[i][j]
                classified = classified + cm[i][j]
            else:
                total = total + cm[i][j]
                misclassified = misclassified + cm[i][j]

    return (float(classified)/float(total), float(misclassified)/float(total))

# Benchmark, just the linear regression
def benchmark(X_train, y_train, X_test, y_test):
    lr = LinearRegression(fit_intercept=True, normalize=False)
    lr = lr.fit(X_train, y_train)
    predict_class = lr.predict(X_test)
    mae = mean_absolute_error(y_test, predict_class)
    rmse = math.sqrt(mean_squared_error(y_test, predict_class))
    return(predict_class, mae, rmse)

#################################################################################################################################################################################
# Train a decision tree ########################################################################################################################################################
#def train_model(df): # can add other inputs (e.g. target column, split fraction, min_samples_split, etc)

df = read_data('dataset_contraceptives')
    
# Select traget column and data columns
features = list(df.columns[:8])
    # features = ['wife_age', 'wife_edu', 'hus_edu', 'num_chd', 'wife_rlgn', 'wife_emp', 'hus_occ', 'sol_index']
target = df.columns[9]
    # target = ['bc_method']
X = df[features] # selects all rows with correct headers
y = df[target]
map_X = pd.get_dummies(X.ix[:, :]) # have to use X not X_train because train_test_split will indices in X not in X_train

# Split data set, CV (k-fold) on the training data
k = 5
cv = KFold(len(X), n_folds=k, shuffle=True)

# Train data
clf = ensemble.RandomForestClassifier(n_estimators=10, min_samples_split=100, random_state=99)
    # min_samples_split=20 requires 20 samples in a node for it to be split
    # random_state=99 to seed the random number generator
kavg_acc = float(0)
kavg_mce = float(0)
kavg_ll = float(0)
kavg_mae = float(0)
kavg_rmse = float(0)
bm_mae = float(0)
bm_rmse = float(0)
for train, vali in cv:
# Get the dataset; this is the way to access values in a pandas DataFrame
    X_train = map_X.ix[train, :]; y_train = y[train]
    X_vali = map_X.ix[vali, :]; y_vali = y[vali]
    #print(X_train)
# Train and evaluate model
    clf = clf.fit(X_train, y_train)
    predict_class = clf.predict(X_vali) # predict class or regression value for X
    predict_class_proba = clf.predict_proba(X_vali) # predicts class probabities of X
# SCORE MIS-CLASSIFICATION ERROR
    cm = confusion_matrix(y_vali, predict_class)
    [acc, mce] = misclassification_error(cm)
    kavg_acc = kavg_acc + acc
    kavg_mce = kavg_mce + mce
# SCORE CROSS ENTROPY
    ll = log_loss(y_vali, predict_class_proba)
    kavg_ll = kavg_ll + ll
# SCORE MEAN ABSOLUTE ERROR
    mae = mean_absolute_error(y_vali, predict_class)
    kavg_mae = kavg_mae + mae
# SCORE ROOT MEAN SQUARED ERROR
    rmse = math.sqrt(mean_squared_error(y_vali, predict_class))
    kavg_rmse = kavg_rmse + rmse
# BENCHMARK
    [benchmark_predict, benchmark_mae, benchmark_rmse] = benchmark(X_train, y_train, X_vali, y_vali)
    bm_mae = bm_mae + benchmark_mae
    bm_rmse = bm_rmse + benchmark_rmse

print("% Correctly Classified Instances: ", 100*kavg_acc/k)
print("% Incorrectly Classified Instances: ", 100*kavg_mce/k)
print("Cross Entropy: ", kavg_ll/k)
print("Random Forest MAE: ", kavg_mae/k, "Benchmark MAE: ", bm_mae/k)
print("Random Forest RMSE: ", kavg_rmse/k, "Benchmark RMSE: ", bm_rmse/k)
