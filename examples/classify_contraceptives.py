# Rebecca Han
# 6/9/16 updated
#####################################################
    # Classifies UCI Machine Learning: Contraceptive Methods (1473 instances, 10 attributes inc class)
    # uses dataframes!
    # creates graphical decision tree as png

import os
import subprocess

import pandas as pd
import numpy as np
import pydot
#import matplotlib.pyplot as plt
from sklearn import tree, preprocessing, cross_validation, metrics
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve

# Load data as DataFrame ########################################################################################################################################################
def read_data(filename):
    df = pd.read_table(filename, sep=",", header=None, names=("wife_age", "wife_edu", "hus_edu", "num_chd", "wife_rlgn", "wife_emp", "hus_occ", "sol_index", "media", "bc_method"))
    return df

# Add column to dataframe with integers for the target ##########################################################################################################################
def encode_target(df, target_column):
    """
    Args:       df -- pandas DataFrame.
                target_column -- column to map to int, producing new Target column.
    Returns:    df_mod -- modified DataFrame.
                targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

# Create tree png using graphviz
def visualize_tree(tree, feature_names):
    """
    Args:       tree -- scikit-learn DecsisionTree
                feature_names -- list of feature names
    """
    with open("clf.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)
    command = ["dot", "-Tpng", "clf.dot", "-o", "clf.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

# Calculate misclassification error ############################################################################################################################################
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

# Train a decision tree ########################################################################################################################################################
def train_model(df, k, min_split):
    # Select traget column and data columns
    features = list(df.columns[:8])
        # features = ['wife_age', 'wife_edu', 'hus_edu', 'num_chd', 'wife_rlgn', 'wife_emp', 'hus_occ', 'sol_index']
    target = df.columns[9]
        # target = ['bc_method']
    X = df[features] # selects all rows with correct headers
    y = df[target]
    map_X = pd.get_dummies(X.ix[:, :]) # have to use X not X_train because train_test_split will indices in X not in X_train
    # Split data set, CV (k-fold) on the training data
    cv = KFold(len(X), n_folds=k, shuffle=True)

    # Train data
    clf = tree.DecisionTreeClassifier(min_samples_split=min_split, random_state=99)
        # min_samples_split=20 requires 20 samples in a node for it to be split
        # random_state=99 to seed the random number generator
    kavg_acc = float(0) # accuracy
    kavg_mce = float(0) # mis-classification error averaged over k folds
    for train, vali in cv:
    # Get the dataset; this is the way to access values in a pandas DataFrame
        X_train = map_X.ix[train, :]; y_train = y[train]
        X_vali = map_X.ix[vali, :]; y_vali = y[vali]
        #print(X_train)
    # Train and evaluate model
        clf = clf.fit(X_train, y_train)
        predict_class = clf.predict(X_vali) # predict class or regression value for X
        #print(predict_class)
        predict_class_proba = clf.predict_proba(X_vali) # predicts class probabities of X
        #print(predict_class_proba)
        train_score = clf.score(X_train, y_train)
        vali_score = clf.score(X_vali, y_vali) # mean accuracy on the given test data and labels
        #print("train score:", train_score)
        #print("vali score:", vali_score)
        cm = confusion_matrix(y_vali, predict_class)
        #print(misclassification_error(cm))
        [acc, mce] = misclassification_error(cm)
        kavg_mce = kavg_mce + mce
        kavg_acc = kavg_acc + acc

    return (kavg_acc/k, kavg_mce/k)

#################################################################################################################################################################################
#def train_model(df): # can add other inputs (e.g. target column, split fraction, min_samples_split, etc)

df = read_data('dataset_contraceptives')

k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
min_split = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]

nCV = 3

# Look at Error vs. k folds
avg_mce = [None]*len(k_range)
avg_acc = [None]*len(k_range)
for i in range(0, len(k_range)):
    c = 1
    sum_mce = float(0)
    sum_acc = float(0)
    k = k_range[i]
    while c <= nCV:
        [acc, mce] = train_model(df, k, 200)
        #print(mce)
        sum_mce = sum_mce + mce
        sum_acc = sum_acc + acc
        c = c + 1
    avg_mce[i] = sum_mce/nCV
    avg_acc[i] = sum_acc/nCV
    #print(mce[i])

print([k_range, avg_mce, avg_acc])

# Look at Error vs. min_split_size
avg_mce = [None]*len(min_split)
avg_acc = [None]*len(min_split)
for i in range(0, len(min_split)):
    c = 1
    sum_mce = float(0)
    sum_acc = float(0)
    m = min_split[i]
    while c <= nCV:
        [acc, mce] = train_model(df, 10, m)
        #print(mce)
        sum_mce = sum_mce + mce
        sum_acc = sum_acc + acc
        c = c + 1
    avg_mce[i] = sum_mce/nCV
    avg_acc[i] = sum_acc/nCV

print([min_split, avg_mce, avg_acc])
