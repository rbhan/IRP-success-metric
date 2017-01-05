# Rebeacca Han
# 6/16/16 updated
#####################################################
	# PREDICTS TTD <= 4YRS
	# uses dataframes!

import os
import subprocess
import math
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import tree, ensemble, linear_model, cross_validation, metrics
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error


# LOAD data as DataFrame ###############################################################################
def read_data(filename):
	df = pd.read_csv(filename, sep=",")
	return df

# Select variables from DataFrame ######################################################################
def choose_vars(df,heads):
	df_vars=df.loc[:,heads]
	return df_vars

# Define cumulative GPA over a subset of courses  ######################################################
def cumulative_GPA(headers):
	df=read_data('cleaned_data_HISTORIC_letter_grades.csv')
	gpa=np.zeros((len(df),1))
	n_courses=np.zeros((len(df),1))
	df_mod=choose_vars(df,headers)
	j=0
	for head in headers:
		i=0
		for line in df_mod[head]:
			grade_val=0
			if line == "A":
				grade_val=4
				n_courses[i] = n_courses[i]+1
			elif line == "B":
				grade_val=3
				n_courses[i] = n_courses[i]+1
			elif line == "C":
				grade_val=2
				n_courses[i] = n_courses[i]+1
			elif line == "D":
				grade_val=1
				n_courses[i] = n_courses[i]+1
			elif line == "F":
				grade_val=0
				n_courses[i] = n_courses[i]+1				
			gpa[i]=gpa[i]+grade_val
			i=i+1

	while j<len(gpa):
		if n_courses[j] > 0:
			gpa[j]=gpa[j]/n_courses[j]
		j=j+1
	
	return gpa

# GRAPH tree as .png ###################################################################################
def visualize_tree(tree, features):
	with open("ttd4.dot", 'w') as f:
		export_graphviz(tree, out_file=f,
						feature_names=features)
	command = ["dot", "-Tpng", "ttd4.dot", "-o", "ttd4.png"]
	try:
		subprocess.check_call(command)
	except:
		exit("Could not run dot, ie graphviz, to "
			 "produce visualization")

# Returns decision tree rules as pseudocode ############################################################
def get_rules(tree, feature_names):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node):
                if (threshold[node] != -2):
                        print "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
                        if left[node] != -1:
                                recurse (left, right, threshold, features,left[node])
                        print "} else {"
                        if right[node] != -1:
                                recurse (left, right, threshold, features,right[node])
                        print "}"
                else:
                        print "return " + str(value[node])

        recurse(left, right, threshold, features, 0)

# SCORE misclassification error ########################################################################
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
	correct = float(classified)/float(total)
	incorrect = float(misclassified)/float(total)

	return (correct, incorrect)

# BENCHMARK using mean #################################################################################
def benchmark_mean(X_train, y_train, X_test, y_test):
	y_mean = np.mean(y_train) # average GPA
	y_predict = np.full(y_test.shape, y_mean, dtype=float)
	mae = mean_absolute_error(y_test, y_predict)
	rmse = math.sqrt(mean_squared_error(y_test, y_predict))

	return(mae, rmse)

# BENCHMARK using logistic regression ####################################################################
def benchmark_logreg(X_train, y_train, X_test, y_test):
	lr = linear_model.LogisticRegression(fit_intercept=True)
	lr = lr.fit(X_train, y_train)
	y_predict = lr.predict(X_test)
	mae = mean_absolute_error(y_test, y_predict)
	rmse = math.sqrt(mean_squared_error(y_test, y_predict))

	return(mae, rmse)

# TRAIN decision tree classifier #######################################################################
def train_tree(X_train, y_train, X_test, y_test):
	dtc = tree.DecisionTreeClassifier(min_samples_split=100, random_state=42)
	dtc = dtc.fit(X_train, y_train)
	y_predict = dtc.predict(X_test)
	y_proba = dtc.predict_proba(X_test)

	# Scores
	cm = confusion_matrix(y_test, y_predict)
	acc, mcr = misclassification_error(cm)
	mae = mean_absolute_error(y_test, y_predict)
	rmse = math.sqrt(mean_squared_error(y_test, y_predict))
	f = dtc.feature_importances_

	return (f, dtc, cm, acc, mcr, mae, rmse)

# TRAIN random forest classifier #######################################################################
def train_forest(X_train, y_train, X_test, y_test):
	rfc = ensemble.RandomForestClassifier(n_estimators=10, min_samples_split=100)
	rfc = rfc.fit(X_train, y_train)
	y_predict = rfc.predict(X_test)
	y_proba = rfc.predict_proba(X_test)

	# Scores
	cm = confusion_matrix(y_test, y_predict)
	acc, mcr = misclassification_error(cm)
	mae = mean_absolute_error(y_test, y_predict)
	rmse = math.sqrt(mean_squared_error(y_test, y_predict))

	return (cm, acc, mcr, mae, rmse)

########################################################################################################
########################################################################################################

df = read_data('cleaned_data_HISTORIC.csv')
# Choose predictor variables of importance
var_list = ['CS 1371', 'BIOL 1510', 'CHEM 1211K', 'CHEM 1212K', 'CHEM 2311', 'CHEM 2312', 'PHYS 2211', 'PHYS 2212', 'MATH 1551', 'MATH 1552', 'MATH 1553', 'MATH 1554', 'MATH 2551', 'MATH 2552', 'CHBE 2100', 'CHBE 2120', 'CHBE 2130', 'CHBE 3130', 'CHBE 3200', 'CHBE 3210', 'CHBE 3225', 'CHBE 4300']
# Find the index of the last element
inx = len(var_list)-1
# Calculate cumulative GPA and add to dataframe
cum_GPA = cumulative_GPA(var_list)
df.insert(inx+1, 'cum_GPA', cum_GPA)
var_list.append('cum_GPA')
# Add TTD6 to variable list
var_list.append('ttd_4')
iny = inx + 2
# select all important variables
df = choose_vars(df,var_list)
	
# Select target column and data columns
features = list(df.columns[:inx+2]) # col 23 for w/o GPA, col 24 for including GPA
target = df.columns[iny] 
XX = df[features] # selects all rows with correct headers
yy = df[target]
map_X = pd.get_dummies(XX.ix[:, :])
X, X_test, y, y_test = train_test_split(XX, yy, test_size=0.30, random_state=42)

acc = [0., 0., 0., 0.]
mcr = [0., 0., 0., 0.]
mae = [0., 0., 0., 0.]
rmse = [0., 0., 0., 0.]

# Benchmark data
[mae[0], rmse[0]] = benchmark_mean(X, y, X_test, y_test)
[mae[1], rmse[1]]  = benchmark_logreg(X, y, X_test, y_test)

# Train tree: split data set, CV (k-fold) on the training data
folds = 5
kf = KFold(len(XX), n_folds=folds, shuffle=True)
for train, vali in kf:
	# Get the dataset; this is the way to aaccess values in a pandas DataFrame
	X_train = map_X.ix[train, :]; y_train = yy[train]
	X_vali = map_X.ix[vali, :]; y_vali = yy[vali]
	[f, dtc, dtc_cm, dtc_acc, dtc_mcr, dtc_mae, dtc_rmse] = train_tree(X_train, y_train, X_vali, y_vali)
	acc[2] = acc[2] + dtc_acc
	mcr[2] = mcr[2] + dtc_mcr
	mae[2] = mae[2] + dtc_mae
	rmse[2] = rmse[2] + dtc_rmse

# Display tree and important rules
visualize_tree(dtc, features)
get_rules(dtc, features)

# Return tree scores
acc[2] = acc[2]/folds
mcr[2] = mcr[2]/folds
mae[2] = mae[2]/folds
rmse[2] = rmse[2]/folds

# Train forest
[rfc_cm, acc[3], mcr[3], mae[3], rmse[3]] = train_forest(X, y, X_test, y_test)

methods = ["Mean", "Logit_Regression", "Decision_Tree", "Random_Forest"]
error = {'Method': methods, 'Accuracy': acc, 'Misclassification_Rate': mcr, 'MAE': mae, 'RMSE': rmse}
compare_methods = pd.DataFrame(data=error, index=None, columns=['Method', 'Accuracy', 'Misclassification_Rate', 'MAE', 'RMSE'])
print(compare_methods)
print(dtc_cm)
print(rfc_cm)
