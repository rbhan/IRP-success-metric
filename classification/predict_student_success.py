# Rebeacca Han
# 6/16/16 updated
#####################################################
	# PREDICTS TTD <= 4YRS
	# uses dataframes!

import os
import copy
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
	
# Select Variables from DataFrame ######################################################################
def choose_vars(df,heads):
	df_vars=df.loc[:,heads]
	return df_vars

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
########################################################################################################	
#var_list = ['CS 1371', 'BIOL 1510', 'CHEM 1211K', 'CHEM 1212K', 'CHEM 2311', 'CHEM 2312', 'PHYS 2211', 'PHYS 2212', 'MATH 1551', 'MATH 1552', 'MATH 1553', 'MATH 1554', 'MATH 2551', 'MATH 2552', 'CHBE 2100', 'CHBE 2120', 'CHBE 2130', 'CHBE 3130', 'CHBE 3200', 'CHBE 3210', 'CHBE 3225', 'CHBE 4300']
#print(generate_models(var_list,'ttd_6'))

def read_in_grades(grades):
	vector = np.zeros((len(grades),1))
	#print(vector)
	i=0
	for score in grades:
		if score == "A":
			vector[i]=4
		elif score == "B":
			vector[i]=3
		elif score == "C":
			vector[i]=2
		elif score == "D":
			vector[i]=1
		elif score == "T":
			vector[i]=-1
		else:
			vector[i]=0
		i=i+1
	return vector
	
########################################################################################################
def predict_success(predictors,grades,predicted):
	df = read_data('cleaned_data_HISTORIC.csv')
	
	var_list=copy.copy(predictors)
	inx=len(predictors)-1
	var_list.append(predicted)
	iny=inx+1
	df = choose_vars(df,var_list)
		
	#Read input grades
	x_student=read_in_grades(grades)
	x_student=np.transpose(x_student)
	#print(x_student)
	
	# Select traget column and data columns
	features = list(df.columns[:inx+1]) # col 23 for w/o GPA, col 24 for including GPA
	#print(features)
	target = df.columns[iny] 
	#print(target)
	XX = df[features] # selects all rows with correct headers
	yy = df[target]
	map_X = pd.get_dummies(XX.ix[:, :])
	X, X_test, y, y_test = train_test_split(XX, yy, test_size=0.30, random_state=42)
	dtc = tree.DecisionTreeClassifier(min_samples_split=100, random_state=42)
	dtc = dtc.fit(X, y)
	y_student_proba = dtc.predict_proba(x_student)
	
	return y_student_proba


#v=['CHBE 2100', 'CHBE 2120', 'CHBE 2130','CHBE 3200']
#g=['C','B','F','A']

#print(predict_success(v,g,'ttd_6'))
#print(predict_success(v,g,'RIP'))
#print(predict_success(v,g,'ttd_4'))
