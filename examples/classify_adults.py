# Rebecca Han
# 6/2/16 updated
#####################################################
	# Classifies UCI Machine Learning: Adult (32561 instances, 16 attributes inc class = categ/int/real)
	# OLD CODE; doesn't use dataframes
	# play with min_samples_split for interesting results

import os
import subprocess

import numpy as np
import pydot
#import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO

# Load data
file = open('dataset_adults', 'r')
adults = file.readlines()
file.close
for row in range(0,len(adults)):
	adults[row] = str(adults[row]).split(', ')

# Converts nominal data into numeric features
def prep_data(d):
	le.fit(d)
	dnum = le.transform(d)
	return dnum

# Preprocess data into numeric features
le = preprocessing.LabelEncoder()
age = ([[int(row[0])] for row in adults])
emp = prep_data([[row[1]] for row in adults]) #
n1 = [[int(row[2])] for row in adults]
edu = prep_data([[row[3]] for row in adults]) #
n2 = [[int(row[4])] for row in adults]
mar = prep_data([[row[5]] for row in adults]) #
prof = prep_data([[row[6]] for row in adults]) #
fam = prep_data([[row[7]] for row in adults]) #
race = prep_data([[row[8]] for row in adults]) #
sex = prep_data([[row[9]] for row in adults]) #
n3 = [[int(row[10])] for row in adults]
n4 = [[int(row[11])] for row in adults] 
n5 = [[int(row[12])] for row in adults]
loc = prep_data([[row[13]] for row in adults]) #
X = np.concatenate([age, emp, n1, edu, n2, mar, prof, fam, race, sex, n3, n4, n5, loc], axis=1)

# Preprocess targets into integers
target = [[row[14]] for row in adults]
y = [None]*len(target)
case0 = target[0]
for t in range(0,len(target)):
	if (case0 == target[t]) == True:
		#print('true')
		y[t] = int(0)
	else:
		#print('false')
		y[t] = int(1)

# Training
clf = tree.DecisionTreeClassifier(min_samples_split=1000, random_state=99)
clf = clf.fit(X, y)

# Creating graphviz
with open("adults.dot", 'w') as f:
	export_graphviz(clf, out_file=f, feature_names=['age','emp','n1','edu','n2','mar','prof','fam','race','sex','n3','n4','n5','loc'])

command = ["dot", "-Tpng", "adults.dot", "-o", "adults.png"]
try:
	subprocess.check_call(command)
except:
	exit("Could not run dot, ie graphviz, to "
             "produce visualization")
