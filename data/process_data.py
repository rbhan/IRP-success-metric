#Processes the IRP data

import pandas as pd
import numpy as np

#Read a csv file 

def read_data(filename):
	df = pd.read_csv(filename,header=0)
	return df

def course_grades(course_code,df_crs):
	i=0
	id=[]
	grd=[]
	while i < len(df_crs):
		if df_crs['Course Code & Number'][i] == course_code:
			id.append(df_crs['Anonymized ID'][i])
			grd.append(df_crs['Sah Final Grade Code'][i])
			
		i=i+1
	id_arr=np.array(id)
	grd_arr=np.array(grd)
	grades=(id_arr,grd_arr)
	return grades
	
#Insert course grades into the data frame
def addgrades(df,course,df_crs):
	df_mod = df.copy()
	g = np.chararray(len(df_mod),1)
	g[:] = 'NaN'
	i = 0
	grades=course_grades(course,df_crs)
	while i < len(grades[0]):
		j = 0
		while j < len(df):
			if grades[0][i] == df['Computed'][j]:
				g[j] = grades[1][i]
			j=j+1
		i=i+1
	df_mod[course] = g
	print('grades done')
	return df_mod
	
def grades(df,course,df_crs):
	df_mod=df.copy()
	df_mod[course]='NaN'
	i=0
	for id in df_mod['Computed']:
		crs=df_crs.loc[df_crs['Anonymized ID'] == id]
		crs=crs.loc[crs['Course Code & Number'] == course]
		l=len(crs)
		
		if l == 0:
			df_mod[course][i] == 'NaN'
		else:
			crs=crs['Sah Final Grade Code'].index.values[0]
			df_mod[course][i] = df_crs['Sah Final Grade Code'][crs]
			
		i=i+1
		
	print('grades done')
	
	return df_mod

def addsubject(df,subject):
	df_mod=df.copy()
	df_crs=read_data('Courses.csv')
	crs=df_crs.loc[df_crs['Sah Subject Code'] == subject]
	print(crs)
	crs = crs['Course Code & Number'].unique()
	print(crs)
	for line in crs:
		df_mod = addgrades(df,line)
	return df_mod
	
#calculate time time to degree

def time_to_degree(df,years):
	ttd = np.zeros((len(df),1))
	current_yr = -201000
	z = 0
	i = 0
	while i < len(df):
		time = float((df['Sdi Term Code Graduated'][i]-df['Term Code Matric'][i])/100)
		if time < current_yr:
			#In school
			ttd[i] = 0
		#elif time < z:
		#	ttd[i] = 3
			#Didn't graduate
		elif time <= years:
			ttd[i] = 1
			#graduated on time
		elif time > years:
			ttd[i]=2
			#graduated not on time
		i=i+1
	
	return ttd

#Add a counter for ap courses

def ap_subj(df,df_crs):
	df_mod=df.copy()
	vector = np.zeros((len(df),1))
	i=0
	for line in df['Computed']:
		d_student = df_crs.loc[df_crs['Anonymized ID'] == line]
		d_ap = d_student.loc[d_student['Sah Transfer Inst Name'].isin(['Advanced Placement','*Advanced Placement, Col Board'])]
		vector[i] = len(d_ap)
		i=i+1
	df_mod['AP_Courses']=vector
	
	return df_mod
	
def RIP(df):
	vector = np.zeros((len(df),1))
	i=0
	for line in df['Sdi Degree']:
		#(line)
		if line == 'BSCHBE':
			vector[i]=1
		i=i+1
		
	return vector

#Select the variables that you want to use for fit - Sex, Race, Transfer, Major, Course grades etc

def choose_vars(df,heads):
	df_vars=df.loc[:,heads]
	return df_vars

def remove_children(df):
	i = 0
	inde = []
	while i < len(df):
		if float(df['Term Code Matric'][i]) > 201000:
			inde.append(i)
		i = i+1
	df_mod=df.drop(df.index[inde])
	return df_mod

#Clean data
def encode_target(df, target_column,df_crs):
	df_mod = df.copy()
	targets = df_crs['Sah Final Grade Code'].unique()
	targets = sorted(targets)
	#print(targets)
	map_to_int = {name: n for n, name in enumerate(targets)}
	print(map_to_int)
	df_mod[target_column] = df_mod[target_column].replace(map_to_int)
	
	return df_mod
	
def Clean_up_grades(df,course):
	df_mod=df.copy()
	i=0
	vector = np.zeros((len(df),1))
	for score in df[course]:
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
	
	df_mod[course]=vector
	return df_mod
	
def Gpa_categorical(df):
	df_mod=df.copy()
	i=0
	gv=np.zeros((len(df),1))
	for g in df['Gpa Gpa']:
		if g <=2:
			gv[i]=0
		elif g<=3:
			gv[i]=1
		elif g>3:
			gv[i]=2
		i=i+1
	return gv

#Define a GPA over a subset of courses
def GPA_current(headers):
	letters='cleaned_data_HISTORIC_letter_grades.csv'
	df=read_data(letters)
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

print(GPA_current(df_a,courses))


#Data Cleaning function
def clean_data(demo,crs):

#Read csv files into dataframes
	df_crs=read_data(crs)
	df_demo=read_data(demo)
	
	df_mod=df_demo.copy()

	#Add a time to degree column - 4 and 6 years
	ttd_4 = time_to_degree(df_mod,4)
	
	ttd_6 = time_to_degree(df_mod,6)
	
	df_mod['ttd_4'] = ttd_4
	df_mod['ttd_6'] = ttd_6
		
	#Retention Rate
	vec=RIP(df_mod)
	df_mod['RIP'] = vec
	
	#Count ap courses
	df_mod = ap_subj(df_mod,df_crs)
	print('ap done')
	#Add a list of courses and append grades
	crs_list = ['CS 1371', 'BIOL 1510', 'CHEM 1211K', 'CHEM 1212K', 'CHEM 2311', 'CHEM 2312', 'PHYS 2211', 'PHYS 2212', 'MATH 1551', 'MATH 1552', 'MATH 1553', 'MATH 1554', 'MATH 2551', 'MATH 2552', 'CHBE 2100', 'CHBE 2120', 'CHBE 2130', 'CHBE 3130', 'CHBE 3200', 'CHBE 3210', 'CHBE 3225', 'CHBE 4300']

		
	for item in crs_list:
		df_mod = grades(df_mod,item,df_crs)
	
	print(df_mod)
	#Choose relevant variables
	vars2=['AP_Courses', 'Gpa Gpa', 'ttd_4','ttd_6', 'RIP', 'Term Code Matric']
	rel_vars=crs_list
	for item in vars2:
		rel_vars.append(item)
	
	df_mod = choose_vars(df_mod,rel_vars)
	print(df_mod)
	#Encode target variables
	df_mod.to_csv('data_almost.csv')
	categorical = ['CS 1371', 'BIOL 1510', 'CHEM 1211K', 'CHEM 1212K', 'CHEM 2311', 'CHEM 2312', 'PHYS 2211', 'PHYS 2212', 'MATH 1551', 'MATH 1552', 'MATH 1553', 'MATH 1554', 'MATH 2551', 'MATH 2552', 'CHBE 2100', 'CHBE 2120', 'CHBE 2130', 'CHBE 3130', 'CHBE 3200', 'CHBE 3210', 'CHBE 3225', 'CHBE 4300']
	
	
	for cat in categorical:
		df_mod=encode_target(df_mod,cat,df_crs)
	
	
	df_mod.to_csv('cleaned_data_all_tcm.csv')
	#Remove students who are too young - less than 6 years but not graduated
	df_mod = remove_children(df_mod)
	
	df_mod.to_csv('cleaned_data_historical.csv')
	
	return df_mod
	
#print(clean_data(d,c))



df_demo=read_data(d)

df_modd=df_a.copy()

ttd_4 = time_to_degree(df_demo,4)
	
ttd_6 = time_to_degree(df_demo,6)
	
df_modd['ttd_4'] = ttd_4
df_modd['ttd_6'] = ttd_6




crs_list = ['CS 1371', 'BIOL 1510', 'CHEM 1211K', 'CHEM 1212K', 'CHEM 2311', 'CHEM 2312', 'PHYS 2211', 'PHYS 2212', 'MATH 1551', 'MATH 1552', 'MATH 1553', 'MATH 1554', 'MATH 2551', 'MATH 2552', 'CHBE 2100', 'CHBE 2120', 'CHBE 2130', 'CHBE 3130', 'CHBE 3200', 'CHBE 3210', 'CHBE 3225', 'CHBE 4300']

for item in crs_list:
	df_modd=Clean_up_grades(df_modd,item)
	
gpacat=Gpa_categorical(df_modd)
df_modd['Gpa Gpa']=gpacat
	
df_modd.to_csv('cleaned_data_ALL.csv')

df_modd=remove_children(df_modd)
df_modd.to_csv('cleaned_data_HISTORIC.csv')	

print(df_modd)




#################################################################################################################################################	
