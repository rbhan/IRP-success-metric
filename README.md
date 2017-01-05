# IRP success metric

## Goals
Train models on historic student data to predict terminal GPA (gpa), retention in program (ret), time to degree (ttd), for current ChBE undergraduates (esp. freshmen/sophomores) given their coursework and grades.

## Raw data (confidential, access restricted to Georgia Tech ChBE undergraduate advisers and members of the project)
* https://github.gatech.edu/rbhan/IRP_success_metric/blob/master/data/Student_Courses.csv
* https://github.gatech.edu/rbhan/IRP_success_metric/blob/master/data/Student_Demographics.csv

### Sources
Historic student data from all undergraduates who entered Georgia Tech's ChBE program as freshmen or transfer in the past 10 years, obtained from GaTech Banner database

### Number of Students
cleaned_data_ALL: 3374 (all)

cleaned_data_HISTORIC: 1474 (subset of ALL that matriculated before 2010)

### Number of Attributes
Number of AP_courses taken

GPA from the following courses:
* 'CS 1371'
* 'BIOL 1510'
* 'CHEM 1211K', 'CHEM 1212K', 'CHEM 2311', 'CHEM 2312'
* 'PHYS 2211', 'PHYS 2212'
* 'MATH 1551', 'MATH 1552', 'MATH 1553', 'MATH 1554', 'MATH 2551', 'MATH 2552'
* 'CHBE 2100', 'CHBE 2120', 'CHBE 2130', 'CHBE 3130', 'CHBE 3200', 'CHBE 3210', 'CHBE 3225', 'CHBE 4300'

### Attribute Information
* Course grades (categorical): 'A'=4, 'B'=3, 'C'=2, 'D'=1, 'T' =-1. anything else = 0	
* Time to degree (categorical): (4/6 yr) 0 = didn't graduate/still in school, 1 = graduated on time, 2 = graduated late
* AP_courses (integer): # ap courses >= 0
* Terminal GPA (categorical): 0 = (GPA <=2), 1 = (2 < GPA <= 3), 2 = (3 < GPA <= 4)
* Retention in program (binary): 0 = Not retained, 1 = ChBE degree

## Cleaning instructions
1. Input demographics file and courses file in .csv format
2. run python function process_data.py(demographics.csv, courses.csv)
