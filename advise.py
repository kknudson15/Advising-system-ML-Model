'''
Model that takes in data and predicts grade for a particular class based off the data that it is trained upon.
'''
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sys

math_courses = ['MATH070', 'MATH072', 'MATH112', 'MATH113', 'MATH115', 'MATH211', 'MATH221', 'MATH222', 'MATH271', 'MATH304', 'MATH312', 'MATH320']
se_courses = ['SE  101', 'SE  210', 'SE  211', 'SE  221', 'SE  231', 'SE  240', 'SE  245', 'SE  276', 'SE  277', 'SE  340', 'SE  341', 'SE  345', 
'SE  444', 'SE  460', 'SE  465', 'SE  466', 'SE  470', 'SE  475', 'SE  476', 'SE  477', 'SE  480', 'SE  490', 'SE  491']

csci_courses = ['CSCI200', 'CSCI201', 'CSCI220', 'CSCI262', 'CSCI301', 'CSCI310', 'CSCI411', 'CSCI430', 'CSCI450']

other_courses = ['AHS 106', 'ANTH101', 'ASTR107', 'ART 130', 'BIOL101', 'BIOL103', 'BIOL104', 'BIOL151', 'BIOL152', 'BIOL262', 
'BIOL339', 'BIOLT001', 'BIOLT004', 'BIOLT009', 'BIOLT010', 'CHEM105', 'CHEM210', 'CMST192', 'CMST220', 'CMST341', 'CMSTT001', 
'CNA 201', 'CNA 267', 'COLL120', 'COMMT002', 'CSCIT001', 'CSCIT09', 'CSD 130', 'ECON206', 'ELECT001', 'ELECT004', 'ELECT006', 
'ELECT008', 'ENGL100', 'ENGL184', 'ENGL190', 'ENGL191', 'ENGL216', 'ENGLT001', 'ENGLT002', 'ETHS111', 'ETHS210', 'ETS 199', 
'FREN201', 'GENG101', 'GER 201', 'GER 202', 'HIST140', 'HISTT002', 'HLTH210', 'HONS100', 'HONS106', 'HONS160', 'HONS170', 
'HONS250', 'HONS260', 'HURL102', 'MME 101', 'MUSM125', 'MUSM126', 'MUSP162', 'MUSP358', 'MUSP362', 'PESST003', 'PHIL194', 
'PHIL212', 'PHYS199', 'PHYS231', 'PHYS232', 'PHYS234', 'PHYS235', 'PHYST002', 'POL 111', 'PSY 115', 'REL 100', 'RESP100', 
'RUSS110','SOC 160', 'SOC 200', 'SOC 268', 'SOC T003', 'STAT239', 'STAT321', 'STAT353', 'TECHT001', 'TECHT005', 'TECHT011', 'TECHT012']

'''
Functions
'''
def read_data():
    '''
    Reads the data from csv file and generates a dataframe
    Output: returns a dataframe generated from the file
    '''
    student_data = pd.read_csv('DARsData.csv')
    columns = student_data.columns.tolist()
    #print(columns)
    return student_data

def add_columns(student_data):
    '''
    Adds columns to the data set
    input: takes in a dataframe
    Output: returns the modified dataframe
    '''
    student_data['MathGPA'] = 0.0
    student_data['MathTaken'] = 0
    student_data['SEGPA'] = 0.0 
    student_data['SETaken'] = 0
    student_data['CSCITaken'] = 0
    student_data['CSCIGPA'] = 0.0
    student_data['OtherTaken'] = 0
    student_data['OtherGPA'] = 0.0

    return student_data

def replace_grade(grade):
    '''
    lambda function that replaces categorical values with discrete ones
    input: the column value
    ouputL returns the discrete value to replace the input with
    '''
    if grade == 'A':
        score = 4.0
    elif grade == 'A-':
        score = 3.7
    elif grade == 'B':
        score = 3.0
    elif grade == 'B-':
        score = 2.7
    elif grade == 'C':
        score = 2.0
    elif grade == 'C-':
        score = 1.7
    elif grade == 'D':
        score = 1.0
    elif grade == 'D-':
        score = 0.7
    elif grade == 'F':
        score = 0.0
    else: 
        score = -1.0
    return score


def count_math_classes(student_data):
    '''
    returns a count of the numbers 
    input: dataframe of values
    output: returns the count of math classes
    '''
    math_count = 0
    for course in math_courses:
        if student_data[course] != -1.0:
            math_count += 1
    return math_count


def calc_math_gpa(student_data):
    '''
    calculates the gpa based off the number of classes taken 
    input: dataframe of values
    output: returns gpa rounded to 2 digits
    '''
    math_score = 0
    for course in math_courses:
        if student_data[course] != -1.0:
            math_score += student_data[course]
    if student_data['MathTaken'] != 0:
        return round(math_score / student_data['MathTaken'], 2)
    else:
        return 0

def count_se_classes(student_data):
    '''
    counts the number of software engineering classes
    input: dataframe of values
    output: returns the number of classes 
    '''
    se_count = 0
    for course in se_courses:
        if student_data[course] != -1.0:
            se_count += 1
    return se_count


def calc_se_gpa(student_data):
    '''
    calculates the gpa in software engineering classes
    input: data frame of valyes
    output: returns the gpa rounded to two digits
    '''
    se_score = 0.0
    for course in se_courses:
        if student_data[course] != -1.0:
            se_score += student_data[course]
    if student_data['SETaken'] != 0:
        return round(se_score / student_data['SETaken'], 2)
    else:
        return 0

def count_csci_classes(student_data):
    '''
    counts the number computer science courses taken by the student
    input: data frame of values 
    output: returns the count of csci courses
    '''
    csci_count = 0
    for course in csci_courses:
        if student_data[course] != -1.0:
            csci_count += 1
    return csci_count

def calc_csci_gpa(student_data):
    '''
    calculates the computer science gpa 
    input: dataframe of values 
    output: returns the gpa rounded to two digits 
    '''
    csci_score = 0.0
    for course in csci_courses:
        if student_data[course] != -1.0:
            csci_score += student_data[course]
    if student_data['CSCITaken'] != 0:
        return round(csci_score / student_data['CSCITaken'], 2)
    else:
        return 0

def count_other_classes(student_data):
    '''
    counts the number of other classes taken by the student
    input: dataframe of values
    outputL returns the count of the other classes taken
    '''
    other_count = 0
    for course in other_courses:
        if student_data[course] != -1.0:
            other_count += 1
    return other_count

def calc_other_gpa(student_data):
    '''
    calculates the gpa in other classes
    input: dataframe of values
    output: returns the gpa rounded to two digits
    '''
    other_score = 0.0
    for course in other_courses:
        if student_data[course] != -1.0:
            other_score += student_data[course]
    if student_data['OtherTaken'] != 0:
        return round(other_score / student_data['OtherTaken'], 2)
    else:
        return 0

def fill_null_math(student_data):
    '''
    fills in null values 
    input: dataframe of values
    return: the modified dataframe
    '''
    for course in math_courses:
        student_data[course].replace(-1.0,round(student_data['MathGPA'].mean(),2),inplace=True)
        #print(student_data[course])
    return student_data
            
def fill_null_se(student_data):
    '''
    fills in null values 
    input: dataframe of values
    return: the modified dataframe
    '''
    for course in se_courses:
        student_data[course].replace(-1.0,round(student_data['SEGPA'].mean(),2),inplace=True)
        #print(student_data[course])
    return student_data
def fill_null_csci(student_data):
    '''
    fills in null values 
    input: dataframe of values
    return: the modified dataframe
    '''
    for course in csci_courses:
        student_data[course].replace(-1.0,round(student_data['CSCIGPA'].mean(),2),inplace=True)
        #print(student_data[course])
    return student_data
def fill_null_other(student_data):
    '''
    fills in null values 
    input: dataframe of values
    return: the modified dataframe
    '''
    for course in other_courses:
        student_data[course].replace(-1.0,round(student_data['OtherGPA'].mean(),2),inplace=True)
        #print(student_data[course])
    return student_data


def process_math (student_data):
    '''
    Creates a column in the dataframe for Math GPA
    input: dataframe of values
    output: returns the modified dataframe
    '''
    student_data['MathTaken'] = student_data.apply(count_math_classes, axis=1)
    student_data['MathGPA'] = student_data.apply(calc_math_gpa, axis=1)
    #print(student_data['MathGPA'])
    return student_data

def process_se(student_data):
    '''
    Creates a column in the dataframe for SE GPA
    input: dataframe of values
    output: returns the modified dataframe
    '''
    student_data['SETaken'] = student_data.apply(count_se_classes, axis=1)
    student_data['SEGPA'] = student_data.apply(calc_se_gpa, axis=1)
    #print(student_data['SEGPA'])
    return student_data

def process_csci(student_data):
    '''
    Creates a column in the dataframe for CSCI GPA
    input: dataframe of values
    output: returns the modified dataframe
    '''
    student_data['CSCITaken'] = student_data.apply(count_csci_classes, axis=1)
    student_data['CSCIGPA'] = student_data.apply(calc_csci_gpa, axis=1)
    #print(student_data['CSCIGPA'])
    return student_data

def process_other(student_data):
    '''
    Creates a column in the dataframe for Other GPA
    input: dataframe of values
    output: returns the modified dataframe
    '''
    student_data['OtherTaken'] = student_data.apply(count_other_classes, axis=1)
    student_data['OtherGPA'] = student_data.apply(calc_other_gpa, axis=1)
    #print(student_data['OtherGPA'])
    return student_data


def math_prediction(lm1, X_test):
    '''
    returns the prediction list for the model and testing data 
    input: linear model object and testing list
    output: returns list of predicted values 
    '''
    return lm1.predict(X_test)

def csci_prediction(lm2, X_test):
    '''
    returns the prediction list for the model and testing data 
    input: linear model object and testing list
    output: returns list of predicted values 
    '''
    return lm2.predict(X_test)

def se_prediction(lm3, X_test):
    '''
    returns the prediction list for the model and testing data 
    input: linear model object and testing list
    output: returns list of predicted values 
    '''
    return lm3.predict(X_test)

def other_prediction(lm4, X_test):
    '''
    returns the prediction list for the model and testing data 
    input: linear model object and testing list
    output: returns list of predicted values 
    '''
    return lm4.predict(X_test)


'''
Main 
'''

#read in data and add necessary columns
student_data = read_data()
add_columns(student_data)


'''
Replace letter grades with numerical values
'''
for course in math_courses:
    student_data[course] = student_data[course].apply(replace_grade)
    #print(student_data[course])

for course in se_courses:
    student_data[course] = student_data[course].apply(replace_grade)

for course in csci_courses:
    student_data[course] = student_data[course].apply(replace_grade)

for course in other_courses:
    student_data[course] = student_data[course].apply(replace_grade)


'''
determine known GPA's and number of classes in the subject taken
'''
student_data = process_math(student_data)
student_data = process_se(student_data)
student_data = process_csci(student_data)
student_data = process_other(student_data)


'''
Replace null values with average from the category 
'''
student_data = fill_null_math(student_data)
student_data = fill_null_se(student_data)
student_data = fill_null_csci(student_data)
student_data = fill_null_other(student_data)



'''
Math Model
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
X1 = student_data[math_courses]
y1 = student_data['MathGPA']


X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3)

lm1 = LinearRegression()
lm1.fit(X_train, y_train)


mpredictions = math_prediction(lm1, X_test)
mpredictions = math_prediction(lm1, X1) [int(sys.argv[1])]
print(mpredictions)

'''
from sklearn import metrics  
print('Math Metrics:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, mpredictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, mpredictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, mpredictions))) 
'''



'''
CSCI Model
'''
X2 = student_data[csci_courses]
y2 = student_data['CSCIGPA']

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3)


lm2 = LinearRegression()
lm2.fit(X_train, y_train)

cpredictions = csci_prediction(lm2, X_test)
cpredictions = csci_prediction(lm2, X2) [int(sys.argv[1])]
print(cpredictions)

'''
from sklearn import metrics  
print('CSCI Metrics:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, cpredictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, cpredictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, cpredictions))) 
'''

'''
SE Model
'''
X3 = student_data[se_courses]
y3 = student_data['SEGPA']

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.3)


lm3 = LinearRegression()
lm3.fit(X_train, y_train)

spredictions = se_prediction(lm3, X_test)
spredictions = se_prediction(lm3, X3) [int(sys.argv[1])]
print(spredictions)

'''
from sklearn import metrics  
print('SE Metrics:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, spredictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, spredictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, spredictions))) 
'''

'''
Other Model
'''
X4 = student_data[other_courses]
y4 = student_data['OtherGPA']

X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size=0.3)


lm4 = LinearRegression()
lm4.fit(X_train, y_train)



opredictions = other_prediction(lm4, X_test)
opredictions = other_prediction(lm4, X4) [int(sys.argv[1])]
print(opredictions)

'''
from sklearn import metrics  
print('Other Metrics:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, opredictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, opredictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, opredictions))) 
'''


