#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:00:59 2019

@author: raphaeluzan
"""

# for some basic operations
import numpy as np
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# for providing path
import os
import matplotlib.pyplot as plt



#### Decouvrir les donnees

# reading the data
data = pd.read_csv('/Users/raphaeluzan/Downloads/StudentsPerformance.csv')
data.columns = [c.replace(' ', '_') for c in data.columns]

# getting the shape of the data
print(data.shape)
# Voir les premieres lignes
data.head()
# Description
data.describe()


## Vizu sur nos donnees
plt.rcParams['figure.figsize'] = (6, 5) 
labels = ['Femme', 'Homme']
sizes = [len(data[(data.gender == "female")]),len(data[(data.gender == "male")])]
colors = ['lightcoral', 'lightskyblue']
plt.pie(sizes ,labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()

seuil = 50
plt.rcParams['figure.figsize'] = (6, 5) 
labels = ['Echec', 'Reussite']
sizes = [len(data[(data.math_score < seuil )]),len(data[(data.math_score >= seuil )])]
colors = ['gold', 'yellowgreen']
explode = (0.3, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()

##Total score
import warnings
warnings.filterwarnings('ignore')

data['total_score'] = data['math_score'] + data['reading_score'] + data['writing_score']
sns.distplot(data['total_score'], color = 'magenta')

plt.title('comparison of total score of all the students', fontweight = 30, fontsize = 20)
plt.xlabel('total score scored by the students')
plt.ylabel('count')
plt.show()


from math import *
data['percentage'] = data['total_score']/3

for i in range(0, 1000):
    data['percentage'][i] = ceil(data['percentage'][i])

#plt.rcParams['figure.figsize'] = (15, 9)
sns.distplot(data['percentage'], color = 'orange')

plt.title('Comparaison du pourcentage noté par tous les étudiants', fontweight = 30, fontsize = 20)
plt.xlabel('Percentage scored')
plt.ylabel('Count')
plt.show()







## Label encoding


from sklearn.preprocessing import LabelEncoder

# creating an encoder
le = LabelEncoder()

# label encoding for test preparation course
data['test_preparation_course'] = le.fit_transform(data['test_preparation_course'])

# label encoding for lunch
data['lunch'] = le.fit_transform(data['lunch'])

# label encoding for race/ethnicity
# we have to map values to each of the categories
data['race/ethnicity'] = data['race/ethnicity'].replace('group A', 1)
data['race/ethnicity'] = data['race/ethnicity'].replace('group B', 2)
data['race/ethnicity'] = data['race/ethnicity'].replace('group C', 3)
data['race/ethnicity'] = data['race/ethnicity'].replace('group D', 4)
data['race/ethnicity'] = data['race/ethnicity'].replace('group E', 5)

# label encoding for parental level of education
data['parental_level_of_education'] = le.fit_transform(data['parental_level_of_education'])

#label encoding for gender
data['gender'] = le.fit_transform(data['gender'])

# label encoding for pass_math
data['pass_math'] = np.where(data['math_score']< seuil, 'Fail', 'Pass')
data['pass_math'] = le.fit_transform(data['pass_math'])

# label encoding for pass_reading
data['pass_reading'] = np.where(data['reading_score']< seuil, 'Fail', 'Pass')
data['pass_reading'] = le.fit_transform(data['pass_reading'])

# label encoding for pass_writing
data['pass_writing'] = np.where(data['writing_score']< seuil, 'Fail', 'Pass')
data['pass_writing'] = le.fit_transform(data['pass_writing'])








# data['status'] = data.apply(lambda x : 0 if x['pass_math'] == 0 or  x['pass_reading'] == 0 or x['pass_writing'] == 0 else 1, axis = 1)

data['status'] = data.apply(lambda x : 0 if x['total_score'] >205 else 1, axis = 1)
data['status'].value_counts(dropna = False).plot.pie(colors = ['grey', 'crimson'])
plt.title('overall results', fontweight = 30, fontsize = 20)
plt.xlabel('status')
plt.ylabel('count')
plt.show()



data['status'] = data.apply(lambda x : 0 if x['total_score'] >150 else 1, axis = 1)
data['status'].value_counts(dropna = False).plot.pie(colors = ['grey', 'crimson'])
plt.title('overall results', fontweight = 30, fontsize = 20)
plt.xlabel('status')
plt.ylabel('count')
plt.show()



#print(data.drop(["total_score","percentage","pass_math","pass_reading","pass_writing","status","label"], axis=1).corr())

#import seaborn as sns
#sns.set()
#sns.pairplot(data.drop(["total_score","percentage","pass_math","pass_reading","pass_writing","status","label"], axis=1));
### -----------------------------------------------


## PCA
data.head()

print(data.shape)
data['math_scoree']=data['math_score']


data_go = data.drop(["math_score","reading_score","writing_score","total_score","percentage","status","pass_math"], axis=1)

data_go.to_csv('/Users/raphaeluzan/Downloads/FormatStudentPerformanceRegression.csv',header=False)




# label encoding for pass_writing
#data['label'] = data['math_score']
#data['label'] = le.fit_transform(data['label'])


data_go = data.drop(["pass_reading","pass_writing","reading_score","writing_score","total_score","percentage","pass_math","math_score","math_scoree"], axis=1)

data_go.to_csv('/Users/raphaeluzan/Downloads/FormatStudentPerformanceClassification.csv',header=False)

