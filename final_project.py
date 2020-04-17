# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:26:37 2020

@author: danie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Reading in data
df = pd.read_csv('C:/Gdrive/UCF/STA5703 Data Mining 1/HW/Final/Data/PHY_TRAIN.csv')
df.head()
df = df.drop('exampleid', axis=1)

#Summary statistics
df.describe()
df.info()
df.isna().sum()

#seperating predictors and response
X = df.iloc[:, 1:] 
y = df.iloc[:, 0]

#Creating missing value indicator for all columns and replace missing values with column mean 
cols = list(X.columns)
X_indicated = X[cols].isnull().astype(int).add_prefix('M_') #Creating df for MVI
X_clean = pd.concat([X.fillna(X.mean()), X_indicated], axis=1) #Joining MVI df with real df and filling na values with column mean
#print (X_clean)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X_clean)

#Building logistic regression model without interaction terms
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=42)
classifier.fit(X_scaled, y)

#Building logistic regression model with at least 3 two-way interaction terms
#Building random forest model 
#Building gradient boosting model 

#Compare fitted model using c-statistics, including AUC/ROC