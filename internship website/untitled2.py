# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:18:48 2022

@author: user
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


mobile_data=pd.read_csv("MobileTrain.csv")
mobile_data_test=pd.read_csv("MobileTest.csv")

mobile_data_new = pd.concat([mobile_data_test.assign(ind="test"), mobile_data.assign(ind="train")])
print("The shape of final dataset is :  ", mobile_data_new.shape)

Q1=np.percentile(mobile_data_new["fc"],25,interpolation="midpoint")
print("Q1)",Q1)
Q2=np.percentile(mobile_data_new["fc"],50)
print("Q2",Q2)
Q3=np.percentile(mobile_data_new["fc"],75)
print("Q3",Q3)

IQR=Q3-Q1
print("IQR",IQR)

low_lim=Q1-1.5*IQR
upp_lim=Q3+1.5*IQR

print("low_lim",low_lim)
print("upp_lim",upp_lim)

outlier=[]
for x in mobile_data_new["fc"]:
    if (x>upp_lim) or (x<low_lim):
        outlier.append(x)

print("outlier", outlier)

ind1=mobile_data_new["fc"]>upp_lim

ind_1=mobile_data_new.loc[ind1].index

#print("index of outlier", ind1)

mobile_data_new.drop(ind_1,inplace=True)



mobile_data_test, mobile_data = mobile_data_new[mobile_data_new["ind"].eq("test")], mobile_data_new[mobile_data_new["ind"].eq("train")]

print("The train set details:", mobile_data.shape)
mobile_data=mobile_data.drop(['id','ind'],axis=1)
print("\n", mobile_data.columns)

print("The test set details:", mobile_data_test.shape)
mobile_data_test=mobile_data_test.drop(['id','ind'],axis=1)
print("\n", mobile_data_test.columns)

"""**Splitting the dataset into independent and dependent variables**"""

#independent variable "x_data"
x_data=mobile_data.drop(["price_range"],axis=1)
#dependent variable "y_data"
y_data=mobile_data["price_range"]
print(x_data.shape)
print(y_data.shape)
x_data_1=x_data.copy()
x_data_1.head(2)

"""**Scaling**"""

#standard scaling
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
x_data=scalar.fit_transform(x_data)
x_data=pd.DataFrame(x_data,columns=x_data_1.columns)
x_data.head(3)

"""**MODEL BUILDING**

Train test split
"""

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,random_state=42,test_size=0.25)


print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

pip install catboost

from sklearn.metrics import make_scorer, accuracy_score
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
clf = CatBoostClassifier()




clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
y_pred
x_test.iloc[1]
y_pred=clf.predict(mobile_data_test)
y_pred
