#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:52:33 2019

@author: fresco
"""
from matplotlib import pyplot as plt
import pandas as pd 
import seaborn as sns
import numpy as np

Pima_indians_DF = pd.read_csv("/home/fresco/Documents/python_projects/Predict_Diabetes/pima-indians-diabetes.csv")


head5 = (Pima_indians_DF.head())



'''
#Pima_indians_DF.hist()
#plt.show()
#plot the variations of each variable for Diabetes and non-Diabetes
plt.subplots(3, 3, figsize=(15, 15))

for i in range(Pima_indians_DF.shape[1]):
    
    ax = plt.subplot(3, 3,i+1)
    ax.yaxis.set_ticklabels([])
    sns.distplot(Pima_indians_DF.loc[Pima_indians_DF['9. Class variable (0 or 1)']==0][Pima_indians_DF.columns[i]],axlabel=False,hist=False,kde_kws={"color": "black", "linestyle": '-', "label": "DIABETES"})
    
    sns.distplot(Pima_indians_DF.loc[Pima_indians_DF['9. Class variable (0 or 1)']==1][Pima_indians_DF.columns[i]],axlabel=False,hist=False,kde_kws={"color": "black", "linestyle": '--', "label": "NO DIABETES"})
    ax.set_xlabel(Pima_indians_DF.columns[i])
    
ax.set_visible(False)'''


#Data preprocessing 
#Null values detection 
#substitution with mean value
Pima_indians_DF.isnull().any()
for col in  Pima_indians_DF.columns.values:
    mis_val=Pima_indians_DF.loc[Pima_indians_DF[col]==0].shape[0]
    print("#of missing values in " + col + ": " +str(mis_val))
          
          
Pima_indians_DF[Pima_indians_DF.columns[1:8]]=Pima_indians_DF[Pima_indians_DF.columns[1:8]].where(cond=Pima_indians_DF[Pima_indians_DF.columns[1:8]]!=0,other=None,axis=1)
others = Pima_indians_DF[Pima_indians_DF.columns[1:8]].mean().copy()
Pima_indians_DF = Pima_indians_DF.where(cond=Pima_indians_DF.notnull(),other=others,axis = 1)

#data standardization
#mean value = 0, var = 1
table = Pima_indians_DF.astype('float').describe().loc[['mean','std','max']]

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

Pima_indians_DF_scaled = preprocessing.scale(Pima_indians_DF)
Pima_indians_DF_scaled= pd.DataFrame(Pima_indians_DF_scaled,columns = Pima_indians_DF.columns)
Pima_indians_DF_scaled[Pima_indians_DF_scaled.columns[-1]] = Pima_indians_DF[Pima_indians_DF.columns[-1]]

std_table = Pima_indians_DF_scaled[Pima_indians_DF.columns[0:8]].describe().loc[['mean','std','max']]

X = Pima_indians_DF_scaled.loc[:][Pima_indians_DF.columns[0:8]]
Y = Pima_indians_DF_scaled.loc[:][Pima_indians_DF.columns[-1]]


X_train , X_test,Y_train, Y_test =  train_test_split(X,Y,test_size = 0.2 )

X_train , X_val,Y_train, Y_val =  train_test_split(X_train,Y_train,test_size = 0.2 )

#model building opt = ADAM , act_f = relu , 2 hidden layers 32 - 16 

from keras.models import Sequential 
from keras.layers import Dense

from keras import optimizers

model = Sequential()
model.add(Dense(units = 32,input_dim = 8,activation = 'relu'))
model.add(Dense(units = 16,activation = 'relu'))
model.add(Dense(units = 1,activation = 'sigmoid'))

print(model.summary())
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])


model.fit( X_train , Y_train, epochs = 200 )

scores = model.evaluate(X_test,Y_test)

print("accuracy (%%): %.2f%%" % (scores[1]*100))

#find confusion_matrix
from sklearn.metrics import confusion_matrix

y_test_pred = model.predict_classes(X_test)
c_matrix = confusion_matrix(Y_test, y_test_pred)
#Visualize confusion_matrix

f,ax = plt.subplots(figsize=(4,4))
ax = sns.heatmap(c_matrix, annot=True,annot_kws={"ha":'center',"va":'top'} ,xticklabels=['No Diabetes','Diabetes'],
yticklabels=['No Diabetes','Diabetes'],cbar=False, cmap='Blues')
ax.set_ylim(bottom = 0.0,top = c_matrix.shape[0],emit=True,auto = True)
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()
plt.clf()

#find the Roc-Curve
from sklearn.metrics import roc_curve 

y_test_pred_probs = model.predict(X_test)
fpr,tpr,_ = roc_curve(Y_test,y_test_pred_probs)

#Visualize Roc-Curve
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],linestyle='dashed')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE")









    



    
    
