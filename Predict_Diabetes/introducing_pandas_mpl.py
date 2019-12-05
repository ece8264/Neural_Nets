#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:03:54 2019

@author: fresco
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(URL,names = ['sepal_length', 'sepal_width','petal_length', 'petal_wid+th', 'class'])

#print(df)

#df.info(verbose = False)
#print(df.describe())
#print(df.head(10))
'''
marks=['*','^','.']
species = df['class'].unique()
c=['#f19340','#cf6898','#0aaaff']


ax = plt.axes()
for i in range(3):
    
    df2=df[df['class']==species[i]]
    df2.plot.scatter(x='sepal_length',y='sepal_width',c=c[i],marker=marks[i],s=50,ax=ax,figsize=(8,5))

#df['petal_length'].plot.hist(title='histogram of petal_length')
#df.plot.box(title='boxplot of sepal_width-length and petal_width-length')

df2=pd.DataFrame({'Day': ['Monday','Tuesday','Wednesday',
'Thursday','Friday','Saturday',
'Sunday']})
    
print(pd.get_dummies(df2))

df.sepal_length[np.random.choice(len(df.index),size=10,replace=False)] = None 
df2 = df.dropna()
df['sepal_length']=df.sepal_length.fillna(df.sepal_length.mean())'''
random_indexes = np.random.choice(len(df.index),size=10,replace=False)
df.loc[random_indexes,'sepal_length'] = None
df.isnull().any()
print("# of cols before deleting: %d" %(df.shape[0]))
df2 = df.dropna()

print("# of cols after deleting: %d" %(df2.shape[0]))
df.sepal_length = df.sepal_length.fillna(df2.sepal_length.mean())
df3 = df.sepal_length.fillna(df.sepal_length.mean())

print(df.isnull().any())



























