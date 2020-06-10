#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:58:48 2019

@author: Marco
"""

# numpy and pandas for data manipulation

from __future__ import print_function
import numpy as np
from sklearn import datasets, linear_model
import genetic_selection
import os
import sys

from genetic_selection import GeneticSelectionCV
import pandas as pd

Number_Monolayers = 773
path = './LASSO_BR2_5/'



# Create target Directory if don't exist
if not os.path.exists(path):
    os.mkdir(path)
    os.mkdir(path+'/DATA_SETS/')
    os.mkdir(path+'/Figs/')
    os.mkdir(path+'/SavedModels/')
    os.mkdir(path+'/LASSO_Converged/')
    print("Directory " , path ,  " Created ")
else:    
    print("Directory " , path ,  " already exists")




monolayer_descriptors = pd.read_csv("1l_atomicPLMF_"+str(Number_Monolayers)+"structures.csv",header=0) # read file with monolayers names and descriptors 
titles = pd.read_csv("1l_atomicPLMF_"+str(Number_Monolayers)+"structures.csv",header=None)
numMonolayerColumns = monolayer_descriptors.shape[1] 
numMonolayerRecords = monolayer_descriptors.shape[0] 

print('numMonolayerColumns',numMonolayerColumns)
print('numMonolayerRecords',numMonolayerRecords)


BilayerProperty = pd.read_csv("C33_DFT.csv",header=0) # read file with bilayers names and target values

print(BilayerProperty)

numBilayerRecords = BilayerProperty.shape[0]
print('numBilayerRecords',numBilayerRecords)
bilayers = BilayerProperty.iloc[:,0]
print('bilayers',bilayers)
monolayers = monolayer_descriptors.iloc[:,0]
print('monolayers',monolayers)


df = pd.read_csv(str(path)+'PLMF.csv')
n_col = df.shape[1]
X = df.iloc[:,1:(n_col-1)]
y = df.iloc[:,n_col-1:]

estimator = linear_model.Lasso(1e-3,normalize=True, max_iter=1e9)

selector = GeneticSelectionCV(estimator,
                              cv=5,
                              verbose=1,
                              scoring="explained_variance",
                              max_features=100,
                              n_population=30,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=20,
                              crossover_independent_proba=0.1,
                              mutation_independent_proba=0.05,
                              tournament_size=5,
                              caching=True,
                              n_jobs=-1)

selector = selector.fit(X, y)
print (selector.score(X,y))
#print(selector.support_)


selection = pd.DataFrame(X.columns , columns = ['features'])
selection['support'] = selector.support_
selection['Flag'] = [1 if x == True else 0 for x in selection['support'] ]

final_selected = list(selection['features'][selection.Flag == 1])

print ("Final Selected Feature list : " , final_selected)
GA_lasso_fields = pd.DataFrame(final_selected).T

new_training_set = open(str(path)+'BR2_GA_training-test_set.csv', 'w')

for j in range(0,numBilayerRecords):
    new_training_set.write("%s," % j)
    for i in GA_lasso_fields: 
        new_training_set.write("%s," % str(X.iloc[j,i]))
    new_training_set.write("%s\n," % str(y.iloc[j,0]))     
new_training_set.close()


for i in range(0,titles.shape[1]-1):
        if i in GA_lasso_fields:
            print(titles.iloc[0,i+1])
    
    
lasso_monolayer_data= open(str(path)+'GA_lasso_monolayer_data.csv', 'w')
for j in range(0,Number_Monolayers-1):                                                #####NUMBER OF MONOLAYERS+1 ######
    lasso_monolayer_data.write("%s," % titles.iloc[j,0])
    for i in range(0,numMonolayerColumns-2):
        if i in GA_lasso_fields:
            lasso_monolayer_data.write("%s," % titles.iloc[j,i+1])
    lasso_monolayer_data.write("\n")
lasso_monolayer_data.close()

GA_lasso_fields.T.to_csv(str(path)+"GA_lasso_fields.csv", index = None, header=None)





