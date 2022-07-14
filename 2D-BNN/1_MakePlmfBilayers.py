print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import datasets
from scipy import interp
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import os
import sys




#import xgboost as xgb
color = sns.color_palette()

Number_Monolayers = 6084

path = './data_proc/'




# Create directory

 
# Create target Directory if don't exist
if not os.path.exists(path):
    os.mkdir(path)
    os.mkdir(path+'/DATA_SETS/')
    os.mkdir(path+'/Figs/')
    os.mkdir(path+'/SavedModels/')
    os.mkdir(path+'/Converged/')
    print("Directory " , path ,  " Created ")
else:    
    print("Directory " , path ,  " already exists")



#%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 3000)

monolayer_descriptors = pd.read_csv("1l_atomicPLMF_"+str(Number_Monolayers)+"structures.csv",header=0) # read file with monolayers names and descriptors 
titles = pd.read_csv("1l_atomicPLMF_"+str(Number_Monolayers)+"structures.csv",header=None)
numMonolayerColumns = monolayer_descriptors.shape[1] 
numMonolayerRecords = monolayer_descriptors.shape[0] 

print('numMonolayerColumns',numMonolayerColumns)
print('numMonolayerRecords',numMonolayerRecords)



BilayerProperty = pd.read_csv("P33_tot1.csv",header=0) # read file with bilayers names and target values

#print(BilayerProperty)

numBilayerRecords = BilayerProperty.shape[0]
bilayers = BilayerProperty.iloc[:,0]
monolayers = monolayer_descriptors.iloc[:,0]



dataset = []
mislabeled =[]



for b in bilayers:
    print(b)
    bt=b.split("_")
    b_d = BilayerProperty.loc[BilayerProperty.Bilayer==b]
    bilayer_record = []
    m1 = monolayer_descriptors.loc[monolayer_descriptors.Monolayer==bt[0]]
    m2 = monolayer_descriptors.loc[monolayer_descriptors.Monolayer==bt[1]]
    i=1
    try:
        sum = m1.iloc[0,i] + m2.iloc[0,i] 
        for i in range(1,numMonolayerColumns):
            sum = m1.iloc[0,i] + m2.iloc[0,i]
            bilayer_record += [sum]
        bilayer_record += [b_d.iloc[0,1]]
        dataset += [bilayer_record]      
    except:
        try:
            m1.iloc[0,1]
            print("cannot find", bt[1], "in 1l_atomicPLMF")
            mislabeled += bt[1]
            
        except:
            print("cannot find", bt[0], "in 1l_atomicPLMF")
            mislabeled += bt[0]



df_dataset=pd.DataFrame(dataset)
df_dataset.to_csv(path+"PLMF.csv",header=True)

df_mislabeled=pd.DataFrame(mislabeled)
df_mislabeled.to_csv(path+"PLMF_mislabeled.csv",header=True)



