print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import binarize
import os
import sys



color = sns.color_palette()

Number_Monolayers = 773

path = './LASSO_BR2_9/'

n_alpha = 2


# Create directory

 
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




pd.options.mode.chained_assignment = None 
pd.set_option('display.max_columns', 3000)

monolayer_descriptors = pd.read_csv("1l_atomicPLMF_"+str(Number_Monolayers)+"structures.csv",header=0) # read file with monolayers names and descriptors 
titles = pd.read_csv("1l_atomicPLMF_"+str(Number_Monolayers)+"structures.csv",header=None)
numMonolayerColumns = monolayer_descriptors.shape[1] 
numMonolayerRecords = monolayer_descriptors.shape[0] 



BilayerProperty = pd.read_csv('C33_DFT.csv',header=0) # read file with bilayers names and target values


numBilayerRecords = BilayerProperty.shape[0]

print('numBilayerRecords',numBilayerRecords)
bilayers = BilayerProperty.iloc[:,0]
monolayers = monolayer_descriptors.iloc[:,0]


df_dataset=pd.read_csv("PLMF.csv",header=0)



x = df_dataset.iloc[:,1:-1]
y = df_dataset.iloc[:,-1]





X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=None)




alphas = [        
    1e-5,
    2e-5,
    3e-5,
    4e-5,
    5e-5,
    6e-5,
    7e-5,
    8e-5,
    9e-5,
    1e-4,
    2e-4,
    3e-4

]





#finding best alpha
thefilecoeff = open(str(path)+'lasso_coefficients.csv', 'w')
print ('stop0')
for i in range(len(alphas)):
    lassoreg = Lasso(alpha=alphas[i],normalize=True, max_iter=1e9)
    lassoreg.fit(x,y)
    y_pred = lassoreg.predict(X_test)
    print(alphas[i],mean_squared_error(y_test, y_pred),r2_score(y_test, y_pred))
    for item in lassoreg.coef_:
        thefilecoeff.write("%s," % item)
    thefilecoeff.write("\n")
	
thefilecoeff.close()




#creating the new_training_set.csv
coeff = pd.read_csv(str(path)+"lasso_coefficients.csv",header=None)
sub=coeff.iloc[0:12,0:numMonolayerColumns-1]
thefile = open(str(path)+'lasso_fields.csv', 'w')
new_training_set = open(str(path)+'BR2_training-test_set.csv', 'w')




lasso_fields=np.array([])
for i in range(0,numMonolayerColumns-2):
    counter=0
    for j in range(0,12):
        if sub[i][j]!=0:
            counter=counter+1;
    if counter>=n_alpha:                          
        print("Found one at ",i)
        lasso_fields=np.append(lasso_fields,i)
        thefile.write("%s\n" % i)
thefile.close()
numFields = lasso_fields.shape[0]


for j in range(0,numBilayerRecords):
    for i in range(0,numMonolayerColumns-1):
        if i in lasso_fields:          
            new_training_set.write("%s," % str(x.iloc[j,i]))
#    new_training_set.write("%s\n" % y[j]) 
    new_training_set.write("%s\n," % str(y.iloc[j]))     
new_training_set.close()
    


for i in range(0,titles.shape[1]-1):
        if i in lasso_fields:
            print(titles.iloc[0,i+1])
    
    
lasso_monolayer_data= open(str(path)+'lasso_monolayer_data.csv', 'w')
for j in range(0,Number_Monolayers-1):                                                
    lasso_monolayer_data.write("%s," % titles.iloc[j,0])
    for i in range(0,numMonolayerColumns-2):
        if i in lasso_fields:
            lasso_monolayer_data.write("%s," % titles.iloc[j,i+1])
    lasso_monolayer_data.write("\n")
lasso_monolayer_data.close()

