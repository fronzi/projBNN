### run this script in two times by commenting and uncommenting phase1 and phase2


import pandas as pd
import csv
import sys

filepath0='./1l_atomicPLMF_773structures.csv'

path = './LASSO_BR2_1/'

filepath1=str(path)+'/lasso_fields.csv'
filepath2= str(path)+'/Monolayers773_FeatSel.csv'


lasso = pd.read_csv(filepath1,header=None,delimiter=',')

fields = lasso.iloc[:,0]




###### select monolayers features 
##########phase 1##############################################################
#mydata=pd.read_csv(filepath0,header=None,delimiter=',')
#rows,columns=mydata.shape

#finalT=[]
##for i in fields:
#for i in range(1,rows):
##    for j in fields:
##    for i in range(i,rows):
#        row=[]
#        row.append(str(mydata.iloc[i,0]))
#        for k in fields:
#            row.append(str(round(float(mydata.loc[i,k+1]),5)))
#            print(mydata.iloc[i,0],k)
#        finalT.append(row)
#df=pd.DataFrame(finalT)
#df.to_csv(str(path)+"Monolayers773_FeatSel.csv", sep=',', index=False,header=False)
###################################################################################



###### sum monolayers features and create bilayers features###################
##########phase 2##############################################################
mydata2=pd.read_csv(filepath2,header=None,delimiter=',')
rows,NewColumns=mydata2.shape 



final=[]
for i in range(0,rows-1):
    for j in range(i,rows-1):
        row=[]
        row.append(str(mydata2.loc[i,0])+'_'+str(mydata2.loc[j,0]))  
        for k in range(1,NewColumns):
            row.append(str(round(float(mydata2.loc[i,k])+float(mydata2.loc[j,k]),5)))
            print(i,j,k)
#        print(i,j,k)
        final.append(row)
df=pd.DataFrame(final)
df.to_csv(str(path)+"/18Mset.csv", sep=',', index=False,header=False)
###################################################################################



