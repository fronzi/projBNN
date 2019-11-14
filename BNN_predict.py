# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.distributions import Bernoulli
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.patches as mpatches
import sys
from sklearn.model_selection import StratifiedKFold




count=2
path = './LASSO_BR2_1'

test_train_df = pd.read_csv(str(path)+"LARGE_DL_SET.csv",header=None)



numColumns = test_train_df.shape[1]-1


X_training_test = test_train_df.iloc[:,1:]

X_training_test.describe().to_csv(str(path)+'X_training_test_stat.csv')


scaler = preprocessing.StandardScaler().fit(X_training_test)

X_training_test = pd.DataFrame(scaler.transform(X_training_test), index=X_training_test.index.values, columns=X_training_test.columns.values)


tf.reset_default_graph()

# Create some variables.


# Add ops to save and restore all the variables.

graph = tf.get_default_graph()


n_post=400

with tf.Session() as sess:  
    
    saver = tf.train.import_meta_graph(str(path)+'model'+str(count)+'.meta')
    saver.restore(sess,tf.train.latest_checkpoint(str(path)))    
    pmodel = graph.get_tensor_by_name('Squeeze:0')
    model_X=graph.get_tensor_by_name('Placeholder:0')   

    Y_post_test = np.zeros((n_post, X_training_test.shape[0]))
    for j in range(n_post):
        Y_post_test[j] = sess.run(pmodel, {model_X: X_training_test})

 


########  Test set  #################################################
           
    
Y_post_mean = Y_post_test.mean(axis=0)
Y_post_std=   Y_post_test.std(axis=0)
#print(Y_post_test.mean(axis=0), Y_post_test.std(axis=0))

test_train_df.insert(1,'C33 (GPa)',Y_post_mean)
test_train_df.insert(2,'Error',(Y_post_std)/2)
res=test_train_df.iloc[:, 0:3]
res.to_csv(str(path)+'LARGE_DL_PREDICTIONS_C33.csv')



        
if True:
    plt.figure(figsize=(10, 10))
    for i in range(n_post):
        plt.plot(X_training_test.iloc[0::100, 0], Y_post_test[i][0::100], "b.", alpha=1. / 200)
    plt.plot(X_training_test.iloc[0::100, 0], Y_post_mean[0::100], "r.")
    plt.grid()
#    plt.xlim(-0,-0.8)
    plt.ylim(Y_post_test[i][0::100].min()-0.1,Y_post_test[i][0::100].max()+0.1)
    plt.ylabel('C33 (Train)', fontsize=24)
    plt.xlabel('Feature 1', fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Test set")
    plt.savefig(str(path)+'C33(TEST)'+str(count), bbox_inches='tight')
    plt.show()

    
if True:
    plt.figure(figsize=(10, 10))    
    plt.errorbar( Y_post_mean[0::100],np.zeros((Y_post_mean[0::100].shape)), yerr=Y_post_std[0::100], fmt='r.')
#    plt.plot(y_pred, y_pred, "r.")
#    plt.plot(y, y, "b-")
    plt.grid() 
    #plt.xlim(-0.1,-0.6)
    plt.xticks(fontsize=16)
    #plt.ylim(-0.1,-0.6)
    plt.yticks(fontsize=16)
    plt.title("Test set")
    plt.ylabel('NN-TEST ($GPa$)', fontsize=24)
    plt.xlabel('DFT ($GPa2$)', fontsize=24)
#    red_patch = mpatches.Patch(label='The red data')
#    plt.legend(handles=[red_patch])    
#    plt.text(-0.05, -0.08, r'MSE ='+str(round(MSE,4)))
#    plt.text(-0.05, -0.02, r'MAE ='+str(round(MAE,4)))
    plt.savefig(str(path)+'UncertaintyTEST'+str(count), bbox_inches='tight')
    plt.show()
    

