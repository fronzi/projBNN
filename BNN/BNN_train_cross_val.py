# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 
import matplotlib.pyplot as plt
from tensorflow.contrib.distributions import Bernoulli
from tensorflow.python.ops.distributions import bernoulli
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.patches as mpatches
import sys
from sklearn.model_selection import StratifiedKFold

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_pred + y_true))) * 100

 

def run_train(session, train_x, train_y):
    print ("\nStart training")
    batch_size=16
    for epoch in range(5000):
        total_batch = int(train_x.shape[0] / batch_size)
    for i in range(total_batch):
        batch_x = train_x[i*batch_size:(i+1)*batch_size]
        batch_y = train_y[i*batch_size:(i+1)*batch_size]
        mse, c = session.run([model_mse,train_step], feed_dict={model_X: batch_x, model_y: batch_y})
        if i % 100 == 0:
            print("Iteration {}. Mean squared error: {:.4f}.".format(i, mse))


def cross_validate(split_size,session,train_x_all,train_y_all):
    retults=[]
    total_folds= int(train_x_all.shape[0]/split_size)
    
    for i in range(split_size):
        train_x = train_x_all[i*total_folds:(i+1)*total_folds]
        train_y = train_y_all[i*total_folds:(i+1)*total_folds]
        run_train(session, train_x, train_y)






class VariationalDense:
    """Variational Dense Layer Class"""
    def __init__(self, n_in, n_out, model_prob, model_lam):
        self.model_prob = model_prob
        self.model_lam = model_lam
        self.model_bern = Bernoulli(probs=self.model_prob, dtype=tf.float32)
        self.model_M = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.001))
        self.model_m = tf.Variable(tf.zeros([n_out]))
        self.model_W = tf.matmul(
            tf.diag(self.model_bern.sample((n_in, ))), self.model_M
        )

    def __call__(self, X, activation=tf.identity):
        output = activation(tf.matmul(X, self.model_W) + self.model_m)
        if self.model_M.shape[1] == 1:
            output = tf.squeeze(output)
        return output

    @property
    def regularization(self):
        return self.model_lam * (
            self.model_prob * tf.reduce_sum(tf.square(self.model_M)) +
            tf.reduce_sum(tf.square(self.model_m))
        )


#tf.Session().reset()


count=3
path = './LASSO_BR2_002/'

train_df = pd.read_csv(str(path)+"kmeans_randomized_trainingset_0_unindexed.csv",header=None)
test_df = pd.read_csv(str(path)+"kmeans_randomized_testset_0_unindexed.csv",header=None)
test_train_df = pd.read_csv(str(path)+"COMPLETE_DL_SET.csv",header=None)



numColumns = train_df.shape[1]


X_training_test = test_train_df.iloc[:,1:numColumns]
#y_training_test = test_train_df.iloc[:,numColumns-1]

X = train_df.iloc[:,0:numColumns-1]
y = train_df.iloc[:,numColumns-1]

X_pred = test_df.iloc[:,0:numColumns-1]
y_pred = test_df.iloc[:,numColumns-1]


X.describe().to_csv(str(path)+'X_train_stat.csv')
X_pred.describe().to_csv(str(path)+'X_test_stat.csv')
X_training_test.describe().to_csv(str(path)+'COMPLETE_DL_SET_stat.csv')
######################



scaler = preprocessing.StandardScaler().fit(X_training_test)
#scaler = preprocessing.Normalizer().fit(X_training_test)
#scaler = preprocessing.Normalizer().fit(X)



X = pd.DataFrame(scaler.transform(X), index=X.index.values, columns=X.columns.values)
X_pred = pd.DataFrame(scaler.transform(X_pred), index=X_pred.index.values, columns=X_pred.columns.values)
X_training_test = pd.DataFrame(scaler.transform(X_training_test), index=X_training_test.index.values, columns=X_training_test.columns.values)



n_samples =  X.shape[0]




# Create the TensorFlow model.
#sys.exit(-1)


n_feats = X.shape[1]
n_hidden = 32
model_prob = 0.9
model_lam = 1e-3
model_X = tf.placeholder(tf.float32, [None, n_feats])
model_y = tf.placeholder(tf.float32, [None])
model_L_1 = VariationalDense(n_feats, n_hidden, model_prob, model_lam)
#model_L_1b = VariationalDense(n_hidden, int(n_hidden/2), model_prob, model_lam)
#model_L_2 = VariationalDense(int(n_hidden), int(n_hidden), model_prob, model_lam)
model_L_3 = VariationalDense(int(n_hidden), int(n_hidden), model_prob, model_lam)
model_L_4 = VariationalDense(int(n_hidden), 1, model_prob, model_lam)
#model_L_5 = VariationalDense(32, 1, model_prob, model_lam)
model_out_1 = model_L_1(model_X, tf.nn.elu)    #relu
#model_out_1b = model_L_1b(model_out_1, tf.nn.relu)
#model_out_2 = model_L_2(model_out_1, tf.nn.elu)
model_out_3= model_L_3(model_out_1,tf.nn.elu)
#model_out_4= model_L_4(model_out_3,tf.nn.relu)
model_pred = model_L_4(model_out_3)
model_sse = tf.reduce_sum(tf.square(model_y - model_pred))
#model_sse = tf.reduce_sum(tf.abs(model_y - model_pred))
#model_sse =  tf.reduce_sum(model_y * tf.log(model_pred))     #cross entropy
model_mse = model_sse / n_samples
model_loss = (
        # Negative log-likelihood.
        model_sse+ 
        # Regularization.
        model_L_1.regularization +
#        model_L_1b.regularization +
#        model_L_2.regularization +
        model_L_3.regularization
        ) / n_samples

 

train_step = tf.train.AdamOptimizer(1e-3).minimize(model_loss)
saver = tf.train.Saver() 
bmse=np.inf
br2=-1
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    cross_validate(split_size=10,session=sess,train_x_all=X,train_y_all=y)
    for i in range(30001):
        sess.run(train_step, {model_X: X, model_y: y})
        if i % 100 == 0:
            mse = sess.run(model_mse, {model_X: X, model_y: y})
    
            

            # Sample from the posterior val set
            n_post = 500
            Y_post = np.zeros((n_post, X_pred.shape[0]))
            for j in range(n_post):
                Y_post[j] = sess.run(model_pred, {model_X: X_pred})
            Y_post_mean = Y_post.mean(axis=0)
            vmse=mean_squared_error(y_pred, Y_post_mean)
            r2=r2_score(y_pred, Y_post_mean)
            

            print("Iteration {}. Train MSE: {:.6f}. Val MSE: {:.6f} Val R2:{:.6f}".format(i, mse,vmse,r2)) 
            
            
            #if vmse<bmse:
            if br2<r2:
                # Sample from the posterior test set
                bmse=vmse
                br2=r2      
                Y_post_val=Y_post.copy()  
                save_path = saver.save(sess, str(path)+'model'+str(count))
                print("best Model saved in path: %s" % save_path) 
            



with tf.Session() as sess:  
    graph = tf.get_default_graph()
    saver = tf.compat.v1.train.import_meta_graph(str(path)+'model'+str(count)+'.meta')
    saver.restore(sess,tf.train.latest_checkpoint(str(path)))    
    pmodel = graph.get_tensor_by_name(model_pred.name) 

    Y_post_train = np.zeros((n_post, X.shape[0]))
    for j in range(n_post):
        Y_post_train[j] = sess.run(pmodel, {model_X: X}) 
       


       

########  Train/Test phase #################################################
    
    
########  Test set #################################################

#    
#
#
Y_post_mean = Y_post_val.mean(axis=0)
Y_post_std=   Y_post_val.std(axis=0)
#print(Y_post_val.mean(axis=0), Y_post_val.std(axis=0))


print('R2-VAL:',r2_score(y_pred, Y_post_mean))
print('MSE-VAL:',mean_squared_error(y_pred, Y_post_mean))
print('MAE-VAL:',mean_absolute_error(y_pred, Y_post_mean))
print('MAPE-VAL:', mean_absolute_percentage_error(y_pred, Y_post_mean)) 



        
if True:
    plt.figure(figsize=(10, 10))
    for i in range(n_post):
        plt.plot(X_pred.iloc[:, 0], Y_post_val[i], "b.", alpha=1. / 200)
    plt.plot(X_pred.iloc[:, 0], y_pred, "g.")
    plt.grid()
#    plt.xlim(-0,-0.8)
#    plt.ylim(-0,-0.8)
    plt.ylabel('Piezo T (Validation)', fontsize=24)
    plt.xlabel('Feature 1', fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("validation test")
    plt.savefig(str(path)+'BG_Feat(VAL)'+str(count), bbox_inches='tight')
    plt.show()
    
    

    
if True:
    plt.figure(figsize=(10, 10))    
    plt.errorbar(y_pred, Y_post_mean, yerr=Y_post_std, fmt='g.')#, "r.", alpha=1. / 200)
#    plt.plot(y_pred, y_pred, "r.")
    plt.plot(y_pred,y_pred, "b-")
    plt.grid()   
    #plt.xlim(-0.1,-0.6)
    #plt.ylim(-0.1,-0.6)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('NN-TEST ($C/m^2$)', fontsize=24)
    plt.xlabel('DFT ($C/m^2$)', fontsize=24) 
    plt.title("validation set")   
#    plt.text(-0.05, -0.08, r'MSE ='+str(round(MSE,4)))
#    plt.text(-0.05, -0.02, r'MAE ='+str(round(MAE,4)))
    plt.savefig(str(path)+'UncertaintyVAL'+str(count), bbox_inches='tight')
    plt.show()
    


########  Train set  #################################################
           
    
Y_post_mean = Y_post_train.mean(axis=0)
Y_post_std=   Y_post_train.std(axis=0)
print(Y_post_train.mean(axis=0))
print(y)


print('R2-TRAIN:',r2_score(y, Y_post_mean))
print('MSE-TRAIN:',mean_squared_error(y, Y_post_mean))
print('MAE-TRAIN:',mean_absolute_error(y, Y_post_mean))
print('MAPE-TRAIN:', mean_absolute_percentage_error(y, Y_post_mean)) 


        
if True:
    plt.figure(figsize=(10, 10))
    for i in range(n_post):
        plt.plot(X.iloc[:, 0], Y_post_train[i], "b.", alpha=1. / 200)
    plt.plot(X.iloc[:, 0], y, "g.")
    plt.grid()
#    plt.xlim(-0,-0.8)
#    plt.ylim(-0,-0.8)
    plt.ylabel('Piezo T (Train)', fontsize=24)
    plt.xlabel('Feature 1', fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Train set")
    plt.savefig(str(path)+'Piezo T (TRAIN)'+str(count), bbox_inches='tight')
    plt.show()

    
if True:
    plt.figure(figsize=(10, 10))    
    plt.errorbar(y, Y_post_mean, yerr=Y_post_std, fmt='g.')#, "r.", alpha=1. / 200)
#    plt.plot(y_pred, y_pred, "r.")
    plt.plot(y, y, "b-")
    plt.grid() 
    #plt.xlim(-0.1,-0.6)
    plt.xticks(fontsize=16)
    #plt.ylim(-0.1,-0.6)
    plt.yticks(fontsize=16)
    plt.title("Train set")
    plt.ylabel('NN-TRAIN ($C/m^2$)', fontsize=24)
    plt.xlabel('DFT ($C/m^2$)', fontsize=24)
#    red_patch = mpatches.Patch(label='The red data')
#    plt.legend(handles=[red_patch])    
#    plt.text(-0.05, -0.08, r'MSE ='+str(round(MSE,4)))
#    plt.text(-0.05, -0.02, r'MAE ='+str(round(MAE,4)))
    plt.savefig(str(path)+'UncertaintyTRAIN'+str(count), bbox_inches='tight')
    plt.show()
    



