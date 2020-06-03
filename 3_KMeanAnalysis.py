#Still a bit disorganized
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

path = './LASSO_BR2_9/'

n_clus = 3

train_df = pd.read_csv(str(path)+"BR2_training-test_set.csv",header=None)
#train_df = pd.read_csv("TOTAL_SET_PREDICTION_RF.csv",header=None)
numColumns = train_df.shape[1]
x_reduced = train_df.iloc[:,0:numColumns-1]
y_reduced = train_df.iloc[:,numColumns-1]

#k-means of data set to extract test set


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clus, random_state=None).fit(x_reduced,y_reduced)
for i in range(x_reduced.shape[0]):
    print(y_reduced.iloc[i],kmeans.labels_[i])

#Silhouette sampling

    
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

print(__doc__)

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]#, 13, 14 , 15 , 16 , 17 , 18 , 19 , 20]
X = x_reduced

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X,y_reduced)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.savefig(str(path)+'DataSet Silhouette '+str(n_clusters), bbox_inches='tight')
    
    
    

########################################################################################################################
########################################################################################################################
########################################################################################################################
#Selecting the number of clusters with the elbow method
########################################################################################################################
########################################################################################################################
########################################################################################################################

#from yellowbrick.cluster import KElbowVisualizer

#model = KMeans()
#visualizer = KElbowVisualizer(model, k=(4,12))

xplot=[]
yplot=[]
for i in range(1,40):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    #kmeans.labels_
    print(i,kmeans.inertia_)
    xplot=xplot+[i]
    yplot=yplot+[kmeans.inertia_]
    
plt.gcf().clear()
plt.plot(xplot,yplot)
plt.show()


########################################################################################################################
########################################################################################################################
########################################################################################################################
#Create randomized k-means training and test sets
########################################################################################################################
########################################################################################################################
########################################################################################################################

count=0


from sklearn.cluster import KMeans
from pandas import DataFrame
train_df = pd.read_csv(str(path)+"BR2_training-test_set.csv",header=None)
numColumns = train_df.shape[1]
x_reduced = train_df.iloc[:,0:numColumns-1]
y_reduced = train_df.iloc[:,numColumns-1]

kmeans = KMeans(n_clusters=n_clus, random_state=None).fit(x_reduced,y_reduced) #n_clusters=3 VARIABLE TO CHANGE BASED ON KMEANS RESULTS of SYLOUETTE AND ELBOW
#for i in range(x_reduced.shape[0]):
#    print(y_reduced.iloc[i],kmeans.labels_[i])

from pandas import DataFrame


train_df['kmeans']=kmeans.labels_
numDescriptors=train_df.shape[1]
cluster0=train_df.loc[train_df.kmeans==0]
cluster1=train_df.loc[train_df.kmeans==1]
cluster2=train_df.loc[train_df.kmeans==2]
cluster3=train_df.loc[train_df.kmeans==3]
cluster4=train_df.loc[train_df.kmeans==4]
cluster5=train_df.loc[train_df.kmeans==5]
cluster6=train_df.loc[train_df.kmeans==6]
#cluster7=train_df.loc[train_df.kmeans==7]
#cluster8=train_df.loc[train_df.kmeans==8]
#cluster9=train_df.loc[train_df.kmeans==9]
#cluster10=train_df.loc[train_df.kmeans==10]


#Randomize
cluster0=cluster0.sample(frac=1)
cluster1=cluster1.sample(frac=1)
cluster2=cluster2.sample(frac=1)
cluster3=cluster3.sample(frac=1)
cluster4=cluster4.sample(frac=1)
cluster5=cluster5.sample(frac=1)
cluster6=cluster6.sample(frac=1)
#cluster7=cluster7.sample(frac=1)
#cluster8=cluster8.sample(frac=1)
#cluster9=cluster9.sample(frac=1)
#cluster10=cluster10.sample(frac=1)


cluster0Portion=cluster0.shape[0]*20/100
cluster1Portion=cluster1.shape[0]*20/100
cluster2Portion=cluster2.shape[0]*20/100
cluster3Portion=cluster3.shape[0]*20/100
cluster4Portion=cluster4.shape[0]*20/100
cluster5Portion=cluster5.shape[0]*20/100
cluster6Portion=cluster6.shape[0]*20/100
#cluster7Portion=cluster7.shape[0]*20/100
#cluster8Portion=cluster8.shape[0]*20/100
#cluster9Portion=cluster9.shape[0]*20/100
#cluster10Portion=cluster10.shape[0]*20/100




kmeansTestSet_df=pd.DataFrame()
kmeansTrainingSet_df=pd.DataFrame()

testportion=cluster0.iloc[0:int(cluster0Portion),0:numColumns]
trainingportion=cluster0.iloc[int(cluster0Portion):cluster0.shape[0],0:numColumns]
kmeansTestSet_df=kmeansTestSet_df.append(testportion)
kmeansTrainingSet_df=kmeansTrainingSet_df.append(trainingportion)
#print(cluster0.shape,kmeansTestSet_df.shape,int(cluster0Portion))

testportion=cluster1.iloc[0:int(cluster1Portion),0:numColumns]
trainingportion=cluster1.iloc[int(cluster1Portion):cluster1.shape[0],0:numColumns]
kmeansTestSet_df=kmeansTestSet_df.append(testportion)
kmeansTrainingSet_df=kmeansTrainingSet_df.append(trainingportion)
#print(cluster1.shape,kmeansTestSet_df.shape,int(cluster1Portion))

testportion=cluster2.iloc[0:int(cluster2Portion),0:numColumns]
trainingportion=cluster2.iloc[int(cluster2Portion):cluster2.shape[0],0:numColumns]
kmeansTestSet_df=kmeansTestSet_df.append(testportion)
kmeansTrainingSet_df=kmeansTrainingSet_df.append(trainingportion)
#print(cluster2.shape,kmeansTestSet_df.shape,int(cluster2Portion))


testportion=cluster3.iloc[0:int(cluster3Portion),0:numColumns]
trainingportion=cluster3.iloc[int(cluster3Portion):cluster3.shape[0],0:numColumns]
kmeansTestSet_df=kmeansTestSet_df.append(testportion)
kmeansTrainingSet_df=kmeansTrainingSet_df.append(trainingportion)
#print(cluster3.shape,kmeansTestSet_df.shape,int(cluster3Portion))
#
##
testportion=cluster4.iloc[0:int(cluster4Portion),0:numColumns]
trainingportion=cluster4.iloc[int(cluster4Portion):cluster4.shape[0],0:numColumns]
kmeansTestSet_df=kmeansTestSet_df.append(testportion)
kmeansTrainingSet_df=kmeansTrainingSet_df.append(trainingportion)
#print(cluster4.shape,kmeansTestSet_df.shape,int(cluster4Portion))
#
##
testportion=cluster5.iloc[0:int(cluster5Portion),0:numColumns]
trainingportion=cluster5.iloc[int(cluster5Portion):cluster5.shape[0],0:numColumns]
kmeansTestSet_df=kmeansTestSet_df.append(testportion)
kmeansTrainingSet_df=kmeansTrainingSet_df.append(trainingportion)
#print(cluster5.shape,kmeansTestSet_df.shape,int(cluster5Portion))


testportion=cluster6.iloc[0:int(cluster6Portion),0:numColumns]
trainingportion=cluster6.iloc[int(cluster6Portion):cluster6.shape[0],0:numColumns]
kmeansTestSet_df=kmeansTestSet_df.append(testportion)
kmeansTrainingSet_df=kmeansTrainingSet_df.append(trainingportion)
#print(cluster6.shape,kmeansTestSet_df.shape,int(cluster6Portion))

#
#testportion=cluster7.iloc[0:int(cluster7Portion),0:numColumns]
#trainingportion=cluster7.iloc[int(cluster7Portion):cluster7.shape[0],0:numColumns]
#kmeansTestSet_df=kmeansTestSet_df.append(testportion)
#kmeansTrainingSet_df=kmeansTrainingSet_df.append(trainingportion)
#print(cluster7.shape,kmeansTestSet_df.shape,int(cluster7Portion))
#  
#testportion=cluster8.iloc[0:int(cluster8Portion),0:numColumns]
#trainingportion=cluster8.iloc[int(cluster8Portion):cluster8.shape[0],0:numColumns]
#kmeansTestSet_df=kmeansTestSet_df.append(testportion)
#kmeansTrainingSet_df=kmeansTrainingSet_df.append(trainingportion)
#print(cluster8.shape,kmeansTestSet_df.shape,int(cluster8Portion))
#  
# 
#
#testportion=cluster9.iloc[0:int(cluster9Portion),0:numColumns]
#trainingportion=cluster9.iloc[int(cluster9Portion):cluster9.shape[0],0:numColumns]
#kmeansTestSet_df=kmeansTestSet_df.append(testportion)
#kmeansTrainingSet_df=kmeansTrainingSet_df.append(trainingportion)
#print(cluster9.shape,kmeansTestSet_df.shape,int(cluster9Portion))

#
#testportion=cluster10.iloc[0:int(cluster10Portion),0:numColumns]
#trainingportion=cluster10.iloc[int(cluster10Portion):cluster10.shape[0],0:numColumns]
#kmeansTestSet_df=kmeansTestSet_df.append(testportion)
#kmeansTrainingSet_df=kmeansTrainingSet_df.append(trainingportion)
#print(cluster10.shape,kmeansTestSet_df.shape,int(cluster10Portion))


kmeansTrainingSet_df.to_csv(str(path)+"kmeans_randomized_trainingset_"+str(count)+"_indexed.csv",header=False)
kmeansTestSet_df.to_csv(str(path)+"kmeans_randomized_testset_"+str(count)+"_indexed.csv",header=False)

kmeansTrainingSet_df.to_csv(str(path)+"kmeans_randomized_trainingset_"+str(count)+"_unindexed.csv",header=False,index=False)
kmeansTestSet_df.to_csv(str(path)+"kmeans_randomized_testset_"+str(count)+"_unindexed.csv",header=False,index=False)



