# projBNN
Bayesian Neural Networks (Dropout)

Each file can be run independently on any Python interpreter (3.6.8)

1_MakePlmfBilayers.py takes a 1l_atomicPLMF_500structures.csv (which contains the full list of descriptors of 2d-monolayers) and C33.csv files and creates the descriptors of the bilayers by summing the fields of each monolayer.
Modify: 
Number_Monolayers  with the number of monolayers included in the PLMF.csv file
path  to the working folder (it will be created if it does not exist)

2_FeatureSelectionLASSO.py takes input the .csv file with the values bilayer property that will be used in BNN training
and the PLMF.csv file to select relevant bilayers descriptors. This will produce a training_set.csv.
Modify:
alphas and n_alpha in the 2_FeatureSelectionLASSO.py 
 	

3_KMeanAnalysis.py takes training.csv file and using a cluster analysis will generate a train and test set choosing representative bilayer structures. 


4_MakeCompleteSet.py select the descriptors listed after 2_FeatureSelectionLASSO.py and create a PLMF file of the full set of bilayers

5_BnnTrainCrossVal.py is the core code and it is used to train the Bayesisn neaural network and then test it by cross-validation.  
This part needs a list of descriptors for the whole set that will be used in the extrapolation. This allows the standardisation of each bilayer descriptor using the whole set as reference. Therefore, 

5_BnnTrainCrossVal.py takes as input COMPLETE_SET.csv and training_test_set.csv 
Parameters to change are: number of nodes, number of hidden layers by modifying “# Create the TensorFlow model”
n_post controls the number of trials are done to create a statistical distribution over the response value. 
model_prob control the probability of nodes dropout. 

6_BnnPredict.py takes the model and the COMPETE_SET.csv to generate a response value for the whole dataset.  

1l_atomicPLMF_500structures.csv and C33.csv can be used as example input file to run the calculations from scratch
