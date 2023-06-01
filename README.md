# 2D-BNN User Guide

This guide will help you navigate the 2D-BNN Bayesian Neural Network Python code. This code is designed to predict the properties of large sets of 2-dimensional materials.

## Getting Started

The 2D-BNN directory contains several pre-processing sub-routines. The Bayesian Neural Network is implemented in BNN_train_cross_val.py. You can load and execute the model using BNN_predict.py. Example input files, 1l_atomicPLMF_773structures.csv and C33.csv, are located in the ./input_example directory and can be used to run computations from scratch.

# Pre-processing Data Workflow:

1. Making PLMF Bilayers
1_MakePlmfBilayers.py uses 1l_atomicPLMF_773structures.csv (which contains a comprehensive list of 2D-monolayer descriptors) and C33.csv files to create bilayer descriptors by adding up the fields of each monolayer. Adjust Number_Monolayers to match the number of monolayers included in the PLMF.csv file and set the path to your working folder (it will be created if it does not exist).

2. Feature Selection
2a_FeatureSelectionLASSO.py requires the .csv file with the bilayer property values to be used in BNN training and the PLMF.csv file to select relevant bilayer descriptors. This will output a training_set.csv file. Adjust alphas and n_alpha in 2_FeatureSelectionLASSO.py.

If the number of features is too large, you can optionally execute 2b_FeatureSelectionGeneticAlgo.py.

3. K-Means Analysis
3_KMeanAnalysis.py uses the training.csv file and a cluster analysis to generate a train and test set, selecting representative bilayer structures.

4. Creating a Complete Set
4_MakeCompleteSet.py selects the descriptors listed after 2_FeatureSelectionLASSO.py and creates a PLMF file of the full set of bilayers. Run this script twice as outlined in the comments in MakeCompleteSet.py.

# Model Optimization:

5. BNN Training and Cross Validation
5_BnnTrainCrossVal.py is the main code that trains the Bayesian neural network and tests it via cross-validation. It needs a list of descriptors for the entire set used in the extrapolation. This allows for each bilayer descriptor's standardisation using the whole set as a reference.

5_BnnTrainCrossVal.py uses COMPLETE_SET.csv and training_test_set.csv as inputs. Changeable parameters include the number of nodes, the number of hidden layers (under # Create the TensorFlow model), n_post (controls the number of trials to create a statistical distribution over the response value), and model_prob (controls the probability of nodes dropout).

Loading the Model and Making Predictions:

6_BnnPredict.py uses the model and COMPETE_SET.csv to generate a response value for the entire dataset.
