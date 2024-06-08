#*****************************************************************************************************************
#                                                                                                                *
#         This function fits a Hellinger Forest, an ensemble of Hellinger Distance Decision Trees (HDDT).        *
#         It randomly selects subsets of features for each tree to improve generalization and robustness.        *
#                                                                                                                *
#*****************************************************************************************************************



import random
import numpy as np
from HDDT import HDDT
from HellingerTreeNode import HellingerTreeNode



def fit_Hellinger_forest(features, labels, numTrees, numBins=100, minFeatureRatio=0.8, cutoff=None, printCount=False, memSplit=1, memThresh=1):
    numInstances, numFeatures = features.shape
    if numInstances <= 1:                                                                                        # Check if the input feature matrix is valid
        raise ValueError("Feature array is empty or only instance exists")
    if numFeatures == 0:
        raise ValueError("No feature data")
    if labels.shape[0] != numInstances:                                                                          # Check if the number of labels matches the number of instances
        raise ValueError("Number of instances in feature matrix and label matrix do not match")
    