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
    
    if len(labelIDs) != 2 or not (0 in labelIDs and 1 in labelIDs):
        raise ValueError("Labels must be either 0 or 1; Label array may only contain a single label value")
    if numBins < 1:                                                                                              # Validate the number of bins
        raise ValueError("Number of bins must be 1 or larger")
    if minFeatureRatio > 1 or minFeatureRatio <= 0:                                                              # Validate the feature ratio
        raise ValueError("minFeatureRatio must be between (0 and 1]")
    if cutoff is None:                                                                                           # Set default cutoff value if not provided
        cutoff = 10 if numInstances > 10 else 1
    
    model = []
    for i in range(numTrees):                                                                                    # Grow each tree in the forest
        if printCount:
            print(f"Growing Tree Number: {i+1}")

        # Randomly select a subset of features for this tree
        reducedFeaturesIndices = random.sample(range(numFeatures), random.randint(ceil(minFeatureRatio * numFeatures), numFeatures))
        reducedFeatures = features[:, reducedFeaturesIndices]

        # Fit the tree with the selected features and add to the model
        tree = HDDT(reducedFeatures, labels, HellingerTreeNode(), numBins, cutoff, memThresh, memSplit)
        model.append((tree, reducedFeaturesIndices))

    return model
