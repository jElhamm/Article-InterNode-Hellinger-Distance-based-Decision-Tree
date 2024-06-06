#****************************************************************************************************************************************************
#                                                                                                                                                   *
#       - This function trains a single Hellinger Distance Decision Tree using the HDDT algorithm.                                                  *
#       - It takes input features and labels, along with optional parameters such as the number of bins for discretizing numeric features,          *
#         the maximum number of instances in a leaf node, and parameters for memory optimization when dealing with large datasets.                  *
#       - The function ensures that the input data is valid, including the proper alignment of features and labels                                  *
#         and the binary nature of the labels.                                                                                                      *
#       - Default values are provided for optional parameters if not specified.                                                                     *
#       - The function initializes a HellingerTreeNode object as the model and uses the HDDT function to train the tree.                            *
#       - It returns the trained model.                                                                                                             *
#                                                                                                                                                   *
#****************************************************************************************************************************************************



import numpy as np
from scipy.stats import mode
from HellingerTreeNode import HellingerTreeNode
from HDDT import HDDT


def fit_Hellinger_tree(features, labels, numBins=100, cutoff=None, memSplit=1, memThresh=1):
    """
    Train a single Hellinger Distance Decision Tree.
    
    Parameters:
        features (numpy.ndarray) : I x F numeric matrix where I is the number of instances and F is the number of features.
                                   Each row represents one training instance and each column represents the value of one of its corresponding features.
        labels (numpy.ndarray)   : I x 1 numeric matrix where I is the number of instances. Each row is the label of a specific training instance 
                                   and corresponds to the same row in features.
        numBins (int, optional)  : Number of bins for discretizing numeric features. Default: 100.
        cutoff (int, optional)   : Maximum number of instances in a leaf node. Default: 10 if more than ten instances, 1 otherwise.
        memSplit (int, optional) : If features matrix is large, compute discretization splits iteratively in batches of size memSplit instead all at once. Default: 1.
        memThresh (int, optional): If features matrix is large, compute discretization splits iteratively in batches of size memSplit only if number of instances in 
                                   branch is greater than memThresh. Default: 1.
    
    Returns:
        model (HellingerTreeNode): A trained Hellinger Distance Decision Tree model.
    """
    
    numInstances, numFeatures = features.shape
    if numInstances <= 1:                                                                                           # Check if the input feature matrix is valid
        raise ValueError("Feature array is empty or only instance exists")
    if numFeatures == 0:
        raise ValueError("No feature data")
    if labels.shape[0] != numInstances:                                                                             # Check if the number of labels matches the number of instances
        raise ValueError("Number of instances in feature matrix and label matrix do not match")
    