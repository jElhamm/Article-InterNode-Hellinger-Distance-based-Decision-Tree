#***************************************************************************************************************
#                                                                                                              *
#               This function implements the Hellinger Distance Decision Tree (HDDT) algorithm.                *
#             It recursively splits the dataset based on the feature and threshold that maximize               *
#            the Hellinger distance, constructing a decision tree for classification. The function             *
#         handles edge cases where all labels are the same or the number of samples is below a cutoff.         *
#                                                                                                              *
#***************************************************************************************************************



import numpy as np
from scipy.stats import mode
from compute_hellinger_distance import compute_hellinger_distance
from HellingerTreeNode import HellingerTreeNode



def HDDT(features, labels, model, num_bins, cutoff, mem_thresh, mem_split):
    num_samples = features.shape[0]
    
    # Check if all labels are the same or if the number of samples is below the cutoff
    if len(np.unique(labels)) == 1 or num_samples <= cutoff:
        model.complete = True
        model.label = mode(labels)[0][0]                                        # Set label to the most frequent class
        model.score = np.sum(labels) / len(labels)                              # Compute the score
        return model
    
    num_features = features.shape[1]
    selected_feature = -1
    selected_threshold = -1
    selected_distance = -1
    
    # Adjust memory split based on the number of samples
    if num_samples <= mem_thresh:
        mem_split = 1
    
    # Iterate over features to find the best feature and threshold for splitting
    for i in range(0, num_features, max(1, num_features // mem_split)):
        max_index = min(num_features, i + max(1, num_features // mem_split))
        feature_indices = np.arange(i, max_index)
        features_temp = features[:, feature_indices]
        
        feature_index, feature_distance, feature_threshold = compute_hellinger_distance(features_temp, labels, num_bins)
        
        if feature_distance > selected_distance:
            selected_feature = feature_indices[feature_index]
            selected_threshold = feature_threshold
            selected_distance = feature_distance
    
    model.threshold = selected_threshold
    model.feature = selected_feature
    
    # Split the dataset into left and right branches
    features_left = features[features[:, selected_feature] <= selected_threshold]
    labels_left = labels[features[:, selected_feature] <= selected_threshold]
    features_right = features[features[:, selected_feature] > selected_threshold]
    labels_right = labels[features[:, selected_feature] > selected_threshold]
    
    # Check for pure split cases
    if features_left.shape[0] == num_samples or features_right.shape[0] == num_samples:
        model.complete = True
        model.label = mode(labels)[0][0]
        model.score = np.sum(labels) / len(labels)
        return model
    
    # Recursively build the left and right branches
    model_left = HellingerTreeNode()
    model_right = HellingerTreeNode()
    
    model.left_branch = HDDT(features_left, labels_left, model_left, num_bins, cutoff, mem_thresh, mem_split)
    model.right_branch = HDDT(features_right, labels_right, model_right, num_bins, cutoff, mem_thresh, mem_split)
    model.complete = False
    return model
