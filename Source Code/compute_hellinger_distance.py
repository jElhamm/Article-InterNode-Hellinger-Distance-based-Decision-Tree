#*****************************************************************************************************************
#                                                                                                                *
#               This function computes the Hellinger distance for each feature in a dataset,                     *
#             identifies the optimal threshold for splitting the data, and returns the feature                   *
#        index, the maximum Hellinger distance, and the best threshold. The Hellinger distance measures          *
#     the difference between the distributions of positive and negative labels, which helps in selecting the     *
#                  best feature and threshold for splitting the data in a decision tree.                         *
#                                                                                                                *
#*****************************************************************************************************************



import numpy as np


def compute_hellinger_distance(features, labels, num_bins):
    # Compute the minimum and maximum values for each feature to determine bin sizes
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    bin_size = (max_vals - min_vals) / num_bins
    
    # Calculate the total number of positive and negative labels
    Tplus = np.sum(labels == 1)
    Tminus = np.sum(labels == 0)
    
    # Generate threshold values for binning the feature values
    thresholds = np.linspace(min_vals, max_vals, num_bins + 1)[1:-1]
    labels = np.expand_dims(labels, axis=1)
    
    distances = []
    
    # Iterate over each feature to calculate Hellinger distances
    for f in range(features.shape[1]):
        feature = features[:, f]
        dists = []
    
            # Calculate the Hellinger distance for each threshold
        for threshold in thresholds[:, f]:
            Tlplus = np.sum((feature <= threshold) & (labels == 1))
            Tlminus = np.sum((feature <= threshold) & (labels == 0))
            Trplus = np.sum((feature > threshold) & (labels == 1))
            Trminus = np.sum((feature > threshold) & (labels == 0))
            
            # Calculate the Hellinger distance for current split
            dist = (np.sqrt(Tlplus / Tplus) - np.sqrt(Tlminus / Tminus)) ** 2 + \
                   (np.sqrt(Trplus / Tplus) - np.sqrt(Trminus / Tminus)) ** 2
            dists.append(dist)
        distances.append(dists)
    
    distances = np.array(distances)
    
    # Determine the feature and threshold that maximize the Hellinger distance
    max_distances = np.max(distances, axis=1)
    feature = np.argmax(max_distances)
    distance = max_distances[feature]
    threshold = thresholds[np.argmax(distances[feature]), feature]
    
    return feature, distance, threshold
    