#********************************************************************************************************************************************
#                                                                                                                                           *
#                 This function predicts labels and scores using a trained Hellinger Distance Decision Tree model.                          *
#         It iterates through each instance in the input feature matrix and traverses the decision tree to predict the label                *
#    and score for each instance. The predicted_classes array stores the predicted labels for each instance, and the predicted_scores       *
#                  array stores the estimated probability of the corresponding instance having a positive label.                            *                
#                                                                                                                                           *
#********************************************************************************************************************************************


import numpy as np


def predict_Hellinger_tree(model, features):
    """
        Predict labels using a trained Hellinger Distance Decision Tree.
        
        Parameters:
            model (HellingerTreeNode): A trained Hellinger Distance Decision Tree model.
            features (numpy.ndarray) : I x F numeric matrix where I is the number of instances and F
                                    is the number of features. Each row represents one training instance
                                    and each column represents the value of one of its corresponding features.
        
        Returns:
            predicted_classes (numpy.ndarray): I x 1 matrix where each row represents a predicted label of the corresponding feature set.
            predicted_scores (numpy.ndarray) : I x 1 matrix where each row represents the estimated
                                            probability of the corresponding feature set having label "1"/"true"/"positive".
    """

    num_instances, num_features = features.shape
    if num_instances <= 0:                                                                      # Check if the input feature matrix is valid
        raise ValueError("Feature array is empty or only one instance exists")
    if num_features == 0:
        raise ValueError("No feature data")

    initial_model = model                                                                       # Initialize the initial model and prediction arrays
    predicted_classes = np.zeros((num_instances, 1))
    predicted_scores = np.zeros((num_instances, 1))
    for i in range(num_instances):                                                              # Iterate through each instance in the feature matrix
        model = initial_model
        complete = model.complete
        while not complete:                                                                     # Traverse the decision tree until a leaf node is reached
            if features[i, model.feature] <= model.threshold:
                model = model.left_branch
            else:
                model = model.right_branch
            complete = model.complete
        predicted_classes[i] = model.label                                                      # Assign the predicted label and score for the current instance
        predicted_scores[i] = model.score

    return predicted_classes, predicted_scores
