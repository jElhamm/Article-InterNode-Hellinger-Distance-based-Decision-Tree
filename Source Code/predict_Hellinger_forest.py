#*******************************************************************************************************************************
#                                                                                                                              *
#            This function predicts labels and scores using a trained Hellinger Distance Decision Forest model.                *
#       It aggregates predictions and scores from multiple decision trees in the forest and returns the majority-voted         *
#                       predicted_classes and the average predicted_scores for positive labels.                                *
#                                                                                                                              *
#*******************************************************************************************************************************



import numpy as np
from scipy.stats import mode


def predict_Hellinger_forest(model, features):
    """
        Predict labels using a trained Hellinger Distance Decision Forest.
        
        Parameters:
            model (list): A trained Hellinger Distance Decision Forest model.
                        Each element of the list contains a tuple (tree_model, feature_indices),
                        where tree_model is a trained Hellinger Distance Decision Tree model
                        and feature_indices is a list of indices indicating which features were used for training the tree.
            f
            eatures (numpy.ndarray): I x F numeric matrix where I is the number of instances and F
                                    is the number of features. Each row represents one training instance
                                    and each column represents the value of one of its corresponding features.
        
        Returns:
            predicted_classes (numpy.ndarray): I x 1 matrix where each row represents a predicted label of the corresponding feature set.
            predicted_scores (numpy.ndarray) : I x 1 matrix where each row represents a score of the corresponding feature set having label "1"/"true"/"positive".
    """


    num_instances, num_features = features.shape
    if num_instances <= 0:                                                                      # Check if the input feature matrix is valid
        raise ValueError("Feature array is empty or only one instance exists")
    if num_features == 0:
        raise ValueError("No feature data")

    num_trees = len(model)
    predictions = np.zeros((num_instances, num_trees))
    scores = np.zeros((num_instances, num_trees))
    for i in range(num_trees):                                                                  # Iterate through each tree in the forest
        tree_model, feature_indices = model[i]
        tree_features = features[:, feature_indices]
        tree_predictions, tree_scores = predict_Hellinger_tree(tree_model, tree_features)
        predictions[:, i] = tree_predictions
        scores[:, i] = tree_scores

    predicted_classes = mode(predictions, axis=1)[0]                                            # Determine majority voted predicted_classes and average predicted_scores
    predicted_scores = np.mean(scores, axis=1)
    return predicted_classes, predicted_scores
