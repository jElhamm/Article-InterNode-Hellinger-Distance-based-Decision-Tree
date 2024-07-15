#********************************************************************************************************************************************************
#                                                                                                                                                       *
#                   This function computes precision, recall, and F1 score to evaluate the performance of a binary classifier                           *
#              based on the provided test labels and corresponding predictions. Precision is the ratio of true positive predictions                     *
#           to the total positive predictions, recall is the ratio of true positive predictions to the total actual positive instances,                 *
#        and F1 score is the harmonic mean of precision and recall. It also handles edge cases where there are no positive predictions or no            *
#      actual positive instances, ensuring that division by zero does not occur. If both precision and recall are zero, F1 score is set to zero         *
#                                                        to avoid undefined values.                                                                     *
#                                                                                                                                                       *
#********************************************************************************************************************************************************



import numpy as np


def get_statistics(test_labels, predictions):
    """
        Calculate precision, recall, and F1 score.
        
        * Parameters:
                    - test_labels (numpy.ndarray): Array of true labels.
                    - predictions (numpy.ndarray): Array of predicted labels.
        
        * Returns:
                    - precision (float): Precision score.
                    - recall (float): Recall score.
                    - f1 (float): F1 score.
    """

    # Calculate precision
    precision = np.sum(test_labels * predictions) / np.sum(predictions)
    if np.sum(predictions) == 0:
        precision = 1.0

    # Calculate recall
    recall = np.sum(test_labels * predictions) / np.sum(test_labels)
    if np.sum(test_labels) == 0:
        recall = 1.0

    # Calculate F1 score
    f1 = np.mean(2 * (precision * recall) / (precision + recall))
    if precision + recall == 0:
        f1 = 0.0

    return precision, recall, f1
