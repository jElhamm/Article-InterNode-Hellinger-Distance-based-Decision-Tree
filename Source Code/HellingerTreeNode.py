# This class defines a node in a decision tree used for classification tasks.
# Each node can split data based on a feature and a threshold value, leading
# to two branches (left and right). The node can also be marked as a leaf node
# when it is complete, and in such cases, it will store a classification label 
# and an associated score.



class HellingerTreeNode:
    def __init__(self):
        self.threshold = None           # Threshold value for the feature to split on
        self.feature   = None           # Feature index used for splitting the data
        self.left_branch  = None        # Left subtree branch
        self.right_branch = None        # Right subtree branch
        self.complete  = False          # Indicator if the node is a leaf node
        self.label = None               # Classification label if node is a leaf
        self.score = None               # Confidence score for the label if node is a leaf
