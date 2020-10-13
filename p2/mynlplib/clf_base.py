from mynlplib.constants import OFFSET
import numpy as np

import operator

# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# argmax = lambda x : max(x.items(),key=operator.itemgetter(1))[0]


# Deliverable 2.1 - can copy from P1
def make_feature_vector(base_features,label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    
    feature_vector = {}
    
    for word, count in base_features.items():
        feature_vector[(label, word)] = count
    
    feature_vector[(label,OFFSET)] = 1
    
    return feature_vector
    

# Deliverable 2.1 - can copy from P1
def predict(base_features,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    
    y = dict.fromkeys(labels, 0)
    
    for label in labels:
        for feature, count in base_features.items():
            y[label] += count * weights[(label, feature)]
        #add offset
        y[label] += weights[(label, OFFSET)]
    
    top_label = argmax(y)
    
    return top_label, y