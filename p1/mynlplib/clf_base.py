from mynlplib.constants import OFFSET
import numpy as np

# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# This will no longer work for our purposes since python3's max does not guarantee deterministic ordering
# argmax = lambda x : max(x.items(),key=lambda y : y[1])[0]

# deliverable 2.1
def make_feature_vector(base_features,label):
    '''
    take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    '''
    
    feature_vector = {}
    
    for word, count in base_features.items():
        feature_vector[(label, word)] = count
    
    feature_vector[(label,OFFSET)] = 1
    
    return feature_vector

# deliverable 2.2
def predict(base_features,weights,labels):
    '''
    prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    '''
    
    #init predict labels
    y = dict.fromkeys(labels, 0)
    
    #not super efficient / brute force loop to get the correct prediction
    for label in labels:
        for feature, count in base_features.items():
            y[label] += count * weights[(label, feature)]
        #add offset
        y[label] += weights[(label, OFFSET)]
    
    #top_label = max(y, key=y.get)
    
    top_label = argmax(y)
    
    return top_label, y
    
    

def predict_all(x,weights,labels):
    '''
    Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    '''
    y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])
    return y_hat