from collections import defaultdict
from mynlplib.clf_base import predict,make_feature_vector

# deliverable 4.1
def perceptron_update(x,y,weights,labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''

    weights_update = defaultdict(float)
    
    pred, _ = predict(x, weights, labels)
    predf = make_feature_vector(x, pred)
    true_pred = make_feature_vector(x, y)
    
    if pred != y:
        
        weights_update.update(true_pred)
        
        for features, value in predf.items():
                predf[features] = -value
        weights_update.update(predf)
    
    return weights_update


# deliverable 4.2
def estimate_perceptron(x,y,N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''

    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    
    for it in range(N_its):
        for x_i,y_i in zip(x,y):
            for s,t in perceptron_update(x_i,y_i,weights,labels).items():
                weights[s] += t
            
        weight_history.append(weights.copy())
    return weights, weight_history
