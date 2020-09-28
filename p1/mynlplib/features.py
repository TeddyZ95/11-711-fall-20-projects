from mynlplib.constants import OFFSET
import numpy as np
import torch

# deliverable 6.1
def get_top_features_for_label(weights,label,k=5):
    '''
    Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    '''

    top_features = []
    
    list_keys = list(weights)
    label_feature = [i for i in list_keys if i[0] == label]
    
    #print(label_feature)
    
    label_weights = []
    for i in label_feature:
        label_weights.append(weights[i])
    
    if len(label_feature) > k:
        indices = np.argpartition(label_weights, -k)[-k:]
    
        for i in indices:
            top_features.append((label_feature[i], weights[label_feature[i]]))
    else: 
        top_features = [(j, weights[j]) for j in label_feature]

    top_features.sort(key= lambda x:x[1], reverse=True)
    
    return top_features    


# deliverable 6.2
def get_top_features_for_label_torch(model,vocab,label_set,label,k=5):
    '''
    Return the five words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted
    :param label_set: set of ordered labels
    :param label: the label you are interested in 
    :returns: list of words
    :rtype: list
    '''

    
    vocab = sorted(vocab)
    
    top_words = []
    
    np_data = list(model.parameters())
    weights = np_data[0].data.numpy()

    indices = np.argpartition(weights[label_set.index(label), :], -k)[-k:]
    
    for it in indices:
        top_words.append((vocab[it], weights[label_set.index(label), it]))

    top_words.sort(key=lambda x: x[1], reverse=True)
    
    top_words_list = []
    for i in top_words:
        top_words_list.append(i[0][0])

    return top_words_list

# deliverable 7.1
def get_token_type_ratio(counts):
    '''
    compute the ratio of tokens to types

    :param counts: bag of words feature for a song, as a numpy array
    :returns: ratio of tokens to types
    :rtype: float

    '''
    
    ratio = np.sum(counts) / np.count_nonzero(counts)
    
    return ratio

# deliverable 7.2
def concat_ttr_binned_features(data):
    '''
    Discretize your token-type ratio feature into bins.
    Then concatenate your result to the variable data

    :param data: Bag of words features (e.g. X_tr)
    :returns: Concatenated feature array [Nx(V+7)]
    :rtype: numpy array

    '''
    
    top = np.sum(data, axis = 1)
    denom = np.count_nonzero(data, axis = 1)
    
    ratio = np.divide(top, denom, out = top, where = denom != 0)
    
    bins = np.array([1, 2, 3, 4, 5, 6, float("inf")])

    new_feature = np.digitize(ratio, bins, right=False)
    new_feature = (np.arange(7) == new_feature[:, np.newaxis]) + 0
    
    result = np.concatenate((data, new_feature), axis = 1)
    
    return result
    
