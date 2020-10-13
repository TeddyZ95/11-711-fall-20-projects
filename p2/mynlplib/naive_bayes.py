from mynlplib.constants import OFFSET
from mynlplib import clf_base, evaluation, preproc

import numpy as np
from collections import defaultdict, Counter


def get_nb_weights(trainfile, smoothing):
    """
    estimate_nb function assumes that the labels are one for each document, where as in POS tagging: we have labels for 
    each particular token. So, in order to calculate the emission score weights: P(w|y) for a particular word and a 
    token, we slightly modify the input such that we consider each token and its tag to be a document and a label. 
    The following helper code converts the dataset to token level bag-of-words feature vector and labels. 
    The weights obtained from here will be used later as emission scores for the viterbi tagger.
    
    inputs: train_file: input file to obtain the nb_weights from
    smoothing: value of smoothing for the naive_bayes weights
    
    :returns: nb_weights: naive bayes weights
    """
    token_level_docs=[]
    token_level_tags=[]
    for words,tags in preproc.conll_seq_generator(trainfile):
        token_level_docs += [{word:1} for word in words]
        token_level_tags +=tags
    nb_weights = estimate_nb(token_level_docs, token_level_tags, smoothing)
    
    return nb_weights


# Can copy from P1
def get_corpus_counts(x,y,label):
    """
    Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    
    corpus_counts = Counter()
    #corpus_counts[label] = 0
    
    for i in range(0, len(y)):
        if y[i] == label:
            corpus_counts += x[i]
            
    corpus_counts = defaultdict(int, corpus_counts)
    
    return corpus_counts
    

    

# Can copy from P1
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''

    corpus_counts = get_corpus_counts(x, y, label)
    
    sum_words = sum(corpus_counts.values())
    denom = np.log(len(vocab) * smoothing + sum_words)
    
    logpw = defaultdict(float)
    for i in vocab:
        logpw[i[0]] = np.log(corpus_counts[i[0]] + smoothing) - denom
        
    return logpw



# Can copy from P1
def estimate_nb(x,y,smoothing):
    """
    estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """

    label_counter = Counter()    
    base_feature_counts = Counter()
    label_counter.update(y)
    labels = set(y)
    
    #counter loop
    for i in range(0, len(x)):
        base_feature_counts += x[i]
    
    vocab = set(base_feature_counts.items())
    weights = defaultdict(float)
    
    for i in labels:
        logpxy = estimate_pxy(x, y, i, smoothing, vocab)
        for j in logpxy:
            weights[(i, j)] = logpxy[j]
        mu = label_counter[i]/len(y)
        #adding offset
        weights[(i, OFFSET)] = np.log(mu)
        
    return weights

    

# Can copy from P1
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''

    labels = list(set(y_tr))
    scores = {}
    best = 0
    best_sm = 0
    
    for i in smoothers:
        weights = estimate_nb(x_tr, y_tr, i)
        y_hat = clf_base.predict_all(x_dv, weights, labels)
        acc = evaluation.acc(y_hat, y_dv)
        scores[i] = acc
        if acc > best:
            best = acc
            best_sm = i
            
    return best_sm, scores
