from collections import Counter

import pandas as pd
import numpy as np

# deliverable 1.1
def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    
    c = Counter()
    res = text.split()
    
    for i in res:
        c[i] += 1
    
    return c

# deliverable 1.2
def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    counts = Counter()
    
    for i in bags_of_words:
        counts += i
    
    return counts

# deliverable 1.3
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2a

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    
    #convert counter to set
    bow1 = set(bow1)
    bow2 = set(bow2)
    
    return bow1.difference(bow2)
    

# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    
    #declare list for new_target_data
    new_target_data = []
    
    #returns new counter of training counts thresholded by min_counts
    new_training_counts = set([i for i in training_counts if training_counts[i] >= min_counts])
    

    #loops through to return pruned target_data & unique vocab
    for i in target_data:
        new_target_data.append(Counter({j:i[j] for j in i if j in new_training_counts}))
        
    #getting set of words in pruned vocabulary
    vocab = set([(i, training_counts[i]) for i in new_training_counts])
    
    return new_target_data, vocab


# deliverable 5.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''
    vocab = sorted(vocab)

    matrix = np.zeros((len(bags_of_words), len(vocab)))
    
    for i, j in enumerate(bags_of_words):
        for (k, v) in enumerate(vocab):
            matrix[i, k] = j[v[0]]
    
    return matrix


### helper code

def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())
