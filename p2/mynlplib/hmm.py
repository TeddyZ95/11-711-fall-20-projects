from mynlplib.preproc import conll_seq_generator
from mynlplib.constants import START_TAG, END_TAG, OFFSET, UNK
from mynlplib import naive_bayes, most_common 
import numpy as np
from collections import defaultdict
import torch
import torch.nn
from torch.autograd import Variable


# Deliverable 4.2
def compute_transition_weights(trans_counts, smoothing):
    """
    Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag)] and weights

    """
    
    weights = defaultdict(float)
    
    all_tags = list(trans_counts.keys())+ [END_TAG]
    
    for i in trans_counts:
        tran = trans_counts[i]
        total = len(list(tran.elements()))
        for tag in all_tags:
            if tag != START_TAG:
                weights[(tag, i)] = np.log(tran[tag] + smoothing) - np.log(total + (len(all_tags) * smoothing))
            else:
                weights[(tag, i)] = -np.inf
    
    
    
    return weights


# Deliverable 3.2
def compute_weights_variables(nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix):
    """
    Computes autograd Variables of two weights: emission_probabilities and the tag_transition_probabilties
    parameters:
    nb_weights: -- a dictionary of emission weights
    hmm_trans_weights: -- dictionary of tag transition weights
    vocab: -- list of all the words
    word_to_ix: -- a dictionary that maps each word in the vocab to a unique index
    tag_to_ix: -- a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    
    :returns:
    emission_probs_vr: torch Variable of a matrix of size Vocab x Tagset_size
    tag_transition_probs_vr: torch Variable of a matrix of size Tagset_size x Tagset_size
    :rtype: autograd Variables of the the weights
    """
    # Assume that tag_to_ix includes both START_TAG and END_TAG
    
    tag_transition_probs = np.full((len(tag_to_ix), len(tag_to_ix)), -np.inf)
    emission_probs = np.full((len(vocab),len(tag_to_ix)), 0.0)
    
    for tag_i in tag_to_ix:
        for tag_j in tag_to_ix:
                tag_transition_probs[tag_to_ix[tag_i], tag_to_ix[tag_j]] = hmm_trans_weights[(tag_i, tag_j)] if (tag_i, tag_j) in hmm_trans_weights else -np.inf
    
    for word in word_to_ix:
        for tag_i in tag_to_ix:
            default_val = 0
            if tag_i == START_TAG or tag_i == END_TAG:
                default_val = -np.inf
            emission_probs[word_to_ix[word]][tag_to_ix[tag_i]] = nb_weights[(tag_i, word)] if (tag_i, word) in nb_weights else default_val    
    
    emission_probs_vr = Variable(torch.from_numpy(emission_probs.astype(np.float32)))
    tag_transition_probs_vr = Variable(torch.from_numpy(tag_transition_probs.astype(np.float32)))
    
    return emission_probs_vr, tag_transition_probs_vr
