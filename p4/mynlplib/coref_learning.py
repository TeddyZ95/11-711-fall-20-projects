import torch
from torch import nn
from torch import autograd as ag
import torch.nn.functional as F

from collections import defaultdict

from . import utils, coref

class FFCoref(nn.Module):
    '''
    A component that scores coreference relations based on a one-hot feature vector
    Architecture: input features -> Linear layer -> tanh -> Linear layer -> score
    '''
    
    ## deliverable 3.2
    def __init__(self, feat_names, hidden_dim):
        '''
        :param feat_names: list of keys to possible pairwise matching features
        :param hidden_dim: dimension of intermediate layer
        '''
        super(FFCoref, self).__init__()
        
        # STUDENT

        self.feat_names = feat_names
        self.net = nn.Sequential(
            nn.Linear(len(self.feat_names), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1))
        
        # END STUDENT
        
        
    ## deliverable 3.2
    def forward(self, features):
        '''
        :param features: defaultdict of pairwise matching features and their values for some pair
        :returns: model score
        :rtype: 1x1 torch Variable
        '''
        return self.net(ag.Variable(torch.FloatTensor([features[feature] for feature in self.feat_names])))

        
    ## deliverable 3.3
    def score_instance(self, markables, feats, i):
        '''
        A function scoring all coref candidates for a given markable
        Don't forget the new-entity option!
        :param markables: list of all markables in the document
        :param i: index of current markable
        :param feats: feature extraction function
        :returns: list of scores for all candidates
        :rtype: torch.FloatTensor of dimensions 1x(i+1)
        '''
        scores = ag.Variable(torch.FloatTensor(1, i + 1))

        for ant_index, ant in enumerate(markables[:i + 1]):
            feature = feats(markables, ant_index, i)
            score_var = self.forward(feature)
            scores[0, ant_index] = score_var[0]

        return scores

    ## deliverable 3.4
    def instance_top_scores(self, markables, feats, i, true_antecedent):
        '''
        Find the top-scoring true and false candidates for i in the markable.
        If no false candidates exist, return (None, None).
        :param markables: list of all markables in the document
        :param i: index of current markable
        :param true_antecedent: gold label for markable
        :param feats: feature extraction function
        :returns trues_max: best-scoring true antecedent
        :returns false_max: best-scoring false antecedent
        '''
        if i == 0 or i == true_antecedent:
            return None, None
        else:
            scores = self.score_instance(markables, feats, i)

            all_trues_indices = torch.LongTensor(
                [index for index in range(0, i) if markables[index].entity == markables[i].entity])
            all_false_indices = torch.LongTensor(
                [index for index in range(0, i) if markables[index].entity != markables[i].entity])

            if not all_trues_indices.shape:
                all_trues_indices = torch.LongTensor([i])

            if all_trues_indices.shape[0] == i:
                return None, None

            zero_tensor = torch.LongTensor([0])
            trues_max_val = torch.max(scores[zero_tensor, all_trues_indices])
            false_max_val = torch.max(scores[zero_tensor, all_false_indices])
            return trues_max_val, false_max_val


def train(model, optimizer, markable_set, feats, margin=1.0, epochs=2):
    _zero = ag.Variable(torch.Tensor([0])) # this var is reusable
    model.train()
    for i in range(epochs):
        tot_loss = 0.0
        instances = 0
        for doc in markable_set:
            true_ants = coref.get_true_antecedents(doc)
            for i in range(len(doc)):
                optimizer.zero_grad()
                max_t, max_f = model.instance_top_scores(doc, feats, i, true_ants[i])
                if max_t is None: continue
                unhinged_loss = -max_t + max_f + margin
                loss = F.relu(unhinged_loss)
                tot_loss += utils.to_scalar(loss)
                loss.backward()
                optimizer.step()
                instances += 1
        print(f'Loss = {tot_loss / instances}')
        
def evaluate(model, markable_set, feats):
    model.eval()
    coref.eval_on_dataset(make_resolver(feats, model), markable_set)
    
# helper
def make_resolver(features, model):
    return lambda markables : [utils.argmax(model.score_instance(markables, features, i)) for i in range(len(markables))]