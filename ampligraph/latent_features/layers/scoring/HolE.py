import tensorflow as tf
from ampligraph.latent_features.layers.scoring import register_layer
from .ComplEx import ComplEx


@register_layer('HolE')
class HolE(ComplEx):
    ''' HolE scoring Layer class
    '''
    def __init__(self, k):
        super(HolE, self).__init__(k)

    def _compute_scores(self, triples):
        ''' compute scores using HolE scoring function.
        
        Parameters:
        -----------
        triples: (n, 3)
            batch of input triples
        
        Returns:
        --------
        scores: 
            tensor of scores of inputs
        '''
        # HolE scoring is 2/k * complex_score
        return (2 / (self.internal_k / 2)) * (super().compute_scores(triples))

    def _get_subject_corruption_scores(self, triples, ent_matrix):
        ''' Compute subject corruption scores.
        Evaluate the inputs against subject corruptions and scores of the corruptions.
        
        Parameters:
        -----------
        triples: (n, k)
            batch of input embeddings
        ent_matrix: (m, k)
            slice of embedding matrix (corruptions)
        
        Returns:
        --------
        scores: (n, 1)
            scores of subject corruptions (corruptions defined by ent_embs matrix)
        '''
        # HolE scoring is 2/k * complex_score
        return (2 / (self.internal_k / 2)) * (super()._get_subject_corruption_scores(triples, 
                                                                                     ent_matrix))

    def _get_object_corruption_scores(self, triples, ent_matrix):
        ''' Compute object corruption scores.
        Evaluate the inputs against object corruptions and scores of the corruptions.
        
        Parameters:
        -----------
        triples: (n, k)
            batch of input embeddings
        ent_matrix: (m, k)
            slice of embedding matrix (corruptions)
        
        Returns:
        --------
        scores: (n, 1)
            scores of object corruptions (corruptions defined by ent_embs matrix)
        '''
        # HolE scoring is 2/k * complex_score
        return (2 / (self.internal_k / 2)) * (super()._get_object_corruption_scores(triples,  
                                                                                    ent_matrix))
