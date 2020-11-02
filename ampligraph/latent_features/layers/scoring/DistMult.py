import tensorflow as tf
from ampligraph.latent_features.layers.scoring import register_layer, AbstractScoringLayer


@register_layer('DistMult')
class DistMult(AbstractScoringLayer):
    ''' DistMult scoring Layer class
    '''
    def __init__(self, k):
        super(DistMult, self).__init__(k)

    def _compute_scores(self, triples):
        ''' compute scores using distmult scoring function.
        
        Parameters:
        -----------
        triples: (n, 3)
            batch of input triples
        
        Returns:
        --------
        scores: 
            tensor of scores of inputs
        '''
        # compute scores as sum(s * p * o)
        scores = tf.reduce_sum(triples[0] * triples[1] * triples[2], 1)
        return scores

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
        rel_emb, obj_emb = triples[1], triples[2]
        # compute the score by broadcasting the corruption embeddings(ent_matrix) and using the scoring function
        # compute scores as sum(s_corr * p * o)
        sub_corr_score = tf.reduce_sum(ent_matrix * tf.expand_dims(rel_emb * obj_emb, 1), 2)
        return sub_corr_score

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
        sub_emb, rel_emb = triples[0], triples[1]
        # compute the score by broadcasting the corruption embeddings(ent_matrix) and using the scoring function
        # compute scores as sum(s * p * o_corr)
        obj_corr_score = tf.reduce_sum(tf.expand_dims(sub_emb * rel_emb, 1) * ent_matrix, 2)
        return obj_corr_score
