import tensorflow as tf
from .AbstractScoringLayer import AbstractScoringLayer, register_layer

@register_layer('TransE')
class TransE(AbstractScoringLayer):

    def __init__(self, k):
        super(TransE, self).__init__(k)
    
    @tf.function
    def _compute_scores(self, triples):
        scores = tf.negative(tf.norm(triples[0] + triples[1] - triples[2], axis=1))
        return scores
    
    @tf.function(experimental_relax_shapes=True)
    def _get_subject_corruption_scores(self, triples, ent_matrix):
        sub_emb, rel_emb, obj_emb = triples[0], triples[1], triples[2]
        sub_corr_score = tf.negative(tf.norm(ent_matrix + tf.expand_dims( rel_emb - obj_emb, 1) , axis=2))
        return sub_corr_score
    
    @tf.function(experimental_relax_shapes=True)
    def _get_object_corruption_scores(self, triples, ent_matrix):
        sub_emb, rel_emb, obj_emb = triples[0], triples[1], triples[2]
        obj_corr_score = tf.negative(tf.norm(tf.expand_dims(sub_emb + rel_emb, 1) - ent_matrix, axis=2))
        return obj_corr_score
    
    
    