import tensorflow as tf
from .AbstractScoringLayer import AbstractScoringLayer, register_layer

@register_layer('DistMult')
class DistMult(AbstractScoringLayer):
    
    def __init__(self, k):
        super(DistMult, self).__init__(k)

    @tf.function
    def compute_scores(self, triples):
        scores = tf.reduce_sum(triples[0] * triples[1] * triples[2], 1)
        return scores
    
    @tf.function(experimental_relax_shapes=True)
    def get_subject_corruption_scores(self, triples, ent_matrix):
        sub_emb, rel_emb, obj_emb = triples[0], triples[1], triples[2]
        sub_corr_score = tf.reduce_sum(ent_matrix * tf.expand_dims( rel_emb * obj_emb, 1) , 2)
        return sub_corr_score
    
    @tf.function(experimental_relax_shapes=True)
    def get_object_corruption_scores(self, triples, ent_matrix):
        sub_emb, rel_emb, obj_emb = triples[0], triples[1], triples[2]
        obj_corr_score = tf.reduce_sum(tf.expand_dims(sub_emb * rel_emb, 1) * ent_matrix, 2)
        return obj_corr_score

