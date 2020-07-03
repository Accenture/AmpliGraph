import tensorflow as tf
from .AbstractScoringLayer import AbstractScoringLayer, register_layer

@register_layer('ComplEx')
class ComplEx(AbstractScoringLayer):
    
    def __init__(self, k):
        super(ComplEx, self).__init__(k)
        self.internal_k = 2 * k

    @tf.function
    def compute_scores(self, triples):
        e_s_real, e_s_img = tf.split(triples[0], 2, axis=1)
        e_p_real, e_p_img = tf.split(triples[1], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)

        scores = tf.reduce_sum(e_s_real * (e_p_real * e_o_real + 
                                           e_p_img * e_o_img) + \
                                e_s_img * (e_p_real * e_o_img - 
                                           e_p_img * e_o_real), axis=1)
        return scores
    
    @tf.function(experimental_relax_shapes=True)
    def get_subject_corruption_scores(self, triples, ent_matrix):
        e_s_real, e_s_img = tf.split(triples[0], 2, axis=1)
        e_p_real, e_p_img = tf.split(triples[1], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)
        
        ent_real, ent_img = tf.split(ent_matrix, 2, axis=1)
        
        sub_corr_score = tf.reduce_sum(
                            ent_real * (tf.expand_dims(e_p_real * e_o_real, 1) + 
                                        tf.expand_dims(e_p_img * e_o_img, 1)) + \
                            ent_img * (tf.expand_dims(e_p_real * e_o_img, 1) - \
                                       tf.expand_dims(e_p_img * e_o_real, 1)), axis=2)
                                          
        return sub_corr_score
    
    @tf.function(experimental_relax_shapes=True)
    def get_object_corruption_scores(self, triples, ent_matrix):
        e_s_real, e_s_img = tf.split(triples[0], 2, axis=1)
        e_p_real, e_p_img = tf.split(triples[1], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)
        
        ent_real, ent_img = tf.split(ent_matrix, 2, axis=1)
        
        obj_corr_score = tf.reduce_sum(
                            (tf.expand_dims(e_s_real * e_p_real, 1) - \
                             tf.expand_dims(e_s_img * e_p_img, 1) ) * ent_real + \
                            (tf.expand_dims(e_s_img * e_p_real, 1) + \
                             tf.expand_dims(e_s_real * e_p_img, 1)) * ent_img , axis=2) 
                                          
        return obj_corr_score
