import tensorflow as tf
from .AbstractScoringLayer import AbstractScoringLayer, register_layer

@register_layer('ComplEx')
class ComplEx(AbstractScoringLayer):
    ''' Complex scoring Layer class
    '''
    
    def __init__(self, k):
        super(ComplEx, self).__init__(k)
        # internally complex uses k embedddings for real part and k embedddings for img part
        # hence internally it uses 2 * k embeddings
        self.internal_k = 2 * k

    @tf.function
    def _compute_scores(self, triples):
        ''' compute scores using ComplEx scoring function.
        
        Parameters:
        -----------
        triples: (n, 3)
            batch of input triples
        
        Returns:
        --------
        scores: 
            tensor of scores of inputs
        '''
        # split the embeddings of s, p, o into 2 parts (real and img part)
        e_s_real, e_s_img = tf.split(triples[0], 2, axis=1)
        e_p_real, e_p_img = tf.split(triples[1], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)

        # apply the complex scoring function
        scores = tf.reduce_sum(e_s_real * (e_p_real * e_o_real + 
                                           e_p_img * e_o_img) + \
                                e_s_img * (e_p_real * e_o_img - 
                                           e_p_img * e_o_real), axis=1)
        return scores
    
    @tf.function(experimental_relax_shapes=True)
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
        # split the embeddings of s, p, o into 2 parts (real and img part)
        e_s_real, e_s_img = tf.split(triples[0], 2, axis=1)
        e_p_real, e_p_img = tf.split(triples[1], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)
        
        # split the corruption entity embeddings into 2 parts (real and img part)
        ent_real, ent_img = tf.split(ent_matrix, 2, axis=1)
        
        # compute the subject corruption score using ent_real, ent_img (corruption embeddings) as subject embeddings
        sub_corr_score = tf.reduce_sum(
                            ent_real * (tf.expand_dims(e_p_real * e_o_real, 1) + 
                                        tf.expand_dims(e_p_img * e_o_img, 1)) + \
                            ent_img * (tf.expand_dims(e_p_real * e_o_img, 1) - \
                                       tf.expand_dims(e_p_img * e_o_real, 1)), axis=2)
                                          
        return sub_corr_score
    
    @tf.function(experimental_relax_shapes=True)
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
        # split the embeddings of s, p, o into 2 parts (real and img part)
        e_s_real, e_s_img = tf.split(triples[0], 2, axis=1)
        e_p_real, e_p_img = tf.split(triples[1], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)
        
        # split the corruption entity embeddings into 2 parts (real and img part)
        ent_real, ent_img = tf.split(ent_matrix, 2, axis=1)
        
        # compute the object corruption score using ent_real, ent_img (corruption embeddings) as object embeddings
        obj_corr_score = tf.reduce_sum(
                            (tf.expand_dims(e_s_real * e_p_real, 1) - \
                             tf.expand_dims(e_s_img * e_p_img, 1) ) * ent_real + \
                            (tf.expand_dims(e_s_img * e_p_real, 1) + \
                             tf.expand_dims(e_s_real * e_p_img, 1)) * ent_img , axis=2) 
                                          
        return obj_corr_score
