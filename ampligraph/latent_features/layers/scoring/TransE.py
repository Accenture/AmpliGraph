import tensorflow as tf

@register_layer('TransE')
class TransE(AbstractScoringLayer):
    ''' TransE scoring Layer class
    '''
    def __init__(self, k):
        super(TransE, self).__init__(k)
    
    @tf.function
    def _compute_scores(self, triples):
        ''' compute scores using transe scoring function.
        
        Parameters:
        -----------
        triples: (n, 3)
            batch of input triples
        
        Returns:
        --------
        scores: 
            tensor of scores of inputs
        '''
        # compute scores as -|| s + p - o|| 
        scores = tf.negative(tf.norm(triples[0] + triples[1] - triples[2], axis=1))
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
        # get the subject, predicate and object embeddings of True positives
        sub_emb, rel_emb, obj_emb = triples[0], triples[1], triples[2]
        # compute the score by broadcasting the corruption embeddings(ent_matrix) and using the scoring function
        # compute scores as -|| s_corr + p - o|| 
        sub_corr_score = tf.negative(tf.norm(ent_matrix + tf.expand_dims( rel_emb - obj_emb, 1) , axis=2))
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
        # get the subject, predicate and object embeddings of True positives:
        sub_emb, rel_emb, obj_emb = triples[0], triples[1], triples[2]
        # compute the score by broadcasting the corruption embeddings(ent_matrix) and using the scoring function
        # compute scores as -|| s + p - o_corr|| 
        obj_corr_score = tf.negative(tf.norm(tf.expand_dims(sub_emb + rel_emb, 1) - ent_matrix, axis=2))
        return obj_corr_score
