import tensorflow as tf
# Precision for floating point comparision
COMPARISION_PRECISION = 1e3

# Scoring layer registry. Every scoring function must be registered in this registry.
SCORING_LAYER_REGISTRY = {}


def register_layer(name, external_params=None, class_params=None):
    '''register the scoring function using this decorator
    
    Parameters:
    -----------
    name: string
        name of the scoring function to be used to register the class
    external_params: list of strings
        if there are any scoring function hyperparams, register their names
    class_params: dict 
        things that may be used internally across various models
    '''
    if external_params is None:
        external_params = []
    if class_params is None:
        class_params = {}

    def insert_in_registry(class_handle):
        assert name not in SCORING_LAYER_REGISTRY.keys(), "Scoring Layer with name {} "
        "already exists!".format(name)
        
        # store the class handle in the registry with name as key
        SCORING_LAYER_REGISTRY[name] = class_handle
        # create a class level variable and store the name
        class_handle.name = name
        
        # store other params related to the scoring function in the registry
        # this will be used later during model selection, etc
        SCORING_LAYER_REGISTRY[name].external_params = external_params
        SCORING_LAYER_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry


class AbstractScoringLayer(tf.keras.layers.Layer):
    ''' Abstract class for scoring layer
    '''
    def __init__(self, k):
        '''Initializes the scoring layer
        
        Parameters:
        -----------
        k: int
            embedding size
        '''
        super(AbstractScoringLayer, self).__init__()
        # store the embedding size. (concrete models may overwrite this)
        self.internal_k = k

    def call(self, triples):
        '''
        Computes the scores of the triples
        
        Parameters:
        -----------
        triples: (n, 3)
            batch of input triples
        
        Returns:
        --------
        scores: 
            tensor of scores of inputs
        '''
        return self._compute_scores(triples)
    
    def _compute_scores(self, triples):
        ''' Abstract function to compute scores. Override this method in concrete classes.
        
        Parameters:
        -----------
        triples: (n, 3)
            batch of input triples
        
        Returns:
        --------
        scores: 
            tensor of scores of inputs
        '''
        raise NotImplementedError('Abstract method not implemented!')
        
    def _get_object_corruption_scores(self, triples, ent_matrix):
        ''' Abstract function to compute object corruption scores.
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
        raise NotImplementedError('Abstract method not implemented!')
        
    def _get_subject_corruption_scores(self, triples, ent_matrix):
        ''' Abstract function to compute subject corruption scores.
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
        raise NotImplementedError('Abstract method not implemented!')

    def get_ranks(self, triples, ent_matrix, start_ent_id, end_ent_id, filters, corrupt_side='s,o'):
        ''' Computes the ranks of triples against their corruptions. 
        Ranks are computed by corruptiong triple s and o side by embeddings in ent_matrix.
        
        Parameters:
        -----------
        triples: (n, k)
            batch of input embeddings
        ent_matrix: (m, k)
            slice of embedding matrix (corruptions)
        
        Returns:
        --------
        ranks: (n, 2)
            ranks of triple against subject and object corruptions (corruptions defined by ent_embs matrix)
        '''
        # compute the score of true positives
        triple_score = self._compute_scores(triples)
        # compute the score by corrupting the subject side of triples by ent_matrix
        sub_corr_score = self._get_subject_corruption_scores(triples, ent_matrix)
        # compute the score by corrupting the object side of triples by ent_matrix
        obj_corr_score = self._get_object_corruption_scores(triples, ent_matrix)
        
        # Handle the floating point comparision by multiplying by reqd precision and casting to int
        # before comparing
        sub_corr_score = tf.cast(sub_corr_score * COMPARISION_PRECISION, tf.int32)
        obj_corr_score = tf.cast(obj_corr_score * COMPARISION_PRECISION, tf.int32)
        triple_score = tf.cast(triple_score * COMPARISION_PRECISION, tf.int32)
        
        out_ranks = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        if tf.strings.regex_full_match(corrupt_side, '.*s.*'):
            
            # compare True positive score against their respective corruptions and get rank.
            sub_rank = tf.reduce_sum(tf.cast(tf.expand_dims(triple_score, 1) <= sub_corr_score, tf.int32), 1)

            if filters.shape[0] > 0:
                # tf.print(tf.shape(triple_score)[0])
                for i in tf.range(tf.shape(triple_score)[0]):
                    # TODO change the hard coded filter index
                    # get the ids of True positives that needs to be filtered
                    filter_ids = filters[0][i]
                    # This is done for patritioning (where the full emb matrix is not used)
                    # this gets only the filter ids of the current partition being used for generating corruption
                    filter_ids_selector = tf.logical_and(filter_ids >= start_ent_id, 
                                                         filter_ids <= end_ent_id)
                    filter_ids = tf.boolean_mask(filter_ids, filter_ids_selector)
                    # from entity id convert to index in the current partition
                    filter_ids = filter_ids - start_ent_id

                    # get the score of the corruptions which are actually True positives
                    score_filter = tf.gather(tf.squeeze(tf.gather_nd(sub_corr_score, [[i]])), filter_ids)
                    # check how many of those were ranked higher than the test triple
                    num_filters_ranked_higher = tf.reduce_sum(
                        tf.cast(tf.gather(triple_score, [i]) <= score_filter, tf.int32))
                    # ajust the rank of the test triple accordingly
                    sub_rank = tf.tensor_scatter_nd_sub(sub_rank, [[i]], [num_filters_ranked_higher])
                
            out_ranks = out_ranks.write(out_ranks.size(), sub_rank)
        
        if tf.strings.regex_full_match(corrupt_side, '.*o.*'):
            obj_rank = tf.reduce_sum(tf.cast(tf.expand_dims(triple_score, 1) <= obj_corr_score, tf.int32), 1)
            if filters.shape[0] > 0:
                for i in tf.range(tf.shape(triple_score)[0]):
                    # TODO change the hard coded filter index
                    # get the ids of True positives that needs to be filtered
                    filter_ids = filters[1][i]
                    # This is done for patritioning (where the full emb matrix is not used)
                    # this gets only the filter ids of the current partition being used for generating corruption
                    filter_ids_selector = tf.logical_and(filter_ids >= start_ent_id, 
                                                         filter_ids <= end_ent_id)
                    filter_ids = tf.boolean_mask(filter_ids, filter_ids_selector)
                    # from entity id convert to index in the current partition
                    filter_ids = filter_ids - start_ent_id

                    # get the score of the corruptions which are actually True positives
                    score_filter = tf.gather(tf.squeeze(tf.gather_nd(obj_corr_score, [[i]])), filter_ids)
                    # check how many of those were ranked higher than the test triple
                    num_filters_ranked_higher = tf.reduce_sum(
                        tf.cast(tf.gather(triple_score, [i]) <= score_filter, tf.int32))
                    # ajust the rank of the test triple accordingly
                    obj_rank = tf.tensor_scatter_nd_sub(obj_rank, [[i]], [num_filters_ranked_higher])
                
            out_ranks = out_ranks.write(out_ranks.size(), obj_rank)
            
        out_ranks = out_ranks.stack()
        return out_ranks
        
    def compute_output_shape(self, input_shape):
        ''' returns the output shape of outputs of call function
        
        Parameters:
        -----------
        input_shape: 
            shape of inputs of call function
        
        Returns:
        --------
        output_shape:
            shape of outputs of call function
        '''
        assert isinstance(input_shape, list)
        batch_size, _ = input_shape
        return [batch_size, 1]
