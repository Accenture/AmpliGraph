import tensorflow as tf

COMPARISION_PRECISION = 1e3

SCORING_LAYER_REGISTRY = {}

def register_layer(name, external_params=None, class_params=None):
    if external_params is None:
        external_params = []
    if class_params is None:
        class_params = {}

    def insert_in_registry(class_handle):
        SCORING_LAYER_REGISTRY[name] = class_handle
        class_handle.name = name
        SCORING_LAYER_REGISTRY[name].external_params = external_params
        SCORING_LAYER_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry


class AbstractScoringLayer(tf.keras.layers.Layer):
    
    def __init__(self, k):
        super(AbstractScoringLayer, self).__init__()
        self.internal_k = k

    def build(self, input_shapes):
        super(AbstractScoringLayer, self).build(input_shapes)

    @tf.function
    def call(self, triples):
        return self.compute_scores(triples)
    
    @tf.function
    def compute_scores(self, triples):
        raise NotImplementedError('Abstract method not implemented!')
        
    @tf.function(experimental_relax_shapes=True)
    def get_object_corruption_scores(self, triples, ent_matrix):
        raise NotImplementedError('Abstract method not implemented!')
        
    @tf.function(experimental_relax_shapes=True)
    def get_subject_corruption_scores(self, triples, ent_matrix):
        raise NotImplementedError('Abstract method not implemented!')

    @tf.function(experimental_relax_shapes=True)
    def get_ranks(self, triples, ent_matrix):
        triple_score = self.compute_scores(triples)
        sub_corr_score = self.get_subject_corruption_scores(triples, ent_matrix)
        obj_corr_score = self.get_object_corruption_scores(triples, ent_matrix)
        
        sub_corr_score = tf.cast(sub_corr_score * COMPARISION_PRECISION, tf.int32)
        obj_corr_score = tf.cast(obj_corr_score * COMPARISION_PRECISION, tf.int32)
        triple_score = tf.cast(triple_score * COMPARISION_PRECISION, tf.int32)
        sub_rank = tf.reduce_sum(tf.cast(tf.expand_dims(triple_score, 1) <= sub_corr_score, tf.int32), 1)
        obj_rank = tf.reduce_sum(tf.cast(tf.expand_dims(triple_score, 1) <= obj_corr_score, tf.int32), 1)
        
        return sub_rank, obj_rank
        
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        batch_size, _ = input_shape
        return [batch_size, 1]