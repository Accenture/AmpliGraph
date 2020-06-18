import tensorflow as tf


class EmbeddingLookupLayer(tf.keras.layers.Layer):

    def __init__(self, max_ent_size, max_rel_size, k, num_partitions=1, **kwargs):
        self.max_ent_size = max_ent_size
        self.max_rel_size = max_rel_size
        self.k = k
#        super(EmbeddingLookupLayer, self).__init__(**kwargs)

#    def build(self, input_shapes):
        self.ent_emb = tf.Variable(tf.initializers.GlorotUniform()(shape=[self.max_ent_size, self.k]), name='ent_emb_layer_1', trainable=True)
        self.rel_emb = tf.Variable(tf.initializers.GlorotUniform()(shape=[self.max_rel_size, self.k]), name='rel_emb_layer_1', trainable=True)
        #self.ent_emb = tf.Variable(shape=[None, self.k], name='ent_emb_layer_1')
        #self.rel_emb = tf.Variable(shape=[None, self.k], name='ent_emb_layer_1')
        
        super(EmbeddingLookupLayer, self).__init__(**kwargs)

#        super(EmbeddingLookupLayer, self).build(input_shapes)  
        
        
    @tf.function
    def partition_change_updates(self, batch_ent_emb, batch_rel_emb):
        paddings_ent = tf.constant([[0, self.max_ent_size - batch_ent_emb.shape[0]], [0, 0]])
        paddings_rel = tf.constant([[0, self.max_rel_size - batch_rel_emb.shape[0]], [0, 0]])

        self.ent_emb.assign(tf.pad(batch_ent_emb, paddings_ent, 'CONSTANT', constant_values=0))
        self.rel_emb.assign(tf.pad(batch_rel_emb, paddings_rel, 'CONSTANT', constant_values=0))
        
    @tf.function
    def call(self, x):
        e_s = tf.nn.embedding_lookup(self.ent_emb, x[:, 0])
        e_p = tf.nn.embedding_lookup(self.rel_emb, x[:, 1])
        e_o = tf.nn.embedding_lookup(self.ent_emb, x[:, 2])
        return [e_s, e_p, e_o]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        batch_size, _ = input_shape
        return [(batch_size, self.k), (batch_size, self.k), (batch_size, self.k)]
    
    