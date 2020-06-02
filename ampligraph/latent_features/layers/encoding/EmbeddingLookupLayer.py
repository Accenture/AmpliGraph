import tensorflow as tf


class EmbeddingLookupLayer(tf.keras.layers.Layer):

    def __init__(self, ent_size, rel_size, k, num_partitions=1, **kwargs):
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.k = k
        super(EmbeddingLookupLayer, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.ent_emb = tf.keras.layers.Embedding(self.ent_size, self.k, name='ent_emb_layer_1')
        self.rel_emb = tf.keras.layers.Embedding(self.rel_size, self.k, name='rel_emb_layer_1')
        super(EmbeddingLookupLayer, self).build(input_shapes)  
        
    @tf.function
    def call(self, x):
        e_s = self.ent_emb(x[:, 0])
        e_p = self.rel_emb(x[:, 1])
        e_o = self.ent_emb(x[:, 2])
        return [e_s, e_p, e_o]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        batch_size, _ = input_shape
        return [(batch_size, self.k), (batch_size, self.k), (batch_size, self.k)]
    
    