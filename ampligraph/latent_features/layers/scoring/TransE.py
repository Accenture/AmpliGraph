import tensorflow as tf


class TransE(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(TransE, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(TransE, self).build(input_shapes)  # Be sure to call this at the end

    @tf.function
    def call(self, triples):
        scores = tf.negative(tf.norm(triples[0] + triples[1] - triples[2], axis=1))
        return scores

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        batch_size, _ = input_shape
        return [batch_size, 1]