import tensorflow as tf


class DistMult(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(DistMult, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(DistMult, self).build(input_shapes)

    @tf.function
    def call(self, triples):
        scores = tf.reduce_sum(triples[0] * triples[1] * triples[2], 1)
        return scores

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        batch_size, _ = input_shape
        return [batch_size, 1]