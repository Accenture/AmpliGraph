import tensorflow as tf


class CorruptionGenerationLayerTrain(tf.keras.layers.Layer):

    def __init__(self, eta, ent_size, **kwargs):
        self.eta=eta
        self.ent_size = ent_size
        super(CorruptionGenerationLayerTrain, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(CorruptionGenerationLayerTrain, self).build(input_shapes)  

    @tf.function
    def call(self, pos):

        batch_size = tf.shape(pos)[0]

        dataset = tf.reshape(tf.tile(tf.reshape(pos, [-1]), [self.eta]), [tf.shape(input=pos)[0] * self.eta, 3])
        keep_subj_mask = tf.tile(tf.cast(tf.random.uniform([tf.shape(input=pos)[0]], 0, 2, dtype=tf.int32, seed=0), tf.bool),
                                     [self.eta])
        keep_obj_mask = tf.logical_not(keep_subj_mask)
        keep_subj_mask = tf.cast(keep_subj_mask, tf.int32)
        keep_obj_mask = tf.cast(keep_obj_mask, tf.int32)
        replacements = tf.random.uniform([tf.shape(dataset)[0]], 0, self.ent_size, dtype=tf.int32, seed=0)
        subjects = tf.math.add(tf.math.multiply(keep_subj_mask, dataset[:, 0]),
                               tf.math.multiply(keep_obj_mask, replacements))
        relationships = dataset[:, 1]
        objects = tf.math.add(tf.math.multiply(keep_obj_mask, dataset[:, 2]),
                              tf.math.multiply(keep_subj_mask, replacements))
        corruptions = tf.transpose(a=tf.stack([subjects, relationships, objects]))
        return corruptions
        

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        batch_size, _ = input_shape
        return [batch_size * self.eta, 3]