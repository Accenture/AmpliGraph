import tensorflow as tf


class CorruptionGenerationLayerEval(tf.keras.layers.Layer):

    def __init__(self, eta, ent_size, **kwargs):
        self.ent_size = ent_size
        super(CorruptionGenerationLayerEval, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(CorruptionGenerationLayerEval, self).build(input_shapes)  # Be sure to call this at the end

    @tf.function
    def call(self, X):
        print('creating eval--------------------------------')
        corrupt_side = 's+o'
        entities_for_corruption =  tf.range(self.ent_size)
        if corrupt_side in ['s+o', 'o']:  # object is corrupted - so we need subjects as it is
            repeated_subjs = tf.keras.backend.repeat(
                tf.slice(X,
                         [0, 0],  # subj
                         [tf.shape(X)[0], 1]),
                tf.shape(entities_for_corruption)[0])
            repeated_subjs = tf.squeeze(repeated_subjs, 2)

        if corrupt_side in ['s+o', 's']:  # subject is corrupted - so we need objects as it is
            repeated_objs = tf.keras.backend.repeat(
                tf.slice(X,
                         [0, 2],  # Obj
                         [tf.shape(X)[0], 1]),
                tf.shape(entities_for_corruption)[0])
            repeated_objs = tf.squeeze(repeated_objs, 2)

        repeated_relns = tf.keras.backend.repeat(
            tf.slice(X,
                     [0, 1],  # reln
                     [tf.shape(X)[0], 1]),
            tf.shape(entities_for_corruption)[0])
        repeated_relns = tf.squeeze(repeated_relns, 2)

        rep_ent = tf.keras.backend.repeat(tf.expand_dims(entities_for_corruption, 0), tf.shape(X)[0])
        rep_ent = tf.squeeze(rep_ent, 0)

        if corrupt_side == 's+o':
            stacked_out = tf.concat([tf.stack([repeated_subjs, repeated_relns, rep_ent], 1),
                                     tf.stack([rep_ent, repeated_relns, repeated_objs], 1)], 0)

        elif corrupt_side == 'o':
            stacked_out = tf.stack([repeated_subjs, repeated_relns, rep_ent], 1)

        else:
            stacked_out = tf.stack([rep_ent, repeated_relns, repeated_objs], 1)

        corruptions = tf.reshape(tf.transpose(stacked_out, [0, 2, 1]), (-1, 3))

        return corruptions
        
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        batch_size, _ = input_shape
        return [self.eta, 3]