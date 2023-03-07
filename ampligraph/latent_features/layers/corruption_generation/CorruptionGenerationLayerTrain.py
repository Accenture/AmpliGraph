# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf


class CorruptionGenerationLayerTrain(tf.keras.layers.Layer):
    """Generates corruptions during training.

    The corruption might involve either subject or object using
    entities sampled uniformly at random from the loaded graph.
    """

    def get_config(self):
        config = super(CorruptionGenerationLayerTrain, self).get_config()
        config.update({"seed": self.seed})
        return config

    def __init__(self, seed=0, **kwargs):
        """
        Initializes the corruption generation layer.

        Parameters
        ----------
        eta: int
            Number of corruptions to generate.
        """
        self.seed = seed
        super(CorruptionGenerationLayerTrain, self).__init__(**kwargs)

    def call(self, pos, ent_size, eta):
        """
        Generates corruption for the positives supplied.

        Parameters
        ----------
        pos: array-like, shape (n, 3)
            Batch of input triples (positives).
        ent_size: int
            Number of unique entities present in the partition.

        Returns
        -------
        corruptions: array-like, shape (n * eta, 3)
            Corruptions of the triples.
        """
        # size and reshape the dataset to sample corruptions
        dataset = tf.reshape(
            tf.tile(tf.reshape(pos, [-1]), [eta]),
            [tf.shape(input=pos)[0] * eta, 3],
        )
        # generate a mask which will tell which subject needs to be corrupted
        # (random uniform sampling)
        keep_subj_mask = tf.cast(
            tf.random.uniform(
                [tf.shape(input=dataset)[0]],
                0,
                2,
                dtype=tf.int32,
                seed=self.seed,
            ),
            tf.bool,
        )
        # If we are not corrupting the subject then corrupt the object
        keep_obj_mask = tf.logical_not(keep_subj_mask)

        # cast it to integer (0/1)
        keep_subj_mask = tf.cast(keep_subj_mask, tf.int32)
        keep_obj_mask = tf.cast(keep_obj_mask, tf.int32)
        # generate the n * eta replacements (uniformly randomly)
        replacements = tf.random.uniform(
            [tf.shape(dataset)[0]], 0, ent_size, dtype=tf.int32, seed=self.seed
        )
        # keep subjects of dataset where keep_subject is 1 and zero it where keep_subject is 0
        # now add replacements where keep_subject is 0 (i.e. keep_object is 1)
        subjects = tf.math.add(
            tf.math.multiply(keep_subj_mask, dataset[:, 0]),
            tf.math.multiply(keep_obj_mask, replacements),
        )
        # keep relations as it is
        relationships = dataset[:, 1]
        # keep objects of dataset where keep_object is 1 and zero it where keep_object is 0
        # now add replacements where keep_object is 0 (i.e. keep_subject is 1)
        objects = tf.math.add(
            tf.math.multiply(keep_obj_mask, dataset[:, 2]),
            tf.math.multiply(keep_subj_mask, replacements),
        )
        # stack the generated subject, reln and object entities and create the
        # corruptions
        corruptions = tf.transpose(
            a=tf.stack([subjects, relationships, objects])
        )
        return corruptions
