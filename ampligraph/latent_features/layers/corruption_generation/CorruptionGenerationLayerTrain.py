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

    def __init__(
            self,
            eta,
            ht_strategy="uniform",
            ent_strategy="uniform",
            seed=0,
            **kwargs
    ):
        """
        Initializes the corruption generation layer.

        Parameters
        ----------
        eta: int
            Number of corruptions to generate.
        ht_strategy: str
            Specifies the strategy to apply in order to pick whether to corrupt the subject (head)
            or the object (tail) when generating negatives. When set to "uniform", head and
            tail have the same probability of being corrupted (default: `"uniform"`).
        ent_strategy: str
            Specifies the strategy to apply in order to randomly sample the entities to use to
            corrupt true triples to generate negatives. When set to "uniform", all the entities
            have the same probability of being picked as corruptions (default: `"uniform"`).
        """
        self.seed = seed
        self.eta = eta

        self.ht_strategy = ht_strategy
        self.ent_strategy = ent_strategy

        self.degree = None
        self.ent_log_probability = None

        super(CorruptionGenerationLayerTrain, self).__init__(**kwargs)


    def set_properties(self, degrees_lst):
        """
        Build the layer passing a list where each element corresponds to the degree
        of the entity with numeric identifier corresponding to its index.

        Using this dictionary we can build relevant quantities in case we need to implement
        alternative negative sampling strategies.

        degrees_lst: list
            List where the degree of each entity is at the index corresponding to the
            numerical identifier of that entity.
        """
        self.degree = tf.constant(degrees_lst, dtype=tf.float32)

        if self.ent_strategy == "inverse_degree":
            # the log-probability of each entity is defined based on the inverse of the degree.
            # probability = (1 / self.degree) / tf.reduce_sum(1 / self.degree)
            self.ent_log_probability = - tf.math.log(self.degree) + tf.math.log(tf.reduce_sum(self.degree))

    def call(self, pos, ent_size, eta=None, ht_probs=None, return_mask=False):
        """
        Generates corruption for the positives supplied.

        Parameters
        ----------
        pos: array-like, shape (n, 3)
            Batch of input triples (positives).
        ent_size: int
            Number of unique entities present in the partition.
        eta: int
            Custom number of negatives in case a number different from
            `self.eta` is needed (e.g., calibration).

        Returns
        -------
        corruptions: array-like, shape (n * eta, 3)
            Corruptions of the triples.
        """
        # size and reshape the dataset to sample corruptions
        eta = self.eta if eta is None else eta
        dataset = tf.tile(pos, [eta, 1])
        # generate a mask which will tell which subject needs to be corrupted
        # (random uniform sampling)

        if self.ht_strategy == "uniform":
            keep_subj_mask = tf.cast(
                tf.random.uniform(
                    [tf.shape(input=dataset)[0]],
                    0, 2, dtype=tf.int32,
                    seed=self.seed
                ),
                tf.bool,
            )
        else:
            log_probs = tf.math.log(
                tf.tile(
                    tf.concat([ht_probs, 1 - ht_probs], axis=1),
                    [eta, 1]
                )
            )
            keep_subj_mask = tf.cast(
                tf.squeeze(
                    tf.random.categorical(
                        log_probs, num_samples=1,
                        dtype=tf.int32, seed=self.seed
                    )
                ), tf.bool
            )
        # If we are not corrupting the subject then corrupt the object
        keep_obj_mask = tf.logical_not(keep_subj_mask)

        # cast it to integer (0/1)
        keep_subj_mask = tf.cast(keep_subj_mask, tf.int32)
        keep_obj_mask = tf.cast(keep_obj_mask, tf.int32)
        # generate the n * eta replacements (uniformly randomly)
        if self.ent_strategy == "uniform":
            replacements = tf.random.uniform(
                [tf.shape(dataset)[0]], 0, ent_size, dtype=tf.int32, seed=self.seed
            )
        else:
            replacements = tf.squeeze(
                tf.random.categorical(
                    self.ent_log_probability[None, :],
                    num_samples=tf.shape(dataset)[0],
                    dtype=tf.int32, seed=self.seed
                )
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
        if return_mask:
            return corruptions, tf.cast(keep_subj_mask, tf.float32)
        return corruptions
