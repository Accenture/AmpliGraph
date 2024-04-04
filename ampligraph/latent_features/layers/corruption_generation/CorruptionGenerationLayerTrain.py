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
            seed=0,
            ontology_sampling=None,
            ontology_sampling_config=None,
            **kwargs
    ):
        """
        Initializes the corruption generation layer.

        Parameters
        ----------
        eta: int
            Number of corruptions to generate.
        """
        self.seed = seed
        self.ontology_sampling = ontology_sampling
        self.idx_class_domain_start = None
        self.idx_class_domain_end = None
        self.idx_class_range_start = None
        self.idx_class_range_end = None
        self.relation_to_domain = None
        self.relation_to_range = None

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
        # if not self.ontology_sampling:
        #     dataset = tf.tile(pos, [eta, 1])
        # else:
        #     number_good_negatives = int(eta * self.ontology_sampling)
        #     dataset = tf.tile(pos, [number_good_negatives, 1])
        dataset = tf.tile(pos, [eta, 1])
        # generate a mask which will tell which subject needs to be corrupted
        # (random uniform sampling)
        keep_subj_mask = tf.cast(
            tf.random.uniform(
                [tf.shape(input=dataset)[0]],
                0, 2,
                dtype=tf.int32, seed=self.seed,
            ),
            tf.bool,
        )
        # If we are not corrupting the subject then corrupt the object
        keep_obj_mask = tf.logical_not(keep_subj_mask)

        # cast it to integer (0/1)
        keep_subj_mask = tf.cast(keep_subj_mask, tf.int32)
        keep_obj_mask = tf.cast(keep_obj_mask, tf.int32)
        # generate the n * eta replacements (uniformly randomly)
        if not self.ontology_sampling:
            replacements = tf.random.uniform(
                [tf.shape(dataset)[0]], 0, ent_size, dtype=tf.int32, seed=self.seed
            )
        else:
            # number of pertinent negatives to generate
            number_good_negatives = int(eta * self.ontology_sampling)
            # we extract the start and end index of classes in self.relation_to_domain
            start_idx_domain = tf.gather(self.idx_class_domain_start, dataset[:, 1])
            end_idx_domain = tf.gather(self.idx_class_domain_end, dataset[:, 1])
            # we now sample the indices to slice the vector of entities divided by class
            idx_domain_replacements = tf.cast(
                tf.floor(
                    tf.random.uniform(
                        [tf.shape(pos)[0] * number_good_negatives], start_idx_domain, end_idx_domain,
                        seed=self.seed)
                ), dtype=tf.int32
            )
            # we use the index to obtain the corruptions we are looking for
            subj_replacements = tf.gather(
                self.relation_to_domain,
                idx_domain_replacements
            )

            # we repeat the same process as above for the objects
            start_idx_range = tf.gather(self.idx_class_range_start, dataset[:, 1])
            end_idx_range = tf.gather(self.idx_class_range_end, dataset[:, 1])
            idx_range_replacements = tf.cast(
                tf.floor(
                    tf.random.uniform(
                        [tf.shape(pos)[0] * number_good_negatives], start_idx_range, end_idx_range,
                        seed=self.seed)
                ), dtype=tf.int32
            )
            obj_replacements = tf.gather(
                self.relation_to_range,
                idx_range_replacements
            )

            # we put in one unique vector the replacements for subject and object
            replacements = subj_replacements * keep_obj_mask + obj_replacements * keep_subj_mask

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
