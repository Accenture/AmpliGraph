# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf
from .AbstractScoringLayer import register_layer, AbstractScoringLayer


@register_layer("Random")
class Random(AbstractScoringLayer):
    r"""Random scoring layer."""

    def get_config(self):
        config = super(Random, self).get_config()
        return config

    def __init__(self, k):
        super(Random, self).__init__(k)

    def _compute_scores(self, triples):
        """Compute scores using the transE scoring function.

        Parameters
        ----------
        triples: array-like, shape (n, 3)
            Batch of input triples.

        Returns
        -------
        scores: tf.Tensor, shape (n,1)
            Tensor of scores of inputs.
        """

        scores = tf.random.uniform(shape=[tf.shape(triples[0])[0]], seed=0)
        return scores

    def _get_subject_corruption_scores(self, triples, ent_matrix):
        """Compute subject corruption scores.

        Evaluate the inputs against subject corruptions and scores of the corruptions.

        Parameters
        ----------
        triples: array-like, shape (n, k)
            Batch of input embeddings.
        ent_matrix: array-like, shape (m, k)
            Slice of embedding matrix (corruptions).

        Returns
        -------
        scores: tf.Tensor, shape (n, 1)
            Scores of subject corruptions (corruptions defined by `ent_embs` matrix).
        """
        scores = tf.random.uniform(
            shape=[tf.shape(triples[0])[0], tf.shape(ent_matrix)[0]], seed=0
        )
        return scores

    def _get_object_corruption_scores(self, triples, ent_matrix):
        """Compute object corruption scores.

        Evaluate the inputs against object corruptions and scores of the corruptions.

        Parameters
        ----------
        triples: array-like, shape (n, k)
            Batch of input embeddings.
        ent_matrix: array-like, shape (m, k)
            Slice of embedding matrix (corruptions).

        Returns
        -------
        scores: tf.Tensor, shape (n, 1)
            Scores of object corruptions (corruptions defined by `ent_embs` matrix).
        """
        scores = tf.random.uniform(
            shape=[tf.shape(triples[0])[0], tf.shape(ent_matrix)[0]], seed=0
        )
        return scores
