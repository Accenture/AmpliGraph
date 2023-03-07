# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf
from .AbstractScoringLayer import register_layer, AbstractScoringLayer


@register_layer("TransE")
class TransE(AbstractScoringLayer):
    r"""Translating Embeddings (TransE) scoring layer.

    The model as described in :cite:`bordes2013translating`.

    The scoring function of TransE computes a similarity between the embedding of the subject
    :math:`\mathbf{e}_{sub}` translated by the embedding of the predicate :math:`\mathbf{e}_{pred}`,
    and the embedding of the object :math:`\mathbf{e}_{obj}`,
    using the :math:`L_1` or :math:`L_2` norm :math:`||\cdot||` (default: :math:`L_1`):

    .. math::
        f_{TransE}=-||\mathbf{e}_{sub} + \mathbf{e}_{pred} - \mathbf{e}_{obj}||

    Such scoring function is then used on positive and negative triples :math:`t^+, t^-` in the loss function.

    """

    def get_config(self):
        config = super(TransE, self).get_config()
        return config

    def __init__(self, k):
        super(TransE, self).__init__(k)

    def _compute_scores(self, triples):
        """Compute scores using transE scoring function.

        Parameters
        ----------
        triples: array-like, (n, 3)
            Batch of input triples.

        Returns
        -------
        scores: tf.Tensor, shape (n,1)
            Tensor of scores of inputs.
        """
        # compute scores as -|| s + p - o||
        scores = tf.negative(
            tf.norm(triples[0] + triples[1] - triples[2], axis=1, ord=1)
        )
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
        # get the subject, predicate and object embeddings of True positives
        rel_emb, obj_emb = triples[1], triples[2]
        # compute the score by broadcasting the corruption embeddings(ent_matrix) and using the scoring function
        # compute scores as -|| s_corr + p - o||
        sub_corr_score = tf.negative(
            tf.norm(
                ent_matrix + tf.expand_dims(rel_emb - obj_emb, 1),
                axis=2,
                ord=1,
            )
        )
        return sub_corr_score

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
        # get the subject, predicate and object embeddings of True positives:
        sub_emb, rel_emb = triples[0], triples[1]
        # compute the score by broadcasting the corruption embeddings(ent_matrix) and using the scoring function
        # compute scores as -|| s + p - o_corr||
        obj_corr_score = tf.negative(
            tf.norm(
                tf.expand_dims(sub_emb + rel_emb, 1) - ent_matrix,
                axis=2,
                ord=1,
            )
        )
        return obj_corr_score
