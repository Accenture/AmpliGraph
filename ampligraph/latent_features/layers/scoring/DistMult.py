# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf
from .AbstractScoringLayer import register_layer, AbstractScoringLayer


@register_layer("DistMult")
class DistMult(AbstractScoringLayer):
    r"""DistMult scoring layer.

    The model as described in :cite:`yang2014embedding`.

    The bilinear diagonal DistMult model uses the trilinear dot product as scoring function:

    .. math::
        f_{DistMult}=\langle \mathbf{r}_p, \mathbf{e}_s, \mathbf{e}_o \rangle

    where :math:`\mathbf{e}_{s}` is the embedding of the subject, :math:`\mathbf{r}_{p}` the embedding
    of the predicate and :math:`\mathbf{e}_{o}` the embedding of the object.
    """

    def get_config(self):
        config = super(DistMult, self).get_config()
        return config

    def __init__(self, k):
        super(DistMult, self).__init__(k)

    def _compute_scores(self, triples):
        """Compute scores using the distmult scoring function.

        Parameters
        ----------
        triples: array-like, shape (n, 3)
            Batch of input triples.

        Returns
        -------
        scores: tf.Tensor, shape (n,1)
            Tensor of scores of inputs.
        """
        # compute scores as sum(s * p * o)
        scores = tf.reduce_sum(triples[0] * triples[1] * triples[2], 1)
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
        rel_emb, obj_emb = triples[1], triples[2]
        # compute the score by broadcasting the corruption embeddings(ent_matrix) and using the scoring function
        # compute scores as sum(s_corr * p * o)
        sub_corr_score = tf.reduce_sum(
            ent_matrix * tf.expand_dims(rel_emb * obj_emb, 1), 2
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
        sub_emb, rel_emb = triples[0], triples[1]
        # compute the score by broadcasting the corruption embeddings(ent_matrix) and using the scoring function
        # compute scores as sum(s * p * o_corr)
        obj_corr_score = tf.reduce_sum(
            tf.expand_dims(sub_emb * rel_emb, 1) * ent_matrix, 2
        )
        return obj_corr_score
