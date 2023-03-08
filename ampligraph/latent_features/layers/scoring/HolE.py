# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from .AbstractScoringLayer import register_layer
from .ComplEx import ComplEx


@register_layer("HolE")
class HolE(ComplEx):
    r"""Holographic Embeddings (HolE) scoring layer.

    The HolE model :cite:`nickel2016holographic` as re-defined by Hayashi et al. :cite:`HayashiS17`:

    .. math::
        f_{HolE}= \frac{2}{k} \, f_{ComplEx}

    where :math:`k` is the size of the embeddings.
    """

    def get_config(self):
        config = super(HolE, self).get_config()
        return config

    def __init__(self, k):
        super(HolE, self).__init__(k)

    def _compute_scores(self, triples):
        """Compute scores using HolE scoring function.

        Parameters
        ----------
        triples: array-like, shape (n, 3)
            Batch of input triples.

        Returns
        -------
        scores: tf.Tensor(n,1)
            Tensor of scores of inputs.
        """
        # HolE scoring is 2/k * complex_score
        return (2 / (self.internal_k / 2)) * (super()._compute_scores(triples))

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
        scores: tf.Tensor, shape (n,1)
            Scores of subject corruptions (corruptions defined by `ent_embs` matrix).
        """
        # HolE scoring is 2/k * complex_score
        return (2 / (self.internal_k / 2)) * (
            super()._get_subject_corruption_scores(triples, ent_matrix)
        )

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
        scores: tf.Tensor, shape (n,1)
            Scores of object corruptions (corruptions defined by `ent_embs` matrix).
        """
        # HolE scoring is 2/k * complex_score
        return (2 / (self.internal_k / 2)) * (
            super()._get_object_corruption_scores(triples, ent_matrix)
        )
