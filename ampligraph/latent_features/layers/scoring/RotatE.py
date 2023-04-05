# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf
from .AbstractScoringLayer import register_layer, AbstractScoringLayer


@register_layer("RotatE")
class RotatE(AbstractScoringLayer):
    r"""Rotate Embeddings (RotatE) scoring layer.

    The RotatE model :cite:`sun2018rotate` is knowledge graph embedding model that tries to map
    the head embedding into the tail embedding through a rotation (relation embedding) in the
    complex space.

    RotatE scoring function is based on the Hadamard product:

    .. math::
        f_{RotatE}= -||\mathbf{e}_{sub} \circ \mathbf{e}_{pred} - \mathbf{e}_{obj}||

    .. note::
        Since RotatE embeddings belong to :math:`\mathbb{C}`, this model uses twice
        as many parameters as :class:`ampligraph.latent_features.DistMult` and as many
        as :class:`ampligraph.latent_features.ComplEx`.
    """

    def get_config(self):
        config = super(RotatE, self).get_config()
        return config

    def __init__(self, k):
        super(RotatE, self).__init__(k)
        # internally complex uses k embeddings for real part and k embeddings for img part
        # hence internally it uses 2 * k embeddings
        self.internal_k = 2 * k

    def _compute_scores(self, triples):
        """Compute scores using the RotatE scoring function.

        Parameters
        ----------
        triples: array, shape (n, 3)
            Batch of input triples.

        Returns
        -------
        scores: tf.Tensor
            Tensor with scores of the inputs.
        """
        # split the embeddings of s, and o into 2 parts (real and img part)
        e_s_real, e_s_img = tf.split(triples[0], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)
        # Construct the complex embedding of the relation using only k parameters
        theta_pred, _ = tf.split(triples[1], 2, axis=1)
        e_p_real = tf.cos(theta_pred)
        e_p_img = tf.sin(theta_pred)

        scores_real = e_s_real * e_p_real - e_s_img * e_p_img - e_o_real
        scores_img = e_s_real * e_p_img + e_s_img * e_p_real - e_o_img
        scores = tf.negative(
            tf.norm(tf.sqrt(scores_real**2 + scores_img**2), axis=1, ord=1)
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
        # split the embeddings of s, p, o into 2 parts (real and img part)
        e_s_real, e_s_img = tf.split(triples[0], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)
        # Construct the complex embedding of the relation using only k parameters
        theta_pred, _ = tf.split(triples[1], 2, axis=1)
        e_p_real = tf.cos(theta_pred)
        e_p_img = tf.sin(theta_pred)

        # split the corruption entity embeddings into 2 parts (real and img
        # part)
        ent_real, ent_img = tf.split(ent_matrix, 2, axis=1)

        # compute the subject corruption score using ent_real, ent_img
        # (corruption embeddings) as subject embeddings
        sub_corr_scores_real =\
            ent_real * tf.expand_dims(e_p_real, 1) - ent_img * tf.expand_dims(e_p_img, 1) - tf.expand_dims(e_o_real, 1)
        sub_corr_scores_img =\
            ent_real * tf.expand_dims(e_p_img, 1) + ent_img * tf.expand_dims(e_p_real, 1) - tf.expand_dims(e_o_img, 1)

        sub_corr_scores = tf.negative(
            tf.norm(tf.sqrt(sub_corr_scores_real**2 + sub_corr_scores_img**2), axis=2, ord=1)
        )
        return sub_corr_scores

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
        # split the embeddings of s, p, o into 2 parts (real and img part)
        e_s_real, e_s_img = tf.split(triples[0], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)
        # Construct the complex embedding of the relation using only k parameters
        theta_pred, _ = tf.split(triples[1], 2, axis=1)
        e_p_real = tf.cos(theta_pred)
        e_p_img = tf.sin(theta_pred)

        # split the corruption entity embeddings into 2 parts (real and img
        # part)
        ent_real, ent_img = tf.split(ent_matrix, 2, axis=1)

        # compute the subject corruption score using ent_real, ent_img
        # (corruption embeddings) as subject embeddings
        obj_corr_scores_real = tf.expand_dims(e_s_real * e_p_real - e_s_img * e_p_img, 1) - ent_real
        obj_corr_scores_img = tf.expand_dims(e_s_real * e_p_img + e_s_img * e_p_real, 1) - ent_img
        obj_corr_scores = tf.negative(
            tf.norm(tf.sqrt(obj_corr_scores_real**2 + obj_corr_scores_img**2), axis=2, ord=1)
        )
        return obj_corr_scores
