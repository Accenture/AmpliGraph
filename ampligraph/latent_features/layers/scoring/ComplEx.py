# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf
from .AbstractScoringLayer import register_layer, AbstractScoringLayer


@register_layer("ComplEx")
class ComplEx(AbstractScoringLayer):
    r"""Complex Embeddings (ComplEx) scoring layer.

    The ComplEx model :cite:`trouillon2016complex` is an extension of
    the :class:`ampligraph.latent_features.DistMult` bilinear diagonal model.

    ComplEx scoring function is based on the trilinear Hermitian dot product in :math:`\mathbb{C}`:

    .. math::
        f_{ComplEx}=Re(\langle \mathbf{r}_p, \mathbf{e}_s, \overline{\mathbf{e}_o}  \rangle)

    .. note::
        Since ComplEx embeddings belong to :math:`\mathbb{C}`, this model uses twice as many parameters as
        :class:`ampligraph.latent_features.DistMult`.
    """

    def get_config(self):
        config = super(ComplEx, self).get_config()
        return config

    def __init__(self, k):
        super(ComplEx, self).__init__(k)
        # internally complex uses k embedddings for real part and k embedddings for img part
        # hence internally it uses 2 * k embeddings
        self.internal_k = 2 * k

    def _compute_scores(self, triples):
        """Compute scores using the ComplEx scoring function.

        Parameters
        ----------
        triples: array, shape (n, 3)
            Batch of input triples.

        Returns
        -------
        scores: tf.Tensor
            Tensor with scores of the inputs.
        """
        # split the embeddings of s, p, o into 2 parts (real and img part)
        e_s_real, e_s_img = tf.split(triples[0], 2, axis=1)
        e_p_real, e_p_img = tf.split(triples[1], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)

        # apply the complex scoring function
        scores = tf.reduce_sum(
            (e_s_real * (e_p_real * e_o_real + e_p_img * e_o_img))
            + (e_s_img * (e_p_real * e_o_img - e_p_img * e_o_real)),
            axis=1,
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
        e_p_real, e_p_img = tf.split(triples[1], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)

        # split the corruption entity embeddings into 2 parts (real and img
        # part)
        ent_real, ent_img = tf.split(ent_matrix, 2, axis=1)

        # compute the subject corruption score using ent_real, ent_img
        # (corruption embeddings) as subject embeddings
        sub_corr_score = tf.reduce_sum(
            ent_real
            * (
                tf.expand_dims(e_p_real * e_o_real, 1)
                + tf.expand_dims(e_p_img * e_o_img, 1)
            )
            + (
                ent_img
                * (
                    tf.expand_dims(e_p_real * e_o_img, 1)
                    - tf.expand_dims(e_p_img * e_o_real, 1)
                )
            ),
            axis=2,
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
        # split the embeddings of s, p, o into 2 parts (real and img part)
        e_s_real, e_s_img = tf.split(triples[0], 2, axis=1)
        e_p_real, e_p_img = tf.split(triples[1], 2, axis=1)
        e_o_real, e_o_img = tf.split(triples[2], 2, axis=1)

        # split the corruption entity embeddings into 2 parts (real and img
        # part)
        ent_real, ent_img = tf.split(ent_matrix, 2, axis=1)

        # compute the object corruption score using ent_real, ent_img
        # (corruption embeddings) as object embeddings
        obj_corr_score = tf.reduce_sum(
            (
                tf.expand_dims(e_s_real * e_p_real, 1)
                - tf.expand_dims(e_s_img * e_p_img, 1)
            )
            * ent_real
            + (
                tf.expand_dims(e_s_img * e_p_real, 1)
                + tf.expand_dims(e_s_real * e_p_img, 1)
            )
            * ent_img,
            axis=2,
        )
        return obj_corr_score
