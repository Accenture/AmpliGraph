# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def sum_pooling(embeddings):
    """Sum pooling function
    Performs pooling by summation of all embeddings along neighbour axis.

    Parameters
    ----------
    embeddings : Tensor, shape [B, max_rel, emb_dim]
        The embeddings of a list of subjects.

    Returns
    -------
    v : TensorFlow operation
        Reduced vector v

    """
    return tf.reduce_sum(embeddings, axis=1)


def avg_pooling(embeddings):
    """Sum pooling function
    Performs pooling by summation of all embeddings along neighbour axis.

    Parameters
    ----------
    embeddings : Tensor, shape [B, max_rel, emb_dim]
        The embeddings of a list of subjects.

    Returns
    -------
    v : TensorFlow operation
        Reduced vector v

    """
    return tf.reduce_mean(embeddings, axis=1)


def max_pooling(embeddings):
    """Sum pooling function
    Performs pooling by summation of all embeddings along neighbour axis.

    Parameters
    ----------
    embeddings : Tensor, shape [B, max_rel, emb_dim]
        The embeddings of a list of subjects.

    Returns
    -------
    v : TensorFlow operation
        Reduced vector v

    """
    return tf.reduce_max(embeddings, axis=1)
