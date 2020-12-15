from ampligraph.latent_features.layers.corruption_generation import CorruptionGenerationLayerTrain

import pytest
import tensorflow as tf
import numpy as np


def test_call():
    tf.random.set_seed(0)
    train_layer = CorruptionGenerationLayerTrain()
    out = train_layer.call(tf.constant([[1,0,5], [3, 0, 7], [1, 1, 9]]), 1000, 2)
    assert (out.numpy() == np.array([[760, 0, 5], [861, 0, 7], [1, 1, 39], [567, 0, 5], [3, 0, 147], [28, 1, 9]])).all(), \
        "CorruptionGenerationLayerTrain: Corruptions not generated correctly"