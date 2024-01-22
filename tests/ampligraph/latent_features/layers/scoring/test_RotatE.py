# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from ampligraph.latent_features.layers.scoring import RotatE

import pytest
import tensorflow as tf
import numpy as np


def test_compute_score():
    model = RotatE(k=3, max_rel_size=2)

    triples = [
        np.array([[1, 1, 1, 2, 2, 2], [10, 10, 10, 11, 11, 11]]).astype(np.float32),
        np.array([[5, 5, 5, 3, 3, 3], [100, 100, 100, 101, 101, 101]]).astype(np.float32),
        np.array([[4, 4, 4, 6, 6, 6], [9, 9, 9, 19, 19, 19]]).astype(np.float32)
    ]

    scores = np.around(model._compute_scores(triples).numpy(), 2)
    assert (scores == np.array([-28.03, -94.19], dtype=np.float32)).all(),\
        'RotatE: Scores don\'t match!'


def test_get_subject_corruption_scores():
    model = RotatE(k=3, max_rel_size=2)

    ent_matrix = np.array([[1, 1, 1, 2, 2, 2], [10, 10, 10, 11, 11, 11]]).astype(np.float32)
    triples = [
        np.array([[1, 1, 1, 2, 2, 2], [10, 10, 10, 11, 11, 11]]).astype(np.float32),
        np.array([[5, 5, 5, 3, 3, 3], [100, 100, 100, 101, 101, 101]]).astype(np.float32),
        np.array([[4, 4, 4, 6, 6, 6], [9, 9, 9, 19, 19, 19]]).astype(np.float32)
    ]

    scores = np.around(model._get_subject_corruption_scores(triples, ent_matrix).numpy(), 2)
    assert (np.diag(scores) == np.array([-28.03, -94.19], dtype=np.float32)).all(),\
        'RotatE: Scores don\'t match!'


def test_get_object_corruption_scores():
    model = RotatE(k=3, max_rel_size=2)

    ent_matrix = np.array([[4, 4, 4, 6, 6, 6], [9, 9, 9, 19, 19, 19]]).astype(np.float32)
    triples = [
        np.array([[1, 1, 1, 2, 2, 2], [10, 10, 10, 11, 11, 11]]).astype(np.float32),
        np.array([[5, 5, 5, 3, 3, 3], [100, 100, 100, 101, 101, 101]]).astype(np.float32),
        np.array([[4, 4, 4, 6, 6, 6], [9, 9, 9, 19, 19, 19]]).astype(np.float32)
    ]

    scores = np.around(model._get_object_corruption_scores(triples, ent_matrix).numpy(), 2)
    assert (np.diag(scores) == np.array([-28.03, -94.19], dtype=np.float32)).all(),\
        'RotatE: Scores don\'t match!'