# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from ampligraph.latent_features.layers.scoring import TransE

import pytest
import tensorflow as tf
import numpy as np

def test_compute_score():
    model = TransE(k=7)
    
    triples = [np.array([[1, 1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10, 10]]).astype(np.float32),
           np.array([[13, 13, 13, 13, 13, 13, 13], [100, 100, 100, 100, 100, 100, 100]]).astype(np.float32),
           np.array([[4, 4, 4, 4, 4, 4, 9], [90, 90, 90, 90, 90, 90, 90]]).astype(np.float32)]

    scores = np.around(model._compute_scores(triples).numpy(), 2)
    assert (scores == np.array([-65., -140.],
                               dtype=np.float32)).all(), 'TransE: Scores don\'t match!'
    
    
def test_get_subject_corruption_scores():
    model = TransE(k=7)
    
    ent_matrix = np.array([[1, 1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10, 10]]).astype(np.float32)
    triples = [np.array([[1, 1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10, 10]]).astype(np.float32),
           np.array([[13, 13, 13, 13, 13, 13, 13], [100, 100, 100, 100, 100, 100, 100]]).astype(np.float32),
           np.array([[4, 4, 4, 4, 4, 4, 9], [90, 90, 90, 90, 90, 90, 90]]).astype(np.float32)]
    scores = np.around(model._get_subject_corruption_scores(triples, ent_matrix).numpy(), 2)
    assert (np.diag(scores) == np.array([-65., -140.],
                               dtype=np.float32)).all(), 'TransE: Scores don\'t match!'    
    
def test_get_object_corruption_scores():
    model = TransE(k=7)
    
    ent_matrix = np.array([[4, 4, 4, 4, 4, 4, 9], [90, 90, 90, 90, 90, 90, 90]]).astype(np.float32)
    triples = [np.array([[1, 1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10, 10]]).astype(np.float32),
           np.array([[13, 13, 13, 13, 13, 13, 13], [100, 100, 100, 100, 100, 100, 100]]).astype(np.float32),
           np.array([[4, 4, 4, 4, 4, 4, 9], [90, 90, 90, 90, 90, 90, 90]]).astype(np.float32)]
    scores = np.around(model._get_object_corruption_scores(triples, ent_matrix).numpy(), 2)
    assert (np.diag(scores) == np.array([-65., -140.],
                               dtype=np.float32)).all(), 'TransE: Scores don\'t match!'  