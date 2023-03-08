# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from ampligraph.latent_features.layers.scoring import DistMult

import pytest
import tensorflow as tf
import numpy as np

def test_compute_score():
    model = DistMult(k=7)
    
    triples = [np.array([[1, 1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10, 10]]).astype(np.float32),
           np.array([[5, 5, 5, 5, 5, 5, 5], [100, 100, 100, 100, 100, 100, 100]]).astype(np.float32),
           np.array([[4, 4, 4, 4, 4, 4, 4], [9, 9, 9, 9, 9, 9, 9]]).astype(np.float32)]
    
    scores = np.around(model._compute_scores(triples).numpy(), 2)
    assert (scores == np.array([140, 63000], 
                               dtype=np.float32)).all(), 'DistMult: Scores don\'t match!'
    
    
def test_get_subject_corruption_scores():
    model = DistMult(k=7)
    
    ent_matrix = np.array([[1, 1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10, 10]]).astype(np.float32)
    triples = [np.array([[1, 1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10, 10]]).astype(np.float32),
           np.array([[5, 5, 5, 5, 5, 5, 5], [100, 100, 100, 100, 100, 100, 100]]).astype(np.float32),
           np.array([[4, 4, 4, 4, 4, 4, 4], [9, 9, 9, 9, 9, 9, 9]]).astype(np.float32)]
    scores = np.around(model._get_subject_corruption_scores(triples, ent_matrix).numpy(), 2)
    assert (np.diag(scores) == np.array([140, 63000], 
                               dtype=np.float32)).all(), 'DistMult: Scores don\'t match!'    
    
def test_get_object_corruption_scores():
    model = DistMult(k=7)
    
    ent_matrix = np.array([[4, 4, 4, 4, 4, 4, 4], [9, 9, 9, 9, 9, 9, 9]]).astype(np.float32)
    triples = [np.array([[1, 1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10, 10]]).astype(np.float32),
           np.array([[5, 5, 5, 5, 5, 5, 5], [100, 100, 100, 100, 100, 100, 100]]).astype(np.float32),
           np.array([[4, 4, 4, 4, 4, 4, 4], [9, 9, 9, 9, 9, 9, 9]]).astype(np.float32)]
    scores = np.around(model._get_object_corruption_scores(triples, ent_matrix).numpy(), 2)
    assert (np.diag(scores) == np.array([140, 63000], 
                               dtype=np.float32)).all(), 'DistMult: Scores don\'t match!'   