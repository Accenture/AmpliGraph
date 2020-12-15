from ampligraph.latent_features.layers.scoring import DistMult

import pytest
import tensorflow as tf
import numpy as np

def test_compute_score():
    model = DistMult(k=3)
    
    triples = [np.array([[1, 1, 1], [2, 2, 2]]).astype(np.float32),
           np.array([[10, 10, 10], [100, 100, 100]]).astype(np.float32),
           np.array([[3, 3, 3], [4, 4, 4]]).astype(np.float32)]
    
    mapping_dict = tf.lookup.experimental.DenseHashTable(tf.int32, tf.int32, -1, -1, -2)
    
    ent_matrix = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).astype(np.float32)
    scores = np.around(model._compute_scores(triples).numpy(), 2)
    
    sub_corr_score = model._get_subject_corruption_scores(triples, ent_matrix)
        
    obj_corr_score = model._get_object_corruption_scores(triples, ent_matrix)
    
    ranks = model.get_ranks(triples, ent_matrix, 0, 4, tf.ragged.constant([], dtype=tf.int32), mapping_dict)
    assert (ranks.numpy() == np.array([[4, 3], [2, 1]], dtype=np.int32)).all(), 'Unfiltered Ranks not correct'
    
    ranks = model.get_ranks(triples, ent_matrix, 0, 4, tf.ragged.constant([[[0], [1]], [[2], [3]]], dtype=tf.int32),
                           mapping_dict)
    
    assert (ranks.numpy() == np.array([[3, 2], [1, 0]], dtype=np.int32)).all(), '(s,o) Filtered Ranks not correct'
    
    ranks = model.get_ranks(triples, ent_matrix, 0, 4, tf.ragged.constant([[[0], [1]], [[2], [3]]], dtype=tf.int32),
                           mapping_dict, corrupt_side = 's')
    assert (ranks.numpy() == np.array([[3, 2]], dtype=np.int32)).all(), '(s) Filtered Ranks not correct'
    
    ranks = model.get_ranks(triples, ent_matrix, 0, 4, tf.ragged.constant([[[2], [3]]], dtype=tf.int32),
                           mapping_dict, corrupt_side = 'o')
    assert (ranks.numpy() == np.array([[1, 0]], dtype=np.int32)).all(), '(o) Filtered Ranks not correct'
    
    ranks = model.get_ranks(triples, ent_matrix, 0, 4, tf.ragged.constant([], dtype=tf.int32),
                           mapping_dict, corrupt_side = 's')
    
    assert (ranks.numpy() == np.array([[4, 3]], dtype=np.int32)).all(), '(s) Unfiltered Ranks not correct'
    
    ranks = model.get_ranks(triples, ent_matrix, 0, 4, tf.ragged.constant([], dtype=tf.int32),
                           mapping_dict, corrupt_side = 'o')
    assert (ranks.numpy() == np.array([[2, 1]], dtype=np.int32)).all(), '(o) Unfiltered Ranks not correct'