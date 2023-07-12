# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from ampligraph.latent_features.layers.encoding import EmbeddingLookupLayer
from ampligraph.latent_features import ScoringBasedEmbeddingModel

import pytest
import tensorflow as tf
import numpy as np

def test_build():
    model = ScoringBasedEmbeddingModel(k=3, eta=1)
    model.compile(optimizer='adam', loss='pairwise')
    
    with pytest.raises(TypeError, 
                       match='Not enough arguments to build Encoding Layer. Please set max_ent_size property.'):
        model.encoding_layer.build((10, 10))

    try:
        model.encoding_layer.max_ent_size = 10
        model.encoding_layer.max_rel_size = 2
        model.encoding_layer.build((10, 10))
    except TypeError:
        pytest.fail("Encoding layer should build if ent size and rel sizes are set")
        
    assert model.encoding_layer.max_ent_size == 10, 'EmbeddingLookupLayer: Max ent size doesn\'t match'
    assert model.encoding_layer.max_rel_size == 2, 'EmbeddingLookupLayer: Max rel size doesn\'t match'
    
    
def test_set_initializer():
    model = ScoringBasedEmbeddingModel(k=3, eta=1)
    model.compile(optimizer='adam', loss='pairwise', entity_relation_initializer=[tf.constant_initializer([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
                                                                  tf.constant_initializer([[10, 11, 12], [10, 11, 12]])])
    
    model.encoding_layer.max_ent_size = 3
    model.encoding_layer.max_rel_size = 2
    model.encoding_layer.build((10, 10))
    assert (model.encoding_layer.ent_emb.numpy() == np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.float32)).all(), \
        'EmbeddingLookupLayer (initializer): Entity embeddings dont match!'
        
    assert (model.encoding_layer.rel_emb.numpy() == np.array([[10, 11, 12], [10, 11, 12]], dtype=np.float32)).all(), \
        'EmbeddingLookupLayer (initializer): Relation embeddings dont match!'
    
    model = ScoringBasedEmbeddingModel(k=3, eta=1)
    
    model.compile(optimizer='adam', loss='pairwise', 
                  entity_relation_initializer=tf.constant_initializer([[0, 1, 2], [0, 1, 2], [0, 1, 2]]))
    model.encoding_layer.max_ent_size = 3
    model.encoding_layer.max_rel_size = 3
    model.encoding_layer.build((10, 10))
    
    assert (model.encoding_layer.ent_emb.numpy() == np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.float32)).all(), \
        'EmbeddingLookupLayer (initializer): Entity embeddings dont match!'
    
    assert (model.encoding_layer.rel_emb.numpy() == np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.float32)).all(), \
        'EmbeddingLookupLayer (initializer): Relation embeddings dont match!'
    
    with pytest.raises(AssertionError, match='Incorrect length for initializer. Assumed 2 got 3'):
        model = ScoringBasedEmbeddingModel(k=3, eta=1)
        model.compile(optimizer='adam', loss='pairwise', 
                      entity_relation_initializer=[tf.constant_initializer([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
                                                   tf.constant_initializer([[10, 11, 12], [10, 11, 12]]), 
                                                   tf.constant_initializer([[10, 11, 12], [10, 11, 12]])])

        model.encoding_layer.max_ent_size = 3
        model.encoding_layer.max_rel_size = 2
        model.encoding_layer.build((10, 10))

def test_call():
    model = ScoringBasedEmbeddingModel(k=3, eta=1)
    model.compile(optimizer='adam', loss='pairwise', entity_relation_initializer=[tf.constant_initializer([[0,0,0], [1,1,1], [2,2,2]]),
                                                                  tf.constant_initializer([[10, 10, 10], [11, 11, 11]])])
    
    model.encoding_layer.max_ent_size = 3
    model.encoding_layer.max_rel_size = 2
    model.encoding_layer.build((10, 10))
    
    out = model.encoding_layer.call(tf.constant([[0, 0, 1], [0, 1, 2]]))
    
    assert (out[0].numpy() == np.array([[0,0,0], [0,0,0]])).all(), \
        'EmbeddingLookupLayer (call): subject embeddings are incorrect'
    assert (out[1].numpy() == np.array([[10, 10, 10], [11, 11, 11]])).all(), \
        'EmbeddingLookupLayer (call): predicate embeddings are incorrect'
    assert (out[2].numpy() == np.array([[1, 1, 1], [2, 2, 2]])).all(), \
        'EmbeddingLookupLayer (call): object embeddings are incorrect'
    
def test_set_ent_rel_initial_value():
    model = ScoringBasedEmbeddingModel(k=3, eta=1)
    model.compile(optimizer='adam', loss='pairwise', 
                  entity_relation_initializer=[tf.constant_initializer([[10,10,10], [10,10,10], [10,10,10], 
                                                                        [10,10,10], [10,10,10]]),
                                               tf.constant_initializer([[10, 10, 10], [11, 11, 11]])])
    
    model.encoding_layer.max_ent_size = 5
    model.encoding_layer.max_rel_size = 2
    
    model.encoding_layer.set_ent_rel_initial_value(np.array([[0,0,0],[1,1,1],[2,2,2]]),
                                                   np.array([[5,5,5]]))
    model.encoding_layer.build((10, 10))
    
    assert (model.encoding_layer.ent_emb.numpy() == np.array([[0,0,0], [1,1,1], [2,2,2], [0,0,0], [0,0,0]])).all(), \
        'EmbeddingLookupLayer (set_ent_rel_initial_value): Entity matrix not correctly initialized'
    
    assert (model.encoding_layer.rel_emb.numpy() == np.array([[5,5,5], [0,0,0]])).all(), \
        'EmbeddingLookupLayer (set_ent_rel_initial_value): Relation matrix not correctly initialized'
    
def test_partition_change_updates():
    model = ScoringBasedEmbeddingModel(k=3, eta=1)
    model.compile(optimizer='adam', loss='pairwise', 
                  entity_relation_initializer=[tf.constant_initializer([[10,10,10], [10,10,10], [10,10,10], 
                                                                        [10,10,10], [10,10,10]]),
                                               tf.constant_initializer([[10, 10, 10], [11, 11, 11]])])
    
    model.encoding_layer.max_ent_size = 5
    model.encoding_layer.max_rel_size = 2
    model.encoding_layer.build((10, 10))
    
    assert (model.encoding_layer.ent_emb.numpy() == np.array([[10,10,10], [10,10,10], [10,10,10], 
                                                                        [10,10,10], [10,10,10]])).all(), \
        'EmbeddingLookupLayer (partition_change_updates): Entity matrix not correctly initialized'
    
    assert (model.encoding_layer.rel_emb.numpy() == np.array([[10, 10, 10], [11, 11, 11]])).all(), \
        'EmbeddingLookupLayer (partition_change_updates): Relation matrix not correctly initialized'
    
    model.encoding_layer.partition_change_updates(np.array([[0,0,0],[1,1,1],[2,2,2]], dtype=np.float32),
                                                   np.array([[5,5,5]], dtype=np.float32))
    
    assert (model.encoding_layer.ent_emb.numpy() == np.array([[0,0,0], [1,1,1], [2,2,2], [0,0,0], [0,0,0]])).all(), \
        'EmbeddingLookupLayer (partition_change_updates): Entity matrix not correctly changed'
    
    assert (model.encoding_layer.rel_emb.numpy() == np.array([[5,5,5], [0,0,0]])).all(), \
        'EmbeddingLookupLayer (partition_change_updates): Relation matrix not correctly changed'
    