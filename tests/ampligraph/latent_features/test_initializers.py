# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from ampligraph.datasets import load_fb15k_237
from ampligraph.latent_features import ScoringBasedEmbeddingModel
import numpy as np
import tensorflow as tf


def test_initializers():
    dataset = load_fb15k_237()

    model = ScoringBasedEmbeddingModel(eta=2, 
                                         k=10,
                                         scoring_type='TransE')

    unique_ent_len = len(set(dataset['train'][:10, 0]).union(set(dataset['train'][:10, 2])))
    init_ent = tf.constant_initializer(
        value=np.ones(shape=(unique_ent_len, 10), dtype=np.float32)
    )

    unique_rel_len = len(set(dataset['train'][:10, 1]))
    init_rel = tf.constant_initializer(
        value=np.ones(shape=(unique_rel_len, 10), dtype=np.float32)
    )

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-10), loss='nll', entity_relation_initializer=[init_ent, init_rel])
    model.fit(dataset['train'][:10], batch_size=10, epochs=1)

    assert np.all(model.encoding_layer.ent_emb.numpy().round() == np.float32(1)), 'Entity Initializer not working!'

    assert np.all(model.encoding_layer.rel_emb.numpy().round() == np.float32(1)), 'Relation Initializer not working!'


    model = ScoringBasedEmbeddingModel(eta=2, 
                                         k=10,
                                         scoring_type='TransE')

    init = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=117)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-10), loss='nll', entity_relation_initializer=init)
    model.fit(dataset['train'][:10], batch_size=10, epochs=1)

    assert np.round(np.mean(model.encoding_layer.ent_emb.numpy()), 3) == np.float32(0), 'Entity Initializer not working! Mean should be 0'
    assert np.round(np.std(model.encoding_layer.ent_emb.numpy()), 3) == np.float32(0.001), 'Entity Initializer not working! Std should be 0.001'
