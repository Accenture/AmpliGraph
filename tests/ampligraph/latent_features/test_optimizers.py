# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from ampligraph.latent_features import optimizers
import pytest
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adagrad, Adam


def test_optimizer_adam():
    ''' test by passing a string'''
    adam = optimizers.get('Adam')
    adam.set_partitioned_training()
    ent = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32), trainable=True)
    rel = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32), trainable=True)
    
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(ent * rel)
    
    adam.minimize(loss, ent, rel, tape)
    curr_weights = adam.get_weights()
    
    # step + 2 hyperparams * 2 trainable vars
    assert len(curr_weights) == (1 + adam.get_hyperparam_count() * adam.num_optimized_vars), \
        'Adam: Lengths dont match!'
    
    assert adam.get_iterations() == 1, 'Adam: Iteration count doesnt match!'
    
    adam.set_weights(curr_weights)
    new_weights = adam.get_weights()
    
    # test whether all params are same
    out = [np.all(i==j) for i, j in zip(curr_weights, new_weights)]
    assert np.all(out), 'Adam: Weights are not the same!'


def test_optimizer_adagrad():
    ''' test the wrapping functionality around keras optimizer'''
    adagrad = optimizers.get(Adagrad(learning_rate = 0.0001))
    adagrad.set_partitioned_training()
    ent = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32), trainable=True)
    rel = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32), trainable=True)
    
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(ent * rel)
    
    adagrad.minimize(loss, ent, rel, tape)
    curr_weights = adagrad.get_weights()
    # step + 2 hyperparams * 2 trainable vars
    assert len(curr_weights) == (1 + adagrad.get_hyperparam_count() * adagrad.num_optimized_vars), \
        'Adagrad: Lengths dont match!'
    
    adagrad.set_weights(curr_weights)
    new_weights = adagrad.get_weights()
    
    # test whether all params are same
    out = [np.all(i==j) for i, j in zip(curr_weights, new_weights)]
    assert np.all(out), 'Adagrad: Weights are not the same!'


def test_entity_relation_hyperparameters():
    '''test the getters and setters of entity relation hyperparams'''
    adam = optimizers.get(Adam(learning_rate = 0.0001))
    ent = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32), trainable=True)
    rel = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32), trainable=True)
    
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(ent * rel)
    
    adam.minimize(loss, ent, rel, tape)
    
    curr_weights = adam.get_weights()
    ent_hyp, rel_hyp = adam.get_entity_relation_hyperparams()

    assert (curr_weights[1] == ent_hyp[0]).all() and (curr_weights[3] == ent_hyp[1]).all(), \
        'ent weights are not correct!'
    assert (curr_weights[2] == rel_hyp[0]).all() and (curr_weights[4] == rel_hyp[1]).all(), \
        'rel weights are not correct!'
