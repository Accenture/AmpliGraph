# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from ampligraph.latent_features import loss_functions
from ampligraph.latent_features.loss_functions import PairwiseLoss, NLLLoss, AbsoluteMarginLoss, \
    SelfAdversarialLoss, NLLMulticlass
import pytest
import tensorflow as tf
import numpy as np


def test_PairwiseLoss():
    lossObj = PairwiseLoss({'margin': 2, 'reduction': 'mean'})
    assert lossObj._loss_parameters['margin'] == 2, 'PairwiseLoss: The internal states dont match the inputs'
    pos_score = tf.constant([10.0, 100.0], dtype=tf.float32)
    corr_score = tf.constant([10.0, 100.0, 12.0, 102.0, 8.0, 98.0], dtype=tf.float32)
    # Loss: 2, 2, 4, 4, 0, 0
    loss = lossObj(pos_score, corr_score, eta=3)
    assert (loss.numpy() > np.sum(np.array([2.0, 2.0], dtype=np.float32)) - 1e-4).all() and \
           (loss.numpy() < np.sum(np.array([2.0, 2.0], dtype=np.float32)) + 1e-4).all(), \
        'PairwiseLoss: Loss function outputs dont match expected outputs'
    
    lossObj = PairwiseLoss({'margin': 2, 'reduction': 'sum'})
    assert lossObj._loss_parameters['margin'] == 2, 'PairwiseLoss: The internal states dont match the inputs'
    pos_score = tf.constant([10.0, 100.0], dtype=tf.float32)
    corr_score = tf.constant([10.0, 100.0, 12.0, 102.0, 8.0, 98.0], dtype=tf.float32)
    # Loss: 2, 2, 4, 4, 0, 0
    loss = lossObj(pos_score, corr_score, eta=3)
    assert (loss.numpy() > np.sum(np.array([6.0, 6.0], dtype=np.float32)) - 1e-4).all() and \
           (loss.numpy() < np.sum(np.array([6.0, 6.0], dtype=np.float32)) + 1e-4).all(), \
        'PairwiseLoss: Loss function outputs dont match expected outputs'


def test_NLLLoss():
    lossObj = NLLLoss({'reduction': 'mean'})
    pos_score = tf.constant([50.0, 30.0], dtype=tf.float32)
    corr_score = tf.constant([51.0, 30.0, -100, -60, 96.0, 30.0], dtype=tf.float32)
    # Loss: 51, 30, 0, 0, 75, 30
    loss = lossObj(pos_score, corr_score, eta=3)
    assert (loss.numpy() > np.sum(np.array([21.0, 10.0], dtype=np.float32)) - 1e-4).all() and \
           (loss.numpy() < np.sum(np.array([21.0, 10.0], dtype=np.float32)) + 1e-4).all(), \
        'NLLLoss: Loss function outputs dont match expected outputs'

    lossObj = NLLLoss({'reduction': 'sum'})
    pos_score = tf.constant([50.0, 30.0], dtype=tf.float32)
    corr_score = tf.constant([51.0, 30.0, -100, -60, 96.0, 30.0], dtype=tf.float32)
    # Loss: 51, 30, 0, 0, 75, 30
    loss = lossObj(pos_score, corr_score, eta=3)
    assert (loss.numpy() > np.sum(np.array([126.0, 60.0], dtype=np.float32)) - 1e-4).all() and (
            loss.numpy() < np.sum(np.array([126.0, 60.0], dtype=np.float32)) + 1e-4).all(), \
        'NLLLoss: Loss function outputs dont match expected outputs'


def test_AbsoluteMarginLoss():
    lossObj = AbsoluteMarginLoss({'margin': 3, 'reduction': 'mean'})
    assert lossObj._loss_parameters['margin'] == 3, \
        'AbsoluteMarginLoss: The internal states dont match the inputs'
    
    pos_score = tf.constant([10.0, -10.0], dtype=tf.float32)
    corr_score = tf.constant([13, -10, 10, -7, 7, -13], dtype=tf.float32)
    # Loss: 6, 10, 3, 10, 0, 10
    loss = lossObj(pos_score, corr_score, eta=3)
    assert (loss.numpy() > np.sum(np.array([3.0, 10.0], dtype=np.float32)) - 1e-4).all() and \
           (loss.numpy() < np.sum(np.array([3.0, 10.0], dtype=np.float32)) + 1e-4).all(), \
        'AbsoluteMarginLoss: Loss function outputs dont match expected outputs'
    
    lossObj = AbsoluteMarginLoss({'margin': 3, 'reduction': 'sum'})
    assert lossObj._loss_parameters['margin'] == 3, \
        'AbsoluteMarginLoss: The internal states dont match the inputs'
    
    pos_score = tf.constant([10.0, -10.0], dtype=tf.float32)
    corr_score = tf.constant([13, -10, 10, -7, 7, -13], dtype=tf.float32)
    # Loss: 6, 10, 3, 10, 0, 10
    loss = lossObj(pos_score, corr_score, eta=3)
    assert (loss.numpy() > np.sum(np.array([9.0, 30.0], dtype=np.float32)) - 1e-4).all() and \
           (loss.numpy() < np.sum(np.array([9.0, 30.0], dtype=np.float32)) + 1e-4).all(), \
        'NLLLoss: Loss function outputs dont match expected outputs'


def test_SelfAdversarialLoss():
    lossObj = SelfAdversarialLoss({'margin': 3, 'alpha':1, 'reduction': 'mean'})
    assert lossObj._loss_parameters['margin'] == 3, \
        'SelfAdversarialLoss: The internal states dont match the inputs'
    assert lossObj._loss_parameters['alpha'] == 1, \
        'SelfAdversarialLoss: The internal states dont match the inputs'
    
    pos_score = tf.constant([3, -10.0], dtype=tf.float32)
    corr_score = tf.constant([np.log(2), np.log(10), np.log(2), np.log(50), np.log(4), np.log(40)], dtype=tf.float32)
    loss = lossObj(pos_score, corr_score, eta=3)
    assert (loss.numpy() > np.sum(np.array([1.3552092, 9.222016], dtype=np.float32)) - 1e-4).all() and\
           (loss.numpy() < np.sum(np.array([1.3552092, 9.222016], dtype=np.float32)) + 1e-4).all(), \
        'SelfAdversarialLoss: Loss function outputs dont match expected outputs'
    
    lossObj = SelfAdversarialLoss({'margin': 3, 'alpha':1, 'reduction': 'sum'})
    assert lossObj._loss_parameters['margin'] == 3, \
        'SelfAdversarialLoss: The internal states dont match the inputs'
    assert lossObj._loss_parameters['alpha'] == 1, \
        'SelfAdversarialLoss: The internal states dont match the inputs'
    
    pos_score = tf.constant([3, -10.0], dtype=tf.float32)
    corr_score = tf.constant([np.log(2), np.log(10), np.log(2), np.log(50), np.log(4), np.log(40)], dtype=tf.float32)
    loss = lossObj(pos_score, corr_score, eta=3)
    assert (loss.numpy() > np.sum(np.array([4.060676, 13.664226], dtype=np.float32)) - 1e-4).all() and \
           (loss.numpy() < np.sum(np.array([4.060676, 13.664226], dtype=np.float32)) + 1e-4).all(), \
        'SelfAdversarialLoss: Loss function outputs dont match expected outputs'


def test_NLLMulticlass():
    lossObj = NLLMulticlass({'reduction': 'mean'})
    
    pos_score = tf.constant([np.log(1), np.log(10)], dtype=tf.float32)
    corr_score = tf.constant([np.log(2), np.log(10), np.log(4), np.log(50), np.log(3), np.log(30)], dtype=tf.float32)
    loss = lossObj(pos_score, corr_score, eta=3)
    assert (loss.numpy() > np.sum(np.array([1.3862944, 1.3862944], dtype=np.float32)) - 1e-4).all() and \
           (loss.numpy() < np.sum(np.array([1.3862944, 1.3862944], dtype=np.float32)) + 1e-4).all(), \
        'NLLMulticlass: Loss function outputs dont match expected outputs'
    
    lossObj = NLLMulticlass({'reduction': 'sum'})
    pos_score = tf.constant([np.log(1), np.log(10)], dtype=tf.float32)
    corr_score = tf.constant([np.log(2), np.log(10), np.log(4), np.log(50), np.log(3), np.log(30)], dtype=tf.float32)
    loss = lossObj(pos_score, corr_score, eta=3)

    assert (loss.numpy() > np.sum(np.array([2.3025851, 2.3025851], dtype=np.float32)) - 1e-4).all() and \
           (loss.numpy() < np.sum(np.array([2.3025851, 2.3025851], dtype=np.float32)) + 1e-4).all(),        \
        'NLLMulticlass: Loss function outputs dont match expected outputs'


def test_LossFunctionWrapper():
    def user_loss(scores_pos, scores_neg):
        neg_exp = tf.exp(scores_neg)
        pos_exp = tf.exp(scores_pos)
        softmax_score = pos_exp / (tf.reduce_sum(neg_exp, 0) + pos_exp)
        loss = -tf.math.log(softmax_score)
        return loss
        
    lossObj = loss_functions.get(user_loss)
    pos_score = tf.constant([np.log(1), np.log(10)], dtype=tf.float32)
    corr_score = tf.constant([np.log(2), np.log(10), np.log(4), np.log(50), np.log(3), np.log(30)], dtype=tf.float32)
    loss = lossObj(pos_score, corr_score, eta=3)
    assert (loss.numpy() > np.sum(np.array([2.3025851, 2.3025851], dtype=np.float32)) - 1e-4).all() and \
           (loss.numpy() < np.sum(np.array([2.3025851, 2.3025851], dtype=np.float32)) + 1e-4).all(), \
        'Get: Loss function outputs dont match expected outputs'


def test_get():
    lossObj = loss_functions.get('nll')
    pos_score = tf.constant([50.0, 30.0], dtype=tf.float32)
    corr_score = tf.constant([51.0, 30.0, -100, -60, 96.0, 30.0], dtype=tf.float32)
    # Loss: 51, 30, 0, 0, 75, 30
    loss = lossObj(pos_score, corr_score, eta=3)
    assert (loss.numpy() > np.sum(np.array([126.0, 60.0], dtype=np.float32)) - 1e-4).all() and\
           (loss.numpy() < np.sum(np.array([126.0, 60.0], dtype=np.float32)) + 1e-4).all(), \
        'Get: Loss function outputs dont match expected outputs'
