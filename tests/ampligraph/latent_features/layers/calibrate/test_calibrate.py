from ampligraph.latent_features.layers.calibration import CalibrationLayer

import pytest
import tensorflow as tf
import numpy as np


def test_init():
    calib_layer = CalibrationLayer(pos_size=5, positive_base_rate=0.5)
    
    calib_layer.build((10,10))
    assert calib_layer.pos_size == calib_layer.neg_size, \
        'CalibrationLayer: pos_size and neg_size must be same if calibrating with corruptions'
    assert calib_layer.calib_w.numpy() == 0, 'CalibrationLayer: w not initialized correctly'
    assert calib_layer.calib_b.numpy() == 0, 'CalibrationLayer: b not initialized correctly'
    
    calib_layer = CalibrationLayer(pos_size=5, neg_size=5, calib_w=10, calib_b=10)
    
    calib_layer.build((10,10))
    assert calib_layer.calib_w.numpy() == 10, 'CalibrationLayer (passed w): w not initialized correctly'
    assert calib_layer.calib_b.numpy() == 10, 'CalibrationLayer (passed b): b not initialized correctly'
    assert calib_layer.positive_base_rate == 0.5, 'Incorrect positive base rate'
    
    with pytest.raises(ValueError, match="Positive_base_rate must be a value between 0 and 1."):
        calib_layer = CalibrationLayer(pos_size=5, positive_base_rate=1.1)
        
    with pytest.raises(AssertionError, match="Positive size must be > 0."):
        calib_layer = CalibrationLayer(pos_size=0)
    
    
def test_call():
    calib_layer = CalibrationLayer(pos_size=5, neg_size=5, calib_w=10, calib_b=10)
    
    calib_layer.build((10,10))
    
    out = calib_layer.call(scores_pos=tf.constant([-2,1,-1], dtype=tf.float32), 
                           scores_neg=tf.constant([10,11,12], dtype=tf.float32), training=0)
    assert (np.around(out.numpy(), 2) == np.array([1, 0, 0.5], dtype=np.float32)).all(), \
        'CalibrationLayer: calibration scores don\'t match'
    
    
    out1 = calib_layer.call(scores_pos=tf.constant([-2,1,-1], dtype=tf.float32), 
                           scores_neg=tf.constant([10,11,12], dtype=tf.float32), training=1)
    assert np.around(out1.numpy(), 2) == np.array([11.78], dtype=np.float32), \
        'CalibrationLayer: calibration scores don\'t match'
    
    calib_layer2 = CalibrationLayer(pos_size=5, positive_base_rate=0.5, calib_w=10, calib_b=10)
    calib_layer2.build((10,10))
    out2 = calib_layer2.call(scores_pos=tf.constant([-2,1,-1], dtype=tf.float32), 
                           scores_neg=tf.constant([10,11,12], dtype=tf.float32), training=1)

    assert np.around(out1.numpy(), 2) == np.around(out2.numpy(), 2), \
        'CalibrationLayer: calibration scores don\'t match'
