import numpy as np
import tensorflow as tf
from ampligraph.latent_features import OPTIMIZER_REGISTRY


def test_sgd_optimizer_const_lr():
    sdg_class = OPTIMIZER_REGISTRY['sgd']
    w = tf.Variable(0.01)
    x = tf.placeholder(shape=(1,), dtype=tf.float32)
    y = tf.placeholder(shape=(1,), dtype=tf.float32)
    pred = w*x
    loss = tf.losses.mean_squared_error(y, pred)
    sgd_optimizer = sdg_class({'lr':0.001}, 10)
    train = sgd_optimizer.minimize(loss)
    with tf.Session() as sess:
        for epoch in range(1,11):
            for batch in range(1, 11):
                feed_dict = {}
                sgd_optimizer.update_feed_dict(feed_dict, batch, epoch)

        assert(list(feed_dict.values())[0]==0.001)
        

def test_sgd_optimizer_fixed_decay():
    sdg_class = OPTIMIZER_REGISTRY['sgd']
    w = tf.Variable(0.01)
    x = tf.placeholder(shape=(1,), dtype=tf.float32)
    y = tf.placeholder(shape=(1,), dtype=tf.float32)
    pred = w*x
    loss = tf.losses.mean_squared_error(y, pred)
    sgd_optimizer = sdg_class({'lr':0.001, 
                               'decay_lr_rate': 2,
                               'cosine_decay':False, 
                               'decay_cycle':10
                              }, 10)
    train = sgd_optimizer.minimize(loss)
    with tf.Session() as sess:
        for epoch in range(1,11):
            for batch in range(1, 11):
                feed_dict = {}
                sgd_optimizer.update_feed_dict(feed_dict, batch, epoch)

        assert(list(feed_dict.values())[0]==0.001)
        sgd_optimizer.update_feed_dict(feed_dict, 1, 11)
        assert(list(feed_dict.values())[0]==0.0005)
        
def test_sgd_optimizer_cosine_decay():
    sdg_class = OPTIMIZER_REGISTRY['sgd']
    w = tf.Variable(0.01)
    x = tf.placeholder(shape=(1,), dtype=tf.float32)
    y = tf.placeholder(shape=(1,), dtype=tf.float32)
    pred = w*x
    loss = tf.losses.mean_squared_error(y, pred)
    sgd_optimizer = sdg_class({'lr':0.001, 
                               'end_lr':0.00001, 
                               'decay_lr_rate': 2,
                               'expand_factor': 2,
                               'cosine_decay':True, 
                               'decay_cycle':10
                              }, 10)
    train = sgd_optimizer.minimize(loss)
    
    with tf.Session() as sess:
        for epoch in range(1,31):
            for batch in range(1, 11):
                feed_dict = {}
                sgd_optimizer.update_feed_dict(feed_dict, batch, epoch)
                if epoch==11 and batch==1:
                    assert(list(feed_dict.values())[0]==0.0005)
                if epoch==6 and batch==1:
                    assert(list(feed_dict.values())[0]==0.000505)
                if epoch==21 and batch==1:
                    assert(list(feed_dict.values())[0]==0.000255)
        
        sgd_optimizer.update_feed_dict(feed_dict, 1, 31)
        assert(list(feed_dict.values())[0]==0.00025)