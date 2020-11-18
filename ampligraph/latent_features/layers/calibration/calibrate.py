import tensorflow as tf
import numpy as np

class CalibrationLayer(tf.keras.layers.Layer):
    
    def __init__(self, pos_size=0, neg_size=0, positive_base_rate=None, **kwargs):
        self.pos_size = pos_size
        self.neg_size = pos_size if neg_size == 0 else neg_size
        
        if positive_base_rate is not None:
            if (positive_base_rate <= 0 or positive_base_rate >= 1):
                raise ValueError("Positive_base_rate must be a value between 0 and 1.")
        else:
            assert pos_size > 0 and neg_size > 0, 'Positive size must be > 0.' 
            
            positive_base_rate = pos_size / (pos_size + neg_size)
            
        self.positive_base_rate = positive_base_rate
        self.w_init = tf.constant_initializer(kwargs.pop('calib_w', 0.0))
        self.b_init = tf.constant_initializer(kwargs.pop('calib_b', 
                                                         np.log((self.neg_size + 1.0) / (
                                                             self.pos_size + 1.0)).astype(np.float32)))
        super(CalibrationLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.calib_w = self.add_weight(
                'calib_w',
                shape=(),
                initializer=self.w_init,
                dtype=tf.float32,
                trainable=True)
        
        self.calib_b = self.add_weight(
                'calib_b',
                shape=(),
                initializer=self.b_init,
                dtype=tf.float32,
                trainable=True)
        self.built = True
        
    def call(self, scores_pos, scores_neg=[], training=0):
        if training:
            scores_all = tf.concat([scores_pos, scores_neg], axis=0)
        else:
            scores_all = scores_pos
            
        logits = -(self.calib_w * scores_all + self.calib_b)
        
        if training:
            labels = tf.concat([tf.cast(tf.fill(scores_pos.shape[0], 
                                            (self.pos_size + 1.0) / (self.pos_size + 2.0)), tf.float32),
                    tf.cast(tf.fill(scores_neg.shape[0], 1 / (self.neg_size + 2.0)), tf.float32)],
                   axis=0)
            weigths_pos = scores_neg.shape[0] / scores_pos.shape[0]
            weights_neg = (1.0 - self.positive_base_rate) / self.positive_base_rate
            weights = tf.concat([tf.cast(tf.fill(scores_pos.shape[0], weigths_pos), tf.float32),
                                 tf.cast(tf.fill(scores_neg.shape[0], weights_neg), tf.float32)], axis=0)
            loss = tf.reduce_mean(weights * tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))
            return loss
        else:
            return tf.math.sigmoid(logits)
        
        
            