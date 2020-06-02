import os
import tensorflow as tf
import pandas as pd
tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(True)
import numpy as np
from ampligraph.latent_features.layers.scoring import DistMult
from ampligraph.latent_features.layers.encoding import EmbeddingLookupLayer
from ampligraph.latent_features.layers.corruption_generation import CorruptionGenerationLayerEval, CorruptionGenerationLayerTrain


class EmbeddingModel(tf.keras.Model):
    def __init__(self, eta, ent_to_idx, rel_to_idx, algo='DistMult'):
        super(EmbeddingModel, self).__init__()
        
        self.corruption_layer = CorruptionGenerationLayerTrain(eta, len(ent_to_idx))
        self.corruption_layer_eval = CorruptionGenerationLayerEval(len(ent_to_idx), len(ent_to_idx))
        self.encoding_layer = EmbeddingLookupLayer(len(ent_to_idx), len(rel_to_idx), 300)
        self.scoring_layer = DistMult()
        
        
    def compute_output_shape(self, inputShape):
        return [(None,3), (None,3)]

    @tf.function
    def call(self, inputs, training=0):
        print('training', training)
        if training==0:
            print('tr')
            corruptions = self.corruption_layer(inputs)
        else:
            print('ev')
            corruptions = self.corruption_layer_eval(inputs)
        print('done')
        inp_emb = self.encoding_layer(inputs)
        corr_emb = self.encoding_layer(corruptions)

        inp_score = self.scoring_layer(inp_emb)
        corr_score = self.scoring_layer(corr_emb)

        return [inp_score, corr_score]