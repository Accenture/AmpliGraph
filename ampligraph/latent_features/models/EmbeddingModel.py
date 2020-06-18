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
    def __init__(self, eta, k, max_ent_size, max_rel_size, algo='DistMult', seed=0):
        super(EmbeddingModel, self).__init__()
        tf.random.set_seed(seed)
        self.max_ent_size = max_ent_size
        self.max_rel_size = max_rel_size
        self.k = k
        
        self.corruption_layer = CorruptionGenerationLayerTrain(eta)
        #self.corruption_layer_eval = CorruptionGenerationLayerEval(len(ent_to_idx), len(ent_to_idx))
        self.encoding_layer = EmbeddingLookupLayer(self.max_ent_size, self.max_rel_size, self.k)
        self.scoring_layer = DistMult()
        
        
    def compute_output_shape(self, inputShape):
        return [(None,3), (None,3)]
    
    @tf.function
    def partition_change_updates(self, unique_entities, ent_emb, rel_emb):
        self.unique_entities = unique_entities
        self.encoding_layer.partition_change_updates(ent_emb, rel_emb)

    @tf.function
    def call(self, inputs, training=0):
        print('training', training)
        #if training==0:
        #    print('tr')
        corruptions = self.corruption_layer(inputs, self.unique_entities)
        #else:
            #print('ev')
            #corruptions = self.corruption_layer_eval(inputs)
        print('done')
        inp_emb = self.encoding_layer(inputs)
        corr_emb = self.encoding_layer(corruptions)

        inp_score = self.scoring_layer(inp_emb)
        corr_score = self.scoring_layer(corr_emb)

        return [inp_score, corr_score]