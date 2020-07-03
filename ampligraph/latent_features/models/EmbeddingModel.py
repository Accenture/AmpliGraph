import os
import tensorflow as tf
import pandas as pd
tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(True)
import numpy as np
from ampligraph.latent_features.layers.scoring import SCORING_LAYER_REGISTRY
from ampligraph.latent_features.layers.encoding import EmbeddingLookupLayer
from ampligraph.latent_features.layers.corruption_generation import CorruptionGenerationLayerEval, CorruptionGenerationLayerTrain


class EmbeddingModel(tf.keras.Model):
    def __init__(self, eta, k, max_ent_size, max_rel_size, algo='DistMult', seed=0):
        super(EmbeddingModel, self).__init__()
        tf.random.set_seed(seed)
        self.max_ent_size = max_ent_size
        self.max_rel_size = max_rel_size
        
        self.scoring_layer = SCORING_LAYER_REGISTRY[algo](k)
        self.k = self.scoring_layer.internal_k
        print('self.scoring_layer.internal_k:', self.scoring_layer.internal_k)
        
        self.corruption_layer = CorruptionGenerationLayerTrain(eta)
        #self.corruption_layer_eval = CorruptionGenerationLayerEval(len(ent_to_idx), len(ent_to_idx))
        self.encoding_layer = EmbeddingLookupLayer(self.max_ent_size, self.max_rel_size, self.k)
        
        
    def compute_output_shape(self, inputShape):
        return [(None,3), (None,3)]
    
    @tf.function
    def partition_change_updates(self, unique_entities, ent_emb, rel_emb):
        self.unique_entities = unique_entities
        self.encoding_layer.partition_change_updates(ent_emb, rel_emb)

    @tf.function
    def call(self, inputs, training=0):
        corruptions = self.corruption_layer(inputs, self.unique_entities)
        inp_emb = self.encoding_layer(inputs)
        corr_emb = self.encoding_layer(corruptions)

        inp_score = self.scoring_layer(inp_emb)
        corr_score = self.scoring_layer(corr_emb)

        return [inp_score, corr_score]
    
    @tf.function(experimental_relax_shapes=True)
    def get_ranks(self, inputs, ent_embs):
        return self.scoring_layer.get_ranks(inputs, ent_embs)