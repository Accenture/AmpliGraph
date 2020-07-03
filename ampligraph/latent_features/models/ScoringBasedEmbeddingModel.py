import os
import tensorflow as tf
import pandas as pd
tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(True)
import numpy as np
from ampligraph.latent_features.layers.scoring import SCORING_LAYER_REGISTRY
from ampligraph.latent_features.layers.encoding import EmbeddingLookupLayer
from ampligraph.latent_features.layers.corruption_generation import CorruptionGenerationLayerTrain


class ScoringBasedEmbeddingModel(tf.keras.Model):
    def __init__(self, eta, k, max_ent_size, max_rel_size, scoring_type='DistMult', seed=0):
        '''
        Initializes the scoring based embedding model
        
        Parameters:
        -----------
        eta: int
            num of negatives to use during training per triple
        k: int
            embedding size
        max_ent_size: int
            max entities that can occur in any partition
        max_rel_size: int
            max relations that can occur in any partition
        algo: string
            name of the scoring layer to use
        seed: int 
            random seed
        '''
        super(ScoringBasedEmbeddingModel, self).__init__()
        # set the random seed
        tf.random.set_seed(seed)
        
        self.max_ent_size = max_ent_size
        self.max_rel_size = max_rel_size
        
        # get the scoring layer
        self.scoring_layer = SCORING_LAYER_REGISTRY[scoring_type](k)
        # get the actual k depending on scoring layer 
        # Ex: complex model uses k embeddings for real and k for img side. 
        # so internally it has 2*k. where as transE uses k.
        self.k = self.scoring_layer.internal_k
        
        # create the corruption generation layer - generates eta corruptions during training
        self.corruption_layer = CorruptionGenerationLayerTrain(eta)
        
        # assume that you have max_ent_size unique entities - if it is single partition
        # this would change if we use partitions - based on which partition is in memory
        # this is used by corruption_layer to sample eta corruptions
        self.unique_entities = self.max_ent_size
        
        # Create the embedding lookup layer. 
        # size of entity emb is max_ent_size * k and relation emb is  max_rel_size * k
        self.encoding_layer = EmbeddingLookupLayer(self.max_ent_size, self.max_rel_size, self.k)
        
        
    def compute_output_shape(self, inputShape):
        ''' returns the output shape of outputs of call function
        
        Parameters:
        -----------
        input_shape: 
            shape of inputs of call function
        
        Returns:
        --------
        output_shape:
            shape of outputs of call function
        '''
        # input triple score (batch_size, 1) and corruption score (batch_size * eta, 1)
        return [(None, 1), (None, 1)]
    
    @tf.function
    def partition_change_updates(self, unique_entities, ent_emb, rel_emb):
        ''' perform the changes that are required when the partition is changed during training
        
        Parameters:
        -----------
        unique_entities: 
            unique entities of the partition
        ent_emb:
            entity embeddings that need to be trained for the partition 
            (all triples of the partition will have embeddings in this matrix)
        rel_emb:
            relation embeddings that need to be trained for the partition 
            (all triples of the partition will have embeddings in this matrix)
        
        '''
        # save the unique entities of the partition : will be used for corruption generation
        self.unique_entities = unique_entities
        # update the trainable variable in the encoding layer
        self.encoding_layer.partition_change_updates(ent_emb, rel_emb)

    @tf.function
    def call(self, inputs):
        '''
        Computes the scores of the triples and returns the corruption scores as well
        
        Parameters:
        -----------
        inputs: (n, 3)
            batch of input triples
        
        Returns:
        --------
        list: 
            list of input scores along with their corruptions
        '''
        # generate the corruptions for the input triples
        corruptions = self.corruption_layer(inputs, self.unique_entities)
        # lookup embeddings of the inputs
        inp_emb = self.encoding_layer(inputs)
        # lookup embeddings of the inputs
        corr_emb = self.encoding_layer(corruptions)
        
        # score the inputs
        inp_score = self.scoring_layer(inp_emb)
        # score the corruptions
        corr_score = self.scoring_layer(corr_emb)

        return [inp_score, corr_score]
    
    @tf.function(experimental_relax_shapes=True)
    def get_ranks(self, inputs, ent_embs):
        '''
        Evaluate the inputs against corruptions and return ranks
        
        Parameters:
        -----------
        inputs: (n, 3)
            batch of input triples
        ent_embs: (m, k)
            slice of embedding matrix (corruptions)
        
        Returns:
        --------
        rank: (n, 2)
            ranks by corrupting against subject corruptions and object corruptions 
            (corruptions defined by ent_embs matrix)
        '''
        return self.scoring_layer.get_ranks(inputs, ent_embs)