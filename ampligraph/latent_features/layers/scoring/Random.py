# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf
from ampligraph.latent_features.layers.scoring import register_layer, AbstractScoringLayer


@register_layer('Random')
class Random(AbstractScoringLayer):
    r''' Random scoring
    '''
    def get_config(self):
        config = super(Random, self).get_config()
        return config
    
    def __init__(self, k):
        super(Random, self).__init__(k)

    def _compute_scores(self, triples):
        ''' compute scores using transe scoring function.
        
        Parameters
        ----------
        triples: (n, 3)
            batch of input triples
        
        Returns
        -------
        scores: tf.Tensor(n,1)
            tensor of scores of inputs
        '''

        scores = tf.random.uniform(shape=[tf.shape(triples[0])[0]], seed=0)
        return scores

    def _get_subject_corruption_scores(self, triples, ent_matrix):
        ''' Compute subject corruption scores.
        Evaluate the inputs against subject corruptions and scores of the corruptions.
        
        Parameters
        ----------
        triples: (n, k)
            batch of input embeddings
        ent_matrix: (m, k)
            slice of embedding matrix (corruptions)
        
        Returns
        -------
        scores: tf.Tensor(n, 1)
            scores of subject corruptions (corruptions defined by ent_embs matrix)
        '''
        scores = tf.random.uniform(shape=[tf.shape(triples[0])[0], tf.shape(ent_matrix)[0]], seed=0)
        return scores

    def _get_object_corruption_scores(self, triples, ent_matrix):
        ''' Compute object corruption scores.
        Evaluate the inputs against object corruptions and scores of the corruptions.
        
        Parameters
        ----------
        triples: (n, k)
            batch of input embeddings
        ent_matrix: (m, k)
            slice of embedding matrix (corruptions)
        
        Returns
        -------
        scores: tf.Tensor(n, 1)
            scores of object corruptions (corruptions defined by ent_embs matrix)
        '''
        scores = tf.random.uniform(shape=[tf.shape(triples[0])[0], tf.shape(ent_matrix)[0]], seed=0)
        return scores
