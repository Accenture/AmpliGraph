import os
import tensorflow as tf

from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.eager import def_function

import pandas as pd
tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(False)
import numpy as np

from ampligraph.datasets import GraphDataLoader
from ampligraph.latent_features.layers.scoring import SCORING_LAYER_REGISTRY
from ampligraph.latent_features.layers.encoding import EmbeddingLookupLayer
from ampligraph.latent_features.layers.corruption_generation import CorruptionGenerationLayerTrain
from tensorflow.python.keras import metrics as metrics_mod
import copy


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
        
        self.eta = eta
        
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
        self.unique_entities = tf.range(self.max_ent_size)
        
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
        corruptions = self.corruption_layer(inputs, len(self.unique_entities))
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
    def _get_ranks(self, inputs, ent_embs):
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
    
    def train_step(self, data):
        '''
        Training step
        
        Parameters:
        -----------
        data: (n, 3)
            batch of input triples (true positives)
        
        Returns:
        --------
        loss: tf.float
            loss computed by evaluating a batch of True positives against corresponding (eta) corruptions
        '''
        with tf.GradientTape() as tape:
            # get the model predictions
            preds = self(data, training=0)
            # compute the loss
            loss = self.loss(preds, self.eta)
            # regularizer - will be in a separate class like ampligraph 1
            loss += (0.0001 * (tf.reduce_sum(tf.pow(tf.abs(self.encoding_layer.ent_emb), 3)) + \
                              tf.reduce_sum(tf.pow(tf.abs(self.encoding_layer.rel_emb), 3))))

        # compute the grads
        gradients = tape.gradient(loss, [self.encoding_layer.ent_emb, self.encoding_layer.rel_emb])
        # update the trainable params
        self.optimizer.apply_gradients(zip(gradients, [self.encoding_layer.ent_emb, self.encoding_layer.rel_emb]))
        self._loss_metric.update_state(loss)
        return {self._loss_metric.name: self._loss_metric.result()}
    
    def make_train_function(self):
        ''' Returns the handle to training step function. The handle takes one batch by iterating over the
        iterator and performs one training step (on the batch) and computes the loss. 
        This method is called by `Model.fit` 
        This function is cached the first time `Model.fit` is called. 
        The cache is cleared whenever `Model.compile` is called.
        
        Returns:
            Function. 
        '''
        if self.train_function is not None:
            return self.train_function
        
        def train_function(iterator):
            ''' This is the function whose handle will be returned.
            
            Parameters:
            -----------
            iterator: tf.data.Iterator
                Data iterator
                
            Returns:
            --------
            out: dict
              return a `dict` containing values that will be passed to `tf.keras.Callbacks.on_train_batch_end`
            '''
            data = next(iterator)
            output = self.train_step(data)
            return output
            
        self.train_function = train_function
        return self.train_function
    
    def fit(self,
          x=None,
          batch_size=None,
          epochs=1,
          verbose=True,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          initial_epoch=0,
          validation_batch_size=None,
          validation_freq=1):
        
        self.compiled_metric = tf.keras.metrics.Mean
        self._assert_compile_was_called()
        if validation_split:
            pass
            # use train test unseen to split training set
            
        with training_utils.RespectCompiledTrainableState(self):
            # create data handler
            self.data_handler = GraphDataLoader(x, batch_size=batch_size, dataset_type="train", epochs=epochs)
        
        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                                                        callbacks,
                                                        add_history=True,
                                                        add_progbar=verbose != 0,
                                                        model=self,
                                                        verbose=verbose,
                                                        epochs=epochs)
        self.stop_training = False
        train_function = self.make_train_function()
        callbacks.on_train_begin()
        
        total_loss = []
        for epoch, iterator in self.data_handler.enumerate_epochs():
            callbacks.on_epoch_begin(epoch)
            with self.data_handler.catch_stop_iteration():
                for step in self.data_handler.steps():
                    callbacks.on_train_batch_begin(step)
                    with tf.device('{}'.format('GPU:0')):
                        logs = train_function(iterator)
                    #total_loss.append(loss)
                    callbacks.on_train_batch_end(step, logs)
                    
            epoch_logs = copy.copy(logs)
                    
            
            callbacks.on_epoch_end(epoch, epoch_logs)
            #print('\n\n\n\nloss------------------{}:{}'.format(epoch, np.mean(total_loss)))
            if self.stop_training:
                break
        
        callbacks.on_train_end()
        return self.history
    
    def _get_optimizer(self, optimizer):
        return tf.optimizers.Adam(lr=0.001)
    
    def compile(self,
          optimizer='adam',
          loss=None,
          metrics=None,
          **kwargs):
        
        
        # self._validate_compile(optimizer, metrics, **kwargs)
        self.optimizer = self._get_optimizer(optimizer)
        self._reset_compile_cache()
        self._is_compiled = True
        self.loss = def_function.function(loss, experimental_relax_shapes=True)
        self._loss_metric = metrics_mod.Mean(name='loss')  # Total loss.
        
        
    def make_test_function(self):

        if self.test_function is not None:
            return self.test_function

        def test_function(iterator):
            inputs = next(iterator)
            sub_rank, obj_rank = self._get_ranks(inputs, self.encoding_layer.ent_emb)
            return [sub_rank, obj_rank]


        #self.test_function = def_function.function(
        #      test_function, experimental_relax_shapes=True)
        
        self.test_function = test_function

        return self.test_function
    
    def evaluate(self,
               x=None,
               batch_size=32,
               verbose=True,
               callbacks=None):
        '''
        Evaluate the inputs against corruptions and return ranks

        Parameters:
        -----------
        x: (n, 3)
            batch of input triples
        batch_size: int
            batch_size
        verbose: bool
            Verbosity mode.
        callbacks: keras.callbacks.Callback
            List of `keras.callbacks.Callback` instances. List of callbacks to apply during evaluation.

        Returns:
        --------
        rank: (n, 2)
            ranks by corrupting against subject corruptions and object corruptions 
            (corruptions defined by ent_embs matrix)
        '''
        #self._assert_compile_was_called()
        
        self.data_handler_test = GraphDataLoader(x, batch_size=batch_size, dataset_type="test", epochs=1, use_indexer = self.data_handler.backend.mapper)
        self.data_handler_test.temperorily_set_emb_matrix(self.encoding_layer.ent_emb.numpy(), self.encoding_layer.rel_emb.numpy())

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=self.data_handler_test.inferred_steps)
        
        test_function = self.make_test_function()
        callbacks.on_test_begin()
        
        self.all_ranks = []
        
        
        for _, iterator in self.data_handler_test.enumerate_epochs(): 
            with self.data_handler_test.catch_stop_iteration():
                for step in self.data_handler_test.steps():
                    callbacks.on_test_batch_begin(step)
                    sub_rank, obj_rank = test_function(iterator)

                    sub_rank = sub_rank.numpy() + 1
                    obj_rank = obj_rank.numpy() + 1
                    self.all_ranks.append(np.array([sub_rank, obj_rank]).T)
                    callbacks.on_test_batch_end(step)
        callbacks.on_test_end()
        return np.concatenate(self.all_ranks)

