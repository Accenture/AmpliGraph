import os
import tensorflow as tf 

from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.eager import def_function

import pandas as pd
tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(False)
import numpy as np
from ampligraph.evaluation.metrics import mrr_score, hits_at_n_score, mr_score
from ampligraph.datasets import data_adapter
from ampligraph.latent_features.layers.scoring import SCORING_LAYER_REGISTRY
from ampligraph.latent_features.layers.encoding import EmbeddingLookupLayer
from ampligraph.latent_features.layers.corruption_generation import CorruptionGenerationLayerTrain
from tensorflow.python.keras import metrics as metrics_mod
import copy
import shelve
import pickle
from ampligraph.datasets import DataIndexer

from ampligraph.latent_features import optimizers
from ampligraph.latent_features import loss_functions

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
        self.num_ents = self.max_ent_size
        
        # Create the embedding lookup layer. 
        # size of entity emb is max_ent_size * k and relation emb is  max_rel_size * k
        self.encoding_layer = EmbeddingLookupLayer(self.max_ent_size, self.max_rel_size, self.k)
        
        self.is_partitioned_training = False
        
        
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
    def partition_change_updates(self, num_ents, ent_emb, rel_emb):
        ''' perform the changes that are required when the partition is changed during training
        
        Parameters:
        -----------
        num_ents: 
            number of unique entities in the partition
        ent_emb:
            entity embeddings that need to be trained for the partition 
            (all triples of the partition will have embeddings in this matrix)
        rel_emb:
            relation embeddings that need to be trained for the partition 
            (all triples of the partition will have embeddings in this matrix)
        
        '''
        # save the unique entities of the partition : will be used for corruption generation
        self.num_ents = num_ents
        # update the trainable variable in the encoding layer
        self.encoding_layer.partition_change_updates(ent_emb, rel_emb)

    @tf.function(experimental_relax_shapes=True)
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
        corruptions = self.corruption_layer(inputs, self.num_ents)
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
    def _get_ranks(self, inputs, ent_embs, start_id, end_id, filters, corrupt_side='s,o'):
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
        return self.scoring_layer.get_ranks(inputs, ent_embs, start_id, end_id, filters, corrupt_side)
    
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
            score_pos, score_neg = self(tf.cast(data, tf.int32), training=0)
            # compute the loss
            loss = self.compiled_loss(score_pos, score_neg, self.eta)
            # regularizer - will be in a separate class like ampligraph 1
            loss += (0.0001 * (tf.reduce_sum(tf.pow(tf.abs(self.encoding_layer.ent_emb), 3)) + \
                              tf.reduce_sum(tf.pow(tf.abs(self.encoding_layer.rel_emb), 3))))

        self.optimizer.minimize(loss, 
                                self.encoding_layer.ent_emb, 
                                self.encoding_layer.rel_emb,
                                tape)
        
        #self.compiled_metrics.update_state(loss)
        return {m.name: m.result() for m in self.metrics}
    
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
            validation_batch_size=100,
            validation_freq=2,
            validation_filter = False,
            use_partitioning=False):
        
        self.compiled_metric = tf.keras.metrics.Mean
        self._assert_compile_was_called()
        if validation_split:
            pass
            # use train test unseen to split training set
            
        with training_utils.RespectCompiledTrainableState(self):
            # create data handler
            self.data_handler = data_adapter.DataHandler(x, 
                                                         model=self, 
                                                         batch_size=batch_size, 
                                                         dataset_type='train', 
                                                         epochs=epochs,
                                                         use_filter=False,
                                                         use_partitioning=use_partitioning)
            self.data_indexer = self.data_handler.get_mapper()
        
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
            self.is_partitioned_training = use_partitioning
            
            if self.is_partitioned_training:
                self.num_buckets = self.data_handler._adapter.num_buckets
                
            train_function = self.make_train_function()
            callbacks.on_train_begin()

            total_loss = []
            for epoch, iterator in self.data_handler.enumerate_epochs():
                # TODO: remove this later
                self.global_epoch = epoch
                callbacks.on_epoch_begin(epoch)
                with self.data_handler.catch_stop_iteration():
                    for step in self.data_handler.steps():
                        callbacks.on_train_batch_begin(step)
                        with tf.device('{}'.format('GPU:0')):
                            logs = train_function(iterator)
                        #total_loss.append(loss)
                        callbacks.on_train_batch_end(step, logs)

                epoch_logs = copy.copy(logs)
                
                if validation_data and self._should_eval(epoch, validation_freq):
                    ranks = self.evaluate(validation_data,
                                          batch_size=validation_batch_size or batch_size,
                                          use_filter=validation_filter)
                    val_logs = {'val_mrr': mrr_score(ranks), 
                                'val_mr': mr_score(ranks),
                                'val_hits@1': hits_at_n_score(ranks, 1),
                                'val_hits@10': hits_at_n_score(ranks, 10),
                               }
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch, epoch_logs)
                #print('\n\n\n\nloss------------------{}:{}'.format(epoch, np.mean(total_loss)))
                if self.stop_training:
                    break

            callbacks.on_train_end()
            
            return self.history
    
    def _get_optimizer(self, optimizer):
        return optimizers.get(optimizer)
        
    def save_weights(self,
               filepath,
               overwrite=True,
               save_format=None,
               options=None):
        super(ScoringBasedEmbeddingModel, self).save_weights(filepath, 
                                                     overwrite, 
                                                     save_format, 
                                                     options)
        with open(filepath+'.ampkl', "wb") as f:
            indexer = self.data_handler.get_mapper()
            metadata = {'inmemory': indexer.in_memory,
                        'entities_dict': indexer.entities_dict,
                        'reversed_entities_dict': indexer.reversed_entities_dict,
                        'relations_dict': indexer.relations_dict,
                        'reversed_relations_dict': indexer.reversed_relations_dict,
                        'is_partitioned_training': self.is_partitioned_training
                       }
            
            if self.is_partitioned_training:
                metadata['num_buckets'] = self.num_buckets
                
            pickle.dump(metadata, f)

        
    def load_weights(self,
                     filepath,
                     by_name=False,
                     skip_mismatch=False,
                     options=None):
        super(ScoringBasedEmbeddingModel, self).load_weights(filepath, 
                                                     by_name, 
                                                     skip_mismatch, 
                                                     options)  
        with open(filepath+'.ampkl', "rb") as f:
            metadata = pickle.load(f)
            self.data_indexer = DataIndexer([], 
                                            metadata['inmemory'],
                                            metadata['entities_dict'],
                                            metadata['reversed_entities_dict'],
                                            metadata['relations_dict'],
                                            metadata['reversed_relations_dict'])
            self.is_partitioned_training = metadata['is_partitioned_training']
            
            if self.is_partitioned_training:
                self.num_buckets = metadata['num_buckets']
            
    
    def compile(self,
          optimizer='adam',
          loss=None,
          **kwargs):
        
        # self._validate_compile(optimizer, metrics, **kwargs)
        self.optimizer = self._get_optimizer(optimizer)
        self._reset_compile_cache()
        self._is_compiled = True
        self.compiled_loss = loss_functions.get(loss)
        self.compiled_metrics = metrics_mod.Mean(name='loss')  # Total loss.
        
    @property
    def metrics(self):
        metrics = []
        if self._is_compiled:
            if self.compiled_loss is not None:
                metrics += self.compiled_loss.metrics
                
        return metrics
        
        
    def get_emb_matrix_test(self, part_number = 1, number_of_parts=1):
        if number_of_parts==1:
            return self.encoding_layer.ent_emb, 0, self.encoding_layer.ent_emb.shape[0] - 1
        else:
            with shelve.open('ent_partition') as ent_partition:
                batch_size = int(np.ceil(len(ent_partition.keys())/number_of_parts))
                indices = np.arange(part_number * batch_size, (part_number + 1) * batch_size).astype(np.str)
                emb_matrix = []
                for idx in indices:
                    try:
                        emb_matrix.append(ent_partition[idx])
                    except KeyError:
                        break
                return np.array(emb_matrix), int(indices[0]), int(indices[-1])
            
        
    def make_test_function(self):

        if self.test_function is not None:
            return self.test_function

        def test_function(iterator):
            number_of_parts = 1
            
            if self.is_partitioned_training:
                number_of_parts = self.num_buckets
            
            if self.use_filter:
                inputs, filters = next(iterator)
            else:
                inputs = next(iterator)
                filters = ()
            
            # compute the output shape based on the type of corruptions to be used
            output_shape = 0
            if 's' in self.corrupt_side:
                output_shape += 1
            
            if 'o' in self.corrupt_side:
                output_shape += 1
            
            # create an array to store the ranks based on output shape
            overall_rank = np.zeros((inputs[0].shape[0], output_shape), dtype=np.int32)
            overall_rank_unf = np.zeros((inputs[0].shape[0], output_shape), dtype=np.int32)
            
            # run the loop based on number of parts in which the original emb matrix was generated
            for j in range(number_of_parts):
                # get the embedding matrix
                emb_mat, start_ent_id, end_ent_id = self.get_emb_matrix_test(j, number_of_parts)
                # compute the rank
                ranks = self._get_ranks(inputs, emb_mat, 
                                        start_ent_id, end_ent_id, filters, self.corrupt_side )
                # store it in the output
                for i in range(ranks.shape[0]):
                    overall_rank[:, i] +=  ranks[i, :]
                    
            # if corruption type is s+o then add s and o ranks and return the added ranks
            if self.corrupt_side == 's+o':
                # add the subject and object ranks
                overall_rank[:, 0] += overall_rank[:, 1]
                # return the added ranks
                return overall_rank[:, :1] 
                
            return overall_rank


        #self.test_function = def_function.function(
        #      test_function, experimental_relax_shapes=True)
        
        self.test_function = test_function

        return self.test_function
    
    def evaluate(self,
                   x=None,
                   batch_size=32,
                   verbose=True,
                   use_filter=False,
                   corrupt_side='s,o',
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
        self.data_handler_test = data_adapter.DataHandler(x, 
                                                          batch_size=batch_size, 
                                                          dataset_type='test', 
                                                          epochs=1, 
                                                          use_filter=use_filter,
                                                          use_indexer = self.data_indexer)
        
        assert corrupt_side in ['s', 'o', 's,o', 's+o'], 'Invalid value for corrupt_side'
        
        self.corrupt_side = corrupt_side
        self.use_filter = use_filter or type(use_filter)==dict
        

        if self.is_partitioned_training:
            self.data_handler_test.temperorily_set_emb_matrix('ent_partition', 'rel_partition')

        else:
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
                    with tf.device('{}'.format('GPU:0')):
                        overall_rank = test_function(iterator)
                    overall_rank += 1
                    self.all_ranks.append(overall_rank)
                    callbacks.on_test_batch_end(step)
        callbacks.on_test_end()
        return np.concatenate(self.all_ranks)