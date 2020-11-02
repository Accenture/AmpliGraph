import tensorflow as tf 
import copy
import shelve
import pickle
import numpy as np

from ampligraph.evaluation.metrics import mrr_score, hits_at_n_score, mr_score
from ampligraph.datasets import data_adapter
from ampligraph.latent_features.layers.scoring import SCORING_LAYER_REGISTRY
from ampligraph.latent_features.layers.encoding import EmbeddingLookupLayer
from ampligraph.latent_features.layers.calibration import CalibrationLayer
from ampligraph.latent_features.layers.corruption_generation import CorruptionGenerationLayerTrain
from ampligraph.datasets import DataIndexer
from ampligraph.latent_features import optimizers
from ampligraph.latent_features import loss_functions
from ampligraph.evaluation import train_test_split_no_unseen
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.eager import def_function
from tensorflow.python.keras import metrics as metrics_mod


tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(False)


class ScoringBasedEmbeddingModel(tf.keras.Model):
    def __init__(self, eta, k, scoring_type='DistMult', seed=0):
        '''
        Initializes the scoring based embedding model
        
        Parameters:
        -----------
        eta: int
            num of negatives to use during training per triple
        k: int
            embedding size
        scoring_type: string
            name of the scoring layer to use
        seed: int 
            random seed
        '''
        super(ScoringBasedEmbeddingModel, self).__init__()
        # set the random seed
        tf.random.set_seed(seed)
        
        self.max_ent_size = None
        self.max_rel_size = None
        
        self.eta = eta
        
        # get the scoring layer
        self.scoring_layer = SCORING_LAYER_REGISTRY[scoring_type](k)
        # get the actual k depending on scoring layer 
        # Ex: complex model uses k embeddings for real and k for img side. 
        # so internally it has 2*k. where as transE uses k.
        self.k = self.scoring_layer.internal_k
        
        # create the corruption generation layer - generates eta corruptions during training
        self.corruption_layer = CorruptionGenerationLayerTrain()
        
        # assume that you have max_ent_size unique entities - if it is single partition
        # this would change if we use partitions - based on which partition is in memory
        # this is used by corruption_layer to sample eta corruptions
        self.num_ents = self.max_ent_size
        
        # Create the embedding lookup layer. 
        # size of entity emb is max_ent_size * k and relation emb is  max_rel_size * k
        self.encoding_layer = EmbeddingLookupLayer(self.k, self.max_ent_size, self.max_rel_size)
        
        # Flag to indicate whether the partitioned training is being done
        self.is_partitioned_training = False
        # Variable related to data indexing (entity to idx mapping)
        self.data_indexer = True
        
        self.is_calibrated = False
        
        self.seed = seed
  
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
        if self.encoding_layer.built:
            # update the trainable variable in the encoding layer
            self.encoding_layer.partition_change_updates(ent_emb, rel_emb)
        else:
            # if the encoding layer has not been built then store it as an initializer
            # this would be the case of during partitioned training (first batch)
            self.encoding_layer.set_ent_rel_initial_value(ent_emb, rel_emb)

    def call(self, inputs, training=False):
        '''
        Computes the scores of the triples and returns the corruption scores as well
        
        Parameters:
        -----------
        inputs: (n, 3)
            batch of input triples
        
        Returns:
        --------
        out: list
            list of input scores along with their corruptions
        '''
        # lookup embeddings of the inputs
        inp_emb = self.encoding_layer(inputs)
        # score the inputs
        inp_score = self.scoring_layer(inp_emb)
        # score the corruptions

        if training:
            # generate the corruptions for the input triples
            corruptions = self.corruption_layer(inputs, self.num_ents, self.eta)
            # lookup embeddings of the inputs
            corr_emb = self.encoding_layer(corruptions)
            corr_score = self.scoring_layer(corr_emb)

            return inp_score, corr_score

        else:
            return inp_score

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
        start_id: int
            original id of the first row of embedding matrix (used during partitioned approach)
        end_id: int 
            original id of the last row of embedding matrix (used during partitioned approach)
        filters: list of lists 
            size of list is either 1 or 2 depending on the corrupt_side. 
            size of the internal list is equal to the size of the input triples.
            Each list contains array of filters(True Positives) related to the specified side for the 
            corresponding triple in inputs. 
        corrupt_side: string
            which side to corrupt during evaluation
        Returns:
        --------
        rank: (n, num of sides being corrupted)
            ranks by corrupting against subject corruptions and object corruptions 
            (corruptions defined by ent_embs matrix)
        '''
        if not self.is_partitioned_training:
            inputs = [tf.nn.embedding_lookup(self.encoding_layer.ent_emb, inputs[:, 0]),
                      tf.nn.embedding_lookup(self.encoding_layer.rel_emb, inputs[:, 1]),
                      tf.nn.embedding_lookup(self.encoding_layer.ent_emb, inputs[:, 2])]
            
        return self.scoring_layer.get_ranks(inputs, ent_embs, start_id, end_id, filters, corrupt_side)
    
    def build(self, input_shape):
        ''' Overide the build function of the Model class. This is called on the first cal; to __call__
        In this function we set some internal parameters of the encoding layers (which are needed to build that layer)
        based on the input data supplied by the user while calling the fit function
        '''
        # set the max number of the entities that will be trained per partition
        # in case of non-partitioned training, it is equal to the total number of entities of the dataset
        self.encoding_layer.max_ent_size = self.max_ent_size
        # set the max number of relations being trained just like above
        self.encoding_layer.max_rel_size = self.max_rel_size
        self.num_ents = self.max_ent_size
        self.built = True
        
    def train_step(self, data):
        '''
        Training step
        
        Parameters:
        -----------
        data: (n, 3)
            batch of input triples (true positives)
        
        Returns:
        --------
        out: dict
            dictionary of metrics computed on the outputs (eg: loss)
        '''
        with tf.GradientTape() as tape:
            # get the model predictions
            score_pos, score_neg = self(tf.cast(data, tf.int32), training=1)
            # compute the loss
            loss = self.compiled_loss(score_pos, score_neg, self.eta, regularization_losses=self.losses)

        # minimize the loss and update the trainable variables
        self.optimizer.minimize(loss, 
                                self.encoding_layer.ent_emb, 
                                self.encoding_layer.rel_emb,
                                tape)

        return {m.name: m.result() for m in self.metrics}
    
    def make_train_function(self):
        ''' Similar to keras lib, this function returns the handle to training step function. 
        It processes one batch of data by iterating over the dataset iterator and computes the loss and optimizes on it.

        Returns:
        --------
        out: Function handle.
              Handle to the training step function  
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
            output: dict
              return a `dict` containing values that will be passed to `tf.keras.Callbacks.on_train_batch_end`
            '''
            data = next(iterator)
            output = self.train_step(data)
            return output
        
        if not self.run_eagerly:
            train_function = def_function.function(
                train_function, experimental_relax_shapes=True)
            
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
            validation_freq=50,
            validation_filter=False,
            use_partitioning=False):
        '''Fit the model of the user data.
        
        Parameters:
        -----------
        x: np.array(n,3), string, GraphDataLoader instance, AbstractGraphPartitioner instance
            Data OR Filename of the data file OR Data Handle - that would be used for training
        batch_size: int
            batch size to use during training. 
            May be overridden if x is GraphDataLoader or AbstractGraphPartitioner instance
        epochs: int
            Number of epochs to train (default: 1)
        verbose: bool
            verbosity (default: True)
        callbacks: list of tf.keras.callbacks.Callback
            list of callbacks to be used during training (default: None)
        validation_split: float
            validation split that would be carved out from x (default: 0.0)
            (currently supported only when x is numpy array)
        validation_data: np.array(n,3), string, GraphDataLoader instance, AbstractGraphPartitioner instance
            Data OR Filename of the data file OR Data Handle - that would be used for validation
        shuffle: bool
            indicates whether to shuffle the data after every epoch during training (default: True)
        initial epoch: int
            initial epoch number (default: 1)
        validation_batch_size: int
            batch size to use during validation. (default: 100)
            May be overridden if validation_data is GraphDataLoader or AbstractGraphPartitioner instance
        validation_freq: int
            indicates how often to validate (default: 50)
        validation_filter: bool or dict
            validation filter to be used. 
        use_partitioning: bool
            flag to indicate whether to use partitioning or not.
            May be overridden if x is an AbstractGraphPartitioner instance
            
        Returns:
        --------
        history: A `History` object. Its `History.history` attribute is a record of training loss values, 
        as well as validation loss values and validation metrics values.
        '''
        # verifies if compile has been called before calling fit
        self._assert_compile_was_called()
        
        # use train test unseen to split training set
        if validation_split:
            assert isinstance(x, np.ndarray), 'Validation split supported for numpy arrays only!'
            x, validation_data = train_test_split_no_unseen(x, 
                                                            test_size=validation_split, 
                                                            seed=self.seed, 
                                                            allow_duplication=False)
            
        with training_utils.RespectCompiledTrainableState(self):
            # create data handler for the data
            self.data_handler = data_adapter.DataHandler(x,
                                                         model=self, 
                                                         batch_size=batch_size, 
                                                         dataset_type='train', 
                                                         epochs=epochs,
                                                         initial_epoch=initial_epoch,
                                                         use_filter=False,
                                                         # if model is already trained use the old indexer
                                                         use_indexer=self.data_indexer,
                                                         use_partitioning=use_partitioning)
            # get the mapping details
            self.data_indexer = self.data_handler.get_mapper()
            # get the maximum entities and relations that will be trained (useful during partitioning)
            self.max_ent_size = self.data_handler._adapter.max_entities  
            self.max_rel_size = self.data_handler._adapter.max_relations

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(callbacks,
                                                          add_history=True,
                                                          add_progbar=verbose != 0,
                                                          model=self,
                                                          verbose=verbose,
                                                          epochs=epochs)
                
            # This variable is used by callbacks to stop training in case of any error
            self.stop_training = False
            self.is_partitioned_training = self.data_handler.using_partitioning
            
            # set some partition related params if it is partitioned training
            if self.is_partitioned_training:
                self.partitioner_k = self.data_handler._adapter.partitioner_k
                self.encoding_layer.max_ent_size = self.max_ent_size
                self.encoding_layer.max_rel_size = self.max_rel_size
                
            # make the train function that will be used to process each batch of data
            train_function = self.make_train_function()
            # before training begins call this callback function
            callbacks.on_train_begin()

            # enumerate over the data
            for epoch, iterator in self.data_handler.enumerate_epochs():
                # current epcoh number
                self.current_epoch = epoch
                # before epoch begins call this callback function
                callbacks.on_epoch_begin(epoch)
                
                # handle the stop iteration of data iterator in this scope
                with self.data_handler.catch_stop_iteration():
                    # iterate over the dataset
                    for step in self.data_handler.steps():
                        
                        # before a batch is processed call this callback function
                        callbacks.on_train_batch_begin(step)

                        # process this batch
                        logs = train_function(iterator) 
                        # after a batch is processed call this callback function
                        callbacks.on_train_batch_end(step, logs)

                # store the logs of the last batch of the epoch
                epoch_logs = copy.copy(logs)

                # if validation is enabled
                if validation_data is not None and self._should_eval(epoch, validation_freq):
                    # evaluate on the validation
                    ranks = self.evaluate(validation_data,
                                          batch_size=validation_batch_size or batch_size,
                                          use_filter=validation_filter)
                    # compute all the metrics
                    val_logs = {'val_mrr': mrr_score(ranks), 
                                'val_mr': mr_score(ranks),
                                'val_hits@1': hits_at_n_score(ranks, 1),
                                'val_hits@10': hits_at_n_score(ranks, 10)}
                    # update the epoch logs with validation details
                    epoch_logs.update(val_logs)

                # after an epoch is completed, call this callback function
                callbacks.on_epoch_end(epoch, epoch_logs)
                if self.stop_training:
                    break

            # on training end call this method
            callbacks.on_train_end()
            # all the training and validation logs are stored in the history object by keras.Model
            return self.history
        
    def save_weights(self,
                     filepath,
                     overwrite=True,
                     save_format=None,
                     options=None):
        ''' Save the trainable weights and other parameters required to load back the model.
        '''
        # TODO: verify other formats
        
        # call the base class method to save the weights
        if not self.is_partitioned_training:
            super(ScoringBasedEmbeddingModel, self).save_weights(filepath, 
                                                                 overwrite, 
                                                                 save_format, 
                                                                 options)
        # store ampligraph specific metadata
        with open(filepath + '.ampkl', "wb") as f:
            metadata = {'inmemory': self.data_indexer.in_memory,
                        'entities_dict': self.data_indexer.entities_dict,
                        'reversed_entities_dict': self.data_indexer.reversed_entities_dict,
                        'relations_dict': self.data_indexer.relations_dict,
                        'reversed_relations_dict': self.data_indexer.reversed_relations_dict,
                        'is_partitioned_training': self.is_partitioned_training,
                        'max_ent_size': self.max_ent_size,
                        'max_rel_size': self.max_rel_size,
                        'eta': self.eta,
                        'k': self.k,
                        'is_calibrated': self.is_calibrated
                        }
            
            if self.is_partitioned_training:
                metadata['partitioner_k'] = self.partitioner_k
                
            if self.is_calibrated:
                metadata['calib_w'] = self.calibration_layer.calib_w.numpy()
                metadata['calib_b'] = self.calibration_layer.calib_b.numpy()

            pickle.dump(metadata, f)

    def build_full_model(self, batch_size=100):
        ''' this method is called while loading the weights to build the model
        '''
        self.build((batch_size, 3))
        for i in range(len(self.layers)):
            self.layers[i].build((batch_size, 3))
            self.layers[i].built = True

    def load_weights(self,
                     filepath,
                     by_name=False,
                     skip_mismatch=False,
                     options=None):
        ''' Load the trainable weights and other parameters.
        '''
        with open(filepath + '.ampkl', "rb") as f:
            metadata = pickle.load(f)
            self.data_indexer = DataIndexer([], 
                                            metadata['inmemory'],
                                            metadata['entities_dict'],
                                            metadata['reversed_entities_dict'],
                                            metadata['relations_dict'],
                                            metadata['reversed_relations_dict'])
            self.is_partitioned_training = metadata['is_partitioned_training']
            self.max_ent_size = metadata['max_ent_size']
            self.max_rel_size = metadata['max_rel_size']
            if self.is_partitioned_training:
                self.partitioner_k = metadata['partitioner_k']
            self.is_calibrated = metadata['is_calibrated']
            if self.is_calibrated:
                self.calibration_layer = CalibrationLayer(calib_w=metadata['calib_w'],
                                                          calib_b=metadata['calib_b'])
        self.build_full_model()
        if not self.is_partitioned_training:
            super(ScoringBasedEmbeddingModel, self).load_weights(filepath, 
                                                                 by_name, 
                                                                 skip_mismatch, 
                                                                 options)

    def compile(self,
                optimizer='adam',
                loss=None,
                entity_relation_initializer='glorot_uniform',
                entity_relation_regularizer=None,
                **kwargs):
        ''' Compile the model
        
        Parameters:
        -----------
        optimizer: String (name of optimizer) or optimizer instance. 
            See `tf.keras.optimizers`.
        loss: String (name of objective function), objective function or
            `ampligraph.latent_features.losses_function.Loss` instance. 
            See `ampligraph.latent_features.losses_function`. 
            An objective function is any callable with the signature `loss = fn(score_true, score_corr, eta)`
        entity_relation_initializer: String (name of objective function), objective function or 
            `tf.keras.initializers.Initializer` instance
            An objective function is any callable with the signature `init = fn(shape)`
            Initializer of the entity and relation embeddings. This is either a single value or a list of size 2.
            If it is a single value, then both the entities and relations will be initialized based on 
            the same initializer. if it is a list, the first initializer will be used for entities and second 
            for relations.
        entity_relation_regularizer: String (name of objective function), objective function or 
            `tf.keras.regularizers.Regularizer` instance
            Regularizer of entities and relations.
            If it is a single value, then both the entities and relations will be regularized based on 
            the same regularizer. if it is a list, the first regularizer will be used for entities and second 
            for relations.
        '''
        # get the optimizer
        self.optimizer = optimizers.get(optimizer)
        self._run_eagerly = kwargs.pop('run_eagerly', None)
        # reset the training/evaluate/predict function
        self._reset_compile_cache()
        
        # get the loss
        self.compiled_loss = loss_functions.get(loss)
        # Only metric supported during the training is mean Loss
        self.compiled_metrics = metrics_mod.Mean(name='loss')  # Total loss.
        
        # set the initializer and regularizer of the embedding matrices in the encoding layer
        self.encoding_layer.set_initializer(entity_relation_initializer)
        self.encoding_layer.set_regularizer(entity_relation_regularizer)
        self._is_compiled = True

    @property
    def metrics(self):
        '''returns all the metrics that will be computed during training'''
        metrics = []
        if self._is_compiled:
            if self.compiled_loss is not None:
                metrics += self.compiled_loss.metrics
        return metrics

    def get_emb_matrix_test(self, part_number=1, number_of_parts=1):
        ''' get the embedding matrix during evaluation
        
        Parameters:
        -----------
        part number: int
            specifies which part of number_of_parts of entire emb matrix to return
        number_of_parts: int
            Total number of parts in which to split the emb matrix
            
        Returns:
        --------
        emb_matrix: np.array(n,k)
            embedding matrix corresponding the part_number out of number_of_parts parts.
        start_index: int
            Original entity index(data dict) of the first row of the emb_matrix.
        end_index: int
            Original entity index(data dict) of the last row of the emb_matrix.
            
        '''
        if number_of_parts == 1:
            return self.encoding_layer.ent_emb, 0, self.encoding_layer.ent_emb.shape[0] - 1
        else:
            with shelve.open('ent_partition') as ent_partition:
                batch_size = int(np.ceil(len(ent_partition.keys()) / number_of_parts))
                indices = np.arange(part_number * batch_size, (part_number + 1) * batch_size).astype(np.str)
                emb_matrix = []
                for idx in indices:
                    try:
                        emb_matrix.append(ent_partition[idx])
                    except KeyError:
                        break
                return np.array(emb_matrix), int(indices[0]), int(indices[-1])

    def make_test_function(self):
        ''' Similar to keras lib, this function returns the handle to test step function. 
        It processes one batch of data by iterating over the dataset iterator and computes the test metrics.

        Returns:
        --------
        out: Function handle.
              Handle to the test step function  
        '''
        if self.test_function is not None:
            return self.test_function

        def test_function(iterator):
            # total number of parts in which to split the embedding matrix 
            # (default 1 ie. use full matrix as it is)
            number_of_parts = 1
            
            # if it is partitioned training 
            if self.is_partitioned_training:
                # split the emb matrix based on number of buckets
                number_of_parts = self.partitioner_k
                
            # if we are using filters then the iterator return 2 outputs
            if self.use_filter:
                # Filters used - batch of test set data and corresponding True positives (filters)
                inputs, filters = next(iterator)
            else:
                # No filter - so just the batch of test set data
                inputs = next(iterator)
                # set the filter to empty
                filters = tf.RaggedTensor.from_row_lengths([], [])

            # compute the output shape based on the type of corruptions to be used
            output_shape = 0
            if 's' in self.corrupt_side:
                output_shape += 1
            
            if 'o' in self.corrupt_side:
                output_shape += 1

            # create an array to store the ranks based on output shape
            overall_rank = tf.zeros((output_shape, tf.shape(inputs)[0]), dtype=np.int32)
            
            if self.is_partitioned_training:
                inputs = self.process_model_inputs_for_test(inputs)

            # run the loop based on number of parts in which the original emb matrix was generated
            for j in range(number_of_parts):
                # get the embedding matrix along with entity ids of first and last row of emb matrix
                emb_mat, start_ent_id, end_ent_id = self.get_emb_matrix_test(j, number_of_parts)
                # compute the rank
                
                ranks = self._get_ranks(inputs, emb_mat, 
                                        start_ent_id, end_ent_id, 
                                        filters, self.corrupt_side)
                # store it in the output
                for i in tf.range(output_shape):
                    overall_rank = tf.tensor_scatter_nd_add(overall_rank, [[i]], [ranks[i, :]])
                    
            overall_rank = tf.reshape(overall_rank, (-1, output_shape))     
            # if corruption type is s+o then add s and o ranks and return the added ranks
            if self.corrupt_side == 's+o':
                # add the subject and object ranks
                overall_rank = tf.reduce_sum(overall_rank, 1)
                # return the added ranks
                return tf.reshape(overall_rank, (-1, 1))
                
            return overall_rank

        if not self.run_eagerly and not self.is_partitioned_training:
            test_function = def_function.function(
                test_function, experimental_relax_shapes=True)

        self.test_function = test_function

        return self.test_function
    
    def process_model_inputs_for_test(self, triples):
        ''' Return the processed triples. 
        Returns:
        --------
        In regular (non partitioned) mode, the triples are returned as it is.
        In case of partitioning, it returns the triple embeddings as a list of size 3 - sub, pred and obj embeddings.
        '''
        if self.is_partitioned_training:
            np_triples = triples.numpy()
            sub_emb_out = []
            obj_emb_out = []
            rel_emb_out = []
            with shelve.open('ent_partition') as ent_emb:
                with shelve.open('rel_partition') as rel_emb:
                    for triple in np_triples:
                        sub_emb_out.append(ent_emb[str(triple[0])])
                        rel_emb_out.append(rel_emb[str(triple[1])])
                        obj_emb_out.append(ent_emb[str(triple[2])])
                        
            emb_out = [np.array(sub_emb_out),
                       np.array(rel_emb_out),
                       np.array(obj_emb_out)]
            return emb_out    
        else:
            return triples

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
        x: np.array(n,3), string, GraphDataLoader instance, AbstractGraphPartitioner instance
            Data OR Filename of the data file OR Data Handle - that would be used for training
        batch_size: int
            batch size to use during training. 
            May be overridden if x is GraphDataLoader or AbstractGraphPartitioner instance
        verbose: bool
            Verbosity mode.
        use_filter: bool or dict
            whether to use filter of not. If a dictionary is specified, the data in the dict is concatenated 
            and used as filter
        corrupt_side: string
            which side to corrupt (can take values: ``s``, ``o``, ``s+o`` or ``s,o``) (default:``s,o``)
        callbacks: keras.callbacks.Callback
            List of `keras.callbacks.Callback` instances. List of callbacks to apply during evaluation.

        Returns:
        --------
        rank: (n, number of corrupted sides)
            ranks by corrupting against subject corruptions and/or object corruptions 
            (corruptions defined by ent_embs matrix)
        '''
        # get teh test set handler
        self.data_handler_test = data_adapter.DataHandler(x,
                                                          batch_size=batch_size,
                                                          dataset_type='test',
                                                          epochs=1,
                                                          use_filter=use_filter,
                                                          use_indexer=self.data_indexer)
        
        assert corrupt_side in ['s', 'o', 's,o', 's+o'], 'Invalid value for corrupt_side'

        self.corrupt_side = corrupt_side
        # flag to indicate if we are using filter or not
        self.use_filter = self.data_handler_test._parent_adapter.backend.use_filter or \
            type(self.data_handler_test._parent_adapter.backend.use_filter) == dict

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

        # before test begins call this callback function
        callbacks.on_test_begin()

        self.all_ranks = []

        # enumerate over the data
        for _, iterator in self.data_handler_test.enumerate_epochs(): 
            # handle the stop iteration of data iterator in this scope
            with self.data_handler_test.catch_stop_iteration():
                # iterate over the dataset
                for step in self.data_handler_test.steps():
                    # before a batch is processed call this callback function
                    callbacks.on_test_batch_begin(step)

                    # process this batch
                    overall_rank = test_function(iterator)
                    # increment the rank by 1 (ranks returned are from (0 - n-1) so increment by 1
                    overall_rank += 1
                    # save the ranks of the batch triples
                    self.all_ranks.append(overall_rank)
                    # after a batch is processed call this callback function
                    callbacks.on_test_batch_end(step)
        # on test end call this method
        callbacks.on_test_end()
        # return ranks
        return np.concatenate(self.all_ranks)

    def predict_step(self, inputs):
        ''' Returns the output of predict step on a batch of data'''
        score_pos = self(inputs, False)
        return score_pos
    
    def predict_step_partitioning(self, inputs):
        ''' Returns the output of predict step on a batch of data'''
        score_pos = self.scoring_layer(inputs)
        return score_pos

    def make_predict_function(self):
        ''' Similar to keras lib, this function returns the handle to predict step function. 
        It processes one batch of data by iterating over the dataset iterator and computes the predict outputs.

        Returns:
        --------
        out: Function handle.
              Handle to the predict step function
        '''
        if self.predict_function is not None:
            return self.predict_function

        def predict_function(iterator):
            inputs = next(iterator)
            if self.is_partitioned_training:
                inputs = self.process_model_inputs_for_test(inputs)
                outputs = self.predict_step_partitioning(inputs)
            else:
                outputs = self.predict_step(inputs)
            return outputs

        if not self.run_eagerly and not self.is_partitioned_training:
            predict_function = def_function.function(predict_function,
                                                     experimental_relax_shapes=True)

        self.predict_function = predict_function
        return self.predict_function

    def predict(self,
                x,
                batch_size=32,
                verbose=0,
                callbacks=None):
        '''
        Compute scores of the input triples

        Parameters:
        -----------
        x: np.array(n,3), string, GraphDataLoader instance, AbstractGraphPartitioner instance
            Data OR Filename of the data file OR Data Handle - that would be used for training
        batch_size: int
            batch size to use during training.
            May be overridden if x is GraphDataLoader or AbstractGraphPartitioner instance
        verbose: bool
            Verbosity mode.
        callbacks: keras.callbacks.Callback
            List of `keras.callbacks.Callback` instances. List of callbacks to apply during evaluation.

        Returns:
        --------
        scores: (n, )
            score of the input triples
        '''

        self.data_handler_test = data_adapter.DataHandler(x,
                                                          batch_size=batch_size,
                                                          dataset_type='test',
                                                          epochs=1,
                                                          use_filter=False,
                                                          use_indexer=self.data_indexer)

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=self.data_handler_test.inferred_steps)

        predict_function = self.make_predict_function()
        callbacks.on_predict_begin()
        outputs = []
        for _, iterator in self.data_handler_test.enumerate_epochs():
            with self.data_handler_test.catch_stop_iteration():
                for step in self.data_handler_test.steps():
                    callbacks.on_predict_batch_begin(step)
                    batch_outputs = predict_function(iterator)
                    outputs.append(batch_outputs)

                    callbacks.on_predict_batch_end(step, {'outputs': batch_outputs})
        callbacks.on_predict_end()
        return np.concatenate(outputs)
    
    def make_calibrate_function(self):
        ''' Similar to keras lib, this function returns the handle to calibrate step function. 
        It processes one batch of data by iterating over the dataset iterator and computes the predict outputs.

        Returns:
        --------
        out: Function handle.
              Handle to the predict step function
        '''

        def calibrate_with_corruption(iterator):
            inputs = next(iterator)
            if self.is_partitioned_training:
                inp_emb = self.process_model_inputs_for_test(inputs)
                inp_score = self.scoring_layer(inp_emb)
                
                corruptions = self.corruption_layer(inputs, self.num_ents, 1)
                corr_emb = self.encoding_layer(corruptions)
                corr_score = self.scoring_layer(corr_emb)
            else:
                inp_emb = self.encoding_layer(inputs)
                inp_score = self.scoring_layer(inp_emb)
                
                corruptions = self.corruption_layer(inputs, self.num_ents, 1)
                corr_emb = self.encoding_layer(corruptions)
                corr_score = self.scoring_layer(corr_emb)
            return inp_score, corr_score
        
        def calibrate_with_negatives(iterator):
            inputs = next(iterator)
            if self.is_partitioned_training:
                inp_emb = self.process_model_inputs_for_test(inputs)
                inp_score = self.scoring_layer(inp_emb)
            else:
                inp_emb = self.encoding_layer(inputs)
                inp_score = self.scoring_layer(inp_emb)
            return inp_score

        if self.is_calibrate_with_corruption:
            calibrate_fn = calibrate_with_corruption
        else:
            calibrate_fn = calibrate_with_negatives
            
        if not self.run_eagerly and not self.is_partitioned_training:
            calibrate_fn = def_function.function(calibrate_fn,
                                                 experimental_relax_shapes=True)

        return calibrate_fn
    
    def calibrate(self, X_pos, X_neg=None, positive_base_rate=None, batch_size=32, epochs=50, verbose=0):
        """Calibrate predictions

        The method implements the heuristics described in :cite:`calibration`,
        using Platt scaling :cite:`platt1999probabilistic`.

        The calibrated predictions can be obtained with :meth:`predict_proba`
        after calibration is done.

        Ideally, calibration should be performed on a validation set that was not used to train the embeddings.

        There are two modes of operation, depending on the availability of negative triples:

        #. Both positive and negative triples are provided via ``X_pos`` and ``X_neg`` respectively. \
        The optimization is done using a second-order method (limited-memory BFGS), \
        therefore no hyperparameter needs to be specified.

        #. Only positive triples are provided, and the negative triples are generated by corruptions \
        just like it is done in training or evaluation. The optimization is done using a first-order method (ADAM), \
        therefore ``batches_count`` and ``epochs`` must be specified.


        Calibration is highly dependent on the base rate of positive triples.
        Therefore, for mode (2) of operation, the user is required to provide the ``positive_base_rate`` argument.
        For mode (1), that can be inferred automatically by the relative sizes of the positive and negative sets,
        but the user can override that by providing a value to ``positive_base_rate``.

        Defining the positive base rate is the biggest challenge when calibrating without negatives. That depends on
        the user choice of which triples will be evaluated during test time.
        Let's take WN11 as an example: it has around 50% positives triples on both the validation set and test set,
        so naturally the positive base rate is 50%. However, should the user resample it to have 75% positives
        and 25% negatives, its previous calibration will be degraded. The user must recalibrate the model now with a
        75% positive base rate. Therefore, this parameter depends on how the user handles the dataset and
        cannot be determined automatically or a priori.

        .. Note ::
            Incompatible with large graph mode (i.e. if ``self.dealing_with_large_graphs=True``).

        .. Note ::
            :cite:`calibration` `calibration experiments available here
            <https://github.com/Accenture/AmpliGraph/tree/paper/ICLR-20/experiments/ICLR-20>`_.


        Parameters
        ----------
        X_pos : np.array(n,3), string, GraphDataLoader instance, AbstractGraphPartitioner instance
            Data OR Filename of the data file OR Data Handle - that would be used as positive triples.
        X_neg : np.array(n,3), string, GraphDataLoader instance, AbstractGraphPartitioner instance
            Data OR Filename of the data file OR Data Handle - that would be used as negative triples.

            If `None`, the negative triples are generated via corruptions
            and the user must provide a positive base rate instead.
        positive_base_rate: float
            Base rate of positive statements.

            For example, if we assume there is a fifty-fifty chance of any query to be true, the base rate would be 50%.

            If ``X_neg`` is provided and this is `None`, the relative sizes of ``X_pos`` and ``X_neg`` will be used to
            determine the base rate. For example, if we have 50 positive triples and 200 negative triples,
            the positive base rate will be assumed to be 50/(50+200) = 1/5 = 0.2.

            This must be a value between 0 and 1.
        batches_size: int
            Batch size for positives
        epochs: int
            Number of epochs used to train the Platt scaling model.
            Only applies when ``X_neg`` is  `None`.
        verbose: bool
            Verbosity
        """
        self.is_calibrated = False
        data_handler_calibrate_pos = data_adapter.DataHandler(X_pos,
                                                              batch_size=batch_size,
                                                              dataset_type='test',
                                                              epochs=epochs,
                                                              use_filter=False,
                                                              use_indexer=self.data_indexer)
        
        pos_size = data_handler_calibrate_pos._parent_adapter.get_data_size()
        neg_size = pos_size
        
        if X_neg is None:
            assert positive_base_rate is not None, 'Please provide the negatives or positive base rate!'
            self.is_calibrate_with_corruption = True
        else:
            self.is_calibrate_with_corruption = False
            
            pos_batch_count = int(np.ceil(pos_size/batch_size))
            
            data_handler_calibrate_neg = data_adapter.DataHandler(X_neg,
                                                                  batch_size=batch_size,
                                                                  dataset_type='test',
                                                                  epochs=epochs,
                                                                  use_filter=False,
                                                                  use_indexer=self.data_indexer)

            neg_size = data_handler_calibrate_neg._parent_adapter.get_data_size()
            neg_batch_count = int(np.ceil(neg_size/batch_size))
            
            if pos_batch_count != neg_batch_count:
                batch_size_neg = int(np.ceil(neg_size/pos_batch_count))
                data_handler_calibrate_neg = data_adapter.DataHandler(X_neg,
                                                                      batch_size=batch_size_neg,
                                                                      dataset_type='test',
                                                                      epochs=epochs,
                                                                      use_filter=False,
                                                                      use_indexer=self.data_indexer)
            
            
            if positive_base_rate is None:
                positive_base_rate = pos_size / (pos_size + neg_size)

        if positive_base_rate is not None and (positive_base_rate <= 0 or positive_base_rate >= 1):
            raise ValueError("positive_base_rate must be a value between 0 and 1.")
        
        self.calibration_layer = CalibrationLayer(pos_size, neg_size, positive_base_rate)
        calibrate_function = self.make_calibrate_function()
        
        optimizer = tf.keras.optimizers.Adam()
        
        pos_outputs = []
        neg_outputs = []
        
        if not self.is_calibrate_with_corruption:
             negative_iterator = iter(data_handler_calibrate_neg.enumerate_epochs())
                
        for _, iterator in data_handler_calibrate_pos.enumerate_epochs():
            if not self.is_calibrate_with_corruption:
                _, neg_handle = next(negative_iterator)
            
            with data_handler_calibrate_pos.catch_stop_iteration():
                for step in data_handler_calibrate_pos.steps():
                    if self.is_calibrate_with_corruption:
                        scores_pos, scores_neg = calibrate_function(iterator)

                    else:
                        scores_pos = calibrate_function(iterator)
                        with data_handler_calibrate_neg.catch_stop_iteration():
                            scores_neg = calibrate_function(neg_handle)
                            
                    
                    
                    with tf.GradientTape() as tape:
                        out = self.calibration_layer(scores_pos, scores_neg, 1)

                    gradients = tape.gradient(out, self.calibration_layer._trainable_weights)
                    # update the trainable params
                    optimizer.apply_gradients(zip(gradients, self.calibration_layer._trainable_weights))
        self.is_calibrated = True
        
        
    def predict_proba(self,
                        x,
                        batch_size=32,
                        verbose=0,
                        callbacks=None):
        '''
        Compute calibrated scores (0 <= score <= 1) of the input triples 

        Parameters:
        -----------
        x: np.array(n,3), string, GraphDataLoader instance, AbstractGraphPartitioner instance
            Data OR Filename of the data file OR Data Handle - that would be used for training
        batch_size: int
            batch size to use during training.
            May be overridden if x is GraphDataLoader or AbstractGraphPartitioner instance
        verbose: bool
            Verbosity mode.
        callbacks: keras.callbacks.Callback
            List of `keras.callbacks.Callback` instances. List of callbacks to apply during evaluation.

        Returns:
        --------
        scores: (n, )
            calibrated score of the input triples
        '''
        if not self.is_calibrated:
            msg = "Model has not been calibrated. \
            Please call `model.calibrate(...)` before predicting probabilities."
            raise RuntimeError(msg)
            
        self.data_handler_test = data_adapter.DataHandler(x,
                                                          batch_size=batch_size,
                                                          dataset_type='test',
                                                          epochs=1,
                                                          use_filter=False,
                                                          use_indexer=self.data_indexer)

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=self.data_handler_test.inferred_steps)

        predict_function = self.make_predict_function()
        callbacks.on_predict_begin()
        outputs = []
        for _, iterator in self.data_handler_test.enumerate_epochs():
            with self.data_handler_test.catch_stop_iteration():
                for step in self.data_handler_test.steps():
                    callbacks.on_predict_batch_begin(step)
                    batch_outputs = predict_function(iterator)
                    probas = self.calibration_layer(batch_outputs, training=0)
                    outputs.append(probas)

                    callbacks.on_predict_batch_end(step, {'outputs': batch_outputs})
        callbacks.on_predict_end()
        return np.concatenate(outputs)
