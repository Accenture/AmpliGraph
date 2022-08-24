# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import contextlib
from .graph_data_loader import GraphDataLoader, DummyBackend
from .sqlite_adapter import SQLiteAdapter
from .graph_partitioner import AbstractGraphPartitioner
from .partitioned_data_manager import get_partition_adapter
from tensorflow.python.framework import errors


class DataHandler():
    def __init__(self, x, 
                 model=None, 
                 batch_size=1, 
                 dataset_type="train", 
                 epochs=1, 
                 initial_epoch=0, 
                 use_indexer = True,
                 use_filter=True,
                 use_partitioning=False,
                 partitioning_k=3):
        '''Initializes the DataHandler
        
        Parameters:
        -----------
        model: tf.keras.Model
            Model instance
        batch_size: int
            batch size to use during training. 
            May be overridden if x is GraphDataLoader or AbstractGraphPartitioner instance
        dataset_type: string
            dataset type that is being passed
        epochs: int
            Number of epochs to train (default: 1)
        initial epoch: int
            initial epoch number (default: 1)
        use_indexer: bool or Mapper
            whether the data needs to be indexed or whether we need to use pre-defined indexer to map 
            the data to index
        use_filter: bool or dict
            whether to use filter of not. If a dictionary is specified, the data in the dict is concatenated 
            and used as filter 
        use_partitioning: bool
            flag to indicate whether to use partitioning or not.
            May be overridden if x is an AbstractGraphPartitioner instance
        partitioning_k: int
            number of partitions to create
        '''
        self._initial_epoch = initial_epoch
        self._epochs = epochs
        self._model = model
        self._inferred_steps = None
        self.using_partitioning = False

        if isinstance(x, GraphDataLoader):
            self._adapter = x
            self._parent_adapter = self._adapter
        elif isinstance(x, AbstractGraphPartitioner):
            self._parent_adapter = x._data
            self._adapter = x
            self.using_partitioning = True
        else:
            # use graph data loader by default
            self._adapter = GraphDataLoader(x,
                                            backend=SQLiteAdapter,
                                            batch_size=batch_size,
                                            dataset_type=dataset_type,
                                            use_indexer=use_indexer,
                                            use_filter=use_filter)
            self._parent_adapter = self._adapter

        if use_partitioning:
            # if use partitioning then pass the graph data loader to partitioner and use
            # partitioned data manager
            assert model is not None, "Please pass the model to datahandler for partitioning!"

            self._adapter = get_partition_adapter(self._adapter,
                                                  self._model,
                                                  strategy='Bucket',
                                                  partitioning_k=partitioning_k)

            self.using_partitioning = True
            
    @contextlib.contextmanager
    def catch_stop_iteration(self):
        """Catches errors when an iterator runs out of data."""
        try:
            yield
        except (StopIteration, errors.OutOfRangeError):
            if self._inferred_steps is None:
                self._inferred_steps = self._current_iter
            
    def steps(self):
        '''Counts the number of steps in an epoch'''
        self._current_iter = 0
        while self._inferred_steps is None or self._current_iter<self._inferred_steps:
            self._current_iter += 1
            yield self._current_iter
            
    @property
    def inferred_steps(self):
        '''Returns the number of steps in the batch'''
        return self._inferred_steps
    
    def enumerate_epochs(self):
        '''Manages the (reloading) data adapter before epoch starts'''
        for epoch in range(self._initial_epoch, self._epochs):
            self._adapter.reload()   
            yield epoch, iter(self._adapter.get_tf_generator())
            self._adapter.on_epoch_end()
            
        self._adapter.on_complete()
        
    def get_mapper(self):
        '''returns the mapper of the main data loader class'''
        return self._parent_adapter.backend.mapper
    
    def get_update_partitioner_metadata(self, filepath):
        out_dict = {}
        if self.using_partitioning:
            out_dict = self._adapter.get_update_metadata(filepath)
        return out_dict