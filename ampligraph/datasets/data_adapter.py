import contextlib
from ampligraph.datasets import GraphDataLoader, DummyBackend
from ampligraph.datasets.graph_partitioner import AbstractGraphPartitioner
import ampligraph.datasets.partitioned_data_manager as partition_manager
from tensorflow.python.framework import errors


class DataHandler():
    def __init__(self, x, 
                 model=None, 
                 batch_size=1, 
                 dataset_type="train", 
                 epochs=1, 
                 initial_epoch=0, 
                 use_indexer = True,
                 train_partitioner=None,
                 use_filter=True,
                 use_partitioning=False):
        
        self._initial_epoch = initial_epoch
        self._epochs = epochs
        self._model = model
        self._inferred_steps = None
        self.train_partitioner = train_partitioner
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
                                            backend=DummyBackend,
                                            batch_size=batch_size,
                                            dataset_type=dataset_type,
                                            use_indexer=use_indexer,
                                            use_filter=use_filter)
            self._parent_adapter = self._adapter

        if use_partitioning:
            # if use partitioning then pass the graph data loader to partitioner and use
            # partitioned data manager
            assert model is not None, "Please pass the model to datahandler for partitioning!"

            self._adapter = partition_manager.get_partition_adapter(self._adapter,
                                                                    self._model,
                                                                    'Bucket')

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
        return self._inferred_steps
    
    def enumerate_epochs(self):
        '''Manages the (reloading) data adapter before epoch starts'''
        for epoch in range(self._initial_epoch, self._epochs):
            self._adapter.reload()   
            yield epoch, iter(self._adapter.get_tf_generator())
            self._adapter.on_epoch_end()
            
        self._adapter.on_complete()
        
    
    def get_mapper(self):
        return self._parent_adapter.backend.mapper

