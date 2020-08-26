# Copyright 2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Graph partitioning strategies.

Module contains several graph partitioning strategies both based 
on vertices split and edges split.

Attributes
----------
PARTITION_ALGO_REGISTRY: dictionary containing strategies' names 
as key and reference to the strategy class as a value.
"""
import numpy as np
import time
import pandas as pd
from abc import ABC, abstractmethod
from ampligraph.utils.profiling import timing_and_memory
from ampligraph.datasets.graph_data_loader import GraphDataLoader
from datetime import datetime
import shelve
import csv
import os
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
PARTITION_ALGO_REGISTRY = {}


def register_partitioning_strategy(name):
    """Decorator responsible for registering partition in the partition registry.
       
       Parameters
       ----------
       name: name of the new partition strategy.
 
       Example
       -------
       >>>@register_partitioning_strategy("NewStrategyName")
       >>>class NewPartitionStrategy(AbstractGraphPartitioner):
       >>>... pass
    """
    def insert_in_registry(class_handle):
        """Checks if partition already exists and if not registers it."""
        if name in PARTITION_ALGO_REGISTRY.keys():
            msg = "Partitioning Strategy with name {} "
            logger.error(msg)
            raise Exception(msg)
        "already exists!".format(name)
        
        PARTITION_ALGO_REGISTRY[name] = class_handle
        class_handle.name = name
        return class_handle

    return insert_in_registry


def get_number_of_partitions(n):
    """Calculates number of partitions for Bucket Partitioner.
    
       Parameters
       ----------
       n: number of buckets with vertices.

       Returns
       -------
       number of partitions
    """
    return int(n*(n+1)/2)


class AbstractGraphPartitioner(ABC):
    """Meta class defining interface for graph partitioning algorithms.

    _data: graph data to split across partitions
    _k [default=2]: number of partitions to split into
    """

    def __init__(self, data, k=2, seed=None, **kwargs):
        """Initialiser for AbstractGraphPartitioner.

           data: input data as a GraphDataLoader.
           k: number of partitions or buckets to split data into.
        """
        self.files = []
        self.partitions = []
        self._data = data
        self._k = k
        self._split(seed=seed, batch_size=data.batch_size, **kwargs)
        self.reload()
        
    def __iter__(self):
        """Function needed to be used as an itertor."""
        return self
    
    def reload(self):
        self.generator = self.partitions_generator()

    def get_data(self):
        return self._data

    def partitions_generator(self):
        """Generates partitions.
           Yields
           ------
           next partition as GraphDataLoader object.
        """
        for partition in self.partitions:
            partition.reload()
            yield partition

    def get_partitions_iterator(self):
        """Reinstantiate partitions generator.
           
           Returns
           -------
           partitions generator.
        """
        return self.partitions_generator()

    def get_partitions_list(self):
        """Returns handler for partitions list."""
        for partition in self.partitions:
            partition.reload()
        return self.partitions       
 
    def __next__(self):
        """Function needed to be used as an itertor."""
        return next(self.generator)

    def _split(self, seed=None, **kwargs):
        """Split data into k equal size partitions.

           Parameters
           ----------
           seed: seed for repeatability purposes, it is only
                 used when certain randomization is required

           Returns
           -------
            partitions: parts of equal size with triples
        """
        pass

    def clean(self):
        for partition in self.partitions:
            partition.clean()
        for f in self.files:
            if f.split(".")[-1] != "shf":
                os.remove(f)
            else:
                os.remove(f + ".bak")
                os.remove(f + ".dir")
                os.remove(f + ".dat")
    
@register_partitioning_strategy("Bucket")
class BucketGraphPartitioner(AbstractGraphPartitioner):
    """Bucket-based partition strategy.

       Example
       -------
       >>>d = np.array([[1,1,2], [1,1,3],[1,1,4],[5,1,3],[5,1,2],[6,1,3],[6,1,2],[6,1,4],[6,1,7]])
       >>>data = GraphDataLoader(d, batch_size=1, dataset_type="test")
       >>>partitioner = BucketGraphPartitioner(data, k=2)
       >>>for i, partition in enumerate(partitioner):
       >>>    print("partition ", i)
       >>>    for batch in partition:
       >>>        print(batch)
        partition  0
        [['0,0,1']]
        [['0,0,2']]
        [['0,0,3']]
        partition  1
        [['4,0,1']]
        [['4,0,2']]
        [['5,0,1']]
        [['5,0,2']]
        [['5,0,3']]
        partition  2
        [['5,0,6']]
    """
    def __init__(self, data, k=2, **kwargs):
        """Initialiser for BucketGraphPartitioner.

           data: input data as a GraphDataLoader.
           k: number of buckets to split entities/vertices into.
        """

        self.partitions = []
        super().__init__(data, k, **kwargs)
       
    def create_single_partition(self, ind1, ind2, timestamp, partition_nb, batch_size=1):
        """Creates partition based on given two indices of buckets.
           It appends created partition to the list of partitions (self.partitions).
          
           Parameters
           ----------
           ind1: index of the first bucket needed to create partition.
           ind2: index of the second bucket needed to create partition.
           timestamp: date and time string that the files are created with (shelves).
           partition_nb: assigned number of partition.           
        """
        #logger.debug("------------------------------------------------")        
        #logger.debug("Creating partition nb: {}".format(partition_nb))
        fname = "bucket_{}_{}.shf".format(ind1, timestamp)
        with shelve.open(fname, writeback=True) as bucket_partition_1:
            indexes_1 = bucket_partition_1['indexes']
        fname = "bucket_{}_{}.shf".format(ind2, timestamp) 
        with shelve.open(fname, writeback=True) as bucket_partition_2:
            indexes_2 = bucket_partition_2['indexes']
            
        #logger.debug("indexes 1: {}".format(ind1, indexes_1))
        #logger.debug("indexes 2: {}".format(ind2, indexes_2))
        
        triples_1_2 = np.array(self._data.get_triples(subjects=indexes_1, objects=indexes_2))[:,:3]
        triples_2_1 = np.array(self._data.get_triples(subjects=indexes_2, objects=indexes_1))[:,:3] # Probably not needed!
        
        logger.debug("triples 1-2: {}".format(triples_1_2))
        logger.debug("triples 2-1: {}".format(triples_2_1))
        triples = np.vstack([triples_1_2, triples_2_1]).astype(np.int32)
        #logger.debug(triples)
        if triples.size != 0:
            triples = np.unique(triples, axis=0)
            #logger.debug("unique triples: {}".format(triples))
            fname = "partition_{}_{}.csv".format(partition_nb, timestamp)
            self.files.append(fname)
            np.savetxt(fname, triples, delimiter="\t", fmt='%d')
            # special case of GraphDataLoader to create partition datasets: with remapped indexes (0, size_of_partition),
            # persisted, with partition number to look up remappings
            partition_loader = GraphDataLoader(fname, 
                                               use_indexer=False, 
                                               batch_size=batch_size, 
                                               remap=True, 
                                               parent=self._data,
                                               name="partition_{}_buckets_{}-{}".format(partition_nb, ind1, ind2))
            self.partitions.append(partition_loader)
            return 0 # status everything went ok
        else:
            return 1 # status not ok, no partition created
    @timing_and_memory
    def _split(self, seed=None, verbose=False, batch_size=1, **kwargs):
        """Split data into self.k buckets based on unique entities and assign 
           accordingly triples to k partitions and intermediate partitions.
        """
        timestamp =  datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        logger.debug(self._data.backend.mapper.entities_dict)
        self.ents_size = self._data.backend.mapper.ents_length
        logger.debug(self.ents_size)
        self.bucket_size = int(np.ceil(self.ents_size / self._k))
        self.buckets_generator = self._data.backend.mapper.get_entities_in_batches(batch_size=self.bucket_size)
        
        for i, bucket in enumerate(self.buckets_generator):
            # dump entities in partition shelve/file
            fname = "bucket_{}_{}.shf".format(i, timestamp)
            self.files.append(fname)
            with shelve.open(fname, writeback=True) as bucket_partition:
                bucket_partition['indexes'] = bucket
            #logger.debug(bucket)
            
        partition_nb = 0
        # ensure that the "same" bucket partitions are generated first
        for i in range(self._k):
            # condition that excludes duplicated partitions 
            # from k x k possibilities, partition 0-1 and 1-0 is the same - not needed
            status_not_ok = self.create_single_partition(i, i, timestamp, partition_nb, batch_size=batch_size)
            if status_not_ok:
                continue
            partition_nb += 1 

        # Now generate across bucket partitions
        for i in range(self._k):
            for j in range(self._k):
                if j > i:
                    # condition that excludes duplicated partitions 
                    # from k x k possibilities, partition 0-1 and 1-0 is the same - not needed
                    status_not_ok = self.create_single_partition(i, j, timestamp, partition_nb, batch_size=batch_size)
                    if status_not_ok:
                        continue
                    partition_nb += 1 

    
@register_partitioning_strategy("RandomVertices")
class RandomVerticesGraphPartitioner(AbstractGraphPartitioner):
    """Partitioning strategy that splits vertices into equal
       sized buckets of random entities from the graph.
    """
    def __init__(self, data, k=2, seed=None, **kwargs):
        """Initialiser for RandomVerticesGraphPartitioner.

           data: input data as a GraphDataLoader.
           k: number of partitions to split data into.
        """
        self._data = data
        self._k = k
        self.partitions = []
        super().__init__(data, k, **kwargs)
       
    @timing_and_memory
    def _split(self, seed=None, batch_size=1, **kwargs):
        """Split data into k equal size partitions by randomly drawing subset of vertices
           of partition size and retrieving triples associated with these vertices.

           Returns
           -------
            partitions: parts of equal size with triples
        """
        timestamp =  datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.ents_size = self._data.backend.mapper.ents_length
       # logger.debug(self.ents_size)
       # logger.debug(backend.mapper.max_ents_index)
        self.partition_size = int(np.ceil(self.ents_size / self._k))
       # logger.debug(self.partition_size)
        self.buckets_generator = self._data.backend.mapper.get_entities_in_batches(batch_size=self.partition_size, random=True, seed=seed)

        for partition_nb, partition in enumerate(self.buckets_generator):
            #logger.debug(partition)
            tmp = np.array(self._data.backend._get_triples(entities=partition))
            if tmp.size != 0:
                triples = np.array(self._data.backend._get_triples(entities=partition))[:,:3].astype(np.int32) 
                #logger.debug("unique triples: {}".format(triples))
                fname = "partition_{}_{}.csv".format(partition_nb, timestamp)
                self.files.append(fname)
                np.savetxt(fname, triples, delimiter="\t", fmt='%d')
                # special case of GraphDataLoader to create partition datasets: with remapped indexes (0, size_of_partition),
                # persisted, with partition number to look up remappings
                partition_loader = GraphDataLoader(fname, 
                                                   use_indexer=False, 
                                                   batch_size=batch_size, 
                                                   remap=True, 
                                                   parent=self._data,
                                                   name="partition_{}".format(partition_nb))
                self.partitions.append(partition_loader)
            else:
                logger.debug("Partition has no triples, skipping!")


class EdgeBasedGraphPartitioner(AbstractGraphPartitioner):
    """Template for edge-based partitioning strategy that splits edges
       into partitions, should be inherited to create different edge-based strategy.
    """
    def __init__(self, data, k=2, batch_size=1, random=False, index_by="", **kwargs):
        """Initialiser for EdgeBasedGraphPartitioner.

           data: input data as a GraphDataLoader.
           k: number of partitions to split data into.
           batch_size: size of the batch for partitions data.
           random: whether to draw edges/triples in random order.
           index_by: which index to use when returning triples (s,o,so,os).
        """

        self.partitions = []
        self._data = data
        self._k = k
        super().__init__(data, k=k, batch_size=batch_size, random=random, index_by=index_by, **kwargs)

    def get_data(self):
        return self._data

    @timing_and_memory
    def _split(self, seed=None, batch_size=1, random=False, index_by="", **kwargs):
        """Split data into k equal size partitions by randomly drawing subset of
        edges from dataset.

           Returns
           -------
            partitions: parts of equal size with triples
        """
        timestamp =  datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.size = self._data.backend.get_data_size()

        self.partition_size = int(np.ceil(self.size / self._k))
        logger.debug(self.partition_size)
        generator = self._data.backend._get_batch_generator(random=random, batch_size=self.partition_size, dataset_type=self._data.dataset_type, index_by=index_by)
       
        for partition_nb, partition in enumerate(generator):
            fname = "partition_{}_{}.csv".format(partition_nb, timestamp)
            self.files.append(fname)
            np.savetxt(fname, np.array(partition, dtype=int), delimiter='\t', fmt='%d')
            # special case of GraphDataLoader to create partition datasets: with remapped indexes (0, size_of_partition),
            # persisted, with partition number to look up remappings
            partition_loader = GraphDataLoader(fname, 
                                               use_indexer=False, 
                                               batch_size=batch_size, 
                                               remap=True, 
                                               parent=self._data,
                                               name="partition_{}".format(partition_nb))
            self.partitions.append(partition_loader)

@register_partitioning_strategy("RandomEdges")
class RandomEdgesGraphPartitioner(EdgeBasedGraphPartitioner):
    """Partitioning strategy that splits edges into equal size
       partitions randomly drawing triples from the data.
    """

    def __init__(self, data, k=2, batch_size=1, **kwargs):
        """Initialiser for RandomEdgesGraphPartitioner.

           data: input data as a GraphDataLoader.
           k: number of partitions to split data into.
           batch_size: size of a batch.
        """
        self.partitions = []
        self._data = data
        self._k = k
        super().__init__(data, k, batch_size=batch_size, random=True, index_by="", **kwargs)

@register_partitioning_strategy("Naive")
class NaiveGraphPartitioner(EdgeBasedGraphPartitioner):
    """Partitioning strategy that splits edges into equal size
       partitions drawing triples from the data sequentially.
    """
    def __init__(self, data, k=2, batch_size=1, **kwargs):
        """Initialiser for NaiveGraphPartitioner.

           data: input data as a GraphDataLoader.
           k: number of partitions to split data into.
           batch_size: size of a batch.
        """
        self.partitions = []
        super().__init__(data, k, batch_size=batch_size, random=False, index_by="", **kwargs)

@register_partitioning_strategy("SortedEdges")
class SortedEdgesGraphPartitioner(EdgeBasedGraphPartitioner):
    """Partitioning strategy that splits edges into equal size
       partitions retriving triples from the data ordered by subject.
    """
    def __init__(self, data, k=2, batch_size=1, **kwargs):
        """Initialiser for SortedEdgesGraphPartitioner.

           data: input data as a GraphDataLoader.
           k: number of partitions to split data into.
           batch_size: size of a batch.
        """

        self.partitions = []
        super().__init__(data, k, batch_size=batch_size, random=False, index_by="s", **kwargs)

@register_partitioning_strategy("DoubleSortedEdges")
class DoubleSortedEdgesGraphPartitioner(EdgeBasedGraphPartitioner):
    """Partitioning strategy that splits edges into equal size
       partitions retriving triples from the data ordered by subject and object.
    """
    def __init__(self, data, k=2, batch_size=1, **kwargs):
        """Initialiser for DoubleSortedEdgesGraphPartitioner.

           data: input data as a GraphDataLoader.
           k: number of partitions to split data into.
           batch_size: size of a batch.
        """
        self.partitions = []
        super().__init__(data, k, batch_size=batch_size, random=False, index_by="so", **kwargs)


def main():
    """Main function - not implemented."""
    pass


if __name__ == "__main__":
    main()
