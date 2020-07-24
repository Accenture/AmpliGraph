# Copyright 2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
import time
import pandas as pd
from abc import ABC, abstractmethod
from ampligraph.utils.profiling import timing_and_memory
from ampligraph.datasets.graph_data_loader import GraphDataLoader
from datetime import datetime
import shelve
import csv

PARTITION_ALGO_REGISTRY = {}


def register_partitioning_strategy(name):
    def insert_in_registry(class_handle):
        assert name not in PARTITION_ALGO_REGISTRY.keys(), "Partitioning Strategy with name {} "
        "already exists!".format(name)
        
        PARTITION_ALGO_REGISTRY[name] = class_handle
        class_handle.name = name
        return class_handle

    return insert_in_registry


def get_number_of_partitions(n):
    return int(n*(n+1)/2)


class AbstractGraphPartitioner(ABC):
    """Meta class defining interface for graph partitioning algorithms.

    _data: graph data to split across partitions
    _k [default=2]: number of partitions to split into
    """

    def __init__(self, data, k=2, seed=None, **kwargs):
        self._data = data
        self._k = k
        self._split(seed=seed, **kwargs)
        self.generator = self.partitions_generator()
        
    def __iter__(self):
        """Function needed to be used as an itertor."""
        return self

    def partitions_generator(self):
        for partition in self.partitions:
            yield partition

    def get_partitions_iterator(self):
        """Reinstantiate partitions generator
           
           Returns
           -------
           partitions generator
        """
        return self.partitions_generator()

    def get_partitions_list(self):
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
    def __init__(self, data, k=2):
        self.partitions = []
        super().__init__(data, k)
       
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
        #print("------------------------------------------------")        
        #print("Creating partition nb: {}".format(partition_nb))
        with shelve.open("bucket_{}_{}.shf".format(ind1, timestamp), writeback=True) as bucket_partition_1:
            indexes_1 = bucket_partition_1['indexes']
        with shelve.open("bucket_{}_{}.shf".format(ind2, timestamp), writeback=True) as bucket_partition_2:
            indexes_2 = bucket_partition_2['indexes']
            
        #print("indexes 1: ", ind1, indexes_1)
        #print("indexes 2: ", ind2, indexes_2)
        
        triples_1_2 = np.array(self._data.get_triples(subjects=indexes_1, objects=indexes_2))[:,:3]
        triples_2_1 = np.array(self._data.get_triples(subjects=indexes_2, objects=indexes_1))[:,:3] # Probably not needed!
        
        print("triples 1-2: ", triples_1_2)
        print("triples 2-1: ", triples_2_1)
        triples = np.vstack([triples_1_2, triples_2_1]).astype(np.int32)
        #print(triples)
        if triples.size != 0:
            triples = np.unique(triples, axis=0)
            #print("unique triples: ", triples)
            np.savetxt("partition_{}_{}.csv".format(partition_nb, timestamp), triples, delimiter="\t", fmt='%d')
            # special case of GraphDataLoader to create partition datasets: with remapped indexes (0, size_of_partition),
            # persisted, with partition number to look up remappings
            partition_loader = GraphDataLoader("partition_{}_{}.csv".format(partition_nb, timestamp), 
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
        print(self._data.backend.mapper.entities_dict)
        self.ents_size = self._data.backend.mapper.ents_length
        print(self.ents_size)
        self.bucket_size = int(np.ceil(self.ents_size / self._k))
        self.buckets_generator = self._data.backend.mapper.get_entities_in_batches(batch_size=self.bucket_size)
        
        for i, bucket in enumerate(self.buckets_generator):
            # dump entities in partition shelve/file
            with shelve.open("bucket_{}_{}.shf".format(i, timestamp), writeback=True) as bucket_partition:
                bucket_partition['indexes'] = bucket
            #print(bucket)
            
        partition_nb = 0
        for i in range(self._k):
            for j in range(self._k):
                if j >= i:
                    # condition that excludes duplicated partitions 
                    # from k x k possibilities, partition 0-1 and 1-0 is the same - not needed
                    status_not_ok = self.create_single_partition(i, j, timestamp, partition_nb, batch_size=batch_size)
                    if status_not_ok:
                        continue
                    partition_nb += 1 

    
@register_partitioning_strategy("RandomVertices")
class RandomVerticesGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2, seed=None, **kwargs):
        self.partitions = []
        super().__init__(data, k)
       
    @timing_and_memory
    def _split(self, seed=None, batch_size=1, **kwargs):
        """Split data into k equal size partitions by randomly drawing subset of vertices
           of partition size and retrieving triples associated with these vertices.

           Returns
           -------
            partitions: parts of equal size with triples
        """
        timestamp =  datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        with self._data.backend as backend:
            self.ents_size = self._data.backend.mapper.ents_length
            self.partition_size = int(np.ceil(self.ents_size / self._k))
            self.buckets_generator = backend.mapper.get_entities_in_batches(batch_size=self.partition_size, random=True, seed=seed)
    
            for partition_nb, partition in enumerate(self.buckets_generator):
                triples = np.array(backend._get_triples(entities=partition))[:,:3].astype(np.int32) 
                #print("unique triples: ", triples)
                np.savetxt("partition_{}_{}.csv".format(partition_nb, timestamp), triples, delimiter="\t", fmt='%d')
                # special case of GraphDataLoader to create partition datasets: with remapped indexes (0, size_of_partition),
                # persisted, with partition number to look up remappings
                partition_loader = GraphDataLoader("partition_{}_{}.csv".format(partition_nb, timestamp), 
                                                   use_indexer=False, 
                                                   batch_size=batch_size, 
                                                   remap=True, 
                                                   parent=self._data,
                                                   name="partition_{}".format(partition_nb))
                self.partitions.append(partition_loader)


class EdgeBasedGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2, batch_size=1, random=False, index_by=""):
        self.partitions = []
        super().__init__(data, k=k, batch_size=batch_size, random=random, index_by=index_by)

    @timing_and_memory
    def _split(self, seed=None, batch_size=1, random=False, index_by="", **kwargs):
        """Split data into k equal size partitions by randomly drawing subset of
        edges from dataset.

           Returns
           -------
            partitions: parts of equal size with triples
        """
        timestamp =  datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        with self._data.backend as backend:
            self.size = backend.get_data_size()

            self.partition_size = int(np.ceil(self.size / self._k))
            print(self.partition_size)
            
            for partition_nb in range(self._k):
                generator = self._data.backend._get_batch(random=random, batch_size=self.partition_size, dataset_type=self._data.dataset_type, index_by=index_by)
                def format_batch():
                    for batch in generator:
                        yield ["\t".join(str(c) for i,c in enumerate(x) if i != 3) for x in batch]
                with open("partition_{}_{}.csv".format(partition_nb, timestamp), "w") as file_csv:
                    writes = csv.writer(file_csv, delimiter='\n', quoting=csv.QUOTE_NONE)
                    writes.writerows(format_batch())
                # special case of GraphDataLoader to create partition datasets: with remapped indexes (0, size_of_partition),
                # persisted, with partition number to look up remappings
                partition_loader = GraphDataLoader("partition_{}_{}.csv".format(partition_nb, timestamp), 
                                                   use_indexer=False, 
                                                   batch_size=batch_size, 
                                                   remap=True, 
                                                   parent=self._data,
                                                   name="partition_{}".format(partition_nb))
                self.partitions.append(partition_loader)

@register_partitioning_strategy("RandomEdges")
class RandomEdgesGraphPartitioner(EdgeBasedGraphPartitioner):
    def __init__(self, data, k=2, batch_size=1):
        self.partitions = []
        super().__init__(data, k, batch_size=batch_size, random=True, index_by="")

@register_partitioning_strategy("Naive")
class NaiveGraphPartitioner(EdgeBasedGraphPartitioner):
    def __init__(self, data, k=2, batch_size=1):
        self.partitions = []
        super().__init__(data, k, batch_size=batch_size, random=False, index_by="")

@register_partitioning_strategy("SortedEdges")
class SortedEdgesGraphPartitioner(EdgeBasedGraphPartitioner):
    def __init__(self, data, k=2, batch_size=1):
        self.partitions = []
        super().__init__(data, k, batch_size=batch_size, random=False, index_by="s")

@register_partitioning_strategy("DoubleSortedEdges")
class DoubleSortedEdgesGraphPartitioner(EdgeBasedGraphPartitioner):
    def __init__(self, data, k=2, batch_size=1):
        self.partitions = []
        super().__init__(data, k, batch_size=batch_size, random=False, index_by="so")


def main():
    pass


if __name__ == "__main__":
    main()
