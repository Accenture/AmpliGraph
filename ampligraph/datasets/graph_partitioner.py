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

    def __init__(self, data, k=2):
        self._data = data
        self._k = k

    def get_triples(self, vertices, **kwargs):
        """Given list of vertices return all triples which contain them.

        Parameters
        ----------
        vertices: list of vertices

        Returns
        -------
        array with triples containing vertices

        """
        return np.array([x for x in self._data 
                         if x[2] in vertices or x[0] in vertices])

    @abstractmethod
    def split(self, seed=None, **kwargs):
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
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, seed=None, verbose=False, **kwargs):
        """Split data into partitions based on bucketed entities as follows:
        
            If you have 2 buckets (0, 1) then you will have 3 partitions i.e.
               a. edges with sub and obj in bucket 0 (0 - 0)
               b. edges with sub and obj in bucket 1 (1 - 1)
               c. edges with sub in 0 and obj in  1 (0 - 1)
        
           Returns
           -------
            partitions: created partitions
        """
        # get the unique entities (s and o)
        unique_ents = set(self._data[:,0])
        unique_ents.update(set(self._data[:,2]))
        unique_ents = np.array(list(unique_ents))
        
        # store entities in buckets
        bucketed_entities = dict()
        
        # Max entities in a bucket 
        max_entities = np.int32(np.ceil(unique_ents.shape[0] / self._k))
        for i in range(self._k):
            # store the entities in the buckets
            bucketed_entities[i] = unique_ents[i * max_entities: (i+1) * max_entities]

        # triples in partitions (based on start and end buckets)
        triples_of_bucket_dict = dict()
        # entities in the partition
        entities_of_bucket_dict = dict()

        start_time = time.time()
        
        
        total_partitions = 0
        for i in range(self._k):
            for j in range(i, self._k):
                try:
                    # based on where the triples start and end, put in respective partitions
                    triples_of_bucket_dict[i][j] =  self._data[np.logical_or(
                                                            np.logical_and(np.isin(self._data[:, 0], bucketed_entities[i]),
                                                                          np.isin(self._data[:, 2], bucketed_entities[j])),
                                                            np.logical_and(np.isin(self._data[:, 0], bucketed_entities[j]),
                                                                          np.isin(self._data[:, 2], bucketed_entities[i]))), :]



                    entities_of_bucket_dict[i][j] = np.array(list(set(triples_of_bucket_dict[i][j][:, 0]).union(
                        set(triples_of_bucket_dict[i][j][:, 2]))))
                except KeyError:
                    triples_of_bucket_dict[i] = dict()
                    entities_of_bucket_dict[i] = dict()
                    triples_of_bucket_dict[i][j] =  self._data[np.logical_or(
                                                            np.logical_and(np.isin(self._data[:, 0], bucketed_entities[i]),
                                                                          np.isin(self._data[:, 2], bucketed_entities[j])),
                                                            np.logical_and(np.isin(self._data[:, 0], bucketed_entities[j]),
                                                                          np.isin(self._data[:, 2], bucketed_entities[i]))), :]

                    entities_of_bucket_dict[i][j] = np.array(list(set(triples_of_bucket_dict[i][j][:, 0]).union(
                        set(triples_of_bucket_dict[i][j][:, 2]))))
                print('{} -> {} : {} triples, {} entities'.format(i, j, triples_of_bucket_dict[i][j].shape,
                                                                 entities_of_bucket_dict[i][j].shape))
                total_partitions +=1

        end_time = time.time()


        if verbose:
            print('Time Taken: {} secs'.format(end_time - start_time) )
            print('Total node partitions:', self._k)
            print('Total edge partitions:', total_partitions)
        
        # triples_of_bucket_dict - nested dictionary
        partitions = [triples_of_bucket_dict[part][sub_part] for part in triples_of_bucket_dict for sub_part in triples_of_bucket_dict[part]]
        return partitions

    
@register_partitioning_strategy("RandomVertices")
class RandomVerticesGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, seed=None, **kwargs):
        """Split data into k equal size partitions by randomly drawing subset of vertices
           of partition size and retrieving triples associated with these vertices.

           Returns
           -------
            partitions: parts of equal size with triples
        """
        vertices = np.asarray(list(set(self._data[:, 0]).union(set(self._data[:, 2]))))
        self.size = len(vertices)
        indexes = range(self.size)
        self.partition_size = int(self.size / self._k)
        vertices_partitions = []
        remaining_data = indexes
        if seed is not None:
            np.random.seed([(seed + i)*i for i in range(self._k) ])
        for part in range(self._k):
            split = np.random.choice(remaining_data, self.partition_size)
            remaining_data = np.setdiff1d(remaining_data, split)
            vertices_partitions.append(self.get_triples(vertices[split]))

        return vertices_partitions


@register_partitioning_strategy("RandomEdges")
class RandomEdgesGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, seed=None, **kwargs):
        """Split data into k equal size partitions by randomly drawing subset of
        edges from dataset.

           Returns
           -------
            partitions: parts of equal size with triples
        """

        self.size = len(self._data)
        indexes = range(self.size)
        self.partition_size = int(self.size / self._k)
        partitions = []
        remaining_data = indexes
        if seed is not None:
            np.random.seed(seed)
        for part in range(self._k):
            split = np.random.choice(remaining_data, self.partition_size)
            remaining_data = np.setdiff1d(remaining_data, split)
            partitions.append(self._data[split])

        return partitions


@register_partitioning_strategy("SortedEdges")
class SortedEdgesGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, seed=None, **kwargs):
        """Split data into k equal size partitions by drawing subset of edges from
        sorted dataset.
           
           Returns
           -------
            partitions: parts of equal size with triples
        """
        
        self.size = len(self._data)
        self.partition_size = int(self.size / self._k)
        partitions = []
        self.sorted_data = self._data[self._data[:,0].argsort()]
        
        for part in range(self._k):
            split = self.sorted_data[part * self.partition_size:self.partition_size * (1 + part), :]
            partitions.append(split)
        
        return partitions        


@register_partitioning_strategy("NaiveGraph")
class NaiveGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, seed=None, **kwargs):
        """Split data into k equal size partitions by drawing subset of edges
        from dataset.
           
           Returns
           -------
            partitions: parts of equal size with triples
        """
        
        self.size = len(self._data)
        self.partition_size = int(self.size / self._k)
        partitions = []
        
        for part in range(self._k):
            split = self._data[part * self.partition_size:self.partition_size * (1 + part), :]
            partitions.append(split)
        
        return partitions      


@register_partitioning_strategy("DoubleSortedEdges")
class DoubleSortedEdgesGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, seed=None, **kwargs):
        """Split data into k equal size partitions by drawing subset of edges
        from double-sorted (according to subject and object) dataset with refinement.
                  
           Returns
           -------
            partitions: parts of equal size with triples
        """

        self.size = len(self._data)
        self.partition_size = int(self.size / self._k)
        partitions = []
        
        self.sorted_data = self._data[np.lexsort((self._data[:, 0], self._data[:, 2]))]
        
        for part in range(self._k):
            split = self.sorted_data[part * self.partition_size:self.partition_size * (1 + part)]
            partitions.append(split)
        
        return partitions  


def main():
    pass


if __name__ == "__main__":
    main()
