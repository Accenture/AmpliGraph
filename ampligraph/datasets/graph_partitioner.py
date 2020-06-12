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


class BucketGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, seed=None, verbose=False, **kwargs):
        """Split data into 
        
           Returns
           -------
            partitions: created partitions
        """
        unique_ents = set(self._data[:,0])
        unique_ents.update(set(self._data[:,2]))
        unique_ents = np.array(list(unique_ents))
        dataset_df = pd.DataFrame(self._data, columns=['s','p', 'o'])
        p_triples_bool = dict()
        p_triples = dict()
        p_ents = dict()

        p_triples_multiple_buckets = dict()
        p_ent_multiple_buckets = dict()
        start_time = time.time()
        for i in range(self._k):
            max_triples = unique_ents.shape[0]//self._k  # size of partition
            p_ents[i] = unique_ents[i * max_triples: (i+1) * max_triples] # chunk vertices based on size
            p_triples_bool[i] = np.logical_and(dataset_df['s'].isin(p_ents[i]).values,
                                                    dataset_df['o'].isin(p_ents[i]).values)
            # filter triples where both subject and predicate are in the created partition
            p_triples[i] = self._data[p_triples_bool[i], :] # take only these triples
            
            if verbose:
                print(p_triples[i].shape)

        total_partitions = 0
        for i in range(self._k):
            for j in range(i, self._k):
                if i==j:
                    try:
                        p_triples_multiple_buckets[i][i] = p_triples[i]
                        p_ent_multiple_buckets[i][i] = p_ents[i]
                    except KeyError:
                        p_triples_multiple_buckets[i] = dict()
                        p_ent_multiple_buckets[i] = dict()
                        p_triples_multiple_buckets[i][i] = p_triples[i]
                        p_ent_multiple_buckets[i][i] = p_ents[i]
                else:
                    try:
                        p_triples_multiple_buckets[i][j] =  self._data[np.logical_or(
                                                                np.logical_and(dataset_df['s'].isin(p_ents[i]).values,
                                                                              dataset_df['o'].isin(p_ents[j]).values),
                                                                np.logical_and(dataset_df['s'].isin(p_ents[j]).values,
                                                                              dataset_df['o'].isin(p_ents[i]).values)), :]





                        p_ent_multiple_buckets[i][j] = np.array(list(set(p_triples_multiple_buckets[i][j][:, 0]).union(
                            set(p_triples_multiple_buckets[i][j][:, 2]))))
                    except KeyError:
                        p_triples_multiple_buckets[i] = dict()
                        p_ent_multiple_buckets[i] = dict()
                        p_triples_multiple_buckets[i][j] =  self._data[np.logical_or(
                                                                np.logical_and(dataset_df['s'].isin(p_ents[i]).values,
                                                                              dataset_df['o'].isin(p_ents[j]).values),
                                                                np.logical_and(dataset_df['s'].isin(p_ents[j]).values,
                                                                              dataset_df['o'].isin(p_ents[i]).values)), :]



                        p_ent_multiple_buckets[i][j] = np.array(list(set(p_triples_multiple_buckets[i][j][:, 0]).union(
                            set(p_triples_multiple_buckets[i][j][:, 2]))))
                if verbose:                        
                    print('{} -> {} : {} triples, {} entities'.format(i, j, p_triples_multiple_buckets[i][j].shape,
                                                                     p_ent_multiple_buckets[i][j].shape))
                total_partitions +=1

        end_time = time.time()


        if verbose:
            print('Time Taken: {} secs'.format(end_time - start_time) )
            print('Total node partitions:', self._k)
            print('Total edge partitions:', total_partitions)
        
        # p_triples_multiple_buckets - nested dictionary
        partitions = [p_triples_multiple_buckets[part][sub_part] for part in p_triples_multiple_buckets for sub_part in p_triples_multiple_buckets[part]]
        return partitions


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
