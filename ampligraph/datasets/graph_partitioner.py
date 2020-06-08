import numpy as np
import matplotlib.pyplot as plt
import ampligraph as amp
from ampligraph import datasets
import tracemalloc
from time import time
import pytest
from abc import ABC, abstractmethod
from functools import wraps
from quality_reporter import timing_and_memory

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
        return np.array([x for x in self._data if x[2] in vertices \
                         or x[0] in vertices])

    @abstractmethod
    def split(self, **kwargs):
        """Split data into k equal size partitions.

           Returns
           -------
            partitions: parts of equal size with triples
        """
        pass

class RandomVerticesGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, **kwargs):
        """Split data into k equal size partitions by randomly drawing subset of vertices
           of partition size and retrieving triples associated with these vertices.

           Returns
           -------
            partitions: parts of equal size with triples
        """
        vertices = np.asarray(list(set(self._data[:,0]).union(set(self._data[:,2]))))
        self.size = len(vertices)
        indexes = range(self.size)
        self.partition_size = int(self.size / self._k)
        vertices_partitions = []
        remaining_data = indexes
        for part in range(self._k):
            split = np.random.choice(indexes, self.partition_size)
            remaining_data = np.setdiff1d(remaining_data, split)
            vertices_partitions.append(self.get_triples(vertices[split]))

        return vertices_partitions

class RandomEdgesGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, **kwargs):
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
        for part in range(self._k):
            split = np.random.choice(indexes, self.partition_size)
            remaining_data = np.setdiff1d(remaining_data, split)
            partitions.append(self._data[split])

        return partitions

class SortedEdgesGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, **kwargs):
        """Split data into k equal size partitions by drawing subset of edges from
        sorted dataset.
           
           Returns
           -------
            partitions: parts of equal size with triples
        """
        
        self.size = len(self._data)
        indexes = range(self.size)
        self.partition_size = int(self.size / self._k)
        partitions = []
        self.sorted_data = np.sort(self._data, axis=0)  # TO OPTIMIZE
        
        remaining_data = indexes
        for part in range(self._k):
            split = self.sorted_data[part*self.partition_size:self.partition_size*(1+part),:]
            partitions.append(split)
        
        return partitions        

class NaiveGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, **kwargs):
        """Split data into k equal size partitions by drawing subset of edges
        from dataset.
           
           Returns
           -------
            partitions: parts of equal size with triples
        """
        
        self.size = len(self._data)
        indexes = range(self.size)
        self.partition_size = int(self.size / self._k)
        partitions = []
        
        remaining_data = indexes
        for part in range(self._k):
            split = self._data[part*self.partition_size:self.partition_size*(1+part),:]
            partitions.append(split)
        
        return partitions      

class DoubleSortedEdgesGraphPartitioner(AbstractGraphPartitioner):
    def __init__(self, data, k=2):
        super().__init__(data, k)

    @timing_and_memory
    def split(self, **kwargs):
        """Split data into k equal size partitions by drawing subset of edges
        from double-sorted (according to subject and object) dataset with refinement.
                  
           Returns
           -------
            partitions: parts of equal size with triples
        """

        self.size = len(self._data)
        indexes = range(self.size)
        self.partition_size = int(self.size / self._k)
        partitions = []
        
        self.sorted_data = self._data[np.lexsort((self._data[:,0], self._data[:,2]))]
        
        remaining_data = indexes
        for part in range(self._k):
            split = self.sorted_data[part*self.partition_size:self.partition_size*(1+part)]
            partitions.append(split)
        
        return partitions  

def main():
    pass

if __name__ == "__main__":
    main()
