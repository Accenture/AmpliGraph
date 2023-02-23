# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import pytest
from ampligraph.datasets.graph_partitioner import AbstractGraphPartitioner, RandomVerticesGraphPartitioner,\
     RandomEdgesGraphPartitioner, SortedEdgesGraphPartitioner, NaiveGraphPartitioner, \
     DoubleSortedEdgesGraphPartitioner, get_number_of_partitions, PARTITION_ALGO_REGISTRY,\
     GraphDataLoader
import numpy as np
import pandas as pd
from scipy import special
from itertools import permutations 
import os

SCOPE= "function"

class dummyGraphGenerator():
    """Generates graph with certain number of nodes, edges
       and unique edges.
       
       Example
       -------
       >>> # Construct graph with 20 nodes, 10 edges and 4 unique edges types
       >>> generator = dummyGraphGenerator(20,10,4)
       >>> generator.get()
       {'train': array([['h', 'r', 'p'],
                        ['a', 't', 'd'],
                        ['a', 's', 'j'],
                        ['d', 'y', 'h'],
                        ['e', 'v', 'j'],
                        ['e', 'z', 'l'],
                        ['c', 'x', 'n']], dtype='<U32'),
        'valid': array([['l', 'x', 'j']], dtype='<U32'),
        'test':  array([['k', 's', 'c'],
                        ['i', 'u', 'o']], dtype='<U32')}
       """
    
    nodes = "abcdefghijklmnop"
    edges = "rstuvwxyz"

    def __init__(self, n_nodes, n_edges, n_unique_edges):
        """Initialize graph generator.
        Parameters
        ----------
        n_nodes: number of nodes in the graph
        n_edges: number of edges in the graph
        n_unique_edges: number of unique edges in the graph        
        """
        assert(n_unique_edges <= n_edges)
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.n_unique_edges = n_unique_edges

    def get_unique(self, n_elems, alphabet):
        """Constructs list of size n_elems with unique substrings from alphabet.
        
        Parameters
        ----------
        n_elems: number of unique elements to be generated from the alphabet
        alphabet: list of unique symbols just like a small alphabet to construct 
                  unique words
        
        Return
        ------
        elems: list of size n_elems with unique elements
        """
        set_size = len(alphabet)
        size = set_size
        perms = list(alphabet)
        i = 1
        while size < n_elems:
            i += 1
            size += special.binom(set_size, i)
            perms.extend(permutations(alphabet, i))

        elems = []
        for elem in perms:
            if type(elem) == str:
                elems.append(elem)
            else: 
                elems.extend(list(elem))
        return elems        
        
    def generate_graph(self, nodes, edges_labels, seed=None):
        """Randomly draws subsets of nodes and edges labels to construct graph
        
        Parameters
        ----------
        nodes: list of unique nodes
        edges_labels: list of unique edges labels
        seed: seed for repeatability purposes
        
        Returns
        -------
        data: numpy array of size (n_edges,3) with graph triples (s,p,o)
        """
        assert(seed >= 0)
        data = np.zeros((self.n_edges,3)).astype(str)
        if seed is not None:
            np.random.seed([seed*2, seed+200, seed+10])
        data[:,0] = np.random.choice(nodes, self.n_edges)
        data[:,1] = np.random.choice(edges_labels, self.n_edges)
        data[:,2] = np.random.choice(nodes, self.n_edges)        
        
        return data
        
    def get(self, seed=None):
        """Delivers constructed graph.
        
        Returns
        -------
        X: dictionary in a format used in datasets with numpy arrays for train, validation 
        and test subsets
        """
        nodes = self.get_unique(self.n_nodes, self.nodes)
        edges_labels = self.get_unique(self.n_unique_edges, self.edges)
        graph = self.generate_graph(nodes, edges_labels, seed=seed)
        size = len(graph)
        
        X = {
            'train': graph[:int(0.7 * size)],
            'valid': graph[int(0.7 * size):int(0.8 * size)],
            'test':  graph[int(0.8 * size):],
        }

        return X


def metaclass_instantiator(metaclass):
    class metaclassInstance(metaclass):
        pass
    metaclassInstance.__abstractmethods__ = frozenset()
    return type('testClass' + metaclass.__name__.title(), (metaclassInstance,), {})


def get_test_graph():
    return np.array([['p', 't', 'e'],
                     ['a', 'r', 'l'],
                     ['i', 'v', 'a'],
                     ['f', 'u', 'g'],
                     ['p', 'v', 'm'],
                     ['n', 'r', 'c'],
                     ['m', 'v', 'f'],
                     ['g', 'u', 'l'],
                     ['g', 't', 'n']])


@pytest.fixture(params = ['test.csv'], scope=SCOPE)
def data(request):
    graph = pd.DataFrame(get_test_graph())
    graph.to_csv(request.param, header=None, index=None, sep='\t') 
    data_loader = GraphDataLoader(request.param)
    yield data_loader
    data_loader.clean()
    try:
        os.remove(request.param)
    except:
        del request.param
    

@pytest.fixture(params = [2, 3], scope=SCOPE)
def k(request):
    return request.param


@pytest.fixture(params = list(PARTITION_ALGO_REGISTRY.keys()), scope=SCOPE)
def graph_partitioner(request, data, k):
    partitioner_cls = PARTITION_ALGO_REGISTRY.get(request.param) 
    partitioner = partitioner_cls(data, k)
    yield partitioner, k
    partitioner.clean()

@pytest.fixture(params = [RandomEdgesGraphPartitioner, NaiveGraphPartitioner, SortedEdgesGraphPartitioner, DoubleSortedEdgesGraphPartitioner], scope=SCOPE)
def edge_graph_partitioner(request, data, k):
    partitioner_cls = request.param
    partitioner = partitioner_cls(data, k)
    yield partitioner, data, k
    partitioner.clean()

def test_get_number_of_partitions():
    n = get_number_of_partitions(3)
    assert n == 6, "Number of partitions should be 6, instead got {}.".format(n)


def test_number_of_partitions_after_graph_partitioning(graph_partitioner):
   n_parts = len(graph_partitioner[0].partitions)
   # BucketGraph return k*(k+1)/2 partitions
   # RandomVertices will return k or less than k partitions depending on the vertex splits.
   if graph_partitioner[0].__class__.__name__ in ['BucketGraphPartitioner', 'RandomVerticesGraphPartitioner']:
       expected = get_number_of_partitions(graph_partitioner[1])
       assert n_parts <= expected, "{}: Requested number of partitions based on buckets should be greater or equal to the actual, expected max of {} got {}".format(graph_partitioner[0].__class__.__name__, expected, n_parts)
   else:
       assert n_parts == graph_partitioner[1], "{}: Requested number of partitions not equal to the actual should be {} got {}".format(graph_partitioner[0].__class__.__name__, graph_partitioner[1], n_parts)


# def test_random_vertices_graph_partitioner(data, k):
      # TODO: this test isn't really meaningful: triples that we obtain with this partitioning strategy
      # are those with both subject AND object in the same partition. So it may happen that the number of
      # entities involved in triples is much smaller than the expected
#     partitioner = RandomVerticesGraphPartitioner(data, k)
#     print('data.shape: ', data)
#     n_nodes = data.backend.mapper.get_entities_count()//k
#     print('n_nodes: ', n_nodes)
#     for partition in partitioner:
#         actual = partition.backend.mapper.get_entities_count()
#         print(actual)
#         accept_range = [x for y in [(n_nodes - i, n_nodes + i) for i in range(k)] for x in y]
#         print(accept_range)
#         assert actual in accept_range, "Nodes in a bucket not equal to expected, got {}, expected {} +- (0, {}).".format(actual, k, n_nodes)
#     partitioner.clean()

def test_partition_size_in_edge_based_graph_partitioner(edge_graph_partitioner):
    partitioner, data, k = edge_graph_partitioner
    expected_size = data.get_data_size()//k
    allowable_sizes =  [x for y in [(expected_size - i, expected_size + i) for i in range(k)] for x in y]
    for partition in partitioner:
        actual = partition.get_data_size()
        assert(actual in allowable_sizes), "Partition size is not allowed, expected {} +- (0,{}), actual {}.".format(expected_size, k, actual)

