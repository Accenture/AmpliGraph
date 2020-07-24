# Copyright 2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
import pytest
from ampligraph.datasets.graph_partitioner import AbstractGraphPartitioner, RandomVerticesGraphPartitioner,\
     RandomEdgesGraphPartitioner, SortedEdgesGraphPartitioner, NaiveGraphPartitioner, \
     DoubleSortedEdgesGraphPartitioner
import numpy as np
from scipy import special
from itertools import permutations 

class dummyGraphGenerator():
    """Generates graph with certain number of nodes, edges
       and unique edges.
       
       Example
       -------
       >>>># Construct graph with 20 nodes, 10 edges and 4 unique edges types
       >>>>generator = dummyGraphGenerator(20,10,4)
       >>>>generator.get()
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


def template_test_edges_graph_partitioner(base_partitioner):
    X = get_test_graph() 
    partitioner = base_partitioner(X, 2)
    partitions = partitioner.split(seed=100)
    partition1_size = len(partitions[0])
    partition2_size = len(partitions[1])
    all_edges = len(X)
    allowable_sizes = [int(all_edges/2) + 1, int(all_edges/2)]
    assert(partition1_size in allowable_sizes  or partition2_size in allowable_sizes)


def test_register_partitioning_strategy():
    assert False, "not implemented"


def test_insert_in_registry():
    assert False, "not implemented"


def test_get_number_of_partitions():
    assert False, "not implemented"


class TestAbstractGraphPartitioner:
    def test___init__(self):
        assert False, "not implemented"


    def test___iter__(self):
        assert False, "not implemented"


    def test_partitions_generator(self):
        assert False, "not implemented"


    def test_get_partitions_iterator(self):
        assert False, "not implemented"


    def test_get_partitions_list(self):
        assert False, "not implemented"


    def test___next__(self):
        assert False, "not implemented"


    def test__split(self):
        assert False, "not implemented"

    def test_abstract_graph_partitioner_get_triples(self):
        """ Test whether get_triples methods returns correct 
            triples given set of vertices.
    
          Triples:
          [['p', 't', 'e'],
           ['g', 'u', 'l'],
           ['m', 'v', 'f'],
           ['c', 'r', 'n'],
           ['n', 'r', 'c'],
           ['i', 'v', 'a'],
           ['g', 'u', 'l']]
        """
        generator = dummyGraphGenerator(20,10,4)
        data = generator.get(seed=100)
        X = data["train"]    
        mockPartitioner = metaclass_instantiator(AbstractGraphPartitioner)
        cut = mockPartitioner(X, 2)
        vertices = ['g', 'e', 'c']
        triples = cut.get_triples(vertices)
    
        subarray = np.array([['p', 't', 'e'],
                             ['g', 'u', 'l'],
                             ['c', 'r', 'n'],
                             ['n', 'r', 'c'],
                             ['g', 'u', 'l']])
    
        np.testing.assert_array_equal(triples, subarray)


class TestBucketGraphPartitioner:
    def test___init__(self):
        assert False, "not implemented"


    def test_create_single_partition(self):
        assert False, "not implemented"


    def test__split(self):
        assert False, "not implemented"


class TestRandomVerticesGraphPartitioner:
    def test___init__(self):
        assert False, "not implemented"

    def test__split(self):
        assert False, "not implemented"   
    
    def test_random_vertices_graph_partitioner(self):
        X = get_test_graph()
        partitioner = RandomVerticesGraphPartitioner(X, 2)
        partitions = partitioner.split(seed=100)
        partition1_nodes_cnt = len(set(partitions[0][:,0]).union(set(partitions[0][:,2])))
        partition2_nodes_cnt = len(set(partitions[1][:,0]).union(set(partitions[1][:,2])))
        all_nodes = set(X[:,0]).union(set(X[:,2]))
        assert(int(len(all_nodes) / 2) <= partition1_nodes_cnt)
        assert(int(len(all_nodes) / 2) <= partition2_nodes_cnt)

     def test_get_triples_random_vertices(self):
        vertices = [1]
        data = np.array([[1,2,3],[4,5,1],[1,2,4],[5,2,4]])
        test_data = np.array([[1,2,3],[4,5,1],[1,2,4]])
        graph_partitioner = RandomVerticesGraphPartitioner(data, 2)
        assert (graph_partitioner.get_triples(vertices) == test_data).all()

    def test_not_equal(self):
        vertices = [1]    
        data = np.array([[1,2,3],[2,3,4],[3,4,1],[1,1,1],[0,3,2]])    
        test_data = np.array([[1,2,3],[4,5,1],[1,2,4]])
        graph_partitioner = RandomVerticesGraphPartitioner(data, 2)
        assert not (graph_partitioner.get_triples(vertices) == test_data).all()


class TestEdgeBasedGraphPartitioner:
    def test___init__(self):
        assert False, "not implemented"


    def test__split(self):
        assert False, "not implemented"


    def test_format_batch(self):
        assert False, "not implemented"


class TestRandomEdgesGraphPartitioner:
    def test___init__(self):
        assert False, "not implemented"

    def test_random_edges_graph_partitioner(self):
        template_test_edges_graph_partitioner(RandomEdgesGraphPartitioner)

    def test_get_triples_random_edges(self):
        vertices = [1]
        data = np.array([[1,2,3],[4,5,1],[1,2,4],[5,2,4]])
        test_data = np.array([[1,2,3],[4,5,1],[1,2,4]])
        graph_partitioner = RandomEdgesGraphPartitioner(data, 2)
        assert (graph_partitioner.get_triples(vertices) == test_data).all()    
        

class TestNaiveGraphPartitioner:
    def test___init__(self):
        assert False, "not implemented"
    
    def test_naive_graph_partitioner(self):
        template_test_edges_graph_partitioner(NaiveGraphPartitioner)


class TestSortedEdgesGraphPartitioner:
    def test___init__(self):
        assert False, "not implemented"

    def test_sorted_edges_graph_partitioner(self):
        template_test_edges_graph_partitioner(SortedEdgesGraphPartitioner)
    

class TestDoubleSortedEdgesGraphPartitioner:
    def test___init__(self):
        assert False, "not implemented"
       
    def test_double_sorted_edges_graph_partitioner(self):
        template_test_edges_graph_partitioner(DoubleSortedEdgesGraphPartitioner)



def test_main():
    assert False, "not implemented"
