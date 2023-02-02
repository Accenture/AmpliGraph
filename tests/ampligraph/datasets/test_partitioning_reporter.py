# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from ampligraph.datasets.partitioning_reporter import PartitioningReporter
from ampligraph.datasets.graph_partitioner import NaiveGraphPartitioner, GraphDataLoader
import pytest
import mock
from pytest_mock import mocker
import pandas as pd
import numpy as np


SCOPE = "function"

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


@pytest.fixture(params = ['test-parent.csv'], scope=SCOPE)
def parent(request):
    graph = pd.DataFrame(np.append(get_test_graph(), np.array([['a','t','a']]*9), axis=0))
    graph.to_csv(request.param, header=None, index=None, sep='\t') 
    data_loader = GraphDataLoader(request.param)
    yield data_loader
    data_loader.clean()
    try:
        os.remove(request.param)
    except:
        del request.param


@pytest.fixture(params = ['test.csv'], scope=SCOPE)
def data_1(request, parent):
    graph = pd.DataFrame(get_test_graph())
    graph.to_csv(request.param, header=None, index=None, sep='\t') 
    data_loader = GraphDataLoader(request.param)
    data_loader.parent = parent 
    yield data_loader
    data_loader.clean()
    try:
        os.remove(request.param)
    except:
        del request.param


@pytest.fixture(params = ['test.csv'], scope=SCOPE)
def data_2(request, parent):
    graph = pd.DataFrame(get_test_graph())
    graph.to_csv(request.param, header=None, index=None, sep='\t') 
    data_loader = GraphDataLoader(request.param)
    data_loader.parent = parent
    yield data_loader
    data_loader.clean()
    try:
        os.remove(request.param)
    except:
        del request.param
    

@pytest.fixture(params = [2, pytest.param(3, marks=pytest.mark.skip)], scope=SCOPE)
def k(request):
    return request.param


@pytest.fixture(scope=SCOPE)
def partitioning(data_1, data_2, k, mocker):
    mocker.patch.object(NaiveGraphPartitioner, 'get_partitions_list')
    mocker.patch.object(NaiveGraphPartitioner, '_split')
    NaiveGraphPartitioner.get_partitions_list.return_value = [data_1, data_2]
    NaiveGraphPartitioner._split.return_value = None
    partitioning = NaiveGraphPartitioner(data_1, k)
    n = data_1.get_data_size()
    partitioning = partitioning.get_partitions_list()
    sizes = [x.get_data_size() for x in partitioning]
    avg_size = np.mean(sizes)
    max_size = np.max(sizes)
    yield partitioning, {'avg_size': avg_size, 'max_size': max_size, 'k': n}


@pytest.fixture(scope=SCOPE)
def reporter(partitioning):
    logs = {}
    partitionings = {"one":(partitioning, logs)}
    reporter = PartitioningReporter(partitionings=partitionings)
    yield reporter


def test_get_edge_cut(reporter, partitioning):
    partitioning, args = partitioning 
    edge_cut, edge_cut_proportion = reporter.get_edge_cut(args['k'], partitioning, args['avg_size'])
    assert edge_cut == 9, "Edge cut should be 9, instead got {} {}.".format(edge_cut, args)


def test_get_edge_imbalance(reporter, partitioning):
    partitioning, args = partitioning 
    edge_imb = reporter.get_edge_imbalance(args['avg_size'], args['max_size'])
    assert edge_imb == 0, "Edge imbalance shuld be 0 got {}".format(edge_imb)


def test_get_vertex_imbalance_and_count(reporter, partitioning):
    partitioning, args = partitioning 
    vertex_imb, vertex_cnt = reporter.get_vertex_imbalance_and_count(partitioning, vertex_count=True)
    assert vertex_imb == 0, "Vertex imbalance should be 0, instead got {}.".format(vertex_imb)
    assert vertex_cnt == [10, 10], "Vertex count is {}, expected [10, 10]".format(vertex_cnt)


def test_get_average_deviation_from_ideal_size_vertices(reporter, partitioning):
    partitioning, args = partitioning 
    dev_vertices = reporter.get_average_deviation_from_ideal_size_vertices(partitioning)
    assert dev_vertices == 100, "Avg deviation from ideal size for vertices should be 100, instead got {}.".format(dev_vertices)


def test_get_average_deviation_from_ideal_size_edges(reporter, partitioning):
    partitioning, args = partitioning 
    dev_edges = reporter.get_average_deviation_from_ideal_size_edges(partitioning)
    assert dev_edges == 0, "Avg deviation from ideal size for edges should be 0, instead got {}.".format(dev_edges)


def test_get_edges_count(reporter, partitioning):
    partitioning, args = partitioning 
    edges_cnt = reporter.get_edges_count(partitioning)
    assert edges_cnt == [9, 9], "Edges counts should be [9, 9] instead got {}.".format(edges_cnt)
