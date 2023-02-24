# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from ampligraph.datasets import GraphDataLoader, NoBackend
from ampligraph.datasets.sqlite_adapter import SQLiteAdapter
import pytest
import numpy as np
import pandas as pd


SCOPE = "function"
np_array =  np.array([['a','b','c'],['c','b','d'],['d','e','f'],['f','e','c'],['a','e','d'], ['a','b','d']])
df = pd.DataFrame(np_array)
df.to_csv('test.csv', index=False, header=False, sep='\t')


@pytest.fixture(params=[np_array, 'test.csv'], scope=SCOPE)
def data_source(request):
    return request.param


@pytest.fixture(params=[NoBackend, SQLiteAdapter], scope=SCOPE)
def graph_data_loader(request, data_source):
    '''Returns a GraphDataLoader instance.'''
    data_loader = GraphDataLoader(data_source, backend=request.param)
    yield data_loader
    data_loader.clean()

   
def test_graph_data_loader_get_batch_generator(graph_data_loader):
    batch_gen = graph_data_loader.get_batch_generator()
    batch = next(batch_gen)
    assert np.shape(batch) == (1,3), "batch size is wrong, got {}".format(batch)

def test_graph_data_loader_get_data_size(graph_data_loader):
    size = graph_data_loader.get_data_size()
    assert size == 6, "Size is not equal to 5"


def test_graph_data_loader_get_complementary_subjects(graph_data_loader):
    sample = np.array([['a','b','d'], ['a','b','d']])
    sample_inds = graph_data_loader.backend.mapper.get_indexes(sample, type_of="t")
    subjects = graph_data_loader.get_complementary_subjects(sample_inds)
    expected = [[0, 1],[0, 1]]
    assert [set(x) for x in subjects] == [set(x) for x in expected], "Subjects differ, expected {}, instead got {}.".format(expected, subjects)


def test_graph_data_loader_get_complementary_objects(graph_data_loader):
    sample = np.array([['a','b','d'], ['a','b','d']])
    sample_inds = graph_data_loader.backend.mapper.get_indexes(sample, type_of="t")
    objects = graph_data_loader.get_complementary_objects(sample_inds)
    expected = [[1, 2], [1, 2]]
    assert [set(x) for x in objects] == [set(x) for x in expected], "Objects differ, expected {}, instead got {} for indexes: {}.".format(expected, objects, sample_inds)


def test_graph_data_loader_get_complementary_entities(graph_data_loader):
    sample = np.array([['a','b','d']])
    sample_inds = graph_data_loader.backend.mapper.get_indexes(sample, type_of="t")
    subjects, objects = graph_data_loader.get_complementary_entities(sample_inds)
    expected = [[0, 1]]
    assert [set(x) for x in subjects] == [set(x) for x in expected], "Subjects differ, expected {}, instead got {}.".format(subjects, expected)
    expected = [[1, 2]]
    assert [set(x) for x in objects] == [set(x) for x in expected], "Objects differ, expected {}, instead got {}.".format(objects, expected)


def test_graph_data_loader_get_triples(graph_data_loader):
    entities = ['a','c']
    entities_inds = graph_data_loader.backend.mapper.get_indexes(entities, type_of="e")
    triples = graph_data_loader.get_triples(entities=entities_inds)
    assert np.array_equal(triples, np.array([[0, 0, 1, 'train']])), "Returned indexes should be , instead got {} for the following indexes: {}.".format(triples, entities_inds)

def test_backends_with_filters():
    train = np.array([[1,1,2], [1,1,3],[1,1,4],[5,1,3],[5,1,4],[6,1,3],[6,1,2],[6,1,4],[6,1,7]])
    test = np.array([[3,1,2], [4,1,3],[5,1,4], [5,1,2], [1,1,5]])
    val = np.array([[3,1,6], [2,1,2],[1,1,6]])
    use_filter = {'train':train,'test':test, 'val':val}

    # complementary objects to 1,1,?: train: 2, 3, 4, test: 5, val: 6 
    # complementary subjects to ?,1,2: train: 1, 6, test: 3, 5, val: 2

    data_dummy = GraphDataLoader(train, batch_size=1, dataset_type="train", use_filter=use_filter)   
    triple = np.array([[0,0,1]])
    data_db = GraphDataLoader(train, batch_size=1, dataset_type="train", backend=SQLiteAdapter, use_filter=use_filter)
    data_db.add_dataset(test, dataset_type='test')
    data_db.add_dataset(val, dataset_type='val')    
    
    dummy = data_dummy.backend.mapper.get_indexes(data_dummy.get_complementary_objects(triple, use_filter=use_filter)[0], type_of='e', order='ind2raw')
    db = data_db.backend.mapper.get_indexes(data_db.get_complementary_objects(triple, use_filter=use_filter)[0], type_of='e', order='ind2raw')
    assert np.array_equal(dummy, db), "Backends differ: dummy: {}, db: {}".format(dummy, db) 
