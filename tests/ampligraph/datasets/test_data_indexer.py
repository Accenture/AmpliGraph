# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from ampligraph.datasets.data_indexer import DataIndexer
import numpy as np
import pytest


def data_generator(data):
    for elem in data:
        yield np.array(elem).reshape((1, 3))

np_array =  np.array([['a', 'b', 'c'], ['c', 'b', 'd'], ['d', 'e', 'f'], ['f', 'e', 'c'], ['a', 'e', 'd']])
generator = lambda: data_generator(np_array) 


@pytest.fixture(params=[np_array, pytest.param(generator, marks=pytest.mark.skip("Can't use generators as parameters in fixtures."))])
def data_type(request):
    '''Returns an in-memory DataIndexer instance with example data.'''
    return request.param


@pytest.fixture(params=['in_memory', 'sqlite', 'shelves'])
def data_indexer(request, data_type):
    '''Returns an in-memory DataIndexer instance with example data.'''
    data_indexer = DataIndexer(data_type, backend=request.param)
    yield data_indexer
    data_indexer.clean()

def test_get_max_ents_index(data_indexer):
    max_ents = data_indexer.backend._get_max_ents_index()
    assert max_ents == 3, "Max index should be 3 for 4 unique entities, instead got {}.".format(max_ents)


def test_get_max_rels_index(data_indexer):
    print(data_indexer)
    max_rels = data_indexer.backend._get_max_rels_index()
    assert max_rels == 1, "Max index should be 1 for 2 unique relations, instead got {}.".format(max_rels)


def test_get_entities_in_batches(data_indexer):
    for batch in data_indexer.get_entities_in_batches(batch_size=3):
        assert len(batch) == 3
        break
    for batch in data_indexer.get_entities_in_batches():
        assert len(batch) == data_indexer.get_entities_count()


def test_get_indexes(data_indexer):
    tmp = np.array([['a', 'b', 'c']])
    indexes = data_indexer.get_indexes(tmp)
    assert np.shape(indexes) == np.shape(tmp), "returned indexes are not the same shape"
    assert np.issubdtype(indexes.dtype,  np.integer), "indexes are not integers"

@pytest.mark.skip(reason="update not implemented for sqlite backend")
def test_update_mappings(data_indexer):
    new_data = np.array([['g', 'i', 'h'], ['g', 'i', 'a']])
    data_indexer.update_mappings(new_data)
    assert data_indexer.backend.ents_length == 6, "entities size should be 6, two new added"
    assert data_indexer.backend.rels_length == 3, "relations size should be 3, one new added"


def test_get_starting_index_ents(data_indexer):
    ind = data_indexer.backend._get_starting_index_ents()
    assert ind == data_indexer.backend.ents_length, "index doesn't match entities length"


def test_get_starting_index_rels(data_indexer):
    ind = data_indexer.backend._get_starting_index_rels()
    assert ind == data_indexer.backend.rels_length, "index doesn't match relations length"
