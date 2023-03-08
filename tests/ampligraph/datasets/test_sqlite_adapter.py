# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from ampligraph.datasets.sqlite_adapter import SQLiteAdapter
from ampligraph.datasets.source_identifier import DataSourceIdentifier
import pytest
import numpy as np
import pandas as pd

SCOPE = "function"
np_array =  np.array([['a','b','c'],['c','b','d'],['d','e','f'],['f','e','c'],['a','e','d'], ['a','b','d']])
indexed_arr =  np.array([[0,0,1],[1,0,2],[2,1,3],[3,1,1],[0,1,2], [0,0,2]])
df = pd.DataFrame(np_array)
df.to_csv('test.csv', index=False, header=False, sep='\t')


@pytest.fixture(scope=SCOPE)
def sqlite_adapter(request):
    '''Returns a SQLiteAdapter instance.'''
    src = DataSourceIdentifier('test.csv')
    backend = SQLiteAdapter('database.db', identifier=src)
    backend._load('test.csv')
    yield backend
    backend._clean()

   
def test_sqlite_adapter_get_batch_generator(sqlite_adapter):
    batch_gen = sqlite_adapter._get_batch_generator()
    batch = next(batch_gen)
    assert np.shape(batch) == (1,3), "batch size is wrong, got {}".format(batch)


def test_sqlite_adapter_get_data_size(sqlite_adapter):
    size = sqlite_adapter.get_data_size()
    assert size == 6, "Size is not equal to 6"
