# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
import pytest
import pandas as pd
from ampligraph.datasets import load_csv, chunks
from ampligraph.datasets.source_identifier import DataSourceIdentifier
import os

SCOPE = "function"

def create_data():
    np_array =  np.array([['a','b','c'],['c','b','d'],['d','e','f'],['f','e','c'],['a','e','d'], ['a','b','d']])
    df = pd.DataFrame(np_array)
    df.to_csv('test.csv', index=False, header=False, sep='\t')
    df.to_csv('test.txt', index=False, header=False, sep='\t')
    df.to_csv('test.gz', index=False, header=False, sep='\t', compression='gzip')
    return np_array, len(df)

def clean_data():
    os.remove('test.csv')
    os.remove('test.txt')
    os.remove('test.gz')

@pytest.fixture(params=['np_array', 'test.csv', 'test.gz', 'test.txt'], scope=SCOPE)
def data_source(request):
    np_array, _ = create_data()
    return request.param if request.param != 'np_array'  else np_array
    clean_data()

@pytest.fixture(scope=SCOPE)
def source_identifier(request, data_source):
    '''Returns a SourceIdentifier instance.'''
    src_identifier = DataSourceIdentifier(data_source)
    yield src_identifier, data_source

   
def test_load_csv():
    _, length = create_data()
    data = load_csv('test.csv')
    assert len(data) == length, "Loaded data differ from what it should be, got {}, expected {}.".format(len(data), len(df))
    clean_data()

def test_data_source_identifier(source_identifier):
    src_identifier, data_src = source_identifier
    if isinstance(data_src, str):
        src = data_src.split('.')[-1]
    elif isinstance(data_src, np.ndarray):
        src = "iter"
    else:
        assert False, "Provided data source is not supported."
    assert src == src_identifier.get_src(), "Source identified not equal to the one provided."


def test_data_source_identifier_fetch_loader(source_identifier):
    src_identifier, data_src = source_identifier
    loader = src_identifier.fetch_loader()
    data = loader(data_src)
    assert isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame) or isinstance(data, type(chunks([]))), "Returned data should be either in numpy array or pandas data frame, instead got {}".format(type(data))

