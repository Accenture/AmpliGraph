# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Helper functions to load knowledge graphs."""

from .datasets import load_from_csv, load_from_rdf, load_fb15k, load_wn18, load_fb15k_237, load_from_ntriples, \
    load_yago3_10, load_wn18rr, load_wn11, load_fb13

from .abstract_dataset_adapter import AmpligraphDatasetAdapter
from .sqlite_adapter import SQLiteAdapter
from .numpy_adapter import NumpyDatasetAdapter
from .oneton_adapter import OneToNDatasetAdapter

__all__ = ['load_from_csv', 'load_from_rdf', 'load_from_ntriples', 'load_wn18', 'load_fb15k',
           'load_fb15k_237', 'load_yago3_10', 'load_wn18rr', 'load_wn11', 'load_fb13',
           'AmpligraphDatasetAdapter', 'NumpyDatasetAdapter', 'SQLiteAdapter', 'OneToNDatasetAdapter']
