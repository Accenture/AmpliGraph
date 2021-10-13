# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from .datasets import load_from_csv, load_from_rdf, load_fb15k, load_wn18, load_fb15k_237, load_from_ntriples, \
    load_yago3_10, load_wn18rr, load_wn11, load_fb13
from .data_indexer import DataIndexer
from .partitioned_data_manager import PartitionDataManager
from .graph_partitioner import AbstractGraphPartitioner, RandomVerticesGraphPartitioner, RandomEdgesGraphPartitioner, \
    SortedEdgesGraphPartitioner, NaiveGraphPartitioner, DoubleSortedEdgesGraphPartitioner, BucketGraphPartitioner, \
    EdgeBasedGraphPartitioner, get_number_of_partitions, PARTITION_ALGO_REGISTRY
from .graph_data_loader import DummyBackend, GraphDataLoader

from .abstract_dataset_adapter import AmpligraphDatasetAdapter
from .sqlite_adapter import SQLiteAdapter
from .numpy_adapter import NumpyDatasetAdapter
from .oneton_adapter import OneToNDatasetAdapter

# from .partitioning_reporter import PartitioningReporter, compare_partitionings

# from .source_identifier import DataSourceIdentifier, load_csv, load_tar, load_gz, chunks
#
#
# __all__ = ['load_from_csv', 'load_from_rdf', 'load_from_ntriples', 'load_wn18', 'load_fb15k',
#            'load_fb15k_237', 'load_yago3_10', 'load_wn18rr', 'load_wn11', 'load_fb13',
#            'AmpligraphDatasetAdapter', 'DataIndexer', 'SQLiteAdapter', 'NumpyDatasetAdapter', 'OneToNDatasetAdapter',
#            'AbstractGraphPartitioner', 'RandomVerticesGraphPartitioner', 'RandomEdgesGraphPartitioner',
#            'SortedEdgesGraphPartitioner', 'NaiveGraphPartitioner', 'DoubleSortedEdgesGraphPartitioner',
#            'BucketGraphPartitioner', 'EdgeBasedGraphPartitioner', 'get_number_of_partitions', 'PartitioningReporter',
#            'compare_partitionings', 'GraphDataLoader', 'DummyBackend', 'PartitionDataManager', 'DataSourceIdentifier',
#            'load_csv', 'load_tar', 'load_gz', 'chunks', 'DataIndexer']
