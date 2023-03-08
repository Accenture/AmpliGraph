# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Support for loading and managing datasets."""
from .datasets import (
    _load_xai_fb15k_237_experiment_log,
    load_cn15k,
    load_codex,
    load_fb13,
    load_fb15k,
    load_fb15k_237,
    load_from_csv,
    load_from_ntriples,
    load_from_rdf,
    load_nl27k,
    load_onet20k,
    load_ppi5k,
    load_wn11,
    load_wn18,
    load_wn18rr,
    load_yago3_10,
)
from .graph_data_loader import DataIndexer, GraphDataLoader, NoBackend
from .graph_partitioner import PARTITION_ALGO_REGISTRY, BucketGraphPartitioner
from .source_identifier import (
    DataSourceIdentifier,
    chunks,
    load_csv,
    load_gz,
    load_json,
    load_tar,
)
from .sqlite_adapter import SQLiteAdapter

__all__ = [
    "load_from_csv",
    "load_from_rdf",
    "load_wn18",
    "load_fb15k",
    "load_fb15k_237",
    "load_from_ntriples",
    "load_yago3_10",
    "load_wn18rr",
    "load_wn11",
    "load_fb13",
    "load_onet20k",
    "load_ppi5k",
    "load_nl27k",
    "load_cn15k",
    "load_codex",
    "chunks",
    "load_json",
    "load_gz",
    "load_tar",
    "load_csv",
    "DataSourceIdentifier",
    "DataIndexer",
    "NoBackend",
    "_load_xai_fb15k_237_experiment_log",
    "SQLiteAdapter",
    "GraphDataLoader",
    "BucketGraphPartitioner",
    "PARTITION_ALGO_REGISTRY",
]
