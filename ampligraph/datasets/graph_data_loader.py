# Copyright 2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from ampligraph.datasets.source_identifier import DataSourceIdentifier
import sqlite3
from sqlite3 import Error
import numpy as np
from urllib.request import pathname2url
import os
import shelve
import pandas as pd
from datetime import datetime
from ampligraph.utils.profiling import get_human_readable_size

class DummyBackend():
    def __init__(self, identifier, verbose=False):
        self.verbose = verbose
        self.identifier = identifier
        
    def __enter__ (self):
        return self
    
    def __exit__ (self, type, value, tb):
        pass
    
    def _load(self, data_source, dataset_type):
        if self.verbose:
            print("Simple in-memory data loading of {} dataset.".format(dataset_type))
        self.data_source = data_source
        loader = self.identifier.fetch_loader()
        self.data = loader(data_source)
        
    def _get_complementary_entities(self, triple):
        if self.verbose:        
            print("Getting complementary entities")
        entities = self._get_complementary_subjects(triple)
        entities.extend(self._get_complementary_objects(triple))
        return list(set(entities))

    def _get_complementary_subjects(self, triple):
        if self.verbose:        
            print("Getting complementary subjects")
        subjects = self.data[self.data[:,2] == triple[2]]
        subjects = list(set(subjects[subjects[:,1] == triple[1]][:,0]))
        return subjects

    def _get_complementary_objects(self, triple):
        if self.verbose:        
            print("Getting complementary objects")
        objects = self.data[self.data[:,0] == triple[0]]
        objects = list(set(objects[objects[:,1] == triple[1]][:,2]))
        return objects
        
    def _get_batch(self, batch_size, dataset_type="train"):
        length = len(self.data)
        for ind in range(0, length, batch_size):
            yield self.data[ind:min(ind + batch_size, length)]

class GraphDataLoader():
    """Desiderata:
        - batch iterator
        - some other functions graph-specific get complementary entities
          previously called "participating entities"
        - support for various backends and in-memory processing through dependency injection
    """    
    def __init__(self, data_source, batch_size=8, dataset_type="train", backend=None, verbose=False):
        """Initialize persistent/in-memory data storage."""   
        self.dataset_type = dataset_type
        self.data_source = data_source
        self.batch_size = batch_size
        self.identifier = DataSourceIdentifier(self.data_source)       
        if isinstance(backend, type):
            self.backend = backend("database_{}.db".format(datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")), 
                                   verbose=verbose)
            print("Initialized Backend with database at: {}".format(self.backend.db_name))
        elif backend is None:
            self.backend = DummyBackend(self.identifier)
        else:
            self.backend = backend
        
        with self.backend as backend:
            backend._load(self.data_source, dataset_type=self.dataset_type)  
        self.batch_iterator = self.get_batch()
      
    def __iter__(self):
        return self

    def __next__(self):
        with self.backend as backend:
            return self.batch_iterator.__next__()
        
    def get_batch(self):
        """Query data for a next batch."""
        with self.backend as backend:
            return backend._get_batch(self.batch_size, dataset_type=self.dataset_type)
    
    def get_complementary_subjects(self, triple):
        """Get subjects complementary to a triple (?,p,o)."""
        with self.backend as backend:
            return backend._get_complementary_subjects(triple)

    def get_complementary_objects(self, triple):
        """Get objects complementary to a triple (s,p,?)."""        
        with self.backend as backend:
            return backend._get_complementary_objects(triple)        
    
    def get_complementary_entities(self, triple):
        """Get subjects and objects complementary to a triple (?,p,?)."""        
        with self.backend as backend:
            return backend._get_complementary_entities(triple)
