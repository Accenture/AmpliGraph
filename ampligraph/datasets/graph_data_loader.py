# Copyright 2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from ampligraph.datasets.source_identifier import DataSourceIdentifier
from ampligraph.datasets import DataIndexer
from datetime import datetime
import numpy as np


class DummyBackend():
    """Class providing artificial backend, that reads data into memory."""
    def __init__(self, identifier, use_indexer=True, verbose=False):
        """Initialise DummyBackend.

           Parameters
           ----------
           identifier: initialize data source identifier, provides loader. 
           use_indexer: flag to tell whether data should be indexed.
        """
        self.verbose = verbose
        self.identifier = identifier
        self.use_indexer = use_indexer
        
    def __enter__ (self):
        """Context manager enter function. Required by GraphDataLoader."""
        return self
    
    def __exit__ (self, type, value, tb):
        """Context manager exit function. Required by GraphDataLoader."""
        pass
    
    def _load(self, data_source, dataset_type):
        """Loads data into self.data.
           
           Parameters
           ----------
           data_source: file with data.
           dataset_type: kind of data to be loaded (train | test | validation).
        """
        if self.verbose:
            print("Simple in-memory data loading of {} dataset.".format(dataset_type))
        self.data_source = data_source
        if isinstance(self.data_source, np.ndarray):
            if self.use_indexer:
                self.mapper = DataIndexer(self.data_source)
                self.data = self.mapper.get_indexes(self.data_source)
            else:
                self.data = self.data_source
        else:
            loader = self.identifier.fetch_loader()
            if self.use_indexer:
                raw_data = loader(self.data_source)
                self.mapper = DataIndexer(raw_data)
                self.data = self.mapper.get_indexes(raw_data)
            else:
                self.data = loader(self.data_source)
        
    def _get_complementary_entities(self, triple):
        """Get subjects and objects complementary to a triple (?,p,?).
           Returns the participating entities in the relation ?-p-o and s-p-?.

           Parameters
           ----------
           x_triple: nd-array (3,)
               triple (s-p-o) that we are querying.

           Returns
           -------
           entities: list of entities participating in the relations s-p-? and ?-p-o.
       """

        if self.verbose:        
            print("Getting complementary entities")
        entities = self._get_complementary_subjects(triple)
        entities.extend(self._get_complementary_objects(triple))
        return list(set(entities))

    def _get_complementary_subjects(self, triple):
        """Get subjects complementary to a triple (?,p,o).
           For a given triple retrive all triples whith same objects and predicates.

           Parameters
           ----------
           triple: list or array with 3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of subjects.
        """

        if self.verbose:        
            print("Getting complementary subjects")
        subjects = self.data[self.data[:,2] == triple[2]]
        subjects = list(set(subjects[subjects[:,1] == triple[1]][:,0]))
        return subjects

    def _get_complementary_objects(self, triple):
        """Get objects complementary to a triple (s,p,?).
           For a given triple retrive all triples whith same subjects and predicates.

           Parameters
           ----------
           triple: list or array with 3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of objects.
        """
        if self.verbose:        
            print("Getting complementary objects")
        objects = self.data[self.data[:,0] == triple[0]]
        objects = list(set(objects[objects[:,1] == triple[1]][:,2]))
        return objects
        
    def _get_batch(self, batch_size, dataset_type="train"):
        """Get next btch of data (generator).
        
           Parameters
           ----------
           batch_size: size of a batch
           dataset_type: kind of dataset that is needed (train | test | validation).

           Returns
           --------
           ndarray(batch_size, 3)
        """
        length = len(self.data)
        for ind in range(0, length, batch_size):
            yield self.data[ind:min(ind + batch_size, length)]

class GraphDataLoader():
    """Data loader for graphs implemented as a batch iterator
        - graph-specific functions: get complementary entities
          (previously called "participating entities"),
        - support for various backends and in-memory processing through dependency injection

       Example
       -------
       >>># with a SQLite backend
       >>>data = GraphDataLoader("./train.csv", backend=SQLiteAdapter, batch_size=4)
       >>># with no backend
       >>>data = GraphDataLoader("./train.csv", batch_size=4)
       >>>for elem in data:
       >>>    process(data)
    """    
    def __init__(self, data_source, batch_size=1, dataset_type="train", backend=None, root_directory="./", use_indexer=True, verbose=False):
        """Initialise persistent/in-memory data storage.
       
           Parameters
           ----------
           data_source: file with data (e.g. CSV).
           batch_size: size of batch,
           dataset_type: kind of data provided (train | test | validation),
           backend: name of backend class or, already initialised backend, 
                    if None, DummyBackend is used (in-memory processing).
           use_indexer: flag to tell whether data should be indexed.          
        """   
        self.dataset_type = dataset_type
        self.data_source = data_source
        self.batch_size = batch_size
        self.root_directory = root_directory
        self.identifier = DataSourceIdentifier(self.data_source)       
        self.use_indexer = use_indexer
        if isinstance(backend, type):
            self.backend = backend("database_{}.db".format(datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")), 
                                   root_directory=self.root_directory, use_indexer=self.use_indexer, verbose=verbose)
            print("Initialized Backend with database at: {}".format(self.backend.db_path))
        elif backend is None:
            self.backend = DummyBackend(self.identifier, use_indexer=self.use_indexer)
        else:
            self.backend = backend
        
        with self.backend as backend:
            backend._load(self.data_source, dataset_type=self.dataset_type)  
        self.batch_iterator = self.get_batch()
      
    def __iter__(self):
        """Function needed to be used as an itertor."""
        return self

    def __next__(self):
        """Function needed to be used as an itertor."""
        with self.backend as backend:
            return self.batch_iterator.__next__()
        
    def get_batch(self):
        """Query data for a next batch."""
        with self.backend as backend:
            return backend._get_batch(self.batch_size, dataset_type=self.dataset_type)
    
    def get_complementary_subjects(self, triple):
        """Get subjects complementary to a triple (?,p,o).
           For a given triple retrive all triples whith same objects and predicates.

           Parameters
           ----------
           triple: list or array with 3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of subjects.
        """
        with self.backend as backend:
            return backend._get_complementary_subjects(triple)

    def get_complementary_objects(self, triple):
        """Get objects complementary to a triple (s,p,?).
           For a given triple retrive all triples whith same subjects and predicates.

           Parameters
           ----------
           triple: list or array with 3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of objects.
        """
        with self.backend as backend:
            return backend._get_complementary_objects(triple)        
    
    def get_complementary_entities(self, triple):
        """Get subjects and objects complementary to a triple (?,p,?).
           Returns the participating entities in the relation ?-p-o and s-p-?.

           Parameters
           ----------
           x_triple: nd-array (3,)
               triple (s-p-o) that we are querying.

           Returns
           -------
           entities: list of entities participating in the relations s-p-? and ?-p-o.
       """

        with self.backend as backend:
            return backend._get_complementary_entities(triple)
