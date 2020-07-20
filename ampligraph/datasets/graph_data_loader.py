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
    def __init__(self, identifier, use_indexer=True, remap=False, name="main_partition", verbose=False):
        """Initialise DummyBackend.

           Parameters
           ----------
           identifier: initialize data source identifier, provides loader. 
           use_indexer: flag to tell whether data should be indexed.
           remap: flag for partitioner to indicate whether to remap previously 
                  indexed data to (0, <size_of_partition>).
           name: identifying name of files for indexer, partition name/id.
        """
        self.verbose = verbose
        self.identifier = identifier
        self.use_indexer = use_indexer
        self.remap = remap
        self.name = name
        
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
            elif self.remap:
                # create a special mapping for partitions, persistent mapping from main indexes to partition indexes
                self.mapper = DataIndexer(self.data_source, name=self.name, in_memory=False)
                self.data = self.mapper.get_indexes(self.data_source)
            else:
                self.data = self.data_source
        else:
            loader = self.identifier.fetch_loader()
            raw_data = loader(self.data_source)
            if self.use_indexer:
                self.mapper = DataIndexer(raw_data)
                self.data = self.mapper.get_indexes(raw_data)
            elif self.remap:
                # create a special mapping for partitions, persistent mapping from main indexes to partition indexes
                self.mapper = DataIndexer(raw_data, name=self.name, in_memory=False)
                self.data = self.mapper.get_indexes(raw_data)
            else:
                self.data = raw_data

    def _get_triples(self, subjects=None, objects=None, entities=None):
        """Get triples that objects belongs to objects and subjects to subjects,
           or if not provided either object or subjet belongs to entities.
        """
        if subjects is None and objects is None:
            msg = "You have to provide either subjects and objects indexes or general entities indexes!"
            assert(entities is not None), msg 
            subjects = entities
            objects = entities
        #check_subjects = np.vectorize(lambda t: t in subjects)
        check_triples = np.vectorize(lambda t, r: (t in objects and r in subjects) or (t in subjects and r in objects))
        triples = self.data[check_triples(self.data[:,2],self.data[:,0])]
        #triples_from_objects = self.data[check_objects(self.data[:,0])]
        #triples = np.vstack([triples_from_subjects, triples_from_objects])
        return triples 
        
    def _get_complementary_entities(self, triples):
        """Get subjects and objects complementary to a triple (?,p,?).
           Returns the participating entities in the relation ?-p-o and s-p-?.

           Parameters
           ----------
           x_triple: nd-array (3,)
               triple (s-p-o) that we are querying.

           Returns
           -------
           entities: list of entities participating in the relations s-p-? and ?-p-o.
           TODO: What exactly this should return?
       """

        if self.verbose:        
            print("Getting complementary entities")
        subjects = self._get_complementary_subjects(triples)
        objects = self._get_complementary_objects(triples)
        return subjects, objects

    def _get_complementary_subjects(self, triples):
        """Get subjects complementary to triples (?,p,o).
           For a given triple retrive all triples whith same objects and predicates.

           Parameters
           ----------
           triple: list or array with 3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of subjects per triple.
        """

        if self.verbose:        
            print("Getting complementary subjects")
        subjects = []
        print(type(self.data))
        for triple in triples:
            tmp = self.data[self.data[:,2] == triple[2]]
            subjects.append(list(set(tmp[tmp[:,1] == triple[1]][:,0])))
        return subjects

    def _get_complementary_objects(self, triples):
        """Get objects complementary to  triples (s,p,?).
           For a given triple retrive all triples whith same subjects and predicates.

           Parameters
           ----------
           triples: list or array with 3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of objects, per triple
        """
        if self.verbose:        
            print("Getting complementary objects")
        objects = []
        for triple in triples:
            tmp = self.data[self.data[:,0] == triple[0]]
            objects.append(list(set(tmp[tmp[:,1] == triple[1]][:,2])))
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
        triples = range(0, length, batch_size)
        for start_index in triples:
            if start_index + batch_size >= length: # if the last batch is smaller than the batch_size
                batch_size = length - start_index
            yield self.data[start_index:start_index + batch_size]

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
    def __init__(self, data_source, batch_size=1, dataset_type="train", backend=None, root_directory="./", use_indexer=True, verbose=False, remap=False, name="main_partition"):
        """Initialise persistent/in-memory data storage.
       
           Parameters
           ----------
           data_source: file with data (e.g. CSV).
           batch_size: size of batch,
           dataset_type: kind of data provided (train | test | validation),
           backend: name of backend class or, already initialised backend, 
                    if None, DummyBackend is used (in-memory processing).
           use_indexer: flag to tell whether data should be indexed.          
           remap: flag to be used by graph partitioner, indicates whether 
                     previously indexed data in partition has to be remapped to
                     new indexes (0, <size_of_partition>), to not be used with 
                     use_indexer=True, the new remappngs will be persisted.
           name: identifying name/id of partition which data loader represents (default main).
        """   
        self.dataset_type = dataset_type
        self.data_source = data_source
        self.batch_size = batch_size
        self.root_directory = root_directory
        self.identifier = DataSourceIdentifier(self.data_source)       
        self.use_indexer = use_indexer
        self.remap = remap
        self.name = name
        assert bool(use_indexer) == (not remap), "Either remap or Indexer should be speciferd at the same time."
        if isinstance(backend, type):
            self.backend = backend("database_{}.db".format(datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")), 
                                   root_directory=self.root_directory, use_indexer=self.use_indexer, remap=self.remap, name=self.name, verbose=verbose)
            print("Initialized Backend with database at: {}".format(self.backend.db_path))
        elif backend is None:
            self.backend = DummyBackend(self.identifier, use_indexer=self.use_indexer, remap=self.remap, name=self.name)
        else:
            self.backend = backend
        
        with self.backend as backend:
            backend._load(self.data_source, dataset_type=self.dataset_type)  
        self.batch_iterator = self.get_batch()
        self.metadata = self.backend.mapper.metadata
      
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
   
    def get_complementary_subjects(self, triples):
        """Get subjects complementary to triples (?,p,o).
           For a given triple retrive all triples whith same objects and predicates.

           Parameters
           ----------
           triple: list or array with 3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of subjects per triple.
        """
        with self.backend as backend:
            return backend._get_complementary_subjects(triples)

    def get_complementary_objects(self, triples):
        """Get objects complementary to triples (s,p,?).
           For a given triple retrive all triples whith same subjects and predicates.

           Parameters
           ----------
           triple: list or array with 3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of objects per triple.
        """
        with self.backend as backend:
            return backend._get_complementary_objects(triples)        
    
    def get_complementary_entities(self, triples):
        """Get subjects and objects complementary to triples (?,p,?).
           Returns the participating entities in the relation ?-p-o and s-p-?.

           Parameters
           ----------
           x_triple: nd-array (N,3,)
              N triples (s-p-o) that we are querying.

           Returns
           -------
           entities: list of entities participating in the relations s-p-? and ?-p-o per triple.
           TODO: What exactly it should return?
       """

        with self.backend as backend:
            return backend._get_complementary_entities(triples)


    def get_triples(self, subjects=None, objects=None, entities=None):
        with self.backend as backend:
            return backend._get_triples(subjects, objects, entities)

