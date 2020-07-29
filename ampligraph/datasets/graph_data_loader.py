# Copyright 2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Data loader for graphs (big and small).

This module provides GraphDataLoader class that can be parametrized with a backend
and in-memory backend (DummyBackend).
"""
from ampligraph.datasets.source_identifier import DataSourceIdentifier
from ampligraph.datasets import DataIndexer
from datetime import datetime
import numpy as np
import shelve


class DummyBackend():
    """Class providing artificial backend, that reads data into memory."""
    def __init__(self, identifier, use_indexer=True, remap=False, name="main_partition", verbose=False, parent=None, in_memory=True):
        """Initialise DummyBackend.

           Parameters
           ----------
           identifier: initialize data source identifier, provides loader. 
           use_indexer: flag to tell whether data should be indexed.
           remap: flag for partitioner to indicate whether to remap previously 
                  indexed data to (0, <size_of_partition>).
           parent: parent data loader that persists data.
           name: identifying name of files for indexer, partition name/id.
        """
        self.verbose = verbose
        self.identifier = identifier
        self.use_indexer = use_indexer
        self.remap = remap
        self.name = name
        self.parent = parent
        self.in_memory = in_memory
        
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
        self.dataset_type = dataset_type
        if isinstance(self.data_source, np.ndarray):
            if self.use_indexer:
                self.mapper = DataIndexer(self.data_source, in_memory=self.in_memory)
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
                self.mapper = DataIndexer(raw_data, in_memory=self.in_memory)
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
        triples = np.append(triples, np.array(len(triples)*[self.dataset_type]).reshape(-1,1), axis=1)
        #triples_from_objects = self.data[check_objects(self.data[:,0])]
        #triples = np.vstack([triples_from_subjects, triples_from_objects])
        return triples 
        
    def get_data_size(self):
        """Returns number of triples."""
        return np.shape(self.data)[0]

    def _get_complementary_entities(self, triples):
        """Get subjects and objects complementary to a triple (?,p,?).
           Returns the participating entities in the relation ?-p-o and s-p-?.
           Function used duriing evaluation.

           WARNING: If the parent is set the triples returened are coming with parent indexing.
           Parameters
           ----------
           x_triple: nd-array (N,3,) of N
               triples (s-p-o) that we are querying.

           Returns
           -------
           entities: two lists, of subjects and objects participating in the relations s-p-? and ?-p-o.
       """

        if self.verbose:        
            print("Getting complementary entities")

        if self.parent is not None:
            print("Parent is set, WARNING: The triples returened are coming with parent indexing.")

            print("Recover original indexes.")
            with shelve.open(self.mapper.entities_dict) as ents:
                with shelve.open(self.mapper.relations_dict) as rels:
                    triples_original_index = np.array([(ents[str(xx[0])], rels[str(xx[1])], ents[str(xx[2])]) for xx in triples], dtype=np.int32)    
            print("Query parent for data.")
            print("Original index: ",triples_original_index)
            subjects = self.parent.get_complementary_subjects(triples_original_index)
            objects = self.parent.get_complementary_objects(triples_original_index)
            print("What to do with this new indexes? Evaluation should happen in the original space, shouldn't it? I'm assuming it does so returning in parent indexing.")
            return subjects, objects
        else:
            subjects = self._get_complementary_subjects(triples)
            objects = self._get_complementary_objects(triples)
        return subjects, objects

    def _get_complementary_subjects(self, triples):
        """Get subjects complementary to triples (?,p,o).
           For a given triple retrive all triples whith same objects and predicates.
           Function used duriing evaluation.

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
        if self.parent is not None:
            print("Parent is set, WARNING: The triples returened are coming with parent indexing.")

            print("Recover original indexes.")
            with shelve.open(self.mapper.reversed_entities_dict) as ents:
                with shelve.open(self.mapper.reversed_relations_dict) as rels:
                    triples_original_index = np.array([(ents[str(xx[0])], rels[str(xx[1])], ents[str(xx[2])]) for xx in triples], dtype=np.int32)    
            print("Query parent for data.")
            subjects = self.parent.get_complementary_subjects(triples_original_index)
            print("What to do with this new indexes? Evaluation should happen in the original space, shouldn't it? I'm assuming it does so returning in parent indexing.")
            return subjects
        else:
            for triple in triples:
                tmp = self.data[self.data[:,2] == triple[2]]
                subjects.append(list(set(tmp[tmp[:,1] == triple[1]][:,0])))
        return subjects

    def _get_complementary_objects(self, triples):
        """Get objects complementary to  triples (s,p,?).
           For a given triple retrive all triples whith same subjects and predicates.
           Function used duriing evaluation.

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
        if self.parent is not None:
            print("Parent is set, WARNING: The triples returened are coming with parent indexing.")

            print("Recover original indexes.")
            with shelve.open(self.mapper.reversed_entities_dict) as ents:
                with shelve.open(self.mapper.reversed_relations_dict) as rels:
                    triples_original_index = np.array([(ents[str(xx[0])], rels[str(xx[1])], ents[str(xx[2])]) for xx in triples], dtype=np.int32)    
            print("Query parent for data.")
            objects = self.parent.get_complementary_objects(triples_original_index)
            print("What to do with this new indexes? Evaluation should happen in the original space, shouldn't it? I'm assuming it does so returning in parent indexing.")
            return objects
        else:
            for triple in triples:
                tmp = self.data[self.data[:,0] == triple[0]]
                objects.append(list(set(tmp[tmp[:,1] == triple[1]][:,2])))
        return objects


    def _intersect(self, dataloader):
        """Intersect between data and dataloader elements.
           Works only when dataloader is of type
           DummyBackend.
        """
        assert(isinstance(dataloader.backend, DummyBackend)), "Intersection can only be calculated between same backends (DummyBackend), instead get {}".format(type(dataloader.backend))
        return np.intersect1d(self.data, dataloader.backend.data)
        
    def _get_batch_generator(self, batch_size, dataset_type="train", random=False, index_by=""):
        """Batch generator of data.
        
           Parameters
           ----------
           batch_size: size of a batch
           dataset_type: kind of dataset that is needed (train | test | validation).
           random: not implemented.
           index_by: not implemented.

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

    def _clean(self):
        del self.data
        self.mapper.clean()

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
    def __init__(self, data_source, batch_size=1, dataset_type="train", backend=None, root_directory="./", use_indexer=True, verbose=False, remap=False, name="main_partition", parent=None, in_memory=True):
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
           in_memory: persist indexes or not.
           name: identifying name/id of partition which data loader represents (default main).
        """   
        self.dataset_type = dataset_type
        self.data_source = data_source
        self.batch_size = batch_size
        self.root_directory = root_directory
        self.identifier = DataSourceIdentifier(self.data_source)       
        self.use_indexer = use_indexer
        self.remap = remap
        self.in_memory = in_memory
        self.name = name
        self.parent = parent
        assert bool(use_indexer) == (not remap), "Either remap or Indexer should be speciferd at the same time."
        if isinstance(backend, type) and backend != DummyBackend:
            self.backend = backend("database_{}.db".format(datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")), 
                                   root_directory=self.root_directory, use_indexer=self.use_indexer, remap=self.remap, name=self.name, parent=self.parent, in_memory=self.in_memory, verbose=verbose)
            print("Initialized Backend with database at: {}".format(self.backend.db_path))
        elif backend is None or backend == DummyBackend:
            self.backend = DummyBackend(self.identifier, use_indexer=self.use_indexer, remap=self.remap, name=self.name, parent=self.parent, in_memory=self.in_memory)
        else:
            self.backend = backend
        
        #with self.backend as backend:
        self.backend._load(self.data_source, dataset_type=self.dataset_type)  
        self.batch_iterator = self.get_batch_generator()
        self.metadata = self.backend.mapper.metadata
      
    def __iter__(self):
        """Function needed to be used as an itertor."""
        return self

    def __next__(self):
        """Function needed to be used as an itertor."""
        with self.backend as backend:
            return self.batch_iterator.__next__()
      
    def reload(self):
        """Reinstantiate batch iterator."""
        self.batch_iterator = self.get_batch_generator()
  
    def get_batch_generator(self):
        """Get batch generator from the backend."""
        return self.backend._get_batch_generator(self.batch_size, dataset_type=self.dataset_type)
  
    def get_data_size(self):
        """Returns number of triples."""
        with self.backend as backend:
            return backend.get_data_size()
 
    def intersect(self, dataloader):
        """Returns intersection between current dataloader elements and another one (argument).

           Parameters
           ----------
           dataloader: dataloader for which to calculate the intersection for.

           Returns
           -------
           intersection: np.array of intersecting elements.
        """

        with self.backend as backend:
            return backend._intersect(dataloader)

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
        """Get triples that subject is in subjects and object is in objects, or
           triples that eiter subject or object is in entities.

           Parameters
           ----------
           subjects: list of entities that triples subject should belong to.
           objects: list of entities that triples object should belong to.
           entities: list of entities that triples subject and object should belong to.
        
           Returns
           -------
           triples: list of triples constrained by subjects and objects.
          
        """
        with self.backend as backend:
            return backend._get_triples(subjects, objects, entities)

    def clean(self):
        self.backend._clean()

