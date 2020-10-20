# Copyright 2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Data indexer. 

This module provides class that maps raw data to indexes and the other way around.
It can be persisted and contains supporting functions.

Example
-------
    >>>data = np.array([['/m/01',
                      '/relation1',
                      '/m/02'],
                     ['/m/01',
                      '/relation2',
                      '/m/07']])
    >>>mapper = DataIndexer(data, in_memory=True)
    >>>mapper.get_indexes(data)        
    >>>mapper.entities_dict[1]
    '/m/02'

.. It extends functionality of to_idx(...) from  AmpliGraph 1:
   https://docs.ampligraph.org/en/1.3.1/generated/ampligraph.evaluation.to_idx.html?highlight=to_idx

"""
from datetime import datetime
import numpy as np
import os
import shelve
import tensorflow as tf
import pandas as pd
import logging
import tempfile


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataIndexer():
    """Index graph unique entities and relations.

       Can support large datasets by two modes one using dictionary for
       in-memory storage (in_memory=True) and the other using persistent 
       dictionary storage - python shelves, for dumping huge indexes.
       
       Methods:
        - create_mappings - core function that create dictionaries or shelves.
        - get_indexes - given array of triples returns it in an indexed form.
        - update_mappings [NotYetImplemented] - update mappings from a new data.
        
       Properties:
        - data - data to be indexed, either a numpy array or a generator.
        - rels, ents, rev_rels, rev_ents - dictionaries (persistent or in-memory).
        - max_ents_index - maximum index in entities, which is also the number 
          of unique entities - 1.
        - max_rels_index - maximum index in relations dictionary, which is also 
          the number of unique relations - 1.
        - [rev_]ents_length - the length of [reversed] entities.
        - [rev_]rels_length - the length of [reversed] relations.
            
        Example
        -------
        
        >>># In-memory mapping
        >>>data = np.array([['a','b','c'],['c','b','d'],['d','e','f']])
        >>>mapper = DataIndexer(data, in_memory=True)
        >>>mapper.get_indexes(data)
        
        Mappings created with: 4 ents, 4 rev_ents, 2 rels and 2 rev_rels

        <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
        array([[0, 0, 1],
               [1, 0, 2],
               [2, 1, 3]], dtype=int32)>
               
        >>># Persistent mapping
        >>>data = np.array([['a','b','c'],['c','b','d'],['d','e','f']])
        >>>mapper = DataIndexer(data, in_memory=False)
        >>>mapper.get_indexes(data)        
       """
    def __init__(self):
        pass
    def create_mappings(self):
        pass
    def update_mappings(self):
        pass
    def get_indexes(self, X):
        pass
    def get_relations_count(self):
        pass
    def get_entities_count(self):
        pass
    def clean(self):
        pass


class DataIndexerInMemory(DataIndexer):
    def __init__(self, data, entities_dict=None, reversed_entities_dict=None,
                 relations_dict=None, reversed_relations_dict=None, root_directory=tempfile.gettempdir(), name="main_partition"):
        """Initialise DataIndexer by creating mappings.

           Parameters
           ----------
           data: data to be indexed.
           in_memory: flag indicating whether to create a persistent
                      or in-memory dictionary of mappings.
           entities_dict: dictionary or shelve path, storing entities mappings,
                          if not provided will be created from data.
           reversed_entities_dict: dictionary or shelve path, storing reversed entities mappings,
                          if not provided will be created from data.
           relations_dict: dictionary or shelve path, storing relations mappings,
                          if not provided will be created from data.
           reversed_relations_dict: dictionary or shelve path, storing reversed relations mappings,
                          if not provided will be created from data.
           root_directory: directory where to store persistent mappings, not used when in_memory set to True.
        """
        self.data = data
        self.metadata = {}
        self.entities_dict = entities_dict
        self.reversed_entities_dict = reversed_entities_dict
        self.relations_dict = relations_dict
        self.reversed_relations_dict = reversed_relations_dict

        self.root_directory = root_directory
        self.name = name

        self.max_ents_index = -1
        self.max_rels_index = -1
        self.ents_length = 0
        self.rev_ents_length = 0
        self.rels_length = 0
        self.rev_rels_length = 0
        self.create_mappings()

    def create_mappings(self):
        """Create mappings of data into indexes. It creates four dictionaries with
           keys as unique entities/relations and values as indexes and reversed
           version of it. Dispatches to the adequate functions to create persistent
           or in-memory dictionaries.
        """

        if isinstance(self.entities_dict, dict) and\
           isinstance(self.reversed_entities_dict, dict) and\
           isinstance(self.relations_dict, dict) and\
           isinstance(self.reversed_relations_dict, dict):
            self._update_properties()
            logger.debug("The mappings initialised from in-memory dictionaries.")
        elif self.entities_dict is None and\
             self.reversed_entities_dict is None and\
             self.relations_dict is None and\
             self.reversed_relations_dict is None:
            logger.debug("The mappings will be created for data in {}.".format(self.name))

            if isinstance(self.data, np.ndarray):
                self.update_dictionary_mappings()
            else:
                self.update_dictionary_mappings_in_chunks()
        else:
            logger.debug("Provided initialization objects are not supported. Can't Initialise mappings.")
   
    def _update_properties(self):
        """Initialise properties from the in-memory dictionary."""
        self.max_ents_index = self._get_max_ents_index()
        self.max_rels_index = self._get_max_rels_index()

        self.ents_length = len(self.entities_dict)
        self.rev_ents_length = len(self.reversed_entities_dict)
        self.rels_length = len(self.relations_dict)
        self.rev_rels_length = len(self.reversed_relations_dict)

    def _get_max_ents_index(self):
        """Get maximum index from entities dictionary."""
        return max(self.reversed_entities_dict.values())

    def _get_max_rels_index(self):
        """Get maximum index from relations dictionary."""
        return max(self.reversed_relations_dict.values())

    def _get_starting_index_ents(self):
        """Returns next index to continue adding elements to entities dictionary."""
        if not self.entities_dict:
            self.entities_dict = {}
            self.reversed_entities_dict = {}
            return 0
        else:
            return self.max_ents_index + 1

    def _get_starting_index_rels(self):
        """Returns next index to continue adding elements to relations dictionary."""
        if not self.relations_dict:
            self.relations_dict = {}
            self.reversed_relations_dict = {}
            return 0
        else:
            return self.max_rels_index + 1    
    
    def update_dictionary_mappings(self, sample=None):
        """Index entities and relations. Creates shelves for mappings between
           entities and relations to indexes and reverse mapping.

           Remember to use mappings for entities with entities and reltions with relations!
        """
        if sample is None:
            sample = self.data
        #logger.debug(sample)
        i = self._get_starting_index_ents()
        j = self._get_starting_index_rels()

        for d in sample:
            if d[0] not in self.reversed_entities_dict:
                self.reversed_entities_dict[d[0]] = i
                self.entities_dict[i] = d[0]
                i += 1
            if d[2] not in self.reversed_entities_dict:
                self.reversed_entities_dict[d[2]] = i
                self.entities_dict[i] = d[2]
                i += 1
            if d[1] not in self.reversed_relations_dict:
                self.reversed_relations_dict[d[1]] = j
                self.relations_dict[j] = d[1]
                j += 1

        self.max_ents_index = i - 1
        self.max_rels_index = j - 1

        self.ents_length = len(self.entities_dict)
        self.rev_ents_length = len(self.reversed_entities_dict)
        self.rels_length = len(self.relations_dict)
        self.rev_rels_length = len(self.reversed_relations_dict)

        if self.rev_ents_length != self.ents_length:
            msg = "Reversed entities index size not equal to index size ({} and {})".format(self.rev_ents_length, self.ents_length)
            logger.error(msg)
            raise Exception(msg)

        if self.rev_rels_length != self.rels_length:
            msg = "Reversed relations index size not equal to index size ({} and {})".format(self.rev_rels_length, self.rels_length)
            logger.error(msg)
            raise Exception(msg)

        logger.debug("Mappings updated with: {} ents, {} rev_ents, {} rels and {} rev_rels".format(self.ents_length,
                                                                                            self.rev_ents_length,
                                                                                            self.rels_length,
                                                                                            self.rev_rels_length))
    def update_dictionary_mappings_in_chunks(self):
        """Update dictionary mappings chunk by chunk."""
        for chunk in self.data:
            if isinstance(chunk, np.ndarray):
                self.update_dictionary_mappings(chunk)
            else:
                self.update_dictionary_mappings(chunk.values)
                
    def get_indexes(self, sample=None, type_of="t", order="raw2ind"):
        """Converts raw data sample to an indexed form according to
           previously created mappings. Dispatches to the adequate functions
           for persistent or in-memory mappings.

           Parameters
           ----------
           sample: numpy array with raw data, that was previously indexed.
           type_of: type of provided sample, one of the following values: {"t", "e", "r"}, indicates whether provided
                 sample is an array of triples ("t"), list of entities ("e") or list of
                 relations ("r").
           order: raw2ind or ind2raw, should it convert raw data to indexes or indexes to raw data?
           Returns
           -------
           array of same size as sample but with indexes of elements instead
           of elements.
        """
        if type_of not in ["t", "e", "r"]:
            msg = "Type (type_of) should be one of the following: t, e, r, instead got {}".format(type_of)
            logger.error(msg)
            raise Exception(msg)

        if type_of == "t":
            if isinstance(sample, pd.DataFrame):
                return self.get_indexes_from_a_dictionary(sample.values, order=order)
            else:
                return self.get_indexes_from_a_dictionary(sample, order=order)
        else:
            return self.get_indexes_from_a_dictionary_single(sample, type_of=type_of, order=order)

    def get_indexes_from_a_dictionary(self, sample, order="raw2ind"):
        """Get indexed triples from a in-memory dictionary.

           Parameters
           ----------
           sample: numpy array with a fragment of data of size (N,3), where each element is:
                  (subject, predicate, object).
           order: raw2ind or ind2raw, should it convert raw data to indexes or indexes to raw data?

           Returns
           -------
           tmp: numpy array of size (N,3) with indexed triples,
                where each element is: (subject index, predicate index, object index).
        """
        if order == "raw2ind":
            entities = self.reversed_entities_dict
            relations = self.reversed_relations_dict
        elif order == "ind2raw":
            entities = self.entities_dict
            relations = self.relations_dict
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(order)
            logger.error(msg)
            raise Exception(msg)
        if entities is None and relations is None:
            msg = "Requested entities and relation mappings are empty."
            logger.error(msg)
            raise Exception(msg)

        subjects   = np.array([entities[x] for x in sample[:,0]],  dtype=np.int32)
        objects    = np.array([entities[x] for x in sample[:,2]],  dtype=np.int32)
        predicates = np.array([relations[x] for x in sample[:,1]],  dtype=np.int32)
        merged = np.stack([subjects, predicates, objects], axis=1)
        return merged            

    def get_indexes_from_a_dictionary_single(self, sample, type_of="e", order="raw2ind"):
        """Get indexed elements (entities, relatiosn) from an in-memory dictionary.

           Parameters
           ----------
           sample: list of entities or relations to get indexes for.
           type_of: "e" or "r", get indexes for entities ("e") or relations ("r").
           order: raw2ind or ind2raw, should it convert raw data to indexes or indexes to raw data?

           Returns
           -------
           tmp: numpy array of indexes.
                where each element is: (subject index, predicate index, object index).
        """
        if order == "raw2ind":
            entities = self.reversed_entities_dict
            relations = self.reversed_relations_dict
            dtype = np.int32
        elif order == "ind2raw":
            entities = self.entities_dict
            relations = self.relations_dict
            dtype = str
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(order)
            logger.error(msg)
            raise Exception(msg)

        if entities is None and relations is None:
            msg = "Requested entities and relations mappings are empty."
            logger.error(msg)
            raise Exception(msg)

        if type_of == "e":
            elements   = np.array([entities[x] for x in sample],  dtype=dtype)
            return elements
        elif type_of == "r":
            elements = np.array([relations[x] for x in sample],  dtype=dtype)
            return elements
        else:
            if type_of not in ["r", "e"]:
                msg = "No such option, should be r (relations) or e (entities), instead got".format(type_of)
                logger.error(msg)
                raise Exception(msg)    
    
    def get_relations_count(self):
        return len(self.relations_dict)
    
    def get_entities_count(self):
        return len(self.entities_dict)
    
    def clean(self):
        del self.entities_dict
        del self.reversed_entities_dict
        del self.relations_dict
        del self.reversed_relations_dict  


class DataIndexerShelves(DataIndexer):
    def __init__(self, data, entities_dict=None, reversed_entities_dict=None,
                 relations_dict=None, reversed_relations_dict=None, root_directory=tempfile.gettempdir(), name="main_partition"):
        """Initialise DataIndexer by creating mappings.

           Parameters
           ----------
           data: data to be indexed.
           in_memory: flag indicating whether to create a persistent
                      or in-memory dictionary of mappings.
           entities_dict: dictionary or shelve path, storing entities mappings,
                          if not provided will be created from data.
           reversed_entities_dict: dictionary or shelve path, storing reversed entities mappings,
                          if not provided will be created from data.
           relations_dict: dictionary or shelve path, storing relations mappings,
                          if not provided will be created from data.
           reversed_relations_dict: dictionary or shelve path, storing reversed relations mappings,
                          if not provided will be created from data.
           root_directory: directory where to store persistent mappings, not used when in_memory set to True.
        """
        self.data = data
        self.metadata = {}
        self.entities_dict = entities_dict
        self.reversed_entities_dict = reversed_entities_dict
        self.relations_dict = relations_dict
        self.reversed_relations_dict = reversed_relations_dict

        self.root_directory = root_directory
        self.name = name

        self.max_ents_index = -1
        self.max_rels_index = -1
        self.ents_length = 0
        self.rev_ents_length = 0
        self.rels_length = 0
        self.rev_rels_length = 0
        self.create_mappings()
        
    def create_mappings(self):
        """Create mappings of data into indexes. It creates four dictionaries with
           keys as unique entities/relations and values as indexes and reversed
           version of it. Dispatches to the adequate functions to create persistent
           or in-memory dictionaries.
        """

        if isinstance(self.entities_dict, str) and self.shelve_exists(self.entities_dict) and\
             isinstance(self.reversed_entities_dict, str) and self.shelve_exists(self.reversed_entities_dict) and\
             isinstance(self.relations_dict, str) and self.shelve_exists(self.relations_dict) and\
             isinstance(self.reversed_relations_dict, str) and self.shelve_exists(self.reversed_relations_dict):
            self._update_properties()
            logger.debug("The mappings initialised from persistent dictionaries (shelves).")
        elif self.entities_dict is None and\
             self.reversed_entities_dict is None and\
             self.relations_dict is None and\
             self.reversed_relations_dict is None:
            logger.debug("The mappings will be created for data in {}.".format(self.name))

            if isinstance(self.data, np.ndarray):
                self.create_persistent_mappings_from_nparray()
            else:
                self.create_persistent_mappings_in_chunks()
        else:
            logger.debug("Provided initialization objects are not supported. Can't Initialise mappings.")

    def create_persistent_mappings_in_chunks(self):
        """Index entities and relations. Creates shelves for mappings between
           entities and relations to indexes and reverse mapping.

           Four shelves are created in root_directory:
           entities_<NAME>_<DATE>.shf - with map entities -> indexes
           reversed_entities_<NAME>_<DATE>.shf - with map indexes -> entities
        for chunk in self.data:
            self.update_existing_
           relations_<NAME>_<DATE>.shf - with map relations -> indexes
           reversed_relations_<NAME>_<DATE>.shf - with map indexes -> relations

           Remember to use mappings for entities with entities and reltions with relations!
        """
        date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.entities_dict = os.path.join(self.root_directory, "entities_{}_{}.shf".format(self.name, date))
        self.reversed_entities_dict = os.path.join(self.root_directory, "reversed_entities_{}_{}.shf".format(self.name, date))
        self.relations_dict = os.path.join(self.root_directory, "relations_{}_{}.shf".format(self.name, date))
        self.reversed_relations_dict = os.path.join(self.root_directory, "reversed_relations_{}_{}.shf".format(self.name, date))

        for chunk in self.data:
            if isinstance(chunk, pd.DataFrame):
                self.update_shelves(chunk.values, rough=True)
            else:
                self.update_shelves(chunk, rough=True)

        logger.debug("We need to reindex all the data now so the indexes are continous among chunks")
        self.reindex()

        self.files_id = "_{}_{}.shf".format(self.name, date)
        files = ["entities", "reversed_entities", "relations", "reversed_relations"]
        logger.debug("Mappings are created in the following files:\n{}\n{}\n{}\n{}".format(*[x + self.files_id for x in files]))
        self.metadata.update({"entities_shelf": self.entities_dict,
                         "reversed_entities_shelf": self.reversed_entities_dict,
                         "relations":self.relations_dict,
                         "reversed_relations_dict":self.reversed_relations_dict,
                         "name": self.name})            
 
    def reindex(self):
        """Reindex the data to continous values from 0 to <MAX UNIQUE ENTIITES/RELATIONS>.
           This is needed where data is provided in chunks as we don't know the overlap
           between chunks upfront ant indexes are not coninous.
           This guarantees that entities and relations have a continous index.
        """
        logger.debug("starting reindexing...")
        remapped_ents_file = "remapped_ents.shf"
        remapped_rev_ents_file = "remapped_rev_ents.shf"
        remapped_rels_file = "remapped_rels.shf"
        remapped_rev_rels_file = "remapped_rev_rels.shf"
        with shelve.open(self.reversed_entities_dict) as ents:
            with shelve.open(remapped_ents_file, writeback=True) as remapped_ents:
                with shelve.open(remapped_rev_ents_file, writeback=True) as remapped_rev_ents:
                    for i, ent in enumerate(ents):
                        remapped_ents[str(i)] = str(ent)
                        remapped_rev_ents[str(ent)] = str(i)

        with shelve.open(self.reversed_relations_dict) as rels:
            with shelve.open(remapped_rels_file, writeback=True) as remapped_rels:
                with shelve.open(remapped_rev_rels_file, writeback=True) as remapped_rev_rels:
                    for i, rel in enumerate(rels):
                        remapped_rels[str(i)] = str(rel)
                        remapped_rev_rels[str(rel)] = str(i)

        self.move_shelve(remapped_ents_file, self.entities_dict)
        self.move_shelve(remapped_rev_ents_file, self.reversed_entities_dict)
        self.move_shelve(remapped_rels_file, self.relations_dict)
        self.move_shelve(remapped_rev_rels_file, self.reversed_relations_dict)
        logger.debug("reindexing done!")
        self._update_properties()
        logger.debug("properties updated")

    def _update_properties(self, rough=False):
        """Initialise properties from the persistent dictionary (shelve)."""

        with shelve.open(self.entities_dict) as ents:
            self.max_ents_index = int(max(ents.keys(), key=lambda x: int(x)))
            self.ents_length = len(ents)
        with shelve.open(self.relations_dict) as rels:
            self.max_rels_index = int(max(rels.keys(), key=lambda x: int(x)))
            self.rels_length = len(rels)
        with shelve.open(self.reversed_entities_dict) as ents:
            self.rev_ents_length = len(ents)
        with shelve.open(self.reversed_relations_dict) as rels:
            self.rev_rels_length = len(rels)
        if not rough:
            if not self.rev_ents_length == self.ents_length:
                msg = "Reversed entities index size not equal to index size ({} and {})".format(self.rev_ents_length, self.ents_length)
                logger.error(msg)
                raise Exception(msg)
            if not self.rev_rels_length == self.rels_length:
                msg = "Reversed relations index size not equal to index size ({} and {})".format(self.rev_rels_length, self.rels_length)
                logger.error(msg)
                raise Exception(msg)
        else:
            logger.debug("In a rough mode, the sizes may not be equal due to duplicates, it will be fixed in reindexing at the later stage.")
            logger.debug("Reversed entities index size and index size {} and {}".format(self.rev_ents_length, self.ents_length))
            logger.debug("Reversed relations index size and index size: {} and {}".format(self.rev_rels_length, self.rels_length))            

    def create_persistent_mappings_from_nparray(self):
        """Index entities and relations from the array.
           Creates shelves for mappings between entities
           and relations to indexes and reverse mappings.

           Four shelves are created in root_directory:
           entities_<NAME>_<DATE>.shf - with map entities -> indexes
           reversed_entities_<NAME>_<DATE>.shf - with map indexes -> entities
           relations_<NAME>_<DATE>.shf - with map relations -> indexes
           reversed_relations_<NAME>_<DATE>.shf - with map indexes -> relations

           Remember to use mappings for entities with entities and reltions with relations!
        """

        date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.entities_dict = os.path.join(self.root_directory, "entities_{}_{}.shf".format(self.name, date))
        self.reversed_entities_dict = os.path.join(self.root_directory, "reversed_entities_{}_{}.shf".format(self.name, date))
        self.relations_dict = os.path.join(self.root_directory, "relations_{}_{}.shf".format(self.name, date))
        self.reversed_relations_dict = os.path.join(self.root_directory, "reversed_relations_{}_{}.shf".format(self.name, date))
        self.files_id = "_{}_{}.shf".format(self.name, date)
        files = ["entities", "reversed_entities", "relations", "reversed_relations"]
        logger.debug("Mappings are created in the following files:\n{}\n{}\n{}\n{}".format(*[x + self.files_id for x in files]))
        self.metadata.update({"entities_shelf": self.entities_dict,
                         "reversed_entities_shelf": self.reversed_entities_dict,
                         "relations":self.relations_dict,
                         "reversed_relations_dict":self.reversed_relations_dict,
                         "name": self.name})
        self.update_shelves()            

    def update_shelves(self, sample=None, rough=False):
        """Update shelves with sample or full data when sample not provided."""
        if sample is None:
            sample = self.data
            if self.shelve_exists(self.entities_dict) or self.shelve_exists(self.reversed_entities_dict) or\
               self.shelve_exists(self.relations_dict) or self.shelve_exists(self.reversed_relations_dict):
                msg =  "Shelves exists for some reason and are not empty!"
                logger.error(msg)
                raise Exception(msg)

        logger.debug("Sample: {}".format(sample))
        entities = set(sample[:,0]).union(set(sample[:,2]))
        predicates = set(sample[:,1])

        start_ents = self._get_starting_index_ents()
        logger.debug("Start index entities: {}".format(start_ents))
        new_indexes_ents = range(start_ents, start_ents + len(entities)) # maximum new index, usually less when multiple chunks provided due to chunks
        if not len(new_indexes_ents) == len(entities):
            msg = "Etimated indexes length for entities not equal to entities length ({} and {})".format(len(new_indexes_ents), len(entities))
            logger.error(msg)
            raise Exception(msg)

        start_rels = self._get_starting_index_rels()
        new_indexes_rels = range(start_rels, start_rels + len(predicates))
        logger.debug("Starts index relations: {}".format(start_rels))
        if not len(new_indexes_rels) == len(predicates):
            msg = "Estimated indexes length for relations not equal to relations length ({} and {})".format(len(new_indexes_rels), len(predicates))
            logger.error(msg)
            raise Exception(msg)
        #print("new indexes rels: ", new_indexes_rels)
        logger.debug("index rels size: {} and rels size: {}".format(len(new_indexes_rels), len(predicates)))
        logger.debug("index ents size: {} and entss size: {}".format(len(new_indexes_ents), len(entities)))

        with shelve.open(self.entities_dict, writeback=True) as ents:
            with shelve.open(self.reversed_entities_dict, writeback=True) as reverse_ents:
                with shelve.open(self.relations_dict, writeback=True) as rels:
                    with shelve.open(self.reversed_relations_dict, writeback=True) as reverse_rels:
                        reverse_ents.update({str(value):str(key) for key, value in zip(new_indexes_ents, entities)})
                        ents.update({str(key):str(value) for key, value in zip(new_indexes_ents, entities)})
                        reverse_rels.update({str(value):str(key) for key, value in zip(new_indexes_rels, predicates)})
                        rels.update({str(key):str(value) for key, value in zip(new_indexes_rels, predicates)})
        self._update_properties(rough=rough)        

    def shelve_exists(self, name):
        """Check if shelve with a given name exists."""
        if not os.path.isfile(name + ".bak"):
            return False
        if not os.path.isfile(name + ".dat"):
            return False
        if not os.path.isfile(name + ".dir"):
            return False
        return True

    def remove_shelve(self, name):
        """Remove shelve with a given name."""
        os.remove(name + ".bak")
        os.remove(name + ".dat")
        os.remove(name + ".dir")        

    def move_shelve(self, source, destination):
        """Move shelve to a different files."""
        os.rename(source + ".dir", destination + ".dir")
        os.rename(source + ".dat", destination + ".dat")
        os.rename(source + ".bak", destination + ".bak")        
        
    def _get_starting_index_ents(self):
        """Returns next index to continue adding elements to entities dictionary."""
        if not self.entities_dict:
            self.entities_dict = {}
            self.reversed_entities_dict = {}
            return 0
        else:
            return self.max_ents_index + 1

    def _get_starting_index_rels(self):
        """Returns next index to continue adding elements to relations dictionary."""
        if not self.relations_dict:
            self.relations_dict = {}
            self.reversed_relations_dict = {}
            return 0
        else:
            return self.max_rels_index + 1        
        
    def update_mappings(self):
        raise NotImplementedError("Use internal function: update_shelves to do the update.")
    
    def get_indexes(self, sample=None, type_of="t", order="raw2ind"):
        """Converts raw data sample to an indexed form according to
           previously created mappings. Dispatches to the adequate functions
           for persistent or in-memory mappings.

           Parameters
           ----------
           sample: numpy array with raw data, that was previously indexed.
           type_of: type of provided sample, one of the following values: {"t", "e", "r"}, indicates whether provided
                 sample is an array of triples ("t"), list of entities ("e") or list of
                 relations ("r").
           order: raw2ind or ind2raw, should it convert raw data to indexes or indexes to raw data?
           Returns
           -------
           array of same size as sample but with indexes of elements instead
           of elements.
        """
        if type_of not in ["t", "e", "r"]:
            msg = "Type (type_of) should be one of the following: t, e, r, instead got {}".format(type_of)
            logger.error(msg)
            raise Exception(msg)

        if type_of == "t":
            return self.get_indexes_from_shelves(sample, order=order)
        else:
            return self.get_indexes_from_shelves_single(sample, type_of=type_of, order=order)

    def get_indexes_from_shelves(self, sample, order="raw2ind"):
        """Get indexed triples.

           Parameters
           ----------
           sample: numpy array with a fragment of data of size (N,3), where each element is:
                  (subject, predicate, object).
           order: raw2ind or ind2raw, should it convert raw data to indexes or indexes to raw data?

           Returns
           -------
           tmp: numpy array of size (N,3) with indexed triples,
                where each element is: (subject index, predicate index, object index).
           """
        if isinstance(sample, pd.DataFrame):
            sample = sample.values
        #logger.debug(sample)
        if order == "raw2ind":
            entities = self.reversed_entities_dict
            relations = self.reversed_relations_dict
            dtype = int
        elif order == "ind2raw":
            entities = self.entities_dict
            relations = self.relations_dict
            dtype = str
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(order)
            logger.error(msg)
            raise Exception(msg)

        with shelve.open(entities) as ents:
            with shelve.open(relations) as rels:
                subjects = [ents[str(elem)] for elem in sample[:,0]]
                objects = [ents[str(elem)] for elem in sample[:,2]]
                predicates = [rels[str(elem)] for elem in sample[:,1]]
                return np.array((subjects, predicates, objects), dtype=dtype).T        

    def get_indexes_from_shelves_single(self, sample, type_of="e", order="raw2ind"):
        """Get indexed elements (entities or relations).

           Parameters
           ----------
           sample: list of entities or relations to get indexes for.
           type_of: "e" or "r", get indexes for entities ("e") or relations ("r").
           order: raw2ind or ind2raw, should it convert raw data to indexes or indexes to raw data?

           Returns
           -------
           tmp: numpy array of indexes
           """
        if order == "raw2ind":
            entities = self.reversed_entities_dict
            relations = self.reversed_relations_dict
            dtype = int
        elif order == "ind2raw":
            entities = self.entities_dict
            relations = self.relations_dict
            dtype = str
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(order)
            logger.error(msg)
            raise Exception(msg)

        if type_of == "e":
            with shelve.open(entities) as ents:
                elements = [ents[str(elem)] for elem in sample]
            return np.array(elements, dtype=dtype)
        elif type_of == "r":
            with shelve.open(relations) as rels:
                elements = [rels[str(elem)] for elem in sample]
            return np.array(elements, dtype=dtype)
        else:
            if not type_of in ["r", "e"]:
                msg = "No such option, should be r (relations) or e (entities), instead got".format(type_of)
                logger.error(msg)
                raise Exception(msg)            
            
    def get_relations_count(self):
        return self.rels_length
    def get_entities_count(self):
        return self.ents_length
    
    def clean(self):
        self.remove_shelve(self.entities_dict)
        self.remove_shelve(self.reversed_entities_dict)
        self.remove_shelve(self.relations_dict)
        self.remove_shelve(self.reversed_relations_dict)


class DataIndexerSQLite(DataIndexer):
    def __init__(self, data, in_memory=True, db_file='entities_relations.db', root_directory=tempfile.gettempdir(), name="main_partition"):
        """Initialise DataIndexer by creating mappings.

           Parameters
           ----------
           data: data to be indexed.
           root_directory: directory where to store persistent mappings.
        """
        self.data = data
        self.metadata = {}
        self.db_file = os.path.join(root_directory, db_file)
        print(self.db_file)
        self.root_directory = root_directory
        self.name = name

        self.max_ents_index = -1
        self.max_rels_index = -1
        self.ents_length = 0
        self.rels_length = 0
        self.create_mappings()
        
    def create_mappings(self):
        """Creates mappings."""
        if isinstance(self.data, np.ndarray):
            self.create_persistent_mappings_from_nparray()
        else:
            self.create_persistent_mappings_in_chunks()      
        
    def update_db(self, sample=None):
        """Update db with sample or full data when sample not provided."""
        if sample is None:
            sample = self.data
        
        subjects = sample[:,0]
        objects = sample[:,2]
        relations = sample[:,1]
        entities = np.concatenate((subjects, objects))

        data = {'entities': entities, 'relations': relations}
        for table, elems in data.items():
            sql_create_table = """ CREATE TABLE IF NOT EXISTS tmp_{} (
                                                name text PRIMARY KEY
                                            );""".format(table)

            with sqlite3.connect(self.db_file) as conn:
                c = conn.cursor()
                c.execute(sql_create_table)
                conn.commit()

            tab = 'tmp_{}'
            values_placeholder = "({})".format(", ".join(["?"]*1))
            query = 'INSERT OR IGNORE INTO {} VALUES {};'.format(tab.format(table), values_placeholder)
            with sqlite3.connect(self.db_file) as conn:
                c = conn.cursor()
                c.executemany(query, [(v,) if isinstance(v, int) or isinstance(v, str) else v for v in elems])
                conn.commit()        

    def create_persistent_mappings_from_nparray(self):
        """Index entities and relations. Creates sqlite db for mappings between
           entities and relations to indexes."""
            self.update_db()
            self.index_data('entities')
            self.index_data('relations')

    def index_data(self, table):    
        """Create new table with persisted id of elements."""
        query = ["CREATE TABLE IF NOT EXISTS {0}(id INTEGER PRIMARY KEY, name TEXT NOT NULL);".format(table),
        "INSERT INTO {0}(id, name) SELECT rowid, name FROM tmp_{0};".format(table),
        "DROP TABLE tmp_{0};".format(table)]
        with sqlite3.connect(self.db_file) as conn:
            c = conn.cursor()
            for q in query:
                c.execute(q)
                conn.commit()

    def create_persistent_mappings_in_chunks(self):
        """Index entities and relations. Creates sqlite db for mappings between
           entities and relations to indexes in chunks.
        """
        for chunk in self.data:
            if isinstance(chunk, pd.DataFrame):
                self.update_db(sample=chunk.values)
            else:
                self.update_db(chunk)


        logger.debug("We need to reindex all the data now so the indexes are continous among chunks")
        self.index_data('entities')
        self.index_data('relations')
                    
    def update_mappings(self):
        raise NotImplementedError("Use internal function: update_db to do the update.")

    def get_indexes(self, sample=None, type_of="t", order="raw2ind"):
        """Converts raw data sample to an indexed form according to
           previously created mappings.

           Parameters
           ----------
           sample: numpy array with raw data, that was previously indexed.
           type_of: type of provided sample, one of the following values: {"t", "e", "r"}, indicates whether provided
                 sample is an array of triples ("t"), list of entities ("e") or list of
                 relations ("r").
           order: raw2ind or ind2raw, should it convert raw data to indexes or indexes to raw data?
           Returns
           -------
           array of same size as sample but with indexes of elements instead
           of elements.
        """
        if type_of not in ["t", "e", "r"]:
            msg = "Type (type_of) should be one of the following: t, e, r, instead got {}".format(type_of)
            logger.error(msg)
            raise Exception(msg)

        if type_of == "t":
            return self.get_indexes_from_db(sample, order=order)
        else:
            return self.get_indexes_from_db_single(sample, type_of=type_of, order=order)
        
    def get_indexes_from_db(self, sample, order="raw2ind"):
        """Get indexed triples.

           Parameters
           ----------
           sample: numpy array with a fragment of data of size (N,3), where each element is:
                  (subject, predicate, object).
           order: raw2ind or ind2raw, should it convert raw data to indexes or indexes to raw data?

           Returns
           -------
           tmp: numpy array of size (N,3) with indexed triples,
                where each element is: (subject index, predicate index, object index).
           """
        if isinstance(sample, pd.DataFrame):
            sample = sample.values

        subjects = self.get_indexes_from_db_single(sample[:,0], type_of='e', order=order)
        objects = self.get_indexes_from_db_single(sample[:,2], type_of='e', order=order)
        predicates = self.get_indexes_from_db_single(sample[:,1], type_of='r', order=order)        
        if order == "raw2ind":
            dtype = int
        elif order == "ind2raw":
            dtype = str
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(order)
            logger.error(msg)
            raise Exception(msg)           
        
        return np.array((subjects, predicates, objects), dtype=dtype).T        

    def get_indexes_from_db_single(self, sample, type_of="e", order="raw2ind"):
        """Get indexed elements (entities or relations).

           Parameters
           ----------
           sample: list of entities or relations to get indexes for.
           type_of: "e" or "r", get indexes for entities ("e") or relations ("r").
           order: raw2ind or ind2raw, should it convert raw data to indexes or indexes to raw data?

           Returns
           -------
           tmp: numpy array of indexes
           """
        if type_of == "e":
            table = "entities"
        elif type_of == 'r':
            table = 'relations'
        else:
            msg = "No such option, should be r (relations) or e (entities), instead got".format(type_of)
            logger.error(msg)
            raise Exception(msg)

        if order == 'raw2ind':
            query = "select name, id from {0} where name in ({1});".format(table,','.join('"{}"'.format(v) for v in sample))
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor() 
                output = None
                try:
                    cursor.execute(query)
                    output = dict(cursor.fetchall())
                    conn.commit()
                    return [output[x] for x in sample]
                except Exception as e:
                    logger.debug("Query failed. The error '{}' occurred".format(e))  
                    return []
        elif order == 'ind2raw':
            query = "select * from {0} where id in ({1});".format(table, ','.join('"{}"'.format(v) for v in sample))
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor() 
                output = None
                try:
                    cursor.execute(query)
                    output = dict(cursor.fetchall())
                    conn.commit()
                    return [output[x] for x in sample]
                except Exception as e:
                    logger.debug("Query failed. The error '{}' occurred".format(e))  
                    return []    
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(order)
            logger.error(msg)
            raise Exception(msg)
      
    def get_count(self, table, condition):
        """Return number of unique elements in a table according to condition."""
        query = "SELECT count(*) from {} {};".format(table, condition)
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor() 
            count = None
            try:
                cursor.execute(query)
                count = cursor.fetchall()
                conn.commit()
            except Exception as e:
                logger.debug("Query failed. The error '{}' occurred".format(e))             

        if count is None:
            logger.debug("Table is empty or not such table exists.")
            return count
        elif not isinstance(count, list) or not isinstance(count[0], tuple):
            raise ValueError("Cannot get count for the table with provided condition.")        
        return count[0][0]   
    
    def get_relations_count(self):
        """Return number of unique relations."""
        return self.get_count('relations', condition)
    
    def get_entities_count(self, condition=""):
        """Return number of unique entities."""
        return self.get_count('entities', condition)
    
    def clean(self):
        """Remove the database file."""
        os.remove(self.db_file)        
        logger.debug("Indexer Database removed.")
