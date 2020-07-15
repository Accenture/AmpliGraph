# Copyright 2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from datetime import datetime
import numpy as np
import os
import shelve
import tensorflow as tf
import pandas as pd

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

    def __init__(self, data, in_memory=True, entities_dict=None, reversed_entities_dict=None, 
                 relations_dict=None, reversed_relations_dict=None, root_directory="./"):
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
        self.in_memory = in_memory
        
        self.entities_dict = entities_dict
        self.reversed_entities_dict = reversed_entities_dict
        self.relations_dict = relations_dict
        self.reversed_relations_dict = reversed_relations_dict
        
        self.root_directory = root_directory
        
        self.max_ents_index = -1
        self.max_rels_index = -1  
        self.ents_length = -1
        self.rev_ents_length = -1
        self.rels_length = -1
        self.rev_rels_length = -1         
        self.create_mappings()
    
    def get_max_ents_index(self):
        """Get maximum index from entities dictionary."""        
        return max(self.entities_dict.values())

    def get_max_rels_index(self):
        """Get maximum index from relations dictionary."""
        return max(self.relations_dict.values())

    def get_entities_in_batches(self, batch_size=-1):
        entities = range(0, self.ents_length, batch_size)
        for start_index in entities:
            if start_index + batch_size >= self.ents_length:
                batch_size = self.ents_length - start_index
            yield np.array(range(start_index, start_index + batch_size))
    
    def update_properties_dictionary(self):
        """Initialise properties from the in-memory dictionary."""
        self.max_ents_index = self.get_max_ents_index()
        self.max_rels_index = self.get_max_rels_index()

        self.ents_length = len(self.entities_dict)
        self.rev_ents_length = len(self.reversed_entities_dict)   
        self.rels_length = len(self.relations_dict)
        self.rev_rels_length = len(self.reversed_relations_dict) 
        
        
    def update_properties_persistent(self):
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
        assert self.rev_ents_length == self.ents_length, "Reversed entities index size not equal to index size"
        assert self.rev_rels_length == self.rels_length , "Reversed relations index size not equal to index size"        
       
    def shelve_exists(self, name):
        print(name)
        """Check if shelve with a given name exists."""
        if not os.path.isfile(name + ".bak"):
            return False
        if not os.path.isfile(name + ".dat"):
            return False
        if not os.path.isfile(name + ".dir"):
            return False
        return True

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
            self.update_properties_dictionary()
            print("The mappings initialised from in-memory dictionaries.")
            in_memory = True
        elif isinstance(self.entities_dict, str) and self.shelve_exists(self.entities_dict) and\
             isinstance(self.reversed_entities_dict, str) and self.shelve_exists(self.reversed_entities_dict) and\
             isinstance(self.relations_dict, str) and self.shelve_exists(self.relations_dict) and\
             isinstance(self.reversed_relations_dict, str) and self.shelve_exists(self.reversed_relations_dict):
            self.update_properties_persistent()            
            print("The mappings initialised from persistent dictionaries (shelves).")
            in_memory = False
        elif self.entities_dict is None and\
             self.reversed_entities_dict is None and\
             self.relations_dict is None and\
             self.reversed_relations_dict is None:
            print("The mappings will be created.")
            
            if self.in_memory:
                if isinstance(self.data, np.ndarray):
                    self.update_dictionary_mappings()
                else:
                    self.update_dictionary_mappings_in_chunks() 
            else:
                if isinstance(self.data, np.ndarray):
                    self.create_persistent_mappings_from_nparray()
                else:
                    self.create_persistent_mappings_in_chunks()       
        else:
            print("Provided initialization objects are not supported. Can't Initialise mappings.")
    
    def get_indexes(self, sample):
        """Converts raw data sample to an indexed form according to 
           previously created mappings. Dispatches to the adequate functions 
           for persistent or in-memory mappings.
        
           Parameters
           ----------
           sample: numpy array with raw data, that was previously indexed.
           
           Returns
           -------
           array of same size as sample but with indexes of elements instead 
           of elements.
        """
        if self.in_memory:
            if isinstance(sample, pd.DataFrame):
                return self.get_indexes_from_a_dictionary(sample.values)
            else:
                return self.get_indexes_from_a_dictionary(sample)
        return self.get_indexes_from_shelves(sample)

    def create_persistent_mappings_from_nparray(self):
        """Index entities and relations from the array. 
           Creates shelves for mappings between entities 
           and relations to indexes and reverse mappings.

           Four shelves are created in root_directory:
           entities_shelf_<DATE>.shf - with map entities -> indexes
           reversed_entities_shelf_<DATE>.shf - with map indexes -> entities
           relations_shelf_<DATE>.shf - with map relations -> indexes
           reversed_relations_shelf_<DATE>.shf - with map indexes -> relations

           Remember to use mappings for entities with entities and reltions with relations!
        """
        
        date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.entities_dict = os.path.join(self.root_directory, "entities_shelf_{}.shf".format(date))
        self.reversed_entities_dict = os.path.join(self.root_directory, "reversed_entities_shelf_{}.shf".format(date))
        self.relations_dict = os.path.join(self.root_directory, "relations_shelf_{}.shf".format(date))
        self.reversed_relations_dict = os.path.join(self.root_directory, "reversed_relations_shelf_{}.shf".format(date))
        self.update_shelves()
        
    def update_shelves(self, sample=None):
        """Update shelves with sample or full data when sample not provided."""
        if sample is None:
            sample = self.data
        start_ents = self.get_starting_index_ents() # deleted + 1
        new_indexes_ents = range(start_ents, start_ents + len(sample))
        start_rels = self.get_starting_index_rels() # deleted + 1
        new_indexes_rels = range(start_rels, start_rels + len(sample))

        with shelve.open(self.entities_dict, writeback=True) as ents:
            with shelve.open(self.reversed_entities_dict, writeback=True) as reverse_ents:
                with shelve.open(self.relations_dict, writeback=True) as rels:
                    with shelve.open(self.reversed_relations_dict, writeback=True) as reverse_rels:
                        entities = set(sample[:,0]).union(set(sample[:,2]))
                        predicates = set(sample[:,1])
                        reverse_ents.update({str(value):str(key) for key, value in zip(new_indexes_ents, entities)})
                        ents.update({str(key):str(value) for key, value in zip(new_indexes_ents, entities)})
                        reverse_rels.update({str(value):str(key) for key, value in zip(new_indexes_rels, predicates)}) 
                        rels.update({str(key):str(value) for key, value in zip(new_indexes_rels, predicates)}) 
        self.update_properties_persistent()
    
    def update_existing_mappings(self, new_data):
        """Update existing mappings with new data."""
        if self.in_memory:
            self.update_dictionary_mappings(new_data)
        else:
            self.update_shelves(new_data)
    
    def create_persistent_mappings_in_chunks(self):
        """Index entities and relations. Creates shelves for mappings between
           entities and relations to indexes and reverse mapping.

           Four shelves are created in root_directory:
           entities_shelf_<DATE>.shf - with map entities -> indexes
           reversed_entities_shelf_<DATE>.shf - with map indexes -> entities
           relations_shelf_<DATE>.shf - with map relations -> indexes
           reversed_relations_shelf_<DATE>.shf - with map indexes -> relations

           Remember to use mappings for entities with entities and reltions with relations!
        """
        date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.entities_dict = os.path.join(self.root_directory, "entities_shelf_{}.shf".format(date))
        self.reversed_entities_dict = os.path.join(self.root_directory, "reversed_entities_shelf_{}.shf".format(date))
        self.relations_dict = os.path.join(self.root_directory, "relations_shelf_{}.shf".format(date))
        self.reversed_relations_dict = os.path.join(self.root_directory, "reversed_relations_shelf_{}.shf".format(date))
        with shelve.open(self.entities_dict, writeback=True) as ents:
            with shelve.open(self.reversed_entities_dict, writeback=True) as reverse_ents:
                with shelve.open(self.relations_dict, writeback=True) as rels:
                    with shelve.open(self.reversed_relations_dict, writeback=True) as reverse_rels:
                        for i, chunk in enumerate(self.data):
                            entities = set(chunk.values[:,0]).union(set(chunk.values[:,2]))
                            predicates = set(chunk.values[:,1])
                            ind = i*len(chunk)
                            reverse_ents.update({str(value):str(key+ind) for key, value in enumerate(entities)})
                            ents.update({str(key+ind):str(value) for key, value in enumerate(entities)})
                            reverse_rels.update({str(value):str(key+ind) for key, value in enumerate(predicates)})        
    
    def get_indexes_from_shelves(self, sample):
        """Get indexed triples.

           Parameters
           ----------
           sample: numpy array with a fragment of data of size (N,3), where each element is:
                  (subject, predicate, object).

           Returns
           -------
           tmp: numpy array of size (N,3) with indexed triples,
                where each element is: (subject index, predicate index, object index).
           """
        if isinstance(sample, pd.DataFrame):
            sample = sample.values()
        with shelve.open(self.reversed_entities_dict) as ents:
            with shelve.open(self.reversed_relations_dict) as rels:
                subjects = [ents[elem] for elem in sample[:,0]]
                objects = [str(ents[elem]) for elem in sample[:,2]]
                predicates = [str(rels[elem]) for elem in sample[:,1]]
                return np.array((subjects, predicates, objects), dtype=int).T
    
    def get_indexes_from_a_dictionary(self, sample):
        """Get indexed triples from a in-memory dictionary.

           Parameters
           ----------
           sample: numpy array with a fragment of data of size (N,3), where each element is:
                  (subject, predicate, object).

           Returns
           -------
           tmp: numpy array of size (N,3) with indexed triples,
                where each element is: (subject index, predicate index, object index).        
        """
        
        assert self.entities_dict is not None and self.relations_dict is not None
        subjects   = tf.convert_to_tensor([self.reversed_entities_dict[x] for x in sample[:,0]],  dtype=tf.int32)
        objects    = tf.convert_to_tensor([self.reversed_entities_dict[x] for x in sample[:,2]],  dtype=tf.int32)
        predicates = tf.convert_to_tensor([self.reversed_relations_dict[x] for x in sample[:,1]],  dtype=tf.int32)
        merged = tf.stack([subjects, predicates, objects], axis=1)
        return merged
    
    
    def get_starting_index_ents(self):
        """Returns next index to continue adding elements to entities dictionary."""        
        if not self.entities_dict:
            self.entities_dict = {}
            self.reversed_entities_dict = {}
            return 0
        else:
            return self.max_ents_index

    def get_starting_index_rels(self):
        """Returns next index to continue adding elements to relations dictionary."""
        if not self.relations_dict:
            self.relations_dict = {}
            self.reversed_relations_dict = {}
            return 0
        else:
            return self.max_rels_index + 1
        

    def update_dictionary_mappings_in_chunks(self):
        """Update dictionary mappings chunk by chunk."""
        for chunk in self.data:
            self.update_dictionary_mappings(chunk.values)
    
    def update_dictionary_mappings(self, sample=None):
        """Index entities and relations. Creates shelves for mappings between
           entities and relations to indexes and reverse mapping.

           Remember to use mappings for entities with entities and reltions with relations!
        """        
        if sample is None:
            sample = self.data
        i = self.get_starting_index_ents() # deleted + 1
        j = self.get_starting_index_rels() # deleted + 1 
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
  
        assert self.rev_ents_length == self.ents_length, "Reversed entities index size not equal to index size"
        assert self.rev_rels_length == self.rels_length , "Reversed relations index size not equal to index size"
                
        print("Mappings updated with: {} ents, {} rev_ents, {} rels and {} rev_rels".format(self.ents_length,
                                                                                            self.rev_ents_length,
                                                                                            self.rels_length,
                                                                                            self.rev_rels_length))
