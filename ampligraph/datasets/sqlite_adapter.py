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

DEFAULT_CHUNKSIZE = 30000
class SQLiteAdapter():
    def __init__(self, db_name, chunk_size=DEFAULT_CHUNKSIZE, verbose=False):
        self.db_name = db_name
        self.verbose = verbose
        self.indexed = False
        if chunk_size is None:
            chunk_size = DEFAULT_CHUNKSIZE
            print("Currently {} only supports data given in chunks. \
            Setting chunksize to {}.".format(self.__name__(), DEFAULT_CHUNKSIZE))
        else:
            self.chunk_size = chunk_size
        
    def __enter__ (self):
        try:
            db_uri = 'file:{}?mode=rw'.format(pathname2url(self.db_name))
            self.connection = sqlite3.connect(db_uri, uri=True)
        except sqlite3.OperationalError:
            print("Missing Database, creating one...")      
            self.connection = sqlite3.connect(self.db_name)        
            self._create_database()
        return self
    
    def __exit__ (self, type, value, tb):
        if tb is None:
            self.connection.commit()
            self.connection.close()
        else:
            # Exception occurred, so rollback.
            self.connection.rollback()
        
    def _get_db_schema(self):
        """Create 2 tables, one with indexes of objects and subjects in 
            other words entities and the other with triples and indexes to 
            navigate eaily on pairs subject-predicate, predicate-object."""

        db_schema = [
        """CREATE TABLE triples_table (subject integer,
                                    predicate integer,
                                    object integer,
                                    dataset_type text(50)
                                    );""",
        "CREATE INDEX triples_table_sp_idx ON triples_table (subject, predicate);",
        "CREATE INDEX triples_table_po_idx ON triples_table (predicate, object);",
        "CREATE INDEX triples_table_type_idx ON triples_table (dataset_type);"
        ]
        return db_schema

    def _get_clean_up(self):
        clean_up = ["drop index IF EXISTS triples_table_po_idx",
                    "drop index IF EXISTS triples_table_sp_idx",
                    "drop index IF EXISTS triples_table_type_idx",
                    "drop table IF EXISTS triples_table"]
        return clean_up

    def _execute_query(self, query):
        cursor = self.connection.cursor()
        output = None
        try:
            cursor.execute(query)
            output = cursor.fetchall()
            self.connection.commit()
            if self.verbose:
                print("Query executed successfully")
        except Error as e:
            print(f"Query failed. The error '{e}' occurred")
        return output

    def _execute_queries(self, list_of_queries):
        for query in list_of_queries:
            self._execute_query(query)

    def _insert_values_to_a_table(self, table, values):
        if self.verbose:
            print("inserting to a table...")
        if len(np.shape(values)) < 2:
            size = 1
        else:
            size = np.shape(values)[1]
        cursor = self.connection.cursor()
        try:
            values_placeholder = "({})".format(", ".join(["?"]*size))
            query = 'INSERT INTO {} VALUES {}'.format(table, values_placeholder)
            cursor.executemany(query, [(v,) if isinstance(v, int) or isinstance(v, str) else v for v in values])
            self.connection.commit()
            if self.verbose:
                print("commited to table: {}".format(table))
        except Error as e:
            print("Error", e)
            self.connection.rollback()
        cursor.close()   

    def _create_database(self):
        self._execute_queries(self._get_db_schema())

    def get_triples(self, chunk, dataset_type="train"): 
        if self.verbose:
            print("getting triples...")
        with shelve.open(self.reversed_entities_shelf) as ents:
            with shelve.open(self.reversed_relations_shelf) as rels:        
                subjects = [ents[elem] for elem in chunk.values[:,0]]
                objects = [str(ents[elem]) for elem in chunk.values[:,2]]
                predicates = [str(rels[elem]) for elem in chunk.values[:,1]]
                tmp = np.array((subjects, predicates, objects), dtype=int).T
                return np.append(tmp, np.array(len(chunk.values)*[dataset_type]).reshape(-1,1), axis=1)

    def index_entities_in_shelf(self):
        if self.verbose:        
            print("indexing entities...")
        date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.entities_shelf = "entities_shelf_{}.shf".format(date)
        self.reversed_entities_shelf = "reversed_entities_shelf_{}.shf".format(date)
        self.relations_shelf = "relations_shelf_{}.shf".format(date)
        self.reversed_relations_shelf = "reversed_relations_shelf_{}.shf".format(date)
        with shelve.open(self.entities_shelf, writeback=True) as ents:
            with shelve.open(self.reversed_entities_shelf, writeback=True) as reverse_ents: 
                with shelve.open(self.relations_shelf, writeback=True) as rels:
                    with shelve.open(self.reversed_relations_shelf, writeback=True) as reverse_rels:             
                        for i, chunk in enumerate(self.data):
                            entities = set(chunk.values[:,0]).union(set(chunk.values[:,2]))
                            predicates = set(chunk.values[:,1])
                            ind = i*len(chunk)
                            reverse_ents.update({str(value):str(key+ind) for key, value in enumerate(entities)})
                            ents.update({str(key+ind):str(value) for key, value in enumerate(entities)})                
                            reverse_rels.update({str(value):str(key+ind) for key, value in enumerate(predicates)})
                            rels.update({str(key+ind):str(value) for key, value in enumerate(predicates)})                                                

    def index_entities(self):
        """Get all keys from the shelf and populate database with them"""
        self.reload_data()
        self.index_entities_in_shelf() # remember to reinitialize data iterator before passing it through        
        # TODO: iteratively add entities to the database tables from shelf
        # maybe a flag letting know the state of the iterator (data source dried out!)
        #print("Warning! Not yet done, TODO in place.")
        #self._insert_values_to_a_table("entity_table", values_entities)
    
    def is_indexed(self):
        """Check if shelves with indexes are set."""
        if not hasattr(self, "entities_shelf"):
            return False
        if not hasattr(self, "reversed_entities_shelf"):
            return False
        if not hasattr(self, "relations_shelf"):
            return False
        if not hasattr(self, "reversed_relations_shelf"):
            return False
        return True
            
    def reload_data(self, verbose=False):
        self.data = self.loader(self.data_source, chunk_size=self.chunk_size)
        if verbose:
            print("Data reloaded", self.data)
        
    def populate(self, data_source, dataset_type="train", get_triples=None, loader=None):
        """Condition: before you can enter triples you have to index data."""
        self.data_source = data_source        
        self.loader = loader
        if loader is None:
            self.identifier = DataSourceIdentifier(self.data_source)
            self.loader = self.identifier.fetch_loader()
        if not self.is_indexed():
            if self.verbose:
                print("indexing...")
            self.index_entities()
        else:
            print("Data is already indexed, using that.")
        if get_triples is None:
            get_triples = self.get_triples
        self.reload_data()
        for chunk in self.data:
            values_triples = get_triples(chunk, dataset_type=dataset_type)
            self._insert_values_to_a_table("triples_table", values_triples)  
        if self.verbose:
            print("data is populated")
    
    def get_size(self, table="triples_table", condition=""):
        query = "SELECT count(*) from {} {};".format(table, condition)
        count = self._execute_query(query)
        if count is None:
            print("Table is empty or not such table exists.")
            return count
        elif not isinstance(count, list) or not isinstance(count[0], tuple):
            raise ValueError("Cannot get count for the table with provided condition.")        
        return count[0][0]

    def clean_up(self):
        status = self._execute_queries(self._get_clean_up())
        
    def remove_db(self):
        os.remove(self.db_name)        
        print("Database removed.")

    def _get_complementary_objects(self, triple):
        return self._execute_query("select {} union select distinct object from triples_table INDEXED BY \
                    triples_table_sp_idx where subject={} and predicate={}".format(triple[2], triple[0], triple[1]))

    def _get_complementary_subjects(self, triple):
        return self._execute_query("select {}  union select distinct subject from triples_table INDEXED BY \
                    triples_table_po_idx where predicate= {}  and object={}".format(triple[0], triple[1], triple[2]))

    def _get_complementary_entities(self, triple):
        """returns the participating entities in the relation ?-p-o and s-p-?
        Parameters
        ----------
        x_triple : nd-array (3,)
            triple (s-p-o) that we are querying
        Returns
        -------
        ent_participating_as_objects : nd-array (n,1)
            entities participating in the relation s-p-?
        ent_participating_as_subjects : nd-array (n,1)
            entities participating in the relation ?-p-o
        """
        entities = self._get_complementary_objects(triple)
        entities.extend(self._get_complementary_subjects(triple))
        return list(set(entities))
    
    def _get_batch(self, batch_size=1, dataset_type="train", use_filter=False):
        """Generator that returns the next batch of data.

        Parameters
        ----------
        dataset_type: string
            indicates which dataset to use
        batches_count: int
            number of batches per epoch (default: -1, i.e. uses batch_size of 1)
        use_filter : bool
            Flag to indicate whether to return the concepts that need to be filtered

        Returns
        -------
        batch_output : nd-array
            yields a batch of triples from the dataset type specified
        participating_objects : nd-array [n,1]
            all objects that were involved in the s-p-? relation. This is returned only if use_filter is set to true.
        participating_subjects : nd-array [n,1]
            all subjects that were involved in the ?-p-o relation. This is returned only if use_filter is set to true.
        """              
        query = "SELECT subject, predicate, object FROM triples_table INDEXED BY \
                                triples_table_type_idx where dataset_type ='{}' LIMIT {}, {}"
        
        if not hasattr(self, "batches_count"):
            size = self.get_size(condition="where dataset_type ='{}'".format(dataset_type))
            self.batches_count = int(size/batch_size)
        
        for i in range(self.batches_count):
            out = self._execute_query(query.format(dataset_type, i * batch_size, batch_size))
            if use_filter:
                # get the filter values
                participating_objects, participating_subjects = self.get_participating_entities(out)
                yield out, participating_objects, participating_subjects
            else:
                yield out                    
                    
    def summary(self, count=True):
        """Desiderata:
            - does the database exists? +
            - what tables does it have how many records, what are fields held and its types + example record? +
            - any triggers?
            - indexes declared?
            - other?
        """
        if os.path.exists(self.db_name):
            print("Summary for Database {}".format(self.db_name))
            file_size = os.path.getsize(self.db_name)
            summary = """File size: {:.5}{}\nTables: {}"""
            tables = self._execute_query("SELECT name FROM sqlite_master WHERE type='table';")
            tables_names = ", ".join(table[0] for table in tables)
            print(summary.format(*get_human_readable_size(file_size), tables_names))            
            types = {"integer":"int", "string":"str"}
            for table_name in tables:
                result = self._execute_query("PRAGMA table_info('%s')" % table_name)
                cols_name_type = ["{} ({}):".format(x[1],types[x[2]] if x[2] in types else x[2]) for x in result]
                length = len(cols_name_type)
                print("-------------\n|" + table_name[0].upper() + "|\n-------------\n")
                formatted_record = "{:7s}{}\n{:7s}{}".format(" ", "{:25s}"*length,"e.g.","{:<25s}"*length)
                msg = ""
                example = ["-"]*length
                if count:
                    nb_records = self.get_size(table_name[0])
                    msg = "\n\nRecords: {}".format(nb_records)                    
                    if nb_records != 0:
                        record = self._execute_query("SELECT * FROM {} LIMIT {};".format(table_name[0],1))[0]
                        example = [str(rec) for rec in record]                        
                else:
                    print("Count is set to False hence no data displayed")

                print(formatted_record.format(*cols_name_type, *example), msg)
        else:
            print("Database does not exist.")
            
    def _load(self, data_source, dataset_type="train"):
        self.data_source = data_source
        self.populate(self.data_source, dataset_type=dataset_type)
