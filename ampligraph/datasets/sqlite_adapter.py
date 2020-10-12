# Copyright 2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""SQLite backend for storing graphs.

This module provides SQLite backend for GraphDataLoader.

Attributes
----------
DEFAULT_CHUNKSIZE: [default 30000] size of data that can be at once loaded to the memory,
                   number of rows, should be set according to available hardware capabilities.
"""
from ampligraph.datasets.source_identifier import DataSourceIdentifier
from ampligraph.datasets import DataIndexer
import sqlite3
from sqlite3 import Error
import numpy as np
from urllib.request import pathname2url
import os
import shelve
from datetime import datetime
from ampligraph.utils.profiling import get_human_readable_size
import pandas as pd
import logging
import tempfile


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_CHUNKSIZE = 30000
class SQLiteAdapter():
    """ Class implementing database connection.
    
        Example
        -------
        >>># using GraphDataLoader
        >>>data = GraphDataLoader("data.csv", backend=SQLiteAdapter)
        >>># using initialised backend
        >>>data = GraphDataLoader("./fb15k/test.txt", backend=SQLiteAdapter("database.db", use_indexer=mapper))
        >>>for elem in data:
        >>>    print(elem)
        >>>    break
        [(1, 1, 2)]
        >>># raw with default indexing
        >>>with SQLiteAdapter("database.db") as backend:
        >>>    backend.populate("./fb15k/test.txt", dataset_type="train")
        >>># raw with previously specified indexing
        >>>mapper = DataIndexer(data.values)
        >>>with SQLiteAdapter("database.db", use_indexer=mapper) as backend:
        >>>    backend.populate("data.csv", dataset_type="train")
    """
    def __init__(self, db_name, identifier=None, chunk_size=DEFAULT_CHUNKSIZE, root_directory=tempfile.gettempdir(),
                 use_indexer=True, verbose=False, remap=False, name='main_partition', parent=None, in_memory=False, use_filter=False):
        """ Initialise SQLiteAdapter.
       
            Parameters
            ----------
            db_name: name of the database.
            chunk_size: size of a chunk to read data from while feeding the database,
                        if not provided will be default (DEFAULT_CHUNKSIZE).
            root_directory: directory where data will be stored - database created and mappings.
            use_indexer: object of type DataIndexer with predifined mapping or bool flag to tell whether data should be indexed.
            remap: wether to remap or not (shouldn't be used here) - NotImplemented here.
            parent: Not Implemented.
            verbose: print status messages.
        """
        self.db_name = db_name
        self.verbose = verbose
        if identifier is None:
           msg = "You need to provide source identifier object"
           logger.error(msg) 
           raise Exception(msg)
        else:
            self.identifier = identifier

        self.flag_db_open = False
        self.root_directory = root_directory
        self.db_path = os.path.join(self.root_directory, self.db_name)
        self.use_indexer = use_indexer
        self.remap = remap
        if self.remap != False:
            msg = "Remapping is not supported for DataLoaders with SQLite Adapter as backend"
            logger.error(msg)
            raise Exception(msg)
        self.name = name
        self.parent = parent
        self.in_memory = in_memory
        self.use_filter = use_filter
        self.sources = {}

        if chunk_size is None:
            chunk_size = DEFAULT_CHUNKSIZE
            logger.debug("Currently {} only supports data given in chunks. \
            Setting chunksize to {}.".format(self.__name__(), DEFAULT_CHUNKSIZE))
        else:
            self.chunk_size = chunk_size

    def open_db(self):
        db_uri = 'file:{}?mode=rw'.format(pathname2url(self.db_path))
        self.connection = sqlite3.connect(db_uri, uri=True)
        self.flag_db_open = True
        logger.debug("----------------DB OPENED - normally -----------------------")

    def open_connection(self):
        """Context manager function to open or create if not exists database connection."""
        if self.flag_db_open == False:
            try:
                self.open_db()
            except sqlite3.OperationalError:
                logger.debug("Database does not exists. Creating one.")      
                self.connection = sqlite3.connect(self.db_path)        
                self.connection.commit()
                self.connection.close()
        
                self._create_database()
                self.open_db()

    def __enter__(self):
        self.open_connection()
        return self
    
    def __exit__ (self, type, value, tb):
        """Context manager exit function, required to used with "with statement", closes
           the connection and do the rollback if required"""
        if self.flag_db_open:
            if tb is None:
                self.connection.commit()
                self.connection.close()
            else:
                # Exception occurred, so rollback.
                self.connection.rollback()
            self.flag_db_open = False
            logger.debug("!!!!!!!!----------------DB CLOSED -----------------------")

    def _add_dataset(self, data_source, dataset_type):
        self._load(data_source, dataset_type)
        
    def _get_db_schema(self):
        """Defines SQL queries to create a table with triples and indexes to 
           navigate easily on pairs subject-predicate, predicate-object.
    
           Returns
           -------
           db_schema: list of SQL commands to create tables and indexes.
        """
        db_schema = [
        """CREATE TABLE triples_table (subject integer,
                                    predicate integer,
                                    object integer,
                                    dataset_type text(50)
                                    );""",
        "CREATE INDEX triples_table_sp_idx ON triples_table (subject, predicate);",
        "CREATE INDEX triples_table_po_idx ON triples_table (predicate, object);",
        "CREATE INDEX triples_table_type_idx ON triples_table (dataset_type);",
        "CREATE INDEX triples_table_sub_obj_idx ON triples_table (subject, object);",
        "CREATE INDEX triples_table_subject_idx ON triples_table (subject);",
        "CREATE INDEX triples_table_object_idx ON triples_table (object);"
        ]
        return db_schema

    def _get_clean_up(self):
        """Defines SQL commands to clean the databse (tables and indexes).
    
           Returns
           -------
           clean_up: list of SQL commands to clean tables and indexes.
        """  
        clean_up = ["drop index IF EXISTS triples_table_po_idx",
                    "drop index IF EXISTS triples_table_sp_idx",
                    "drop index IF EXISTS triples_table_type_idx",
                    "drop table IF EXISTS triples_table"]
        return clean_up

    def _execute_query(self, query):
        """Connects to the database and execute given query.
    
           Parameters
           ----------
           query: SQLite query to be executed.
     
           Returns
           -------
           output: result of a query with fetchall().
        """
        with self:
            cursor = self.connection.cursor()
            output = None
            try:
                cursor.execute(query)
                output = cursor.fetchall()
                self.connection.commit()
                if self.verbose:
                    logger.debug("Query executed successfully, {}".format(query))
            except Error as e:
                logger.debug("Query failed. The error '{}' occurred".format(e))
            return output

    def _execute_queries(self, list_of_queries):
        """Executes given list of queries one by one.

           Parameters
           ----------
           query: list of SQLite queries to be executed.
     
           Returns
           -------
           output: TODO! result of queries with fetchall().
          
        """
        for query in list_of_queries:
            self._execute_query(query)

    def _insert_values_to_a_table(self, table, values):
        """Insert data into a given table in a database.
    
           Parameters
           ----------
           table: table where to input data.
           values: array of data with shape (N,3) to be written to the database, 
                   where N is a number of entries.      
        """
        with self:
            if self.verbose:
                logger.debug("inserting to a table...")
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
                    logger.debug("commited to table: {}".format(table))
            except Error as e:
                logger.debug("Error: {}".format(e))
                #self.connection.rollback()
            logger.debug("Values were inserted!")

    def _create_database(self):
        """Creates database."""
        self._execute_queries(self._get_db_schema())

    def _get_triples(self, subjects=None, objects=None, entities=None):
        """Get triples that objects belongs to objects and subjects to subjects,
           or if not provided either object or subjet belongs to entities.
        """
        if subjects is None and objects is None:
            if entities is None:
                msg = "You have to provide either subjects and objects indexes or general entities indexes!"
                logger.error(msg)
                raise Exception(msg)

            subjects = entities
            objects = entities

        query = "select * from triples_table where (subject in ({0}) and object in ({1})) or (subject in ({1}) and object in ({0}));".format(",".join(str(v) for v in  subjects), ",".join(str(v) for v in  objects))
        triples = np.array(self._execute_query(query))
        triples = np.append(triples[:,:3].astype('int'), triples[:,3].reshape(-1,1), axis=1)
        return triples 

    def get_indexed_triples(self, chunk, dataset_type="train"): 
        """Get indexed triples.
    
           Parameters
           ----------
           chunk: numpy array with a fragment of data of size (N,3), where each element is:
                  (subject, predicate, object).
           dataset_type: defines what kind of data is it (train, test, validation).
           
           Returns
           -------
           tmp: numpy array of size (N,4) with indexed triples,
                where each element is: (subject index, predicate index, object index, dataset_type).
           """
        if self.verbose:
            logger.debug("getting triples...")
        if isinstance(chunk, pd.DataFrame):
            chunk = chunk.values
        if self.use_indexer != False:
            #logger.debug(chunk)
            triples = self.mapper.get_indexes(chunk)
            return np.append(triples, np.array(len(chunk)*[dataset_type]).reshape(-1,1), axis=1)
        else:
            return np.append(chunk, np.array(len(chunk)*[dataset_type]).reshape(-1,1), axis=1)


    def index_entities(self):
        """Index data. It reloads data before as it is an iterator."""
        self.reload_data()
        if self.use_indexer == True:
            self.mapper = DataIndexer(self.data, in_memory=self.in_memory, root_directory=self.root_directory)
        elif self.use_indexer == False:
            logger.debug("Data won't be indexed")
        elif isinstance(self.use_indexer, DataIndexer):
            self.mapper = self.use_indexer
    
    def is_indexed(self):
        """Check if adapter has indexer.
        
           Returns
           -------
           True/False - flag indicating whether indexing took place.
        """
        if not hasattr(self, "mapper"):
            return False
        return True
            
    def reload_data(self, verbose=False):
        """Reinitialise an iterator with data."""
        self.data = self.loader(self.data_source, chunk_size=self.chunk_size)
        if verbose:
            logger.debug("Data reloaded: {}".format(self.data))
        
    def populate(self, data_source, dataset_type="train", get_indexed_triples=None, loader=None):
        """Condition: before you can enter triples you have to index data.
    
           Parameters
           ----------
           data_source: file with data (e.g. csv file).
           dataset_type: what type of data is it? (train | test | validation).
           get_indexed_triples: function to obtain indexed triples.
           loader: loading function to be used to load data, if None, the
                   DataSourceIdentifier will try to identify type and return
                   adequate loader.
        """
        self.data_source = data_source        
        self.loader = loader
        if loader is None:
            self.loader = self.identifier.fetch_loader()
        if not self.is_indexed() and self.use_indexer != False:
            if self.verbose:
                logger.debug("indexing...")
            self.index_entities()
        else:
            logger.debug("Data is already indexed or no indexing is required.")
        if get_indexed_triples is None:
            get_indexed_triples = self.get_indexed_triples
        data = self.loader(data_source, chunk_size=self.chunk_size)

        self.reload_data()
        for chunk in data:
            values_triples = get_indexed_triples(chunk, dataset_type=dataset_type)
            self._insert_values_to_a_table("triples_table", values_triples)  
        if self.verbose:
            logger.debug("data is populated")
    
    def get_data_size(self, table="triples_table", condition=""):
        """Gets the size of the given table [with specified condition].
    
           Parameters
           ----------
           table: table for which to obtain the size.
           condition: condition to count only a subset of data.
    
           Returns
           -------
           count: number of records in the table.
        """
        query = "SELECT count(*) from {} {};".format(table, condition)
        count = self._execute_query(query)
        if count is None:
            logger.debug("Table is empty or not such table exists.")
            return count
        elif not isinstance(count, list) or not isinstance(count[0], tuple):
            raise ValueError("Cannot get count for the table with provided condition.")        
        #logger.debug(count)
        return count[0][0]

    def clean_up(self):
        """Clean the database."""
        status = self._execute_queries(self._get_clean_up())
        
    def remove_db(self):
        """Remove the database file."""
        os.remove(self.db_path)        
        logger.debug("Database removed.")

    def _get_complementary_objects(self, triples, use_filter=None):
        """For a given triple retrive all triples whith same subjects and predicates.

           Parameters
           ----------
           triples: list or array with Nx3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of objects.
        """
        results = []
        if use_filter == False or use_filter is None:
            use_filter = {'train': self.data}
        filtered = []
        valid_filters = [x[0] for x in self._execute_query("SELECT DISTINCT dataset_type FROM triples_table")]
        for filter_name, filter_source in use_filter.items():
            if filter_name in valid_filters:                
                tmp_filter = []
                for triple in triples:
                    query = 'select distinct object from triples_table INDEXED BY\
                             triples_table_sp_idx where subject in ({}) and predicate in ({})  and dataset_type ="{}"'

                    query = query.format(triple[0], triple[1], filter_name)
                    q = self._execute_query(query)
                    tmp = list(set([y for x in q for y in x ]))
                    tmp_filter.append(tmp)
                filtered.append(tmp_filter)
        # Unpack data into one  list per triple no matter what filter it comes from
        unpacked = list(zip(*filtered))
        for k in unpacked:
            lst = [j for i in k for j in i]
            results.append(lst)

        return results

    def _get_complementary_subjects(self, triples, use_filter=None):
        """For a given triple retrive all triples whith same objects and predicates.

           Parameters
           ----------
           triple: list or array with 3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of subjects.
        """
        results = []
        if use_filter == False or use_filter is None:
            use_filter = {'train': self.data}

        filtered = []
        valid_filters = [x[0] for x in self._execute_query("SELECT DISTINCT dataset_type FROM triples_table")]
        for filter_name, filter_source in use_filter.items():
                if filter_name in valid_filters:
                    tmp_filter = []
                    for triple in triples:
                        query = 'select distinct subject from triples_table INDEXED BY \
                                 triples_table_po_idx where predicate in ({})  and object in ({})  and dataset_type ="{}"'
                        query = query.format(triple[1], triple[2], filter_name)
                        q = self._execute_query(query)
                        tmp = list(set([y for x in q for y in x ]))
                        tmp_filter.append(tmp)
                    filtered.append(tmp_filter)
        # Unpack data into one  list per triple no matter what filter it comes from
        unpacked = list(zip(*filtered))
        for k in unpacked:
            lst = [j for i in k for j in i]
            results.append(lst)
        return results

    def _get_complementary_entities(self, triples, use_filter=None):
        """Returns the participating entities in the relation ?-p-o and s-p-?.

        Parameters
        ----------
        x_triple: nd-array (3,)
            triple (s-p-o) that we are querying.

        Returns
        -------
        entities: list of entities participating in the relations s-p-? and ?-p-o.
        """
        objects = self._get_complementary_objects(triples, use_filter=use_filter)
        subjects = self._get_complementary_subjects(triples, use_filter=use_filter)
        return subjects, objects
    
    def _get_batch_generator(self, batch_size=1, dataset_type="train", random=False, use_filter=False, index_by=""):
        """Generator that returns the next batch of data.

        Parameters
        ----------
        dataset_type: string
            indicates which dataset to use (train | test | validation).
        batch_size: int
            number of elements in a batch (default: 1).
        use_filter : bool
            Flag to indicate whether to return the concepts that need to be filtered
        index_by: possible values:  {"", so, os, s, o}, indicates whether to use index and which to use,
                                   index by subject, object or both. Indexes were created for the fields so 
                                   SQLite should use them here to speed up, see example below:
                  sqlite> EXPLAIN QUERY PLAN SELECT * FROM triples_table ORDER BY subject, object LIMIT 7000, 30;
                  QUERY PLAN
                  `--SCAN TABLE triples_table USING INDEX triples_table_sub_obj_idx

        random: get records from database in a random order.

        Returns
        -------
        batch_output : nd-array
            yields a batch of triples from the dataset type specified
        participating_entities : list of all entities that were involved in the s-p-? and ?-p-o relations. 
                                 This is returned only if use_filter is set to true.
        """              
        size = self.get_data_size(condition="where dataset_type ='{}'".format(dataset_type))
        self.batches_count = int(size/batch_size)
        logger.debug("batches count: {}".format(self.batches_count))
        logger.debug("size of data: {}".format(size))
        index = ""
        if index_by != "":
            if (index_by == "s" or index_by == "o" or index_by == "so" or index_by == "os") and random != False:       
                msg = "Field index_by can only be used with random set to False and can only take values \
                       from this set: {{s,o,so,os,''}}, instead got: {}".format(index_by)
                logger.error(msg)
                raise Exception(msg)

            if index_by == "s":
                index = "ORDER BY subject"
            if index_by == "o":
                index = "ORDER BY object"
            if index_by == "so" or index_by == "os":
                index = "ORDER BY subject, object"
            if index == "" and random:
                index = "ORDER BY random()"
        query_template = "SELECT * FROM triples_table INDEXED BY \
                 triples_table_type_idx where dataset_type ='{}' {} LIMIT {}, {};"

        for i in range(self.batches_count):
            #logger.debug("BATCH NUMBER: {}".format(i))
            #logger.debug(i * batch_size)
            query = query_template.format(dataset_type, index, i * batch_size, batch_size)
            #logger.debug(query)
            out = self._execute_query(query)
            #logger.debug(out)
            if out:
                out = np.array(out)[:,:3]                   
            if use_filter:
                # get the filter values
                participating_entities = self._get_complementary_entities(out)
                yield out, participating_entities
            else:
                yield out                    
                    
    def summary(self, count=True):
        """Prints summary of the database, whether it exists, what
           tables does it have and how many records (count=True),
           what are fields held and their types with an example record.

           Parameters
           ----------
           count: whether to count number of records per table (can be time consuming)

           Example
           -------
           >>>adapter = SQLiteAdapter("database_24-06-2020_03-51-12_PM.db")
           >>>with adapter as db:
           >>>    db.summary()
           Summary for Database database_29-06-2020_09-37-20_AM.db
           File size: 3.9453MB
           Tables: triples_table
           -------------
           |TRIPLES_TABLE|
           -------------
           
                  subject (int):   predicate (int):   object (int):   dataset_type (text(50)): 
           e.g.   34321            29218              38102           train                     
           
           Records: 59070

        """
        if os.path.exists(self.db_path):
            print("Summary for Database {}".format(self.db_name))
            print("Located in {}".format(self.db_path))
            file_size = os.path.getsize(self.db_path)
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
                    nb_records = self.get_data_size(table_name[0])
                    msg = "\n\nRecords: {}".format(nb_records)                    
                    if nb_records != 0:
                        record = self._execute_query("SELECT * FROM {} LIMIT {};".format(table_name[0],1))[0]
                        example = [str(rec) for rec in record]                        
                else:
                    print("Count is set to False hence no data displayed")

                print(formatted_record.format(*cols_name_type, *example), msg)
        else:
            logger.debug("Database does not exist.")
            
    def _load(self, data_source, dataset_type="train"):
        """Loads data from the data source to the database. Wrapper around populate method,
           required by the GraphDataLoader interface.
           
           Parameters
           ----------
           data_source: file from where to read data (e.g. csv file).
           dataset_type: kind of dataset that is being loaded (train | test | validation).
        """
        self.data_source = data_source
        self.populate(self.data_source, dataset_type=dataset_type)

    def _intersect(self, dataloader):
        if not isinstance(dataloader.backend, SQLiteAdapter):
            msg = "Provided dataloader should be of type SQLiteAdapter backend, instead got {}.".format(type(dataloader.backend)) 
            logger.error(msg)
            raise Exception(msg)
        raise NotImplementedError

    def _clean(self):
        os.remove(self.db_path)
        self.mapper.clean()
