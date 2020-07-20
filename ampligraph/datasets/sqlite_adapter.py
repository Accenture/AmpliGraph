# Copyright 2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
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
    def __init__(self, db_name, chunk_size=DEFAULT_CHUNKSIZE, root_directory="./", use_indexer=True, verbose=False, remap=False, name='main_partition'):
        """ Initialise SQLiteAdapter.
       
            Parameters
            ----------
            db_name: name of the database.
            chunk_size: size of a chunk to read data from while feeding the database,
                        if not provided will be default (DEFAULT_CHUNKSIZE).
            root_directory: directory where data will be stored - database created and mappings.
            use_indexer: object of type DataIndexer with predifined mapping or bool flag to tell whether data should be indexed.
            remap: wether to remap or not (shouldn't be used here) - NotImplemented here.
            verbose: print status messages.
        """
        self.db_name = db_name
        self.root_directory = root_directory
        self.db_path = os.path.join(self.root_directory, self.db_name)
        self.use_indexer = use_indexer
        self.remap = remap
        assert self.remap == False, "Remapping is not supported for DataLoaders with SQLite Adapter as backend"
        self.name = name
        self.verbose = verbose
        if chunk_size is None:
            chunk_size = DEFAULT_CHUNKSIZE
            print("Currently {} only supports data given in chunks. \
            Setting chunksize to {}.".format(self.__name__(), DEFAULT_CHUNKSIZE))
        else:
            self.chunk_size = chunk_size
        
    def __enter__ (self):
        """Context manager function to open or create if not exists database connection."""
        try:
            db_uri = 'file:{}?mode=rw'.format(pathname2url(self.db_path))
            self.connection = sqlite3.connect(db_uri, uri=True)
        except sqlite3.OperationalError:
            print("Missing Database, creating one...")      
            self.connection = sqlite3.connect(self.db_path)        
            self._create_database()
        return self
    
    def __exit__ (self, type, value, tb):
        """Context manager exit function, required to used with "with statement", closes
           the connection and do the rollback if required"""
        if tb is None:
            self.connection.commit()
            self.connection.close()
        else:
            # Exception occurred, so rollback.
            self.connection.rollback()
        
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
        "CREATE INDEX triples_table_type_idx ON triples_table (dataset_type);"
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
        """Creates database."""
        self._execute_queries(self._get_db_schema())

    def _get_triples(self, subjects=None, objects=None, entities=None):
        """Get triples that objects belongs to objects and subjects to subjects,
           or if not provided either object or subjet belongs to entities.
        """
        if subjects is None and objects is None:
            msg = "You have to provide either subjects and objects indexes or general entities indexes!"
            assert(entities is not None), msg 
            subjects = entities
            objects = entities

        query = "select * from triples_table where (subject in ({0}) and object in ({1})) or (subject in ({1}) and object in ({0}));".format(",".join(str(v) for v in  subjects), ",".join(str(v) for v in  objects))
        triples = self._execute_query(query)
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
            print("getting triples...")
        if self.use_indexer != False:
            #print(chunk)
            triples = self.mapper.get_indexes(chunk)
            return np.append(triples, np.array(len(chunk.values)*[dataset_type]).reshape(-1,1), axis=1)
        else:
            return np.append(chunk.values, np.array(len(chunk.values)*[dataset_type]).reshape(-1,1), axis=1)

#        with shelve.open(self.reversed_entities_shelf) as ents:
#            with shelve.open(self.reversed_relations_shelf) as rels:        
#                subjects = [ents[elem] for elem in chunk.values[:,0]]
#                objects = [str(ents[elem]) for elem in chunk.values[:,2]]
#                predicates = [str(rels[elem]) for elem in chunk.values[:,1]]
#                tmp = np.array((subjects, predicates, objects), dtype=int).T
#                return np.append(tmp, np.array(len(chunk.values)*[dataset_type]).reshape(-1,1), axis=1)

#    def index_entities_in_shelf(self):
#        """Index entities and relations. Creates shelves for mappings between
#           entities and relations to indexes and reverse mapping. 
#    
#           Four shelves are created in root_directory:
#           entities_shelf_<DATE>.shf - with map entities -> indexes
#           reversed_entities_shelf_<DATE>.shf - with map indexes -> entities
#           relations_shelf_<DATE>.shf - with map relations -> indexes
#           reversed_relations_shelf_<DATE>.shf - with map indexes -> relations
#    
#           Rememer to use mappings for entities with entities and reltions with relations!
#        """
#        if self.verbose:        
#            print("indexing entities...")
#        date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
#        self.entities_shelf = os.path.join(self.root_directory, "entities_shelf_{}.shf".format(date))
#        self.reversed_entities_shelf = os.path.join(self.root_directory, "reversed_entities_shelf_{}.shf".format(date))
#        self.relations_shelf = os.path.join(self.root_directory, "relations_shelf_{}.shf".format(date))
#        self.reversed_relations_shelf = os.path.join(self.root_directory, "reversed_relations_shelf_{}.shf".format(date))
#        with shelve.open(self.entities_shelf, writeback=True) as ents:
#            with shelve.open(self.reversed_entities_shelf, writeback=True) as reverse_ents: 
#                with shelve.open(self.relations_shelf, writeback=True) as rels:
#                    with shelve.open(self.reversed_relations_shelf, writeback=True) as reverse_rels:             
#                        for i, chunk in enumerate(self.data):
#                            entities = set(chunk.values[:,0]).union(set(chunk.values[:,2]))
#                            predicates = set(chunk.values[:,1])
#                            ind = i*len(chunk)
#                            reverse_ents.update({str(value):str(key+ind) for key, value in enumerate(entities)})
#                            ents.update({str(key+ind):str(value) for key, value in enumerate(entities)})                
#                            reverse_rels.update({str(value):str(key+ind) for key, value in enumerate(predicates)})
#                            rels.update({str(key+ind):str(value) for key, value in enumerate(predicates)})                                                

    def index_entities(self):
        """Index data. It reloads data before as it is an iterator."""
        self.reload_data()
        if self.use_indexer == True:
            self.mapper = DataIndexer(self.data)
        elif self.use_indexer == False:
            print("Data won't be indexed")
        elif isinstance(self.use_indexer, DataIndexer):
            self.mapper = self.use_indexer
#        self.reload_data()
#        self.index_entities_in_shelf()
    
    def is_indexed(self):
        """Check if adapter has indexer.
        
           Returns
           -------
           True/False - flag indicating whether indexing took place.
        """
        if not hasattr(self, "mapper"):
            return False
#        if not hasattr(self, "reversed_entities_shelf"):
#            return False
#        if not hasattr(self, "relations_shelf"):
#            return False
#        if not hasattr(self, "reversed_relations_shelf"):
#            return False
        return True
            
    def reload_data(self, verbose=False):
        """Reinitialise an iterator with data."""
        self.data = self.loader(self.data_source, chunk_size=self.chunk_size)
        if verbose:
            print("Data reloaded", self.data)
        
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
            self.identifier = DataSourceIdentifier(self.data_source)
            self.loader = self.identifier.fetch_loader()
        if not self.is_indexed() and self.use_indexer != False:
            if self.verbose:
                print("indexing...")
            self.index_entities()
        else:
            print("Data is already indexed or no indexing is required.")
        if get_indexed_triples is None:
            get_indexed_triples = self.get_indexed_triples
        self.reload_data()
        for chunk in self.data:
            values_triples = get_indexed_triples(chunk, dataset_type=dataset_type)
            self._insert_values_to_a_table("triples_table", values_triples)  
        if self.verbose:
            print("data is populated")
    
    def get_size(self, table="triples_table", condition=""):
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
            print("Table is empty or not such table exists.")
            return count
        elif not isinstance(count, list) or not isinstance(count[0], tuple):
            raise ValueError("Cannot get count for the table with provided condition.")        
        return count[0][0]

    def clean_up(self):
        """Clean the database."""
        status = self._execute_queries(self._get_clean_up())
        
    def remove_db(self):
        """Remove the database file."""
        os.remove(self.db_path)        
        print("Database removed.")

    def _get_complementary_objects(self, triples):
        """For a given triple retrive all triples whith same subjects and predicates.

           Parameters
           ----------
           triples: list or array with Nx3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of objects.
        """
        query = "select distinct object from triples_table INDEXED BY triples_table_sp_idx where subject in ({}) and predicate in ({});"
        query = query.format(",".join(str(v) for v in triples[:,0]),",".join(str(v) for v in triples[:,1]))
        return self._execute_query(query)

    def _get_complementary_subjects(self, triple):
        """For a given triple retrive all triples whith same objects and predicates.

           Parameters
           ----------
           triple: list or array with 3 elements (subject, predicate, object).

           Returns
           -------
           result of a query, list of subjects.
        """

        return self._execute_query("select {}  union select distinct subject from triples_table INDEXED BY \
                    triples_table_po_idx where predicate= {}  and object={}".format(triple[0], triple[1], triple[2]))

    def _get_complementary_entities(self, triple):
        """Returns the participating entities in the relation ?-p-o and s-p-?.

        Parameters
        ----------
        x_triple: nd-array (3,)
            triple (s-p-o) that we are querying.

        Returns
        -------
        entities: list of entities participating in the relations s-p-? and ?-p-o.
        """
        entities = self._get_complementary_objects(triple)
        entities.extend(self._get_complementary_subjects(triple))
        return list(set(entities))
    
    def _get_batch(self, batch_size=1, dataset_type="train", use_filter=False):
        """Generator that returns the next batch of data.

        Parameters
        ----------
        dataset_type: string
            indicates which dataset to use (train | test | validation).
        batch_size: int
            number of elements in a batch (default: 1).
        use_filter : bool
            Flag to indicate whether to return the concepts that need to be filtered

        Returns
        -------
        batch_output : nd-array
            yields a batch of triples from the dataset type specified
        participating_entities : list of all entities that were involved in the s-p-? and ?-p-o relations. 
                                 This is returned only if use_filter is set to true.
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
                participating_entities = self.get_complementary_entities(out)
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
        """Loads data from the data source to the database. Wrapper around populate method,
           required by the GraphDataLoader interface.
           
           Parameters
           ----------
           data_source: file from where to read data (e.g. csv file).
           dataset_type: kind of dataset that is being loaded (train | test | validation).
        """
        self.data_source = data_source
        self.populate(self.data_source, dataset_type=dataset_type)
