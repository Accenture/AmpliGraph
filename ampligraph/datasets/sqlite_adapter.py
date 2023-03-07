# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
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
DEFAULT_CHUNKSIZE: int
    Size of data that can be at once loaded to the memory, number of rows,
    should be set according to available
    hardware capabilities (default: 30000).
"""
import logging
import os
import sqlite3
import tempfile
from sqlite3 import Error
from urllib.request import pathname2url

import numpy as np
import pandas as pd
import tensorflow as tf

from ampligraph.utils.profiling import get_human_readable_size

from .data_indexer import DataIndexer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_CHUNKSIZE = 30000


class SQLiteAdapter:
    """Class implementing database connection.

    Example
    -------
    >>> AMPLIGRAPH_DATA_HOME='/your/path/to/datasets/'
    >>> # Initialize GraphDataLoader from .csv file
    >>> data = GraphDataLoader("data.csv", backend=SQLiteAdapter)
    >>> # Initialize GraphDataLoader from .txt file using indexer
    >>> # to map entities to integers
    >>> data = GraphDataLoader(AMPLIGRAPH_DATA_HOME + "fb15k/test.txt",
    >>>                        backend=SQLiteAdapter("database.db",
    >>>                        use_indexer=True))
    >>> for elem in data:
    >>>     print(elem)
    >>>     break
    [(1, 1, 2)]
    >>> # Populate the database with raw triples for training
    >>> with SQLiteAdapter("database.db") as backend:
    >>>     backend.populate(AMPLIGRAPH_DATA_HOME + "fb15k/train.txt",
    >>>                      dataset_type="train")
    >>> # Populate the database with indexed triples for training
    >>> with SQLiteAdapter("database.db", use_indexer=True) as backend:
    >>>     backend.populate(AMPLIGRAPH_DATA_HOME + "fb15k/train.txt",
    >>>                      dataset_type="train")
    """

    def __init__(
        self,
        db_name,
        identifier=None,
        chunk_size=DEFAULT_CHUNKSIZE,
        root_directory=None,
        use_indexer=True,
        verbose=False,
        remap=False,
        name="main_partition",
        parent=None,
        in_memory=True,
        use_filter=False,
    ):
        """Initialise SQLiteAdapter.

        Parameters
        ----------
        db_name: str
            Name of the database.
        chunk_size: int
            Size of a chunk to read data from while feeding
            the database (default: DEFAULT_CHUNKSIZE).
        root_directory: str
            Path to a directory where the database will be created,
            and the data and mappings will be stored. If `None`, the root
            directory is obtained through the :meth:`tempfile.gettempdir()
            method (default: `None`).
        use_indexer: DataIndexer or bool
            Object of type DataIndexer with pre-defined mapping or bool
            flag to tell whether data
            should be indexed.
        remap: bool
            Whether to remap or not (shouldn't be used here) -
            - NotImplemented here.
        parent:
            Not Implemented.
        verbose:
            Print status messages.
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

        if root_directory is None:
            self.root_directory = tempfile.gettempdir()
        else:
            self.root_directory = root_directory

        self.db_path = os.path.join(self.root_directory, self.db_name)
        self.use_indexer = use_indexer
        self.remap = remap
        if self.remap:
            msg = "Remapping is not supported for DataLoaders with SQLite\
                   Adapter as backend"

            logger.error(msg)
            raise Exception(msg)
        self.name = name
        self.parent = parent
        self.in_memory = in_memory
        self.use_filter = use_filter
        self.sources = {}

        if chunk_size is None:
            chunk_size = DEFAULT_CHUNKSIZE
            logger.debug(
                "Currently {} only supports data given in chunks. \
            Setting chunksize to {}.".format(
                    self.__name__(), DEFAULT_CHUNKSIZE
                )
            )
        else:
            self.chunk_size = chunk_size

    def get_output_signature(self):
        """Get the output signature of the tf.data.Dataset object."""
        triple_tensor = tf.TensorSpec(shape=(None, 3), dtype=tf.int32)

        # focusE
        if self.data_shape > 3:
            weights_tensor = tf.TensorSpec(
                shape=(None, self.data_shape - 3), dtype=tf.float32
            )
            if self.use_filter:
                return (
                    triple_tensor,
                    tf.RaggedTensorSpec(shape=(2, None, None), dtype=tf.int32),
                    weights_tensor,
                )
            return (triple_tensor, weights_tensor)
        if self.use_filter:
            return (
                triple_tensor,
                tf.RaggedTensorSpec(shape=(2, None, None), dtype=tf.int32),
            )
        return triple_tensor

    def open_db(self):
        """Open the database."""
        db_uri = "file:{}?mode=rw".format(pathname2url(self.db_path))
        self.connection = sqlite3.connect(db_uri, uri=True)
        self.flag_db_open = True
        logger.debug("----------------DB OPENED - normally -----------------")

    def open_connection(self):
        """Context manager function to open (or create if it does not exist)
        a database connection.

        """
        if not self.flag_db_open:
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

    def __exit__(self, type, value, tb):
        """Context manager exit function, to be used with "with statement",
        closes the connection and do the rollback if required.

        """
        if self.flag_db_open:
            if tb is None:
                self.connection.commit()
                self.connection.close()
            else:
                # Exception occurred, so rollback.
                self.connection.rollback()
            self.flag_db_open = False
            logger.debug("!!!!!!!!----------------DB CLOSED ----------------")

    def _add_dataset(self, data_source, dataset_type):
        """Load the data."""
        self._load(data_source, dataset_type)

    def _get_db_schema(self):
        """Defines SQL queries to create a table with triples and indexes to
        navigate easily on pairs subject-predicate, predicate-object.

        Returns
        -------
        db_schema: list
             List of SQL commands to create tables and indexes.
        """
        if self.data_shape < 4:
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
                "CREATE INDEX triples_table_object_idx ON triples_table (object);",
            ]
        else:  # focusE
            db_schema = [
                """CREATE TABLE triples_table (subject integer,
                                        predicate integer,
                                        object integer,
                                        weight float,
                                        dataset_type text(50)
                                        );""",
                "CREATE INDEX triples_table_sp_idx ON triples_table (subject, predicate);",
                "CREATE INDEX triples_table_po_idx ON triples_table (predicate, object);",
                "CREATE INDEX triples_table_type_idx ON triples_table (dataset_type);",
                "CREATE INDEX triples_table_sub_obj_idx ON triples_table (subject, object);",
                "CREATE INDEX triples_table_subject_idx ON triples_table (subject);",
                "CREATE INDEX triples_table_object_idx ON triples_table (object);",
            ]
        return db_schema

    def _get_clean_up(self):
        """Defines SQL commands to clean the database (tables and indexes).

        Returns
        -------
        clean_up: list
             List of SQL commands to clean tables and indexes.
        """
        clean_up = [
            "drop index IF EXISTS triples_table_po_idx",
            "drop index IF EXISTS triples_table_sp_idx",
            "drop index IF EXISTS triples_table_type_idx",
            "drop table IF EXISTS triples_table",
        ]
        return clean_up

    def _execute_query(self, query):
        """Connects to the database and execute given query.

        Parameters
        ----------
        query: str
             SQLite query to be executed.

        Returns
        -------
        output:
             Result of a query with fetchall().
        """
        with self:
            cursor = self.connection.cursor()
            output = None
            try:
                cursor.execute(query)
                output = cursor.fetchall()
                self.connection.commit()
                if self.verbose:
                    logger.debug(f"Query executed successfully, {query}")
            except Error as e:
                logger.debug(f"Query failed. The error '{e}' occurred")
            return output

    def _execute_queries(self, list_of_queries):
        """Executes given list of queries one by one.

        Parameters
        ----------
        query: list
             List of SQLite queries to be executed.

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
        table: str
             Table where to input data.
        values: ndarray
             Numpy array of data with shape (N,m) to be written to
             the database. `N` is a number of entries, :math:`m=3`
             if we only have triples and :math:`m>3` if we have numerical
                weights associated with each triple.
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
                values_placeholder = "({})".format(", ".join(["?"] * size))
                query = "INSERT INTO {} VALUES {}".format(
                    table, values_placeholder
                )
                precompute = [
                    (v,) if isinstance(v, int) or isinstance(v, str) else v
                    for v in values
                ]
                cursor.executemany(query, precompute)
                self.connection.commit()
                if self.verbose:
                    logger.debug("commited to table: {}".format(table))
            except Error as e:
                logger.debug("Error: {}".format(e))
                # self.connection.rollback()
            logger.debug("Values were inserted!")

    def _create_database(self):
        """Creates database."""
        self._execute_queries(self._get_db_schema())

    def _get_triples(self, subjects=None, objects=None, entities=None):
        """Get triples whose objects belong to objects and subjects to
        subjects, or, if not provided either object or subject, belong to
        entities.

        """
        if subjects is None and objects is None:
            if entities is None:
                msg = "You have to provide either subjects and objects\
                        indexes or general entities indexes!"

                logger.error(msg)
                raise Exception(msg)

            subjects = entities
            objects = entities
        if subjects is not None and objects is not None:
            query = "select * from triples_table where (subject in ({0}) and\
                    object in \
            ({1})) or (subject in ({1}) and object in ({0}));".format(
                ",".join(str(v) for v in subjects),
                ",".join(str(v) for v in objects),
            )
        elif objects is None:
            query = (
                "select * from triples_table where (subject in ({0}));".format(
                    ",".join(str(v) for v in subjects)
                )
            )
        elif subjects is None:
            query = (
                "select * from triples_table where (object in ({0}));".format(
                    ",".join(str(v) for v in objects)
                )
            )
        triples = np.array(self._execute_query(query))
        triples = np.append(
            triples[:, :3].astype("int"), triples[:, 3].reshape(-1, 1), axis=1
        )
        return triples

    def get_indexed_triples(self, chunk, dataset_type="train"):
        """Get indexed triples.

        Parameters
        ----------
        chunk: ndarray
             Numpy array with a fragment of data of size (N,3), where each
             element is: (subject, predicate, object).
        dataset_type: str
             Defines what kind of data we are considering
             (`"train"`, `"test"`, `"validation"`).

        Returns
        -------
        tmp: ndarray
             Numpy array of size (N, 4) with indexed triples, where each
             element is of the form
             (subject index, predicate index, object index, dataset_type).
        """
        if self.verbose:
            logger.debug("getting triples...")
        if isinstance(chunk, pd.DataFrame):
            chunk = chunk.values
        if self.use_indexer:
            # logger.debug(chunk)
            triples = self.mapper.get_indexes(chunk[:, :3])
            if self.data_shape > 3:
                weights = chunk[:, 3:]
                # weights = preprocess_focusE_weights(data=triples,
                #                                     weights=weights)
                return np.hstack(
                    [
                        triples,
                        weights,
                        np.array(len(triples) * [dataset_type]).reshape(-1, 1),
                    ]
                )
            return np.append(
                triples,
                np.array(len(triples) * [dataset_type]).reshape(-1, 1),
                axis=1,
            )
        else:
            return np.append(
                chunk,
                np.array(len(chunk) * [dataset_type]).reshape(-1, 1),
                axis=1,
            )

    def index_entities(self):
        """Index the data via the definition of the DataIndexer."""
        self.reload_data()
        if self.use_indexer is True:
            self.mapper = DataIndexer(
                self.data,
                backend="in_memory" if self.in_memory else "sqlite",
                root_directory=self.root_directory,
            )
        elif self.use_indexer is False:
            logger.debug("Data won't be indexed")
        elif isinstance(self.use_indexer, DataIndexer):
            self.mapper = self.use_indexer

    def is_indexed(self):
        """Check if the current data adapter has already been indexed.

        Returns
        -------
        Flag : bool
             Flag indicating whether indexing took place.

        """
        if not hasattr(self, "mapper"):
            return False
        return True

    def reload_data(self, verbose=False):
        """Reinitialise an iterator with data."""
        self.data = self.loader(self.data_source, chunk_size=self.chunk_size)
        if verbose:
            logger.debug("Data reloaded: {}".format(self.data))

    def populate(
        self,
        data_source,
        dataset_type="train",
        get_indexed_triples=None,
        loader=None,
    ):
        """Populate the database with data.

         Condition: before you can store triples, you have to index data.

        Parameters
        ----------
        data_source: ndarray or str
             Numpy array or file (e.g., csv file)  with data.
        dataset_type: str
             What type of data is it?
             options (`"train"` | `"test"` | `"validation"`).
        get_indexed_triples: func
             Function to obtain indexed triples.
        loader: func
             Loading function to be used to load data; if `None`, the
             `DataSourceIdentifier` will try to identify the type of
             ``data_source`` and return an adequate loader.

        """
        self.data_source = data_source
        self.loader = loader
        if loader is None:
            self.loader = self.identifier.fetch_loader()
        if not self.is_indexed() and self.use_indexer is not False:
            if self.verbose:
                logger.debug("indexing...")
            self.index_entities()
        else:
            logger.debug(
                "Data is already indexed or no\
                    indexing is required."
            )
        if get_indexed_triples is None:
            get_indexed_triples = self.get_indexed_triples
        data = self.loader(data_source, chunk_size=self.chunk_size)

        self.reload_data()
        for chunk in data:  # chunk is a numpy array of size (n,m) with m=3/4
            if chunk.shape[1] > 3:
                # weights = preprocess_focusE_weights(data=chunk[:, :3],
                # weights=chunk[:, 3:]) # weights normalization
                weights = chunk[:, 3:]
                chunk = np.concatenate([chunk[:, :3], weights], axis=1)
            self.data_shape = chunk.shape[1]
            values_triples = get_indexed_triples(
                chunk, dataset_type=dataset_type
            )
            self._insert_values_to_a_table("triples_table", values_triples)
        if self.verbose:
            logger.debug("data is populated")

        query = "SELECT count(*) from triples_table;"
        _ = self._execute_query(query)

        if isinstance(self.use_filter, dict):
            for key in self.use_filter:
                present_filters = [
                    x[0]
                    for x in self._execute_query(
                        "SELECT\
                        DISTINCT dataset_type FROM triples_table"
                    )
                ]
                if key not in present_filters:
                    # to allow users not to pass weights in test and validation
                    if (
                        self.data_shape > 3
                        and self.use_filter[key].shape[1] == 3
                    ):
                        nan_weights = np.empty(
                            (self.use_filter[key].shape[0], 1)
                        )
                        nan_weights.fill(np.nan)
                        self.use_filter[key] = np.concatenate(
                            [self.use_filter[key], nan_weights], axis=1
                        )
                    self.populate(self.use_filter[key], key)
        query = "SELECT count(*) from triples_table;"
        _ = self._execute_query(query)

    def get_data_size(self, table="triples_table", condition=""):
        """Gets the size of the given table [with specified condition].

        Parameters
        ----------
        table: str
             Table for which to obtain the size.
        condition: str
             Condition to count only a subset of data.

        Returns
        -------
        count: int
             Number of records in the table.

        """
        query = "SELECT count(*) from {} {};".format(table, condition)
        count = self._execute_query(query)
        if count is None:
            logger.debug("Table is empty or not such table exists.")
            return count
        elif not isinstance(count, list) or not isinstance(count[0], tuple):
            raise ValueError(
                "Cannot get count for the table with\
                    provided condition."
            )
        # logger.debug(count)
        return count[0][0]

    def clean_up(self):
        """Clean the database."""
        _ = self._execute_queries(self._get_clean_up())

    def remove_db(self):
        """Remove the database file."""
        os.remove(self.db_path)
        logger.debug("Database removed.")

    def _get_complementary_objects(self, triples, use_filter=None):
        """For a given triple retrieve all triples with same subjects
        and predicates.

           Parameters
           ----------
           triples: list or array
                List or array with Nx3 elements (subject, predicate, object).

           Returns
           -------
           objects : list
                Result of a query, list of objects.

        """
        results = []
        if self.use_filter is False or self.use_filter is None:
            self.use_filter = {"train": self.data}
        filtered = []
        valid_filters = [
            x[0]
            for x in self._execute_query(
                "SELECT DISTINCT\
                dataset_type FROM triples_table"
            )
        ]
        for filter_name, filter_source in self.use_filter.items():
            if filter_name in valid_filters:
                tmp_filter = []
                for triple in triples:
                    query = 'select distinct object from triples_table\
                            INDEXED BY triples_table_sp_idx where subject in \
                            ({}) and predicate in ({})  and dataset_type ="{}"'

                    query = query.format(triple[0], triple[1], filter_name)
                    q = self._execute_query(query)
                    tmp = list(set([y for x in q for y in x]))
                    tmp_filter.append(tmp)
                filtered.append(tmp_filter)
        # Unpack data into one list per triple no matter what filter
        # it comes from
        unpacked = list(zip(*filtered))
        for k in unpacked:
            lst = [j for i in k for j in i]
            results.append(lst)

        return results

    def _get_complementary_subjects(self, triples, use_filter=None):
        """For a given triple retrieve all triples with same objects
        and predicates.

           Parameters
           ----------
           triple: list or array
                List or array with elements (subject, predicate, object).

           Returns
           -------
           subjects : list
                Result of a query, list of subjects.

        """
        results = []
        if self.use_filter is False or self.use_filter is None:
            self.use_filter = {"train": self.data}

        filtered = []
        valid_filters = [
            x[0]
            for x in self._execute_query(
                "SELECT DISTINCT\
                dataset_type FROM triples_table"
            )
        ]
        for filter_name, filter_source in self.use_filter.items():
            if filter_name in valid_filters:
                tmp_filter = []
                for triple in triples:
                    query = 'select distinct subject from triples_table \
                            INDEXED BY triples_table_po_idx where predicate \
                            in ({})  and object in ({})\
                            and dataset_type ="{}"'

                    query = query.format(triple[1], triple[2], filter_name)
                    q = self._execute_query(query)
                    tmp = list(set([y for x in q for y in x]))
                    tmp_filter.append(tmp)
                filtered.append(tmp_filter)
        # Unpack data into one  list per triple no matter what
        # filter it comes from
        unpacked = list(zip(*filtered))
        for k in unpacked:
            lst = [j for i in k for j in i]
            results.append(lst)
        return results

    def _get_complementary_entities(self, triples, use_filter=None):
        """Returns the participating entities in the relation
        ?-p-o and s-p-?.

        Parameters
        ----------
        triples: ndarray of shape (N,3)
            Triples (s-p-o) that we are querying.

        Returns
        -------
        entities: list, list
                Two lists of subjects and objects participating in the
                relations ?-p-o and s-p-?.

        """
        objects = self._get_complementary_objects(
            triples, use_filter=use_filter
        )
        subjects = self._get_complementary_subjects(
            triples, use_filter=use_filter
        )
        return subjects, objects

    def _get_batch_generator(
        self, batch_size=1, dataset_type="train", random=False, index_by=""
    ):
        """Generator that returns the next batch of data.

        Parameters
        ----------
        dataset_type: str
            Indicates which dataset to use
            (`"train"` | `"test"` | `"validation"`).
        batch_size: int
            Number of elements in a batch (default: :math:`1`).
        index_by: str
            Possible values:  `{"", "so", "os", "s", "o"}`. It indicates
            whether to use index and which to use:
            index by subject (`"s"`), object (`"o"`) or both (`"so"`, `"os"`).
            Indexes were created for the fields so SQLite should use them
            here to speed up, see example below:
           sqlite> EXPLAIN QUERY PLAN SELECT * FROM triples_table\
                   ORDER BY subject, object LIMIT 7000, 30;
           QUERY PLAN
           `--SCAN TABLE triples_table USING INDEX triples_table_sub_obj_idx
        random: bool
            Whether to get records from database in a random order.

        Yields
        -------
        batch_output : ndarray
            Yields a batch of triples from the dataset type specified
        participating_entities : list
            List of all entities that were involved in the s-p-? and
            ?-p-o relations. This is returned only if ``use_filter=True``.

        """
        if not isinstance(dataset_type, str):
            dataset_type = dataset_type.decode("utf-8")
        cond = f"where dataset_type ='{dataset_type}'"
        size = self.get_data_size(condition=cond)
        # focusE: size ppi55k = 230929
        self.batches_count = int(np.ceil(size / batch_size))
        logger.debug("batches count: {}".format(self.batches_count))
        logger.debug("size of data: {}".format(size))
        index = ""
        if index_by != "":
            if (
                index_by == "s"
                or index_by == "o"
                or index_by == "so"
                or index_by == "os"
            ) and random:
                msg = "Field index_by can only be used with random set\
                        to False and can only take values from this\
                        set: {{s,o,so,os,''}},\
                        instead got: {}".format(
                    index_by
                )
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
                 triples_table_type_idx where\
                 dataset_type ='{}' {} LIMIT {}, {};"

        for i in range(self.batches_count):
            # logger.debug("BATCH NUMBER: {}".format(i))
            # logger.debug(i * batch_size)
            query = query_template.format(
                dataset_type, index, i * batch_size, batch_size
            )
            # logger.debug(query)
            out = self._execute_query(query)
            # logger.debug(out)
            if out:
                triples = np.array(out)[:, :3].astype(np.int32)
                # focusE
                if self.data_shape > 3:
                    weights = np.array(out)[:, 3:-1]

            else:
                weights = np.array([])

            if self.use_filter:
                # get the filter values
                participating_entities = self._get_complementary_entities(
                    triples
                )
                if self.data_shape > 3:
                    yield triples, tf.ragged.constant(participating_entities),
                    weights
                else:
                    yield triples, tf.ragged.constant(participating_entities)
            else:
                if self.data_shape > 3:
                    yield triples, weights
                else:
                    yield triples

    def summary(self, count=True):  # FocusE fix types
        """Prints summary of the database.

        The information that is displayed is: whether it exists, what tables
        does it have, how many records it contains (if ``count=True``),
        what are fields held and their types with an example record.

        Parameters
        ----------
        count: bool
             Whether to count number of records per table
             (can be time consuming).

        Example
        -------
        >>> adapter = SQLiteAdapter("database_24-06-2020_03-51-12_PM.db")
        >>> with adapter as db:
        >>>     db.summary()
        Summary for Database database_29-06-2020_09-37-20_AM.db
        File size: 3.9453MB
        Tables: triples_table
        +-----------------------------------------------------------------------------------+
        |                                TRIPLES_TABLE                                      |
        +---------------------+------------------+---------------+--------------------------+
        |    | subject (int)  | predicate (int)  | object (int)  | dataset_type (text(50))  |
        +----+----------------+------------------+---------------+--------------------------+
        |e.g.| 34321          | 29218            | 38102         | train                    |
        +----+----------------+------------------+---------------+--------------------------+
        Records: 59070

        """
        if os.path.exists(self.db_path):
            print("Summary for Database {}".format(self.db_name))
            print("Located in {}".format(self.db_path))
            file_size = os.path.getsize(self.db_path)
            summary = """File size: {:.5}{}\nTables: {}"""
            tables = self._execute_query(
                "SELECT name FROM sqlite_master\
                    WHERE type='table';"
            )
            tables_names = ", ".join(table[0] for table in tables)
            print(
                summary.format(
                    *get_human_readable_size(file_size), tables_names
                )
            )
            types = {"integer": "int", "float": "float", "string": "str"}
            # float aggiunto per focusE
            for table_name in tables:
                result = self._execute_query(
                    "PRAGMA table_info('%s')" % table_name
                )
                cols_name_type = [
                    "{} ({}):".format(
                        x[1], types[x[2]] if x[2] in types else x[2]
                    )
                    for x in result
                ]  # FocusE
                length = len(cols_name_type)
                print(
                    "-------------\n|"
                    + table_name[0].upper()
                    + "|\n-------------\n"
                )
                formatted_record = "{:7s}{}\n{:7s}{}".format(
                    " ", "{:25s}" * length, "e.g.", "{:<25s}" * length
                )
                msg = ""
                example = ["-"] * length
                if count:
                    nb_records = self.get_data_size(table_name[0])
                    msg = "\n\nRecords: {}".format(nb_records)
                    if nb_records != 0:
                        record = self._execute_query(
                            f"SELECT * FROM\
                                {table_name[0]} LIMIT {1};"
                        )[0]
                        example = [str(rec) for rec in record]
                else:
                    print("Count is set to False hence no data displayed")

                print(formatted_record.format(*cols_name_type, *example), msg)
        else:
            logger.debug("Database does not exist.")

    def _load(self, data_source, dataset_type="train"):
        """Loads data from the data source to the database. Wrapper
        around populate method, required by the GraphDataLoader interface.

        Parameters
        ----------
        data_source: str or ndarray
             Numpy array or path to a file (e.g. csv file) from where
             to read data.
        dataset_type: str
             Kind of dataset that is being loaded
             (`"train"` | `"test"` | `"validation"`).

        """

        self.data_source = data_source
        self.populate(self.data_source, dataset_type=dataset_type)

    def _intersect(self, dataloader):
        if not isinstance(dataloader.backend, SQLiteAdapter):
            msg = "Provided dataloader should be of type SQLiteAdapter\
                    backend, instead got {}.".format(
                type(dataloader.backend)
            )
            logger.error(msg)
            raise Exception(msg)
        raise NotImplementedError

    def _clean(self):
        os.remove(self.db_path)
        self.mapper.clean()
