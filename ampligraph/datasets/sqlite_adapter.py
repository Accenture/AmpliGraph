# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
from ..datasets import AmpligraphDatasetAdapter
import tempfile
import sqlite3
import time
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SQLiteAdapter(AmpligraphDatasetAdapter):
    '''SQLLite adapter
    '''

    def __init__(self, existing_db_name=None, ent_to_idx=None, rel_to_idx=None):
        """Initialize the class variables
        Parameters
        ----------
        existing_db_name : string
            Name of an existing database to use.
            Assumes that the database has schema as required by the adapter and the persisted data is already mapped
        ent_to_idx : dictionary of mappings
            Mappings of entity to idx
        rel_to_idx : dictionary of mappings
            Mappings of relation to idx
        """
        super(SQLiteAdapter, self).__init__()
        # persistance status of the data
        self.persistance_status = {}
        self.dbname = existing_db_name
        # flag indicating whether we are using existing db
        self.using_existing_db = False
        self.temp_dir = None
        if self.dbname is not None:
            # If we are using existing db then the mappings need to be passed
            assert (self.rel_to_idx is not None)
            assert (self.ent_to_idx is not None)

            self.using_existing_db = True
            self.rel_to_idx = rel_to_idx
            self.ent_to_idx = ent_to_idx

    def get_db_name(self):
        """Returns the db name
        """
        return self.dbname

    def _create_schema(self):
        """Creates the database schema
        """
        if self.using_existing_db:
            return
        if self.dbname is not None:
            self.cleanup()

        self.temp_dir = tempfile.TemporaryDirectory(suffix=None, prefix='ampligraph_', dir=None)
        self.dbname = os.path.join(self.temp_dir.name, 'Ampligraph_{}.db'.format(int(time.time())))

        conn = sqlite3.connect("{}".format(self.dbname))
        cur = conn.cursor()
        cur.execute("CREATE TABLE entity_table (entity_type integer primary key);")
        cur.execute("CREATE TABLE triples_table (subject integer, \
                                                    predicate integer, \
                                                    object integer, \
                                                    dataset_type text(50), \
                                                    foreign key (object) references entity_table(entity_type), \
                                                    foreign key (subject) references entity_table(entity_type) \
                                                    );")

        cur.execute("CREATE INDEX triples_table_sp_idx ON triples_table (subject, predicate);")
        cur.execute("CREATE INDEX triples_table_po_idx ON triples_table (predicate, object);")
        cur.execute("CREATE INDEX triples_table_type_idx ON triples_table (dataset_type);")

        cur.execute("CREATE TABLE integrity_check (validity integer primary key);")

        cur.execute('INSERT INTO integrity_check VALUES (0)')
        conn.commit()
        cur.close()
        conn.close()

    def generate_mappings(self, use_all=False, regenerate=False):
        """Generate mappings from either train set or use all dataset to generate mappings
        Parameters
        ----------
        use_all : boolean
            If True, it generates mapping from all the data. If False, it only uses training set to generate mappings
        regenerate : boolean
            If true, regenerates the mappings.
            If regenerating, then the database is created again(to conform to new mapping)
        Returns
        -------
        rel_to_idx : dictionary
            Relation to idx mapping dictionary
        ent_to_idx : dictionary
            entity to idx mapping dictionary
        """
        if (len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0 or (regenerate is True)) \
                and (not self.using_existing_db):
            from ..evaluation import create_mappings
            self._create_schema()
            if use_all:
                complete_dataset = []
                for key in self.dataset.keys():
                    complete_dataset.append(self.dataset[key])
                self.rel_to_idx, self.ent_to_idx = create_mappings(np.concatenate(complete_dataset, axis=0))

            else:
                self.rel_to_idx, self.ent_to_idx = create_mappings(self.dataset["train"])

            self._insert_entities_in_db()
        return self.rel_to_idx, self.ent_to_idx

    def _insert_entities_in_db(self):
        """Inserts entities in the database
        """
        # TODO: can change it to just use the values of the dictionary
        pg_entity_values = np.arange(len(self.ent_to_idx)).reshape(-1, 1).tolist()
        conn = sqlite3.connect("{}".format(self.dbname))
        cur = conn.cursor()
        try:
            cur.executemany('INSERT INTO entity_table VALUES (?)', pg_entity_values)
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
        cur.close()
        conn.close()

    def use_mappings(self, rel_to_idx, ent_to_idx):
        """Use an existing mapping with the datasource.
        """
        # cannot change mappings for an existing database.
        if self.using_existing_db:
            raise Exception('Cannot change the mappings for an existing DB')
        super().use_mappings(rel_to_idx, ent_to_idx)
        self._create_schema()

        for key in self.dataset.keys():
            self.mapped_status[key] = False
            self.persistance_status[key] = False

        self._insert_entities_in_db()

    def get_size(self, dataset_type="train"):
        """Returns the size of the specified dataset
        Parameters
        ----------
        dataset_type : string
            type of the dataset

        Returns
        -------
        size : int
            size of the specified dataset
        """
        select_query = "SELECT count(*) from triples_table where dataset_type ='{}'"
        conn = sqlite3.connect("{}".format(self.dbname))
        cur1 = conn.cursor()
        cur1.execute(select_query.format(dataset_type))
        out = cur1.fetchall()
        cur1.close()
        return out[0][0]

    def get_next_batch(self, batches_count=-1, dataset_type="train", use_filter=False):
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
        if (not self.using_existing_db) and (not self.mapped_status[dataset_type]):
            self.map_data()

        if batches_count == -1:
            batch_size = 1
            batches_count = self.get_size(dataset_type)
        else:
            batch_size = int(np.ceil(self.get_size(dataset_type) / batches_count))

        select_query = "SELECT subject, predicate,object FROM triples_table INDEXED BY \
                            triples_table_type_idx where dataset_type ='{}' LIMIT {}, {}"

        for i in range(batches_count):
            conn = sqlite3.connect("{}".format(self.dbname))
            cur1 = conn.cursor()
            cur1.execute(select_query.format(dataset_type, i * batch_size, batch_size))
            out = np.array(cur1.fetchall(), dtype=np.int32)
            cur1.close()
            if use_filter:
                # get the filter values
                participating_objects, participating_subjects = self.get_participating_entities(out)
                yield out, participating_objects, participating_subjects
            else:
                yield out

    def _insert_triples(self, triples, key=""):
        """inserts triples in the database for the specified category
        """
        conn = sqlite3.connect("{}".format(self.dbname))
        key = np.array([[key]])
        for j in range(int(np.ceil(triples.shape[0] / 500000.0))):
            pg_triple_values = triples[j * 500000:(j + 1) * 500000]
            pg_triple_values = np.concatenate((pg_triple_values, np.repeat(key,
                                                                           pg_triple_values.shape[0], axis=0)), axis=1)
            pg_triple_values = pg_triple_values.tolist()
            cur = conn.cursor()
            cur.executemany('INSERT INTO triples_table VALUES (?,?,?,?)', pg_triple_values)
            conn.commit()
            cur.close()

        conn.close()

    def map_data(self, remap=False):
        """map the data to the mappings of ent_to_idx and rel_to_idx
        Parameters
        ----------
        remap : boolean
            remap the data, if already mapped. One would do this if the dictionary is updated.
        """
        if self.using_existing_db:
            # since the assumption is that the persisted data is already mapped for an existing db
            return
        from ..evaluation import to_idx
        if len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0:
            self.generate_mappings()

        for key in self.dataset.keys():
            if isinstance(self.dataset[key], np.ndarray):
                if (not self.mapped_status[key]) or (remap is True):
                    self.dataset[key] = to_idx(self.dataset[key],
                                               ent_to_idx=self.ent_to_idx,
                                               rel_to_idx=self.rel_to_idx)
                    self.mapped_status[key] = True
                if not self.persistance_status[key]:
                    self._insert_triples(self.dataset[key], key)
                    self.persistance_status[key] = True

        conn = sqlite3.connect("{}".format(self.dbname))
        cur = conn.cursor()
        # to maintain integrity of data
        cur.execute('Update integrity_check set validity=1 where validity=0')
        conn.commit()

        cur.execute('''CREATE TRIGGER IF NOT EXISTS triples_table_ins_integrity_check_trigger
                        AFTER INSERT ON triples_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                            ;
                    ''')
        cur.execute('''CREATE TRIGGER IF NOT EXISTS triples_table_upd_integrity_check_trigger
                        AFTER UPDATE ON triples_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                            ;
                    ''')
        cur.execute('''CREATE TRIGGER IF NOT EXISTS triples_table_del_integrity_check_trigger
                        AFTER DELETE ON triples_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                            ;
                    ''')

        cur.execute('''CREATE TRIGGER IF NOT EXISTS entity_table_upd_integrity_check_trigger
                        AFTER UPDATE ON entity_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                        ;
                    ''')
        cur.execute('''CREATE TRIGGER IF NOT EXISTS entity_table_ins_integrity_check_trigger
                        AFTER INSERT ON entity_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                        ;
                    ''')
        cur.execute('''CREATE TRIGGER IF NOT EXISTS entity_table_del_integrity_check_trigger
                        AFTER DELETE ON entity_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                        ;
                    ''')
        cur.close()
        conn.close()

    def _validate_data(self, data):
        """validates the data
        """
        if type(data) != np.ndarray:
            msg = 'Invalid type for input data. Expected ndarray, got {}'.format(type(data))
            raise ValueError(msg)

        if (np.shape(data)[1]) != 3:
            msg = 'Invalid size for input data. Expected number of column 3, got {}'.format(np.shape(data)[1])
            raise ValueError(msg)

    def set_data(self, dataset, dataset_type=None, mapped_status=False, persistance_status=False):
        """set the dataset based on the type.
            Note: If you pass the same dataset type it will be appended

            #Usage for extremely large datasets:
            from ampligraph.datasets import SQLiteAdapter
            adapt = SQLiteAdapter()

            #compute the mappings from the large dataset.
            #Let's assume that the mappings are already computed in rel_to_idx, ent_to_idx.
            #Set the mappings
            adapt.use_mappings(rel_to_idx, ent_to_idx)

            #load and store parts of data in the db as train test or valid
            #if you have already mapped the entity names to index, set mapped_status = True
            adapt.set_data(load_part1, 'train', mapped_status = True)
            adapt.set_data(load_part2, 'train', mapped_status = True)
            adapt.set_data(load_part3, 'train', mapped_status = True)

            #if mapped_status = False, then the adapter will map the entities to index before persisting
            adapt.set_data(load_part1, 'test', mapped_status = False)
            adapt.set_data(load_part2, 'test', mapped_status = False)

            adapt.set_data(load_part1, 'valid', mapped_status = False)
            adapt.set_data(load_part2, 'valid', mapped_status = False)

            #create the model
            model = ComplEx(batches_count=10000, seed=0, epochs=10, k=50, eta=10)
            model.fit(adapt)

        Parameters
        ----------
        dataset : nd-array or dictionary
            dataset of triples
        dataset_type : string
            if the dataset parameter is an nd- array then this indicates the type of the data being based
        mapped_status : bool
            indicates whether the data has already been mapped to the indices
        persistance_status : bool
            indicates whether the data has already been written to the database
        """
        if self.using_existing_db:
            raise Exception('Cannot change the existing DB')

        if isinstance(dataset, dict):
            for key in dataset.keys():
                self._validate_data(dataset[key])
                self.dataset[key] = dataset[key]
                self.mapped_status[key] = mapped_status
                self.persistance_status[key] = persistance_status
        elif dataset_type is not None:
            self._validate_data(dataset)
            self.dataset[dataset_type] = dataset
            self.mapped_status[dataset_type] = mapped_status
            self.persistance_status[dataset_type] = persistance_status
        else:
            raise Exception("Incorrect usage. Expected a dictionary or a combination of dataset and it's type.")

        if not (len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0):
            self.map_data()

    def get_participating_entities(self, x_triple):
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
        x_triple = np.squeeze(x_triple)
        conn = sqlite3.connect("{}".format(self.dbname))
        cur1 = conn.cursor()
        cur2 = conn.cursor()
        cur_integrity = conn.cursor()
        cur_integrity.execute("SELECT * FROM integrity_check")

        if cur_integrity.fetchone()[0] == 0:
            raise Exception('Data integrity is corrupted. The tables have been modified.')

        query1 = "select " + str(x_triple[2]) + " union select distinct object from triples_table INDEXED BY \
                    triples_table_sp_idx  where subject=" + str(x_triple[0]) + " and predicate=" + str(x_triple[1])
        query2 = "select  " + str(x_triple[0]) + " union select distinct subject from triples_table INDEXED BY \
                    triples_table_po_idx where predicate=" + str(x_triple[1]) + " and object=" + str(x_triple[2])

        cur1.execute(query1)
        cur2.execute(query2)

        ent_participating_as_objects = np.array(cur1.fetchall())
        ent_participating_as_subjects = np.array(cur2.fetchall())
        '''
        if ent_participating_as_objects.ndim>=1:
            ent_participating_as_objects = np.squeeze(ent_participating_as_objects)

        if ent_participating_as_subjects.ndim>=1:
            ent_participating_as_subjects = np.squeeze(ent_participating_as_subjects)
        '''
        cur1.close()
        cur2.close()
        cur_integrity.close()
        conn.close()

        return ent_participating_as_objects, ent_participating_as_subjects

    def cleanup(self):
        """Clean up the database
        """
        if self.using_existing_db:
            # if using an existing db then dont remove
            self.dbname = None
            self.using_existing_db = False
            return

        # Drop the created tables
        if self.dbname is not None:
            conn = sqlite3.connect("{}".format(self.dbname))
            cur = conn.cursor()
            cur.execute("drop trigger IF EXISTS entity_table_del_integrity_check_trigger")
            cur.execute("drop trigger IF EXISTS entity_table_ins_integrity_check_trigger")
            cur.execute("drop trigger IF EXISTS entity_table_upd_integrity_check_trigger")

            cur.execute("drop trigger IF EXISTS triples_table_del_integrity_check_trigger")
            cur.execute("drop trigger IF EXISTS triples_table_upd_integrity_check_trigger")
            cur.execute("drop trigger IF EXISTS triples_table_ins_integrity_check_trigger")
            cur.execute("drop table IF EXISTS integrity_check")
            cur.execute("drop index IF EXISTS triples_table_po_idx")
            cur.execute("drop index IF EXISTS triples_table_sp_idx")
            cur.execute("drop index IF EXISTS triples_table_type_idx")
            cur.execute("drop table IF EXISTS triples_table")
            cur.execute("drop table IF EXISTS entity_table")
            cur.close()
            conn.close()
            try:
                if self.temp_dir is not None:
                    self.temp_dir.cleanup()
            except OSError:
                logger.warning('Unable to remove the created temperory files.')
                logger.warning('Filename:{}'.format(self.dbname))

            self.dbname = None
