
import numpy as np
from ..datasets import AmpligraphDatasetAdapter

import sqlite3
import time

class SQLiteAdapter(AmpligraphDatasetAdapter):
    '''
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
    
    '''
    def __init__(self, existing_db_name=None, ent_to_idx=None, rel_to_idx=None):
        super(SQLiteAdapter, self).__init__()
        self.mapped_status = {}
        self.persistance_status = {}
        self.dbname = existing_db_name
        self.using_existing_db = False
        if self.dbname is not None:
            assert(self.rel_to_idx is not None)
            assert(self.ent_to_idx is not None)
            
            self.using_existing_db = True
            self.rel_to_idx = rel_to_idx
            self.ent_to_idx = ent_to_idx
            
    def get_db_name(self):
        return self.dbname
    
    def _create_schema(self):
        if self.using_existing_db:
            return
        if self.dbname is not None:
            self.cleanup()
        self.dbname = 'Ampligraph_{}.db'.format(int(time.time()))

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

        conn.commit()
        
        cur.execute('INSERT INTO integrity_check VALUES (0)')
        conn.commit()
        conn.close()
    
    def generate_mappings(self, use_all=False, regenerate=False):
        if (len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0 or regenerate==True) and not self.using_existing_db:
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
        #TODO: can change it to just use the values of the dictionary
        pg_entity_values = np.arange(len(self.ent_to_idx)).reshape(-1,1).tolist()
        conn = sqlite3.connect("{}".format(self.dbname))
        cur = conn.cursor()
        try:
            cur.executemany('INSERT INTO entity_table VALUES (?)', pg_entity_values)
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            
        conn.close()
    
    def use_mappings(self, rel_to_idx, ent_to_idx):
        if self.using_existing_db:
            raise Exception('Cannot change the mappings for an existing DB')
        super().use_mappings(rel_to_idx, ent_to_idx)
        self._create_schema()
        for key in self.dataset.keys():
            self.mapped_status[key] = False
            #TODO: drop and recreate tables
            self.persistance_status[key] = False
            
        self._insert_entities_in_db()
        
        
    def get_size(self, dataset_type="train"):
        select_query = "SELECT count(*) from triples_table where dataset_type ='{}'"
        conn = sqlite3.connect("{}".format(self.dbname))         
        cur1 = conn.cursor()
        cur1.execute(select_query.format(dataset_type))
        out = cur1.fetchall()
        return out[0][0]
    
    def get_next_train_batch(self, batch_size, dataset_type="train"):
        if (not self.using_existing_db) and self.mapped_status[dataset_type] == False:
            self.map_data()
            
        batches_count = int(np.ceil(self.get_size(dataset_type)/batch_size))
        select_query = "SELECT subject, predicate,object FROM triples_table INDEXED BY triples_table_type_idx where dataset_type ='{}' LIMIT {}, {}"
        for i in range(batches_count):
            conn = sqlite3.connect("{}".format(self.dbname))         
            cur1 = conn.cursor()
            cur1.execute(select_query.format(dataset_type, i*batch_size, batch_size))
            out = np.array(cur1.fetchall(), dtype=np.int32)
            cur1.close()
            yield out
            
    def get_next_eval_batch(self, batch_size, dataset_type="test"):
        if (not self.using_existing_db) and self.mapped_status[dataset_type] == False:
            self.map_data()
            
        batches_count = int(np.ceil(self.get_size(dataset_type)/batch_size))
        select_query = "SELECT subject, predicate,object FROM triples_table INDEXED BY triples_table_type_idx where dataset_type ='{}' LIMIT {}, {}"
        for i in range(batches_count):
            conn = sqlite3.connect("{}".format(self.dbname))         
            cur1 = conn.cursor()
            cur1.execute(select_query.format(dataset_type, i*batch_size, batch_size))
            out = np.array(cur1.fetchall(), dtype=np.int32)
            cur1.close()
            yield out
    
    def get_next_batch_with_filter(self, batch_size=1, dataset_type="test"):
        if (not self.using_existing_db) and self.mapped_status[dataset_type] == False:
            self.map_data()
            
        batches_count = int(np.ceil(self.get_size(dataset_type)/batch_size))
        select_query = "SELECT subject, predicate,object FROM triples_table INDEXED BY triples_table_type_idx where dataset_type = '{}' LIMIT {}, {}"
        for i in range(batches_count):
            conn = sqlite3.connect("{}".format(self.dbname))         
            cur1 = conn.cursor()
            cur1.execute(select_query.format(dataset_type, i*batch_size, batch_size))
            out = np.array(cur1.fetchall(), dtype=np.int32)
            cur1.close()
            participating_objects, participating_subjects = self.get_participating_entities(out)
            yield out, participating_objects, participating_subjects
            
    def _insert_triples(self, triples , key=""):
        conn = sqlite3.connect("{}".format(self.dbname))
        key= np.array([[key]])
        for j in range(int(np.ceil(triples.shape[0]/500000.0))):
            pg_triple_values = triples[j*500000:(j+1)*500000]
            pg_triple_values = np.concatenate((pg_triple_values, np.repeat(key, pg_triple_values.shape[0], axis=0)), axis=1)
            pg_triple_values = pg_triple_values.tolist()
            cur = conn.cursor()
            cur.executemany('INSERT INTO triples_table VALUES (?,?,?,?)', pg_triple_values)
            conn.commit()
            
        conn.close()
    
    def map_data(self, remap=False):
        if self.using_existing_db:
            return
        from ..evaluation import to_idx
        if len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0:
            self.generate_mappings()
            
        for key in self.dataset.keys():
            if isinstance(self.dataset[key], np.ndarray):
                if self.mapped_status[key] == False or remap == True:
                    self.dataset[key] = to_idx(self.dataset[key], 
                                               ent_to_idx=self.ent_to_idx, 
                                               rel_to_idx=self.rel_to_idx)
                    self.mapped_status[key] = True
                if self.persistance_status[key] == False:
                    self._insert_triples(self.dataset[key], key)
                    self.persistance_status[key] = True
                    
        conn = sqlite3.connect("{}".format(self.dbname))   
        cur = conn.cursor()
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
            
        conn.close()
                
            
    def _validate_data(self, data):
        if type(data) != np.ndarray:
            msg = 'Invalid type for input data. Expected ndarray, got {}'.format(type(data))
            raise ValueError(msg)

        if (np.shape(data)[1]) != 3:
            msg = 'Invalid size for input data. Expected number of column 3, got {}'.format(np.shape(data)[1])
            raise ValueError(msg)
            
    def set_data(self, dataset, dataset_type=None, mapped_status=False, persistance_status=False):
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
        x_triple = np.squeeze(x_triple)
        conn = sqlite3.connect("{}".format(self.dbname))         
        cur1 = conn.cursor()
        cur2 = conn.cursor()
        cur_integrity = conn.cursor()
        cur_integrity.execute("SELECT * FROM integrity_check")
        if cur_integrity.fetchone()[0] == 0:
            raise Exception('Data integrity is corrupted. The tables have been modified.')

        query1 = "select object from triples_table INDEXED BY triples_table_sp_idx where subject=" + str(x_triple[0]) + \
                    " and predicate="+ str(x_triple[1])
        query2 = "select subject from triples_table INDEXED BY triples_table_po_idx where predicate=" + str(x_triple[1]) + \
                    " and object="+ str(x_triple[2])
        
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
        
        conn.close()
        
        return ent_participating_as_objects, ent_participating_as_subjects
        
    def cleanup(self):
        if self.using_existing_db:
            self.dbname = None
            self.using_existing_db=False
            return
        
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
        conn.commit()
        conn.execute("VACUUM")
        conn.close()
        self.dbname = None