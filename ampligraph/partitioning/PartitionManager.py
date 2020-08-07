class PartitionManager():
    def __init__(self, dataset, strategy, k, num_devices=1):
        
        self.num_buckets = 
        self.num_partitions = 
        # create partitions
        partitioner = PARTITION_ALGO_REGISTRY.get(strategy)(dataset, k=num_buckets)
        
        
        # needs to go into the shelves
        self.entity_embeddings = xavier(num_entities, k)
        self.rel_embeddings = xavier(num_rels, k)
        
        # needs to go into the shelves
        self.optimizer_hyperparams_ent = np.zeros(shape=(self.num_ents, 2, self.k), 
                                                  dtype=np.float32)
        self.optimizer_hyperparams_rel = np.zeros(shape=(self.num_rels, 2, self.k), 
                                                  dtype=np.float32)
        
        self.num_devices = 1
        
        
        def get_next_partition(self):
            ''' This function returns an iterator which can be used to iterate over partitions.'''
            for i, partitioned_data_iterator in enumerate(partitioner):
                # Load the shelf related to the the buckets of the partitions and return the corresponding embeddings.
                
                yield partitioned_data_iterator, ent_emb, rel_emb
                
                
        def update_trained_partion_embeddings(self):
            pass
        
        def change_partition(self):
            pass
        
        def get_next_batch(self)
            pass