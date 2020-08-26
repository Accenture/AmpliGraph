from ampligraph.datasets import SQLiteAdapter
from ampligraph.datasets import GraphDataLoader
from ampligraph.datasets.graph_partitioner import PARTITION_ALGO_REGISTRY, AbstractGraphPartitioner
import numpy as np
import shelve   

class PartitionedDataManager():
    def __init__(self, dataset_loader, model, epochs=1):
        
        self._model = model
        self.k = self._model.k
        self.eta = self._model.eta
        self.num_ents = 14505
        self.num_rels = 237
        self.max_ent_size = 10000
        
        self._inferred_steps = None
        self._initial_epoch = 0

        strategy='Bucket'
        self.num_buckets = 3
        self._epochs=epochs

        if isinstance(dataset_loader, AbstractGraphPartitioner):
            self.partitioner = dataset_loader
        else:
            print('Partitioning may take a while...')
            self.partitioner = PARTITION_ALGO_REGISTRY.get(strategy)(dataset_loader, k=num_buckets)
        self._generate_partition_params()
        
        
    def _generate_partition_params(self):
        ''' Generates the metadata needed for persisting and loading partition embeddings and other params'''
        
        chunk_size = 10000
        def xavier(in_shape, out_shape, part_in_shape=None):
            if part_in_shape is None:
                part_in_shape = in_shape,
            std = np.sqrt(2 / (in_shape + out_shape))
            return np.random.normal(0, std, size=(part_in_shape, out_shape)).astype(np.float32)
        
        # create entity embeddings and optimizer hyperparams for all entities
        for i in range(self.num_buckets):
            with shelve.open('ent_partition', writeback=True) as ent_partition:
                with shelve.open(self.partitioner.files[i]) as bucket:
    
                    out_dict_keys = str(i)
                    num_ents_bucket = bucket['indexes'].shape[0]
                    opt_param = np.zeros(shape=(num_ents_bucket, 2, self.k), dtype=np.float32)
                    ent_emb = xavier(self.num_ents, self.k, num_ents_bucket)
                    ent_partition.update({out_dict_keys: [opt_param, ent_emb]})
         
        # create relation embeddings and optimizer hyperparams for all relations
        for i in range(1):
            with shelve.open('rel_partition', writeback=True) as rel_partition:
                out_dict_keys = str(i)
                opt_param = np.zeros(shape=(self.num_rels, 2, self.k), dtype=np.float32)
                rel_emb = xavier(self.num_rels, self.k, self.num_rels)
                rel_partition.update({out_dict_keys: [opt_param, rel_emb]})
                
        # for every partition
        for i in range(len(self.partitioner.partitions)):
            # get the source and dest bucket
            splits = self.partitioner.partitions[i].backend.mapper.metadata['name'].split('-')
            source_bucket = splits[0][-1]
            dest_bucket = splits[1]
            all_keys_merged_buckets = []
            # get all the unique entities present in the partition
            with shelve.open(self.partitioner.files[int(source_bucket)]) as bucket:
                all_keys_merged_buckets.extend(bucket['indexes'])
            if source_bucket != dest_bucket: 
                with shelve.open(self.partitioner.files[int(dest_bucket)]) as bucket:
                    all_keys_merged_buckets.extend(bucket['indexes'])


            # since we would be concatenating the bucket embeddings, let's find what 0, 1, 2 etc indices of 
            # embedding matrix means.
            # bucket entity value to ent_emb matrix index mappings eg: 2001 -> 0, 2002->1, 2003->2, ...
            merged_bucket_to_ent_mat_mappings = {}
            for key, val in zip(all_keys_merged_buckets, np.arange(0, len(all_keys_merged_buckets))):
                merged_bucket_to_ent_mat_mappings[key] = val
            #print(merged_bucket_to_ent_mat_mappings)
            emb_mat_order = []

            # partitions do not contain all entities of the bucket they belong to.
            # they will produce data from 0->n idx. So we need to remap the get position of the 
            # entities of the partition in the concatenated emb matrix
            # data_index -> original_ent_index -> ent_emb_matrix mappings (a->b->c) 0->2002->1, 1->2003->2 
            # (because 2001 may not exist in this partition)
            with shelve.open(self.partitioner.partitions[i].backend.mapper.metadata['entities_shelf']) as ent_sh:
                sorted_partition_keys = np.sort(np.array(list(ent_sh.keys())).astype(np.int32))
                # a : 0 to n
                for key in sorted_partition_keys:
                    # a->b mapping
                    a_to_b = int(ent_sh[str(key)])
                    # a->b->c mapping
                    emb_mat_order.append(merged_bucket_to_ent_mat_mappings[a_to_b])

            # store it 
            with shelve.open('ent_partition_metadata', writeback=True) as metadata:
                metadata[str(i)] = emb_mat_order      
                
            rel_mat_order = []
            with shelve.open(self.partitioner.partitions[i].backend.mapper.metadata['relations']) as rel_sh:
                sorted_partition_keys = np.sort(np.array(list(rel_sh.keys())).astype(np.int32))
                # a : 0 to n
                for key in sorted_partition_keys:
                    # a->b mapping
                    rel_mat_order.append(int(rel_sh[str(key)]))

            with shelve.open('rel_partition_metadata', writeback=True) as metadata:
                metadata[str(i)] = rel_mat_order      


    def update_partion_embeddings(self, graph_data_loader, partition_number):
        '''Persists the embeddings and other params after the partition is trained'''
        self.all_ent_embs[self.ent_original_ids] = \
            self._model.encoding_layer.ent_emb.numpy()[:len(self.ent_original_ids), :]
        self.all_rel_embs[self.rel_original_ids] = \
            self._model.encoding_layer.rel_emb.numpy()[:len(self.rel_original_ids), :]

        opt_weights = self._model.optimizer.get_weights()

        if len(opt_weights)>0:

            self.all_rel_opt_params[self.rel_original_ids, :, :] = \
                np.concatenate([opt_weights[2][:len(self.rel_original_ids)][:, np.newaxis, :], 
                                opt_weights[4][:len(self.rel_original_ids)][:, np.newaxis, :]], 1)
            self.all_ent_opt_params[self.ent_original_ids, :, :] = \
                np.concatenate([opt_weights[1][:len(self.ent_original_ids)][:, np.newaxis, :], 
                                opt_weights[3][:len(self.ent_original_ids)][:, np.newaxis, :]], 1)
            
        # Open the buckets related to the partition and concat
        splits = graph_data_loader.backend.mapper.metadata['name'].split('-')
        source_bucket = splits[0][-1]
        dest_bucket = splits[1]
        
        try:
            s = shelve.open('ent_partition', writeback=True)
            source_bucket_params = s[source_bucket]
            dest_source_bucket_params = s[dest_bucket]

            # split and save self.all_ent_opt_params and self.all_ent_embs into respective buckets

            opt_params = [self.all_ent_opt_params[:self.split_opt_idx],
                          self.all_ent_opt_params[self.split_opt_idx:]]
            emb_params = [self.all_ent_embs[:self.split_emb_idx],
                          self.all_ent_embs[self.split_emb_idx:]]
            
            s[source_bucket] = [opt_params[0], emb_params[0]]
            s[dest_bucket] = [opt_params[1], emb_params[1]]
            
        finally:
            s.close()
            
        try:
            
            s = shelve.open('rel_partition', writeback=True)
            s['0'] = [self.all_rel_opt_params, self.all_rel_embs]
            
        finally:
            s.close()
            


    def change_partition(self, graph_data_loader, partition_number):
        '''Gets a new partition to train and loads all the params of the partition'''
        try:
            #s = shelve.open(graph_data_loader.backend.mapper.metadata['entities_shelf'])
            #self.ent_original_ids = np.array(list(s.values())).astype(np.int32)
            
            s = shelve.open('ent_partition_metadata')
            # entities mapping ids
            self.ent_original_ids = s[str(partition_number)]
        finally:
            s.close()

        try:
            #s = shelve.open(graph_data_loader.backend.mapper.metadata['relations'])
            #self.rel_original_ids = np.array(list(s.values())).astype(np.int32)
            s = shelve.open('rel_partition_metadata')
            # entities mapping ids
            self.rel_original_ids = s[str(partition_number)]
            
        finally:
            s.close()
            
        # Open the buckets related to the partition and concat
        splits = graph_data_loader.backend.mapper.metadata['name'].split('-')
        source_bucket = splits[0][-1]
        dest_bucket = splits[1]
        
        try:
            s = shelve.open('ent_partition')
            source_bucket_params = s[source_bucket]
            dest_source_bucket_params = s[dest_bucket]
            # full ent embs
            self.all_ent_embs = np.concatenate([source_bucket_params[1], dest_source_bucket_params[1]])
            self.split_emb_idx = source_bucket_params[1].shape[0]
            
            self.all_ent_opt_params = np.concatenate([source_bucket_params[0], dest_source_bucket_params[0]])
            self.split_opt_idx = source_bucket_params[0].shape[0]
            
            # now select only partition embeddings
            ent_embs = self.all_ent_embs[self.ent_original_ids]
            ent_opt_params = self.all_ent_opt_params[self.ent_original_ids]
        finally:
            s.close()
            
        try:
            s = shelve.open('rel_partition')
            self.all_rel_embs = s['0'][1]
            self.all_rel_opt_params =s['0'][0]
            rel_embs = self.all_rel_embs[self.rel_original_ids]
            rel_opt_params = self.all_rel_opt_params[self.rel_original_ids]
        finally:
            s.close()
        
            
        
        #ent_embs = self.entity_embeddings[self.ent_original_ids, :]
        #rel_embs = self.entity_embeddings[self.rel_original_ids, :]

        self._model.partition_change_updates(len(self.ent_original_ids), ent_embs, rel_embs)
        if self._model.global_epoch >1:
            # needs to be better handled
            #optimizer_rel_weights_updates_beta1 = self.optimizer_hyperparams_rel[self.rel_original_ids, 0, :]
            #optimizer_rel_weights_updates_beta2 = self.optimizer_hyperparams_rel[self.rel_original_ids, 1, :]
            #optimizer_ent_weights_updates_beta1 = self.optimizer_hyperparams_ent[self.ent_original_ids, 0, :]
            #optimizer_ent_weights_updates_beta2 = self.optimizer_hyperparams_ent[self.ent_original_ids, 1, :]
            
            optimizer_rel_weights_updates_beta1 = rel_opt_params[:, 0, :]
            optimizer_rel_weights_updates_beta2 = rel_opt_params[:, 1, :]
            optimizer_ent_weights_updates_beta1 = ent_opt_params[: , 0, :]
            optimizer_ent_weights_updates_beta2 = ent_opt_params[: , 1, :]

            optimizer_rel_weights_updates_beta1 = np.pad(optimizer_rel_weights_updates_beta1, 
                                                         ((0, self.num_rels - optimizer_rel_weights_updates_beta1.shape[0]), 
                                                          (0,0)), 
                                                         'constant',
                                                         constant_values=(0))
            optimizer_rel_weights_updates_beta2 = np.pad(optimizer_rel_weights_updates_beta2, 
                                                         ((0, self.num_rels - optimizer_rel_weights_updates_beta2.shape[0]), 
                                                          (0,0)), 
                                                         'constant', 
                                                         constant_values=(0))
            optimizer_ent_weights_updates_beta1 = np.pad(optimizer_ent_weights_updates_beta1, 
                                                         ((0, self.max_ent_size - optimizer_ent_weights_updates_beta1.shape[0]), 
                                                          (0,0)), 
                                                         'constant', 
                                                         constant_values=(0))
            optimizer_ent_weights_updates_beta2 = np.pad(optimizer_ent_weights_updates_beta2, 
                                                         ((0, self.max_ent_size - optimizer_ent_weights_updates_beta2.shape[0]), 
                                                          (0,0)), 
                                                         'constant', 
                                                         constant_values=(0))

            self._model.optimizer.set_weights([self._model.optimizer.iterations.numpy(), 
                                         optimizer_ent_weights_updates_beta1,
                                         optimizer_rel_weights_updates_beta1,
                                         optimizer_ent_weights_updates_beta2,
                                         optimizer_rel_weights_updates_beta2
                                        ])

    def data_generator(self):
        for i, partition_data in enumerate(self.partitioner):
            # partition_data is an object of graph data loader
            self.change_partition(partition_data, i)
            try:
                while True:
                    batch_data_from_current_partition = next(partition_data)
                    yield batch_data_from_current_partition
            except StopIteration:
                self.update_partion_embeddings(partition_data, i)
            finally:
                pass
                
    def __iter__(self):
        """Function needed to be used as an itertor."""
        return self

    def __next__(self):
        """Function needed to be used as an itertor."""
        return next(self.batch_iterator)
    
    def reload(self):
        self.partitioner.reload()
        self.batch_iterator = iter(self.data_generator())
 