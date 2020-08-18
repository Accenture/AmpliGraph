from ampligraph.datasets import SQLiteAdapter
from ampligraph.datasets import GraphDataLoader
from ampligraph.datasets.graph_partitioner import PARTITION_ALGO_REGISTRY
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
        num_buckets = 3
        self._epochs=epochs
        print('Partitioning may take a while...')
        self.partitioner = PARTITION_ALGO_REGISTRY.get(strategy)(dataset_loader, k=num_buckets)
        

        # needs to go into the shelves
        self.optimizer_hyperparams_ent = np.zeros(shape=(self.num_ents, 2, self.k), 
                                                  dtype=np.float32)
        self.optimizer_hyperparams_rel = np.zeros(shape=(self.num_rels, 2, self.k), 
                                              dtype=np.float32)

        def xavier(in_shape, out_shape):
            std = np.sqrt(2 / (in_shape + out_shape))
            return np.random.normal(0, std, size=(in_shape, out_shape)).astype(np.float32)

        # needs to go into the shelves
        self.entity_embeddings = xavier(self.num_ents, self.k)
        self.rel_embeddings = xavier(self.num_rels, self.k)


    def update_partion_embeddings(self):
        self.entity_embeddings[self.ent_original_ids, :] = \
            self._model.encoding_layer.ent_emb.numpy()[:len(self.ent_original_ids), :]

        self.rel_embeddings[self.rel_original_ids, :] = \
            self._model.encoding_layer.rel_emb.numpy()[:len(self.rel_original_ids), :]

        opt_weights = self._model.optimizer.get_weights()
        if len(opt_weights)>0:
            self.optimizer_hyperparams_rel[self.rel_original_ids, :, :] = \
                np.concatenate([opt_weights[2][:len(self.rel_original_ids)][:, np.newaxis, :], 
                                opt_weights[4][:len(self.rel_original_ids)][:, np.newaxis, :]], 1)

            self.optimizer_hyperparams_ent[self.ent_original_ids, :, :] = \
                np.concatenate([opt_weights[1][:len(self.ent_original_ids)][:, np.newaxis, :], 
                                opt_weights[3][:len(self.ent_original_ids)][:, np.newaxis, :]], 1)


    def change_partition(self, graph_data_loader):
        # load a new partition and update the trainable params and optimizer hyperparams
        
        try:
            s = shelve.open(graph_data_loader.backend.mapper.metadata['entities_shelf'])
            self.ent_original_ids = np.array(list(s.values())).astype(np.int32)
        finally:
            s.close()

        try:
            s = shelve.open(graph_data_loader.backend.mapper.metadata['relations'])
            self.rel_original_ids = np.array(list(s.values())).astype(np.int32)
        finally:
            s.close()

        ent_embs = self.entity_embeddings[self.ent_original_ids, :]
        rel_embs = self.entity_embeddings[self.rel_original_ids, :]

        self._model.partition_change_updates(len(self.ent_original_ids), ent_embs, rel_embs)
        if self._model.global_epoch >1:
            # needs to be better handled
            optimizer_rel_weights_updates_beta1 = self.optimizer_hyperparams_rel[self.rel_original_ids, 0, :]
            optimizer_rel_weights_updates_beta2 = self.optimizer_hyperparams_rel[self.rel_original_ids, 1, :]
            optimizer_ent_weights_updates_beta1 = self.optimizer_hyperparams_ent[self.ent_original_ids, 0, :]
            optimizer_ent_weights_updates_beta2 = self.optimizer_hyperparams_ent[self.ent_original_ids, 1, :]

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
            self.change_partition(partition_data)
            try:
                while True:
                    batch_data_from_current_partition = next(partition_data)
                    yield batch_data_from_current_partition
            except StopIteration:
                self.update_partion_embeddings()
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
 