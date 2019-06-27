
import numpy as np
from ..datasets import AmpligraphDatasetAdapter, SQLiteAdapter

class NumpyDatasetAdapter(AmpligraphDatasetAdapter):
    def __init__(self):
        super(NumpyDatasetAdapter, self).__init__()
        self.mapped_status = {}
        self.filter_adapter = None
    
    def generate_mappings(self, use_all=False):
        from ..evaluation import create_mappings
        if use_all:
            complete_dataset = []
            for key in self.dataset.keys():
                complete_dataset.append(self.dataset[key])
            self.rel_to_idx, self.ent_to_idx = create_mappings(np.concatenate(complete_dataset, axis=0))

        else:
            self.rel_to_idx, self.ent_to_idx = create_mappings(self.dataset["train"])
            
        return self.rel_to_idx, self.ent_to_idx
    
    def use_mappings(self, rel_to_idx, ent_to_idx):
        super().use_mappings(rel_to_idx, ent_to_idx)
        for key in self.dataset.keys():
            self.mapped_status[key] = False
        
    def get_size(self, dataset_type="train"):
        return self.dataset[dataset_type].shape[0]
    
    def get_next_train_batch(self, batch_size=1, dataset_type="train"):
        if self.mapped_status[dataset_type] == False:
            self.map_data()
            
        batches_count = int(np.ceil(self.get_size(dataset_type)/batch_size))
        for i in range(batches_count):
            out = np.int32(self.dataset[dataset_type][(i*batch_size) : ((i+1)*batch_size), :])
            yield out
            
    def get_next_eval_batch(self, batch_size=1, dataset_type="test"):
        if self.mapped_status[dataset_type] == False:
            self.map_data()
            
        batches_count = int(np.ceil(self.get_size(dataset_type)/batch_size))
        for i in range(batches_count):
            out = np.int32(self.dataset[dataset_type][(i*batch_size) : ((i+1)*batch_size), :])
            yield out
    
    def map_data(self, remap=False):
        from ..evaluation import to_idx
        if len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0:
            self.generate_mappings()
            
        for key in self.dataset.keys():
            if self.mapped_status[key] == False or remap == True:
                self.dataset[key] = to_idx(self.dataset[key], 
                                           ent_to_idx=self.ent_to_idx, 
                                           rel_to_idx=self.rel_to_idx)
                self.mapped_status[key] = True
                
            
    def _validate_data(self, data):
        if type(data) != np.ndarray:
            msg = 'Invalid type for input data. Expected ndarray, got {}'.format(type(data))
            raise ValueError(msg)

        if (np.shape(data)[1]) != 3:
            msg = 'Invalid size for input data. Expected number of column 3, got {}'.format(np.shape(data)[1])
            raise ValueError(msg)
            
    def set_data(self, dataset, dataset_type=None, mapped_status=False):
        if isinstance(dataset, dict):
            for key in dataset.keys():
                self._validate_data(dataset[key])
                self.dataset[key] = dataset[key]
                self.mapped_status[key] = mapped_status
        elif dataset_type is not None:
            self._validate_data(dataset)
            self.dataset[dataset_type] = dataset
            self.mapped_status[dataset_type] = mapped_status
        else:
            raise Exception("Incorrect usage. Expected a dictionary or a combination of dataset and it's type.")
            
        if not (len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0):
            self.map_data()
            
    def set_filter(self, filter_triples, mapped_status=False):
        self.filter_adapter = SQLiteAdapter()
        self.filter_adapter.use_mappings(self.rel_to_idx, self.ent_to_idx)
        self.filter_adapter.set_data(filter_triples, "filter", mapped_status)
        
    def get_next_batch_with_filter(self, batch_size=1, dataset_type="test"):
        if self.mapped_status[dataset_type] == False:
            self.map_data()
            
        batches_count = int(np.ceil(self.get_size(dataset_type)/batch_size))
        for i in range(batches_count):
            out = np.int32(self.dataset[dataset_type][(i*batch_size) : ((i+1)*batch_size), :])
            participating_objects, participating_subjects = self.filter_adapter.get_participating_entities(out)
            yield out, participating_objects, participating_subjects
            
    def cleanup(self):
        if self.filter_adapter is not None:
            self.filter_adapter.cleanup()
            self.filter_adapter = None