import abc
class AmpligraphDatasetAdapter(abc.ABC):
    def __init__(self):
        self.dataset = {}
        self.rel_to_idx = {}
        self.ent_to_idx = {}
        
    def use_mappings(self, rel_to_idx, ent_to_idx):
        self.rel_to_idx = rel_to_idx
        self.ent_to_idx = ent_to_idx
    
    def generate_mappings(self, use_train=True):
        raise NotImplementedError('Abstract Method not implemented!')
        
    def get_size(self, dataset_type="train"):
        raise NotImplementedError('Abstract Method not implemented!')
        
    def set_data(self, dataset, dataset_type=None, mapped_status=False):
        raise NotImplementedError('Abstract Method not implemented!')
        
    def map_data(self, remap=False):
        raise NotImplementedError('Abstract Method not implemented!')
    
    def set_filter(self, filter_triples):
        raise NotImplementedError('Abstract Method not implemented!')
        
    def get_next_batch(self, batch_size, dataset_type="train"):
        raise NotImplementedError('Abstract Method not implemented!')
        
    def get_next_batch_with_filter(self, batch_size=1, dataset_type="test"):
        raise NotImplementedError('Abstract Method not implemented!')
            
    def cleanup(self):
        raise NotImplementedError('Abstract Method not implemented!')