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
         
    def get_next_batch(self, batch_size, dataset_type="train"):
        raise NotImplementedError('Abstract Method not implemented!')
        
    def set_data(self, x_train=None, x_valid=None, x_test=None):
        raise NotImplementedError('Abstract Method not implemented!')
        
    def map_data(self, remap=False):
        raise NotImplementedError('Abstract Method not implemented!')
