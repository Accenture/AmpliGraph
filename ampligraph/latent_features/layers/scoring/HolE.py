import tensorflow as tf
from .AbstractScoringLayer import AbstractScoringLayer, register_layer
from .ComplEx import ComplEx

@register_layer('HolE')
class HolE(ComplEx):
    
    def __init__(self, k):
        super(HolE, self).__init__(k)

    @tf.function
    def _compute_scores(self, triples):
        return (2 / (self.internal_k / 2)) * (super().compute_scores(triples))
    
    @tf.function(experimental_relax_shapes=True)
    def _get_subject_corruption_scores(self, triples, ent_matrix):
        return (2 / (self.internal_k / 2)) * (super()._get_subject_corruption_scores(triples, 
                                                                                     ent_matrix))
    
    @tf.function(experimental_relax_shapes=True)
    def _get_object_corruption_scores(self, triples, ent_matrix):
        return (2 / (self.internal_k / 2)) * (super()._get_object_corruption_scores(triples,  
                                                                                    ent_matrix))

