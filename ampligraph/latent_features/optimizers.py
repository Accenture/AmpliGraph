import tensorflow as tf
import abc
import logging
import six
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OptimizerWrapper(abc.ABC):
    """Wrapper around tensorflow optimizer
    """
    def __init__(self, name=None, optimizer=None, **kwargs):
        """Initialize the Optimizer
        """
        if name is not None:
            config = {'class_name': name, 'config': kwargs}
            optimizer = tf.keras.optimizers.deserialize(config)
            
        self._optimizer = optimizer
        self.__num_optimized_vars = 0
        self.__number_hyperparams = -1

    def apply_gradients(self, grads_and_vars):
        """Wrapper around apply_gradients. 
        See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers for more details.
        """
        self._optimizer.apply_gradients(grads_and_vars)
        
    def minimize(self, loss, ent_emb, rel_emb, gradient_tape, other_vars=[]):
        all_trainable_vars = [ent_emb, rel_emb]
        all_trainable_vars.extend(other_vars)
        self.__num_optimized_vars = len(all_trainable_vars)
        gradients = gradient_tape.gradient(loss, all_trainable_vars)
        # update the trainable params
        self._optimizer.apply_gradients(zip(gradients, all_trainable_vars))
        
        if self.__number_hyperparams == -1:
            optim_weights = self._optimizer.get_weights()
            self.__number_hyperparams = 0
            for i in range(1, len(optim_weights), self.__num_optimized_vars):
                self.__number_hyperparams += 1

    def get_hyperparam_count(self):
        return self.__number_hyperparams
    
    def get_entity_relation_hyperparams(self):
        optim_weights = self._optimizer.get_weights()
        ent_hyperparams = []
        rel_hyperparams = []
        for i in range(1, len(optim_weights), self.__num_optimized_vars):
            ent_hyperparams.append(optim_weights[i])
            rel_hyperparams.append(optim_weights[i + 1])
            
        return ent_hyperparams, rel_hyperparams
    
    def set_entity_relation_hyperparams(self, ent_hyperparams, rel_hyperparams):
        optim_weights = self._optimizer.get_weights()
        for i, j in zip(range(1, len(optim_weights), self.__num_optimized_vars), range(len(ent_hyperparams))):
            optim_weights[i] = ent_hyperparams[j]
            optim_weights[i + 1] = rel_hyperparams[j]
        self._optimizer.set_weights(optim_weights)
        
    def get_weights(self):
        """Wrapper around get weights.
        See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers for more details.
        """
        return self._optimizer.get_weights()
        
    def set_weights(self, weights):
        """Wrapper around set weights.
        See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers for more details.
        """
        self._optimizer.set_weights(weights)

    def get_iterations(self):
        return self._optimizer.iterations.numpy()
    
    def get_config(self):
        return self._optimizer.get_config()
        

def get(identifier):
    if isinstance(identifier, tf.optimizers.Optimizer):
        return OptimizerWrapper(optimizer=identifier)
    elif isinstance(identifier, OptimizerWrapper):
        return identifier
    elif isinstance(identifier, six.string_types):
        return OptimizerWrapper(name=identifier)
    else:
        raise ValueError('Could not interpret optimizer identifier:', identifier)

