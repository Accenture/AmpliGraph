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
    def __init__(self, optimizer):
        """Initialize the Optimizer
        """
        self._optimizer = optimizer


    def apply_gradients(self, grads_and_vars):
        """Wrapper around apply_gradients. 
        See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers for more details.
        """
        self._optimizer.apply_gradients(grads_and_vars)
        
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

        
def get(identifier):
    if isinstance(identifier, tf.optimizers.Optimizer):
        return OptimizerWrapper(identifier)
    elif isinstance(identifier, six.string_types):
        optim = tf.optimizers.get(identifier)
        return OptimizerWrapper(optim)
    else:
        raise ValueError('Could not interpret optimizer identifier:', identifier)