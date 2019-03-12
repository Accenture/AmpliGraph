import tensorflow as tf
import numpy as np
import abc

REGULARIZER_REGISTRY = {}


def register_regularizer(name, external_params=[], class_params= {}):
    def insert_in_registry(class_handle):
        REGULARIZER_REGISTRY[name] = class_handle
        class_handle.name = name
        REGULARIZER_REGISTRY[name].external_params = external_params
        REGULARIZER_REGISTRY[name].class_params = class_params
        return class_handle
    return insert_in_registry


class Regularizer(abc.ABC):
    """Abstract class for a regularizer

    """

    """The name of the regularizer"""
    name = ""
    external_params = []
    class_params = {}

    def __init__(self, verbose=False, **kwargs):
        """Initialize the regularizer

        Parameters
        ----------
        verbose : bool
            Verbose mode
        kwargs : reguarizer hyperparameters
        """

        self._regularizer_parameters = {}

    def get_state(self, param_name):
        """Get the state value

        Parameters
        ----------
        param_name : string
            name of the state for which one wants to query the value

        Returns
        -------
        param_value:
            the value of the corresponding state

        """

        try:
            param_value = REGULARIZER_REGISTRY[self.name].class_params.get(param_name)
            return param_value
        except KeyError:
            raise Exception('Invalid Key')

    def _apply(self, X):
        """ Apply the regularization function. Every inherited class must implement this function.
        
        (All the TF code must go in this function.)
        
        Parameters
        ----------
        X : list, shape [n]
            List of trainable params that should be regularized
        
        Returns
        -------
        loss : float
            Regularization Loss
        """
        NotImplementedError("This function is a placeholder in an abstract class")

    def _inputs_check(self, X):
        """ Creates any dependencies that need to be checked before performing regularization.
        
        Parameters
        ----------
        X: list, shape [n]
            List of trainable params that should be regularized
        """
        pass

    def apply(self, X):
        """ Interface to external world. This function performs input checks, input pre-processing, and
        and applies the loss function.

        Parameters
        ----------
        X : list, shape [n]
            List of trainable params that should be regularized
        
        Returns
        -------
        loss : float
            Regularization Loss
        """
        self._inputs_check(X)
        loss = self._apply(X)
        return loss


@register_regularizer("None")
class NoRegularizer(Regularizer):
    """No regularization"""

    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def _init_hyperparams(self):
        """Verifies and stores the hyperparameters needed by the algorithm."""
        pass

    def _inputs_check(self, X):
        """ Creates any dependencies that need to be checked before performing regularization .
        
        Parameters
        ----------
        X: list, shape [n]
            List of trainable params that should be regularized
        """
        pass

    def _apply(self, X):
        """ Apply the loss function.

        Parameters
        ----------
        X : list, shape [n]
            List of trainable params that should be regularized
        
        Returns
        -------
        loss : float
            Regularization Loss

        """
        return tf.constant(0.0)


@register_regularizer("L1", ['lambda'])
class L1Regularizer(Regularizer):
    """L1 regularization

    Parameters
    ----------
    lambda_weight : float
        weight for regularizer loss for each parameter (default: 1e-5)
    verbose : bool
        verbose mode

    """

    def __init__(self, lambda_weight=1e-5, verbose=False):
        super().__init__(verbose=verbose)
        self._regularizer_parameters['lambda'] = lambda_weight

    def _inputs_check(self, X):
        """ Creates any dependencies that need to be checked before performing regularization.
        
        Parameters
        ----------
        X: list, shape [n]
            List of trainable params that should be regularized
        """
        if np.isscalar(self._regularizer_parameters['lambda']):
            self._regularizer_parameters['lambda'] = [self._regularizer_parameters['lambda']] * len(X)
        elif isinstance(self._regularizer_parameters['lambda'], list) and len(self._regularizer_parameters['lambda']) == len(X):
            pass
        else:
            raise ValueError("Regularizer weight must be a scalar or a "
                             "list with length equal to number of params passes")

    def _apply(self, X):
        """ Apply the loss function.

        Parameters
        ----------
        X : list, shape [n]
            List of trainable params that should be regularized.
        
        Returns
        -------
        loss : float
            Regularization Loss

        """

        self._inputs_check(X)

        loss_reg = 0
        for i in range(len(X)):
            loss_reg += (self._regularizer_parameters['lambda'][i] * tf.reduce_sum(tf.abs(X[i])))

        return loss_reg


@register_regularizer("L2", ['lambda'])
class L2Regularizer(Regularizer):
    """L2 regularization
    
    Hyperparameters:
    
    'lambda' - weight for regularizer loss for each parameter(default: 1e-5)
    """

    def __init__(self, lambda_weight=1e-5, verbose=False):
        super().__init__(verbose=verbose)
        self._regularizer_parameters['lambda'] = lambda_weight

    def _inputs_check(self, X):
        """ Creates any dependencies that need to be checked before performing regularization.
        
        Parameters
        ----------
        X: list, shape [n]
            List of trainable params that should be regularized
        """
        if np.isscalar(self._regularizer_parameters['lambda']):
            self._regularizer_parameters['lambda'] = [self._regularizer_parameters['lambda']] * len(X)
        elif isinstance(self._regularizer_parameters['lambda'], list) and len(self._regularizer_parameters['lambda']) == len(X):
            pass
        else:
            raise ValueError("Regularizer weight must be a scalar or a list "
                             "with length equal to number of params passes")

    def _apply(self, X):
        """ Apply the loss function.

        Parameters
        ----------
        X : list, shape [n]
            List of trainable params that should be regularized
        
        Returns
        -------
        loss : float
            Regularization Loss

        """

        self._inputs_check(X)

        loss_reg = 0
        for i in range(len(X)):
            loss_reg += (self._regularizer_parameters['lambda'][i] * tf.reduce_sum(tf.square(X[i])))

        return loss_reg


@register_regularizer("L3", ['lambda'])
class L3Regularizer(Regularizer):
    """L3 regularization
    
    Hyperparameters:
    
    'lambda' - weight for regularizer loss for each parameter(default: 1e-5)
    """

    def __init__(self, lambda_weight=1e-5, verbose=False):
        super().__init__(verbose=verbose)
        self._regularizer_parameters['lambda'] = lambda_weight

    def _inputs_check(self, X):
        """ Creates any dependencies that need to be checked before performing regularization.

        Parameters
        ----------
        X: list, shape [n]
            List of trainable params that should be regularized
        """
        if np.isscalar(self._regularizer_parameters['lambda']):
            self._regularizer_parameters['lambda'] = [self._regularizer_parameters['lambda']] * len(X)
        elif isinstance(self._regularizer_parameters['lambda'], list) and len(
                self._regularizer_parameters['lambda']) == len(X):
            pass
        else:
            raise ValueError("Regularizer weight must be a scalar or a list "
                             "with length equal to number of params passes")

    def _apply(self, X):
        """ Apply the loss function.

        Parameters
        ----------
        X : list, shape [n]
            List of trainable params that should be regularized.
        
        Returns
        -------
        loss : float
            Regularization Loss

        """
        self._inputs_check(X)

        loss_reg = 0
        for i in range(len(X)):
            loss_reg += (self._regularizer_parameters['lambda'][i] * tf.reduce_sum(tf.pow(tf.abs(X[i]), 3)))

        return loss_reg
