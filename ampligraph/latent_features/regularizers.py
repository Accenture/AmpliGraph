import tensorflow as tf
import numpy as np
import abc

def l1_regularizer(trainable_params, hyperparam_dict):
    """L1 regularization

    Parameters
    ----------
    trainable_params : list, shape [n]
        List of trainable params that should be reqularized
    hyperparam_dict : dict of hyperparams
        must contain lambda value for reqularization, float or list, shape [n if list, 1 if float]
        If it is a float, same lambda will be used for all params

def register_regularizer(name, external_params=[], class_params= {}):
    def insert_in_registry(class_handle):
        REGULARIZER_REGISTRY[name] = class_handle
        class_handle.name = name
        REGULARIZER_REGISTRY[name].external_params = external_params
        REGULARIZER_REGISTRY[name].class_params = class_params
        return class_handle
    return insert_in_registry

#@register_regularizer("None")  
class Regularizer(abc.ABC):
    """Abstract class for loss function.
    """
    lambda_reg = hyperparam_dict.get('lambda', 1e-5)
    if np.isscalar(lambda_reg):
        lambda_reg = [lambda_reg] * len(trainable_params)
    elif isinstance(lambda_reg, list) and len(lambda_reg) == len(trainable_params):
        pass
    else:
        raise ValueError("Regularizer weight must be a scalar or a list with length equal to number of params passes") 

    def _init_hyperparams(self, hyperparam_dict):
        """ Verifies and stores the hyperparams needed by the algorithm
        
        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The regulairzer will check the keys to get the corresponding params
        """
        NotImplementedError("This function is a placeholder in an abstract class")
        
        
    def _apply(self, trainable_params):
        """ Apply the regularization function. Every inherited class must implement this function. (All the TF code must go in this function.)
        
        Parameters
        ----------
        trainable_params : list, shape [n]
            List of trainable params that should be reqularized
        
        Returns
        -------
        loss : float
            Regularization Loss
        """
        NotImplementedError("This function is a placeholder in an abstract class")
        
        
    def _inputs_check(self, trainable_params):
        """ Creates any dependencies that need to be checked before performing regularization 
        
        Parameters
        ----------
        trainable_params: list, shape [n]
            List of trainable params that should be reqularized
        """
        pass
        
    def apply(self, trainable_params):
        """ Interface to external world. This function does the input checks, preprocesses input and finally applies loss fn.
        Parameters
        ----------
        trainable_params : list, shape [n]
            List of trainable params that should be reqularized
        
        Returns
        -------
        loss : float
            Regularization Loss
        """
        self._inputs_check(trainable_params)
        loss = self._apply(trainable_params)
        return loss
    
    
@register_regularizer("None" )      
class NoRegularizer(Regularizer):
    """No regularization
    """
    
    def __init__(self, hyperparam_dict):
        super().__init__(hyperparam_dict)
        
    def _init_hyperparams(self, hyperparam_dict):
        """ Verifies and stores the hyperparams needed by the algorithm
        
        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The regularizer will check the keys to get the corresponding params('lambda')
        """
        pass
        
        
    def _inputs_check(self, trainable_params):
        """ Creates any dependencies that need to be checked before performing regularization 
        
        Parameters
        ----------
        trainable_params: list, shape [n]
            List of trainable params that should be reqularized
        """
        pass
        
    def _apply(self, trainable_params):
        """ Apply the loss function
        Parameters
        ----------
        trainable_params : list, shape [n]
            List of trainable params that should be reqularized
        
        Returns
        -------
        loss : float
            Regularization Loss

        """
        return tf.constant(0.0)
    
    
@register_regularizer("L1", ['lambda'] )      
class L1Regularizer(Regularizer):
    """L1 regularization
    """
    
    def __init__(self, hyperparam_dict):
        super().__init__(hyperparam_dict)
        
    def _init_hyperparams(self, hyperparam_dict):
        """ Verifies and stores the hyperparams needed by the algorithm
        
        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The regularizer will check the keys to get the corresponding params('lambda')
        """
        self._regularizer_parameters['lambda'] = hyperparam_dict.get('lambda', 1e-5)
        
        
    def _inputs_check(self, trainable_params):
        """ Creates any dependencies that need to be checked before performing regularization 
        
        Parameters
        ----------
        trainable_params: list, shape [n]
            List of trainable params that should be reqularized
        """
        if np.isscalar(self._regularizer_parameters['lambda']):
            self._regularizer_parameters['lambda'] = [self._regularizer_parameters['lambda']] * len(trainable_params)
        elif isinstance(self._regularizer_parameters['lambda'], list) and len(self._regularizer_parameters['lambda']) == len(trainable_params):
            pass
        else:
            raise ValueError("Regularizer weight must be a scalar or a list with length equal to number of params passes") 

        
    def _apply(self, trainable_params):
        """ Apply the loss function
        Parameters
        ----------
        trainable_params : list, shape [n]
            List of trainable params that should be reqularized
        
        Returns
        -------
        loss : float
            Regularization Loss

        """
        loss_reg = 0
        for i in range(len(trainable_params)):
            loss_reg += (self._regularizer_parameters['lambda'][i] * tf.reduce_sum(tf.abs(trainable_params[i])))

def l2_regularizer(trainable_params, hyperparam_dict):
    """L2 regularization
    """
    lambda_reg = hyperparam_dict.get('lambda', 1e-5)
    if np.isscalar(lambda_reg):
        lambda_reg = [lambda_reg] * len(trainable_params)
    elif isinstance(lambda_reg, list) and len(lambda_reg) == len(trainable_params):
        pass
    else:
        raise ValueError("Regularizer weight must be a scalar or a list with length equal to number of params passes") 


        """
        loss_reg = 0
        for i in range(len(trainable_params)):
            loss_reg += (self._regularizer_parameters['lambda'][i] * tf.reduce_sum(tf.square(trainable_params[i]))) 

        return loss_reg 