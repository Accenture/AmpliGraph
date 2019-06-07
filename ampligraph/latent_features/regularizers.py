# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf
import numpy as np
import abc
import logging

REGULARIZER_REGISTRY = {}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def register_regularizer(name, external_params=[], class_params={}):
    def insert_in_registry(class_handle):
        REGULARIZER_REGISTRY[name] = class_handle
        class_handle.name = name
        REGULARIZER_REGISTRY[name].external_params = external_params
        REGULARIZER_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry


# defalut lambda to be used in L1, L2 and L3 regularizer
DEFAULT_LAMBDA = 1e-5

# default regularization - L2
DEFAULT_NORM = 2


class Regularizer(abc.ABC):
    """Abstract class for Regularizer.
    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, hyperparam_dict, verbose=False):
        """Initialize the regularizer.

        Parameters
        ----------
        hyperparam_dict : dict
            dictionary of hyperparams
            (Keys are described in the hyperparameters section)
        """
        self._regularizer_parameters = {}

        # perform check to see if all the required external hyperparams are passed
        try:
            self._init_hyperparams(hyperparam_dict)
            if verbose:
                logger.info('\n------ Regularizer -----')
                logger.info('Name : {}'.format(self.name))
                for key, value in self._regularizer_parameters.items():
                    logger.info('{} : {}'.format(key, value))

        except KeyError as e:
            msg = 'Some of the hyperparams for regularizer were not passed.\n{}'.format(e)
            logger.error(msg)
            raise Exception(msg)

    def get_state(self, param_name):
        """Get the state value.

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
        except KeyError as e:
            msg = 'Invalid Key.\n{}'.format(e)
            logger.error(msg)
            raise Exception(msg)

    def _init_hyperparams(self, hyperparam_dict):
        """ Initializes the hyperparameters needed by the algorithm.
        
        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The regularizer will check the keys to get the corresponding params
        """
        logger.error('This function is a placeholder in an abstract class')
        NotImplementedError("This function is a placeholder in an abstract class")

    def _apply(self, trainable_params):
        """ Apply the regularization function. Every inherited class must implement this function.
        
        (All the TF code must go in this function.)
        
        Parameters
        ----------
        trainable_params : list, shape [n]
            List of trainable params that should be reqularized
        
        Returns
        -------
        loss : tf.Tensor
            Regularization Loss
        """
        logger.error('This function is a placeholder in an abstract class')
        NotImplementedError("This function is a placeholder in an abstract class")

    def apply(self, trainable_params):
        """ Interface to external world. This function performs input checks, input pre-processing, and
        and applies the loss function.

        Parameters
        ----------
        trainable_params : list, shape [n]
            List of trainable params that should be reqularized
        
        Returns
        -------
        loss : tf.Tensor
            Regularization Loss
        """
        loss = self._apply(trainable_params)
        return loss


@register_regularizer("LP", ['p', 'lambda'])
class LPRegularizer(Regularizer):
    """ Performs LP regularization
    
        .. math::

               \mathcal{L}(Reg) =  \sum_{i=1}^{n}  \lambda_i * \mid w_i \mid_p
           
        where n is the number of model parameters, :math:`p \in{1,2,3}` is the p-norm and
        :math:`\lambda` is the regularization weight.
           
        Example: if :math:`p=1` the function will perform L1 regularization.
        L2 regularization is obtained with :math:`p=2`.
          
    """

    def __init__(self, regularizer_params={'lambda': DEFAULT_LAMBDA, 'p': DEFAULT_NORM}, verbose=False):
        """ Initializes the hyperparameters needed by the algorithm.

        Parameters
        ----------
        regularizer_params : dictionary
            Consists of key-value pairs. The regularizer will check the keys to get the corresponding params:

            - **'lambda'**: (float). Weight of regularization loss for each parameter (default: 1e-5)
            - **'p'**: (int): norm (default: 2)

            Example: ``regularizer_params={'lambda': 1e-5, 'p': 1}``
            
        """
        super().__init__(regularizer_params, verbose)

    def _init_hyperparams(self, hyperparam_dict):
        """ Initializes the hyperparameters needed by the algorithm.
        
        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The regularizer will check the keys to get the corresponding params:
            
            'lambda': list or float
            
                weight for regularizer loss for each parameter(default: 1e-5). If list, size must be equal to no. of parameters.
                
            'p': int
            
                Norm of the regularizer (``1`` for L1 regularizer, ``2`` for L2 and so on.) (default:2) 
                
        """
        self._regularizer_parameters['lambda'] = hyperparam_dict.get('lambda', DEFAULT_LAMBDA)
        self._regularizer_parameters['p'] = hyperparam_dict.get('p', DEFAULT_NORM)
        if type(self._regularizer_parameters['p']) is not int:
            msg = 'Invalid value for regularizer parameter p:{}. Supported type int'.format(
                self._regularizer_parameters['p'])
            logger.error(msg)
            raise Exception(msg)

    def _apply(self, trainable_params):
        """ Apply the regularizer to the params.

        Parameters
        ----------
        trainable_params : list, shape [n]
            List of trainable params that should be reqularized.
        
        Returns
        -------
        loss : tf.Tensor
            Regularization Loss

        """
        if np.isscalar(self._regularizer_parameters['lambda']):
            self._regularizer_parameters['lambda'] = [self._regularizer_parameters['lambda']] * len(trainable_params)
        elif isinstance(self._regularizer_parameters['lambda'], list) and len(
                self._regularizer_parameters['lambda']) == len(trainable_params):
            pass
        else:
            logger.error('Regularizer weight must be a scalar or a list with length equal to number of params passes')
            raise ValueError(
                "Regularizer weight must be a scalar or a list with length equal to number of params passes")

        loss_reg = 0
        for i in range(len(trainable_params)):
            loss_reg += (self._regularizer_parameters['lambda'][i] * tf.reduce_sum(
                tf.pow(tf.abs(trainable_params[i]), self._regularizer_parameters['p'])))

        return loss_reg
