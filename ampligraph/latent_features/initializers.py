# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf
import abc
import logging
import numpy as np
from sklearn.utils import check_random_state

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

INITIALIZER_REGISTRY = {}

# Default value of lower bound for uniform sampling
DEFAULT_UNIFORM_LOW = -0.05

# Default value of upper bound for uniform sampling
DEFAULT_UNIFORM_HIGH = 0.05

# Default value of mean for Gaussian sampling
DEFAULT_NORMAL_MEAN = 0

# Default value of std for Gaussian sampling
DEFAULT_NORMAL_STD = 0.05

# Default value indicating whether to use xavier uniform or normal
DEFAULT_XAVIER_IS_UNIFORM = False


def register_initializer(name, external_params=[], class_params={}):
    def insert_in_registry(class_handle):
        INITIALIZER_REGISTRY[name] = class_handle
        class_handle.name = name
        INITIALIZER_REGISTRY[name].external_params = external_params
        INITIALIZER_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry


class Initializer(abc.ABC):
    """Abstract class for initializer .
    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, initializer_params={}, verbose=True, seed=0):
        """Initialize the Class

        Parameters
        ----------
        initializer_params : dict
            dictionary of hyperparams that would be used by the initializer.
        verbose : bool
            set/reset verbose mode
        seed : int/np.random.RandomState
            random state for random number generator
        """
        self.verbose = verbose
        self._initializer_params = {}
        if isinstance(seed, int):
            self.random_generator = check_random_state(seed)
        else:
            self.random_generator = seed
        self._init_hyperparams(initializer_params)

    def _display_params(self):
        """Display the parameter values
        """
        logger.info('\n------ Initializer -----')
        logger.info('Name : {}'.format(self.name))
        for key, value in self._initializer_params.items():
            logger.info('{} : {}'.format(key, value))

    def _init_hyperparams(self, hyperparam_dict):
        """ Initializes the hyperparameters.

        Parameters
        ----------
        hyperparam_dict: dictionary
            Consists of key value pairs. The initializer will check the keys to get the corresponding params
        """
        raise NotImplementedError('Abstract Method not implemented!')

    def _get_tf_initializer(self, in_shape=None, out_shape=None, concept='e'):
        """Create a tensorflow node for initializer

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        concept: char
            concept type (e for entity, r for relation)

        Returns
        -------
        initializer_instance: An Initializer instance.
        """
        raise NotImplementedError('Abstract Method not implemented!')

    def _get_np_initializer(self, in_shape=None, out_shape=None, concept='e'):
        """Create an initialized numpy array

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        concept: char
            concept type (e for entity, r for relation)

        Returns
        -------
        initialized_values: n-d array
            Initialized weights
        """
        raise NotImplementedError('Abstract Method not implemented!')

    def get_entity_initializer(self, in_shape=None, out_shape=None, init_type='tf'):
        """ Initializer for entity embeddings

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        init_type: string
            Type of initializer ('tf' for tensorflow, 'np' for numpy)

        Returns
        -------
        initialized_values: tf.Op or n-d array
            Weights initializer
        """
        assert init_type in ['tf', 'np'], 'Invalid initializer type!'
        if init_type == 'tf':
            return self._get_tf_initializer(in_shape, out_shape, 'e')
        else:
            return self._get_np_initializer(in_shape, out_shape, 'e')

    def get_relation_initializer(self, in_shape=None, out_shape=None, init_type='tf'):
        """ Initializer for relation embeddings

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        init_type: string
            Type of initializer ('tf' for tensorflow, 'np' for numpy)

        Returns
        -------
        initialized_values: tf.Op or n-d array
            Weights initializer
        """
        assert init_type in ['tf', 'np'], 'Invalid initializer type!'
        if init_type == 'tf':
            return self._get_tf_initializer(in_shape, out_shape, 'r')
        else:
            return self._get_np_initializer(in_shape, out_shape, 'r')


@register_initializer("normal", ["mean", "std"])
class RandomNormal(Initializer):
    r"""Initializes from a normal distribution with specified ``mean`` and ``std``

    .. math::

        \mathcal{N} (\mu, \sigma)

    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, initializer_params={}, verbose=True, seed=0):
        """Initialize the Random Normal initialization strategy

        Parameters
        ----------
        initializer_params : dict
            Consists of key-value pairs. The initializer will check the keys to get the corresponding params:

            - **mean**: (float). Mean of the weights(default: 0)
            - **std**: (float): std of the weights (default: 0.05)

            Example: ``initializer_params={'mean': 0, 'std': 0.01}``
        verbose : bool
            Enable/disable verbose mode
        seed : int/np.random.RandomState
            random state for random number generator
        """

        super(RandomNormal, self).__init__(initializer_params, verbose, seed)

    def _init_hyperparams(self, hyperparam_dict):
        """ Initializes the hyperparameters.

        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The initializer will check the keys to get the corresponding params
        """
        self._initializer_params['mean'] = hyperparam_dict.get('mean', DEFAULT_NORMAL_MEAN)
        self._initializer_params['std'] = hyperparam_dict.get('std', DEFAULT_NORMAL_STD)

        if self.verbose:
            self._display_params()

    def _get_tf_initializer(self, in_shape=None, out_shape=None, concept='e'):
        """Create a tensorflow node for initializer

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        concept: char
            concept type (e for entity, r for relation)

        Returns
        -------
        initializer_instance: An Initializer instance.
        """
        return tf.random_normal_initializer(mean=self._initializer_params['mean'],
                                            stddev=self._initializer_params['std'],
                                            dtype=tf.float32)

    def _get_np_initializer(self, in_shape, out_shape, concept='e'):
        """Create an initialized numpy array

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        concept: char
            concept type (e for entity, r for relation)

        Returns
        -------
        out: n-d array
            matrix initialized from a normal distribution of specified mean and std
        """
        return self.random_generator.normal(self._initializer_params['mean'],
                                            self._initializer_params['std'],
                                            size=(in_shape, out_shape)).astype(np.float32)


@register_initializer("uniform", ["low", "high"])
class RandomUniform(Initializer):
    r"""Initializes from a uniform distribution with specified ``low`` and ``high``

    .. math::

        \mathcal{U} (low, high)

    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, initializer_params={}, verbose=True, seed=0):
        """Initialize the Uniform initialization strategy

        Parameters
        ----------
        initializer_params : dict
            Consists of key-value pairs. The initializer will check the keys to get the corresponding params:

            - **low**: (float). lower bound for uniform number (default: -0.05)
            - **high**: (float): upper bound for uniform number (default: 0.05)

            Example: ``initializer_params={'low': 0, 'high': 0.01}``
        verbose : bool
            Enable/disable verbose mode
        seed : int/np.random.RandomState
            random state for random number generator
        """

        super(RandomUniform, self).__init__(initializer_params, verbose, seed)

    def _init_hyperparams(self, hyperparam_dict):
        """ Initializes the hyperparameters.

        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The initializer will check the keys to get the corresponding params
        """
        self._initializer_params['low'] = hyperparam_dict.get('low', DEFAULT_UNIFORM_LOW)
        self._initializer_params['high'] = hyperparam_dict.get('high', DEFAULT_UNIFORM_HIGH)

        if self.verbose:
            self._display_params()

    def _get_tf_initializer(self, in_shape=None, out_shape=None, concept='e'):
        """Create a tensorflow node for initializer

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        concept: char
            concept type (e for entity, r for relation)

        Returns
        -------
        initializer_instance: An Initializer instance.
        """
        return tf.random_uniform_initializer(minval=self._initializer_params['low'],
                                             maxval=self._initializer_params['high'],
                                             dtype=tf.float32)

    def _get_np_initializer(self, in_shape, out_shape, concept='e'):
        """Create an initialized numpy array

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        concept: char
            concept type (e for entity, r for relation)

        Returns
        -------
        out: n-d array
            matrix initialized from a uniform distribution of specified low and high bounds
        """
        return self.random_generator.uniform(self._initializer_params['low'],
                                             self._initializer_params['high'],
                                             size=(in_shape, out_shape)).astype(np.float32)


@register_initializer("xavier", ["uniform"])
class Xavier(Initializer):
    r"""Follows the xavier strategy for initialization of layers :cite:`glorot2010understanding`.

    If ``uniform`` is set to True, then it initializes the layer from the following uniform distribution:

    .. math::

        \mathcal{U} ( - \sqrt{ \frac{6}{ fan_{in} + fan_{out} } }, \sqrt{ \frac{6}{ fan_{in} + fan_{out} } } )

    If ``uniform`` is False, then it initializes the layer from the following normal distribution:

    .. math::

        \mathcal{N} ( 0, \sqrt{ \frac{2}{ fan_{in} + fan_{out} } } )

    where :math:`fan_{in}` and :math:`fan_{out}` are number of input units and output units of the layer respectively.

    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, initializer_params={}, verbose=True, seed=0):
        """Initialize the Xavier strategy

        Parameters
        ----------
        initializer_params : dict
            Consists of key-value pairs. The initializer will check the keys to get the corresponding params:

            - **uniform**: (bool). indicates whether to use Xavier Uniform or Xavier Normal initializer.

            Example: ``initializer_params={'uniform': False}``
        verbose : bool
            Enable/disable verbose mode
        seed : int/np.random.RandomState
            random state for random number generator
        """

        super(Xavier, self).__init__(initializer_params, verbose, seed)

    def _init_hyperparams(self, hyperparam_dict):
        """ Initializes the hyperparameters.

        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The initializer will check the keys to get the corresponding params
        """
        self._initializer_params['uniform'] = hyperparam_dict.get('uniform', DEFAULT_XAVIER_IS_UNIFORM)

        if self.verbose:
            self._display_params()

    def _get_tf_initializer(self, in_shape=None, out_shape=None, concept='e'):
        """Create a tensorflow node for initializer

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        concept: char
            concept type (e for entity, r for relation)

        Returns
        -------
        initializer_instance: An Initializer instance.
        """
        return tf.contrib.layers.xavier_initializer(uniform=self._initializer_params['uniform'],
                                                    dtype=tf.float32)

    def _get_np_initializer(self, in_shape, out_shape, concept='e'):
        """Create an initialized numpy array

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        concept: char
            concept type (e for entity, r for relation)

        Returns
        -------
        out: n-d array
            matrix initialized using xavier uniform or xavier normal initializer
        """
        if self._initializer_params['uniform']:
            limit = np.sqrt(6 / (in_shape + out_shape))
            return self.random_generator.uniform(-limit, limit, size=(in_shape, out_shape)).astype(np.float32)
        else:
            std = np.sqrt(2 / (in_shape + out_shape))
            return self.random_generator.normal(0, std, size=(in_shape, out_shape)).astype(np.float32)


@register_initializer("constant", ["entity", "relation"])
class Constant(Initializer):
    r"""Initializes with the constant values provided by the user

    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, initializer_params={}, verbose=True, seed=0):
        """Initialize the the constant values provided by the user

        Parameters
        ----------
        initializer_params : dict
            Consists of key-value pairs. The initializer will check the keys to get the corresponding params:

            - **entity**: (np.ndarray.float32). Initial values for entity embeddings
            - **relation**: (np.ndarray.float32). Initial values for relation embeddings

            Example: ``initializer_params={'entity': ent_init_value, 'relation': rel_init_value}``
        verbose : bool
            Enable/disable verbose mode
        seed : int/np.random.RandomState
            random state for random number generator
        """

        super(Constant, self).__init__(initializer_params, verbose, seed)

    def _init_hyperparams(self, hyperparam_dict):
        """ Initializes the hyperparameters.

        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The initializer will check the keys to get the corresponding params
        """
        try:
            self._initializer_params['entity'] = hyperparam_dict['entity']
            self._initializer_params['relation'] = hyperparam_dict['relation']
        except KeyError:
            raise Exception('Initial values of both entity and relation embeddings need to '
                            'be passed to the initializer!')
        if self.verbose:
            self._display_params()

    def _get_tf_initializer(self, in_shape=None, out_shape=None, concept='e'):
        """Create a tensorflow node for initializer

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        concept: char
            concept type (e for entity, r for relation)

        Returns
        -------
        initializer_instance: An Initializer instance.
        """

        if concept == 'e':
            assert self._initializer_params['entity'].shape[0] == in_shape and \
                self._initializer_params['entity'].shape[1] == out_shape, \
                "Invalid shape for entity initializer!"

            return tf.compat.v1.constant_initializer(self._initializer_params['entity'], dtype=tf.float32)
        else:
            assert self._initializer_params['relation'].shape[0] == in_shape and \
                self._initializer_params['relation'].shape[1] == out_shape, \
                "Invalid shape for relation initializer!"

            return tf.compat.v1.constant_initializer(self._initializer_params['relation'], dtype=tf.float32)

    def _get_np_initializer(self, in_shape, out_shape, concept='e'):
        """Create an initialized numpy array

        Parameters
        ----------
        in_shape: int
            number of inputs to the layer (fan in)
        out_shape: int
            number of outputs of the layer (fan out)
        concept: char
            concept type (e for entity, r for relation)

        Returns
        -------
        out: n-d array
            matrix initialized using constant values supplied by the user
        """
        if concept == 'e':
            assert self._initializer_params['entity'].shape[0] == in_shape and \
                self._initializer_params['entity'].shape[1] == out_shape, \
                "Invalid shape for entity initializer!"

            return self._initializer_params['entity']
        else:
            assert self._initializer_params['relation'].shape[0] == in_shape and \
                self._initializer_params['relation'].shape[1] == out_shape, \
                "Invalid shape for relation initializer!"

            return self._initializer_params['relation']
