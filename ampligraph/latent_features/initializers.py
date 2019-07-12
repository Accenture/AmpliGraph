import tensorflow as tf
import abc
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

INITIALIZER_REGISTRY = {}

# Default value of lower bound for uniform sampling
DEFAULT_UNIFORM_LOW = -0.05

# Default value of upper bound for uniform sampling
DEFAULT_UNIFORM_HIGH = 0.05

# Default value of mean for Gaussian sampling
DEFAULT_NORMAL_MEAN = 0

# Default value of mean for Gaussian sampling
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
        """
        self.verbose = verbose
        self._initializer_params = {}
        self.seed = seed
        self._init_hyperparams(initializer_params)
        
    def _init_hyperparams(self, hyperparam_dict):
        """ Initializes the hyperparameters.
        
        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The initializer will check the keys to get the corresponding params
        """
        raise NotImplementedError('Abstract Method not implemented!')
        
    def get_tf_initializer(self):
        raise NotImplementedError('Abstract Method not implemented!')
        
    def get_np_initializer(self, in_shape, out_shape):
        raise NotImplementedError('Abstract Method not implemented!')


@register_initializer("normal", ["mean", "std"])             
class RandomNormal(Initializer):
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
            Consists of key-value pairs. The initializer will check the keys to get the corresponding params:

            - **'mean'**: (float). Mean of the weights(default: 0)
            - **'std'**: (float): std if the weights (default: 0.05)

            Example: ``initializer_params={'mean': 0, 'std': 0.01}``
        verbose : bool
            Enable/disable verbose mode
        seed : int
            random number generator seed
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
            logger.info('\n------ Initializer -----')
            logger.info('Name : {}'.format(self.name))
            for key, value in self._initializer_params.items():
                logger.info('{} : {}'.format(key, value))
                
    def get_tf_initializer(self):
        return tf.random_normal_initializer(mean=self._initializer_params['mean'],
                                            stddev=self._initializer_params['std'],
                                            seed=self.seed)
        
    def get_np_initializer(self, in_shape, out_shape):
        return np.random.normal(self._initializer_params['mean'],
                                self._initializer_params['std'], 
                                size=(in_shape, out_shape))


@register_initializer("uniform", ["low", "high"])             
class RandomUniform(Initializer):
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
            Consists of key-value pairs. The initializer will check the keys to get the corresponding params:

            - **'low'**: (float). lower bound for uniform number (default: -0.05)
            - **'high'**: (float): upper bound for uniform number (default: 0.05)

            Example: ``initializer_params={'low': 0, 'high': 0.01}``
        verbose : bool
            Enable/disable verbose mode
        seed : int
            random number generator seed
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
            logger.info('\n------ Initializer -----')
            logger.info('Name : {}'.format(self.name))
            for key, value in self._initializer_params.items():
                logger.info('{} : {}'.format(key, value))
                
    def get_tf_initializer(self):
        return tf.random_uniform_initializer(minval=self._initializer_params['low'], 
                                             maxval=self._initializer_params['high'], 
                                             seed=self.seed)
        
    def get_np_initializer(self, in_shape, out_shape):
        return np.random.uniform(self._initializer_params['low'],
                                 self._initializer_params['high'], 
                                 size=(in_shape, out_shape))


@register_initializer("xavier", ["uniform"])             
class Xavier(Initializer):
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
            Consists of key-value pairs. The initializer will check the keys to get the corresponding params:

            - **'uniform'**: (bool). indicates whether to use Xavier Uniform or Xavier Normal initializer.

            Example: ``initializer_params={'uniform': False}``
        verbose : bool
            Enable/disable verbose mode
        seed : int
            random number generator seed
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
            logger.info('\n------ Initializer -----')
            logger.info('Name : {}'.format(self.name))
            for key, value in self._initializer_params.items():
                logger.info('{} : {}'.format(key, value))
                
    def get_tf_initializer(self):
        return tf.contrib.layers.xavier_initializer(uniform=self._initializer_params['uniform'], 
                                                    seed=self.seed)
        
    def get_np_initializer(self, in_shape, out_shape):
        if self._initializer_params['uniform']:
            limit = np.sqrt(6 / (in_shape + out_shape))
            return np.random.uniform(-limit,
                                     limit, 
                                     size=(in_shape, out_shape))
        else:
            std = np.sqrt(2 / (in_shape + out_shape))
            return np.random.normal(0,
                                    std, 
                                    size=(in_shape, out_shape))
