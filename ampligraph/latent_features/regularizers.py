from functools import partial
import tensorflow as tf

def LP_regularizer(trainable_param, regularizer_parameters={}):
    return regularizer_parameters.get('lambda', 0.001) * tf.reduce_sum(
        tf.pow(tf.abs(trainable_param), regularizer_parameters.get('p', 3)))
    
def get(identifier, hyperparams={}):
    '''
    Get the optimizer specified by the identifier
    
    Parameters:
    -----------
    identifier: string, tf.keras.regularizer instance or a callable
        Instance of tf.keras.regularizer or name of the regularizer to use (will use default parameters) or a 
        callable function
        
    Returns:
    --------
    regularizer: instance of tf.keras.regularizer
        Regularizer instance
        
    '''
    if isinstance(identifier, str) and identifier == 'LP':
        identifier = partial(LP_regularizer, regularizer_parameters=hyperparams)
    return tf.keras.regularizers.get(identifier)