import tensorflow as tf
import numpy as np

def l1_regularizer(trainable_params, hyperparam_dict):
    """L1 regularization

    Parameters
    ----------
    trainable_params : list, shape [n]
        List of trainable params that should be reqularized
    hyperparam_dict : dict of hyperparams
        must contain lambda value for reqularization, float or list, shape [n if list, 1 if float]
        If it is a float, same lambda will be used for all params

    Returns
    -------
    loss_reg : Tensor
        Regularization Loss - this must be added to the loss function

    """
    lambda_reg = hyperparam_dict.get('lambda', 1e-5)
    if np.isscalar(lambda_reg):
        lambda_reg = [lambda_reg] * len(trainable_params)
    elif isinstance(lambda_reg, list) and len(lambda_reg) == len(trainable_params):
        pass
    else:
        raise ValueError("Regularizer weight must be a scalar or a list with length equal to number of params passes") 


    loss_reg = 0
    for i in range(len(trainable_params)):
        loss_reg += (lambda_reg[i] * tf.reduce_sum(tf.abs(trainable_params[i])))
        
    return loss_reg



def l2_regularizer(trainable_params, hyperparam_dict):
    """L2 regularization

    Parameters
    ----------
    trainable_params : list, shape [n]
        List of trainable params that should be reqularized
    hyperparam_dict : dict of hyperparams
        must contain lambda value for reqularization, float or list, shape [n if list, 1 if float]
        If it is a float, same lambda will be used for all params
        
    Returns
    -------
    loss_reg : Tensor
        Regularization Loss - this must be added to the loss function

    """
    lambda_reg = hyperparam_dict.get('lambda', 1e-5)
    if np.isscalar(lambda_reg):
        lambda_reg = [lambda_reg] * len(trainable_params)
    elif isinstance(lambda_reg, list) and len(lambda_reg) == len(trainable_params):
        pass
    else:
        raise ValueError("Regularizer weight must be a scalar or a list with length equal to number of params passes") 


    loss_reg = 0
    for i in range(len(trainable_params)):
        loss_reg += (lambda_reg[i] * tf.reduce_sum(tf.square(trainable_params[i]))) 

    return loss_reg