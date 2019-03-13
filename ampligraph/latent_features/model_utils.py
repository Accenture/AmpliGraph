import os
import pickle 
import importlib
import logging

SAVED_MODEL_FILE_NAME = 'model.pickle'

"""This module contains utility functions for neural knowledge graph embedding models.
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def save_model(model, loc):
    """ Save a trained model to disk.

    Parameters
    ----------
    model: A trained neural knowledge graph embedding model, the model must be an instance of TransE, DistMult or ComplEx classes
    loc: Directory into which user expects to save the model

    """
    logger.debug('Saving model {}.'.format(model.__class__.__name__))
    if not os.path.exists(loc):
        logger.debug('Creating path to {}.'.format(loc))
        os.makedirs(loc)

    hyperParamPath = os.path.join(loc, SAVED_MODEL_FILE_NAME)

    obj = {
        'class_name': model.__class__.__name__,
        'hyperparams': model.all_params,
        'is_fitted': model.is_fitted,
        'ent_to_idx': model.ent_to_idx,
        'rel_to_idx': model.rel_to_idx,
    }
    model.get_embedding_model_params(obj)
    logger.debug('Saving parameters: hyperparams:{}\n\tis_fitted:{}'.format(model.all_params,model.is_fitted))
    with open(hyperParamPath, 'wb') as fw:
        pickle.dump(obj, fw)

    #dump model tf
def restore_model(loc):
    """ Restore a saved model into an instance of TransE, DistMult or ComplEx classes.

    Parameters
    ----------
    loc: Directory containing the saved model

    Returns:
    model: a neural knowledge graph embedding model
        
    """
    restore_loc = os.path.join(loc, SAVED_MODEL_FILE_NAME)
    model = None
    logger.debug('Loading model from {}.'.format(loc))
    restored_obj = None
    with open(restore_loc, 'rb') as fr:
        restored_obj = pickle.load(fr)

    if restored_obj:
        logger.debug('Restoring model.')
        module = importlib.import_module("ampligraph.latent_features.models")
        class_ = getattr(module, restored_obj['class_name'])
        model = class_(**restored_obj['hyperparams'])
        model.is_fitted = restored_obj['is_fitted']
        model.ent_to_idx = restored_obj['ent_to_idx']
        model.rel_to_idx = restored_obj['rel_to_idx']
        model.restore_model_params(restored_obj)
    else:
        logger.debug('No model found.')
    return model
