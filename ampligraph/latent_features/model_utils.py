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
    
        Examples
        --------
        >>> import numpy as np
        >>> from ampligraph.latent_features import ComplEx, save_model, restore_model
        >>> X = np.array([['a', 'y', 'b'],
        >>>               ['b', 'y', 'a'],
        >>>               ['a', 'y', 'c'],
        >>>               ['c', 'y', 'a'],
        >>>               ['a', 'y', 'd'],
        >>>               ['c', 'y', 'd'],
        >>>               ['b', 'y', 'c'],
        >>>               ['f', 'y', 'e']])
        >>> model.fit(X)
        >>> y_pred_before = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
        >>> EXAMPLE_LOC = 'saved_models'
        >>> save_model(model, EXAMPLE_LOC)
        >>> print(y_pred_before)
        [1.261404, -1.324778]

        Parameters
        ----------
        model: EmbeddingModel
            A trained neural knowledge graph embedding model, the model must be an instance of TransE,
            DistMult, ComplEx, or HolE.
        loc: string
            Directory where the model will be saved.

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
    logger.debug('Saving parameters: hyperparams:{}\n\tis_fitted:{}'.format(model.all_params, model.is_fitted))
    with open(hyperParamPath, 'wb') as fw:
        pickle.dump(obj, fw)

        # dump model tf


def restore_model(loc):
    """ Restore a saved model from disk.
    
        Examples
        --------
        >>> from ampligraph.latent_features import restore_model
        >>> import numpy as np
        >>> EXAMPLE_LOC = 'saved_models' # Assuming that the model is present at this location
        >>> restored_model = restore_model(EXAMPLE_LOC)
        >>> y_pred_after = restored_model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
        >>> print(y_pred_after)
        [1.261404, -1.324778]

        Parameters
        ----------
        loc: string
            Directory where the saved model is located.

        Returns
        -------
        model: EmbeddingModel
            the neural knowledge graph embedding model restored from disk.
        
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
