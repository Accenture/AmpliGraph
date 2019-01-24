import os
import pickle 
import importlib

SAVED_MODEL_FILE_NAME = 'model.pickle'

"""This module contains utils function manipulating around a neural knowledge graph embedding model
"""

def save_model(model, loc):
    '''Save a trained model into disk

    Parameters
    ----------
    model: A trained neural knowledge graph embedding model, the model must be an instance of TransE, DistMult or ComplEx classes
    loc: Folder location into which user expects to save the model

    '''
    if not os.path.exists(loc):
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
    with open(hyperParamPath, 'wb') as fw:
        pickle.dump(obj, fw)

    #dump model tf
def restore_model(loc):
    '''Restore a saved model into an instance of TransE, DistMult or ComplEx classes

    Parameters
    ----------
    loc: Folder location containing the saved model

    Returns:
    model: a neural knowledge graph embedding model
        
    '''
    restore_loc = os.path.join(loc, SAVED_MODEL_FILE_NAME)
    
    restored_obj = None
    with open(restore_loc, 'rb') as fr:
        restored_obj = pickle.load(fr)

    if (restored_obj):
        module = importlib.import_module("ampligraph.latent_features.models")
        class_ = getattr(module, restored_obj['class_name'])
        model = class_(**restored_obj['hyperparams'])
        model.is_fitted = restored_obj['is_fitted']
        model.ent_to_idx = restored_obj['ent_to_idx']
        model.rel_to_idx = restored_obj['rel_to_idx']
        model.restore_model_params(restored_obj)
        return model
    return None

