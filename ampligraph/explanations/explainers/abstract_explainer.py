""""Abstract explainer."""
import abc
import numpy as np
import logging
from abc import abstractmethod
from ampligraph.explanations.utils import  GraphNavigator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

EXPLAINERS_REGISTRY = {}

def register_explainer(name):
    """Decorator responsible for registering explainer in the explainers registry.
       
       Parameters
       ----------
       name: str
           name of the new partition strategy.
 
       Example
       -------
       >>> @register_explainer("NewExplainer")
       >>> class NewExplainer(AbstractExplainer):
       >>> ... pass
    """
    def insert_in_registry(class_handle):
        """Checks if explainer already exists and if not registers it."""
        if name in EXPLAINERS_REGISTRY.keys():
            msg = "Explainer with name {} already exists!".format(name)
            logger.error(msg)
            raise Exception(msg)
        
        EXPLAINERS_REGISTRY[name] = class_handle
        class_handle.name = name
        
        return class_handle

    return insert_in_registry


class AbstractExplainer(metaclass=abc.ABCMeta):
    """Abstract explainer."""
    def __init__(self, X, model):
        if isinstance(X, np.ndarray):
            X = { 'train': X, 
                  'test':np.array([]), 
                  'valid': np.array([]) }
        self.X = X
        self.model = model
        self.navigator = GraphNavigator(self.X)

    @abstractmethod
    def predict_explain(self, target_triple):
        """Explain instances.

           Parameters
           ----------
           target_triple: np.array
               triples to be explained.
        """

    @abstractmethod
    def visualize(self, explanation, fmt='pdf', fname='graph'):
        """Visualize explanation.

           Parameters
           ----------
           explanation: dict
               dictionary object with explanation and context.
           fname: str
               name of file where the graph figure should be saved.
           fmt: str
               format to save graph in.

           Return
           ------
           viz: Source
               pygraphviz Source object.
        """
