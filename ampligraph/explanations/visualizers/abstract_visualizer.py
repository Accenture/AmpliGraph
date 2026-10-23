"""Abstract Visualizer."""
from abc import abstractmethod
import abc
import logging

VISUALIZERS_REGISTRY = {}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def register_visualizer(name):
    """Decorator responsible for registering visualizer in the visualizers registry.

       Parameters
       ----------
       name: str
           name of the new visualizer in the following format:
           <lowercase_explaining_method_name>_viz,
           e.g. random_explainer_viz for visualizer for RandomExplainer.

       Example
       -------
       >>> @register_visualizer("new_explainer_viz")
       >>> class NewVisualizer(AbstractGraphVisualizer):
       >>> ... pass
    """
    def insert_in_registry(class_handle):
        """Checks if visualizer already exists and if not registers it."""
        if name in VISUALIZERS_REGISTRY.keys():
            msg = "Visualizer with name {} already exists!".format(name)
            logger.error(msg)
            raise Exception(msg)

        VISUALIZERS_REGISTRY[name] = class_handle
        class_handle.name = name

        return class_handle

    return insert_in_registry


class AbstractGraphVisualizer(metaclass=abc.ABCMeta):
    """Abstract visualizer."""
    def __init__(self, X):
        self.X = X

    @abstractmethod
    def visualize(self, explanation, fname='graph', fmt='pdf'):
        """Visualize explaining graph.

           Parameters
           ----------
           explanation: dict
               dictionary object with explanation and context.
           fname: str
               name of file where the graph figure should be saved.
           fmt: str
               format to save graph in.

        """
