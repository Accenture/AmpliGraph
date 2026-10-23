"""Baseline random explainer."""
import os
import numpy as np
import networkx as nx
from ampligraph.explanations.explainers import AbstractExplainer, register_explainer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import pygraphviz
    import graphviz
    from ampligraph.explanations.visualizers import VISUALIZERS_REGISTRY
except ImportError:
    logger.warning("[random_explainer] Without pygraphviz, pydot, and graphviz you won't be able to use visualizers module.")


def get_explaining_subgraph(G, triple, ratio=0.2):
    """Helper function to draw random triples from neighbourhood.

       Parameters
       ----------
       G: nx.Graph
           neighbourhood graph rooted at target triple.
       triple: np.array
           target triple.
       ratio: float
           ratio of how many triples to draw from neighbourhood graph.

       Return
       ------
       triple: np.array
           numpy array of triples.
    """
    n_edges = int(ratio * len(G.edges())) if  int(ratio * len(G.edges())) != 0 else 1
    data = np.array(list(nx.bfs_edges(G, triple[0])))
    triples_inds = np.random.choice(list(range(len(data))), n_edges)
    triples = []
    for edge in data[triples_inds]:
        triples.append((edge[0], G[edge[0]][edge[1]]['rel'], edge[1]))
    return np.array(triples)

@register_explainer("random_explainer")
class RandomExplainer(AbstractExplainer):
    """RandomExplainer - random explainer for Knowledge Graph Embeddings models.

       Explains model's prediction for a target triple by randomly drawing a sample of
       triples from target triple neighbourhood.

       Example
       -------
       >>> X = {'train': np.array([['a','b','c'],...]),
                'test':np.array(['d','e','f']...]),
                'valid':...}
       >>> model = restore_model(model_name_path='./model.pkl') # needs to be calibrated
       >>> target_triple = ['d','e','f']
       >>> explainer = RandomExplainer(X, model)
       >>> expl = explainer.predict_and_explain(target_triple)
       >>> expl
       {'prediction': array([-15.234796], dtype=float32),
        'probability': array([0.66178256], dtype=float32),
        'relevant_triples': array([...]),
        'query_triple': array([...]),
        'N-hood': array([...])}
    """

    def __init__(self, X, model):
        """Initialize explainer.

           Parameters
           ----------
           X: dict
              data dictionary with triples of form:
              {'train': np.array(...), 'test': np.array(...), 'valid': np.array(...)}.
           model: AmpliGraph model
              model for which predictions and explanations will be made.
        """
        super().__init__(X, model)
        self.__name__ = "random_explainer"
        try:
            self.visualizer = VISUALIZERS_REGISTRY.get(self.__name__ + "_viz")(X)
        except:
            logger.warning("Visualizer for {} not found!".format(self.__name__))
        self.model = model
        self.nodes = set(self.X['train'][:,0]).union(set(self.X['train'][:,2]))
        self.nodes = list(self.nodes)
        self.embedded_ents = self.model.get_embeddings(self.nodes)
        self.k = len(self.embedded_ents[0])
        self.nodes_emb_dict = dict(zip(self.nodes, self.embedded_ents))

    def explain_(self, triple, name="test_triple", root='triples', n_hop=1, ratio=0.2):
        """Explain instance.

           Parameters
           ----------
           triple: np.array
               triple to be explained.
           name: str
               prefix name for saving explanation examples.
           n_hop: int
               how many hops of neighbourhood to consider while drawing explanations.
           ratio: float
               fraction of triples to take as an explanation.
        """

        if not os.path.exists(root):
            os.mkdir(root)
        prefix = name
        G = self.navigator.get_n_hop_neighbourhood_subgraph(triple, n_hop=n_hop)
        triples = get_explaining_subgraph(G, triple, ratio=ratio)

        ### Target triple graph saved in G
        with open("{}/{}_triples.txt".format(root, prefix),'w') as f:
            f.write(str(triples))
        logger.info("Relevant triples: {}".format(len(triples)))
        n_hood = G.edges(data=True)

        return triples, n_hood

    def predict_explain(self, triples):
        """Explain given triple.

           Parameters
           ----------
           target_triple: np.array
               triple for which explanations was requested.

           Returns
           -------
           explanation: dict
               explanation object containing explanation graph.
        """

        # single triple case
        target_triple = triples
        relevant_triples, n_hood = self.explain_(triples, name="prefix_short", ratio=0.3)
        prediction = self.model.predict(target_triple)
        proba = self.model.predict_proba(target_triple)
        n_hood = np.asarray([[t[0], t[2]['rel'], t[1]] for t in n_hood])
        explanation_dictionary = {'prediction':prediction,
                                  'probability': proba,
                                  'query_triple': target_triple,
                                  'relevant_triples': relevant_triples,
                                  'N-hood': n_hood}
        return explanation_dictionary

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
        viz = self.visualizer.visualize(explanation, fmt=fmt, fname=fname)
        return viz
