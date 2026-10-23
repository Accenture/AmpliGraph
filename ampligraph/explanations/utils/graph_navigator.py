"""Graph Navigator."""
import numpy as np
import networkx as nx


class GraphNavigator():
    """Class responsible for accessing different parts of graph.
       Methods:
        - get_neighbours: retrieves neighbours of a given node.
        - get_n_hop_neighbourhood_subgraph: get n-hop neighbourhood of a given triple.

    """
    def __init__(self, X):
        """Initializes GraphNavigator
           Parameters
           ----------
           X: dict
               dataset dicionary
               (e.g. X = {'train':np.array([[1,2,3],[2,1,3]]), 'test': ..., 'valid':...} )

        """
        self.X = X

    def get_neighbours(self, node, subset='train'):
        """Retrieves neighbours of a node.
           Parameters
           ----------
           node: str
               node for which the neighbours need to be retrieved.
           subset ['train']: from which dataset to retrieve neighbours ('train', 'test', 'valid').

           Returns
           -------
           []: list
              neighbours of a node in a given format.

        """
        sub = np.asarray([elem for elem in self.X[subset] if elem[0] == node and elem[2]])
        obj = np.asarray([elem for elem in self.X[subset] if elem[2] == node and elem[0]])
        if isinstance(sub, list):
            sub = np.asarray(sub)
        if isinstance(obj, list):
            obj = np.asarray(obj)

        if sub.size:
            if obj.size:
                return np.concatenate((sub,obj))
            return sub
        if obj.size:
            return obj
        return []

    def get_n_hop_neighbourhood_subgraph(self, triple, n_hop=1):
        """Returns n-hop neighbourhood subgraph in a specified format.

           Parameters
           ----------
           triple: np.array
               triple for which nhop neighbourhood needs to be returned.
           n_hop [1]: int the hop number to retrieve.

           Returns
           -------
           G: nx.Graph
               networkx graph of n-hop neighbourhood in a given format.
        """
        G = nx.DiGraph()
        G.add_edge(triple[0], triple[2], rel=triple[1])

        sub = self.get_neighbours(triple[0])
        for elem in sub:
            G.add_edge(elem[0], elem[2], rel=elem[1])
            if n_hop != 1:
                G_sub = self.get_n_hop_neighbourhood_subgraph(elem, n_hop=n_hop-1)
                G = nx.compose(G, G_sub)
        obj = self.get_neighbours(triple[2])
        for elem in obj:
            G.add_edge(elem[0], elem[2], rel=elem[1])
            if n_hop != 1:
                G_obj = self.get_n_hop_neighbourhood_subgraph(elem, n_hop=n_hop-1)
                G =  nx.compose(G, G_obj)
        return G
