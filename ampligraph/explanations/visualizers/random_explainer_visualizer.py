"""Visualizer for random explainer."""
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import to_pydot
from graphviz import Source
import pygraphviz as pgv
from ampligraph.explanations.visualizers import AbstractGraphVisualizer, register_visualizer
from ampligraph.explanations.utils import create_cluster_with_nodes,\
                                           format_dot_attrs, apply_edges_attributes

@register_visualizer("random_explainer_viz")
class GraphVisualizerRandom(AbstractGraphVisualizer):
    """Visualizer for random explainer."""
    def __init__(self, X):
        super().__init__(X)

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

        target_triple = explanation['query_triple']
        n_hood = explanation['N-hood']
        relevant_triples = explanation['relevant_triples']
        subject_ = target_triple[0]
        object_ = target_triple[2]
        G = nx.DiGraph()
        G.add_edges_from([(e[0], e[2], {'rel':e[1]})for e in n_hood])

        edges = [(e[0], e[2]) for e in relevant_triples]
        nodes = set(np.asarray(edges)[:,0])
        nodes.update(set(np.asarray(edges)[:,1]))
        exceptions = [subject_, object_]
        exceptions.extend(list(nodes))
        G_cp = format_dot_attrs(G) # apply graph properties for graphviz display
        # color relevant relations (as returned by ExamplE) with same color
        G_cp = apply_edges_attributes(G_cp, edges, attributes={'color':['purple']*len(edges)})

        G_cp = apply_edges_attributes(G_cp,
                                      [[subject_,
                                      object_]],
                                      attributes={'color':['orange'],
                                                  "penwidth": [5],
                                                  'style':['dashed']}) # highlight target triple

        g = to_pydot(G_cp)
        g = pgv.AGraph(str(g), name='root')
        # spatially cluster TT nodes
        g =  create_cluster_with_nodes(g, [subject_, object_], name='TT')
        s = Source(g)
        s.render(format=fmt, filename=fname)
        return s
