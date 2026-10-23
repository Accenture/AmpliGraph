"""Visualizer for ExamplE."""
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
from graphviz import Source
import pygraphviz as pgv
from ampligraph.explanations.visualizers import AbstractGraphVisualizer, register_visualizer
from ampligraph.explanations.utils import create_cluster_with_edges, create_cluster_with_nodes,\
                                           format_dot_attrs, apply_edges_attributes
@register_visualizer("example_viz")
class GraphVisualizerExamplE(AbstractGraphVisualizer):
    """Visualizer for ExamplE."""
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

        G = nx.DiGraph()
        target_triple = explanation['query_triple']
        G.add_edges_from([(e[0], e[2], {'rel':e[1]})for e in explanation['resembles']])
        G.add_edges_from([(e[0], e[2], {'rel':e[1]})for e in explanation['examples']])
        G.add_edges_from([(e[0], e[2], {'rel':e[1]})for e in explanation['prototype']])
        G.add_edges_from([(target_triple[0], target_triple[2], {'rel':target_triple[1]})])

        # all resembles coming from subject -> pink
        subject_related_color = 'pink'
        example_edges_subject = [(e[0], e[2]) for e in explanation['resembles']\
                if e[1] == 'resembles' and e[0] == target_triple[0]]
        G = apply_edges_attributes(G, example_edges_subject,
                                   {'color':[subject_related_color]*len(example_edges_subject),
                                    'resemblance':['subject']*len(example_edges_subject)})

        # all resembles coming from object -> blue
        object_related_color = 'lightblue'
        example_edges_object = [(e[0], e[2]) for e in explanation['resembles']\
                                if e[1] == 'resembles' and e[0] == target_triple[2]]
        G = apply_edges_attributes(G, example_edges_object,
                                  {'color':[object_related_color]*len(example_edges_object),
                                   'resemblance':['object']*len(example_edges_object)})
        sep = "-|||-"
        # color target_triples
        target_triple_color = 'orange'
        G = apply_edges_attributes(G, [(target_triple[0],target_triple[2])],
                                   attributes={'color':[target_triple_color],
                                               "penwidth": [5], 'style':['dashed']})
        # highlight weights if available
        # TODO

        # cluster spatially
        G = format_dot_attrs(G) # apply graph properties for graphviz display
        g = to_pydot(G)

        g = pgv.AGraph(str(g), name='root')

        nodes =[e[0] for e in  G.nodes(data='resemblance') if e[1] == 'subject']
        nodes.extend([e[0] for e in  G.nodes(data='resemblance') if e[1] == 'object'])

        # spatially cluster examples nodes
        g =  create_cluster_with_nodes(g, nodes, name='examples_nodes')
        # spatially cluster TT nodes
        g =  create_cluster_with_nodes(g, [target_triple[0], target_triple[2]], name='TT')
        edges_resemblance = [e for e in G.edges(data=True) if 'resemblance' in e[2]]
        edges_examples = [e for e in G.edges(data=True) if 'resemblance' not in e[2]]
        g = create_cluster_with_edges(g, edges_resemblance, name='resemblance')
        g = create_cluster_with_edges(g, edges_examples, name='examples')
        g = create_cluster_with_edges(g,
                                      [e for e in G.edges(data=True)\
                                       if e not in edges_resemblance and e not in edges_examples],
                                      name='normal')

        g = g.to_string().replace(sep,"\n")
        path_g = 'graph.dot'
        with open(path_g, 'w') as f:
            f.write(g)
        s = Source.from_file(path_g)
        s.view()    
        return s
