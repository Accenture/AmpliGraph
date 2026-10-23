"""Example-based explanation heuristic."""
import copy
import itertools
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ampligraph.explanations.explainers import AbstractExplainer, register_explainer
from collections import Counter
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import pygraphviz
    import graphviz
    from ampligraph.explanations.visualizers import VISUALIZERS_REGISTRY
except ImportError:
    logger.warning("[example] Without pygraphviz, pydot, and graphviz you won't be able to use visualizers module.")


def explanation_graph_assembler(examples, prototype, target_triple):
    """Assembles examples, prototype and target triple into a final explanation graph.

       Parameters
       ----------
       examples: np.array
           examples graph in a form of list of triples belonging to the examples graph
                 coming from examples_filter.
       prototype:  np.array
           prototype graph in a form of list of triples coming from prototype aggregator.
       target_triple: np.array
           triple for which explanations was requested.

       Returns
       -------
       explanation: np.array
           explantion graph triples.
    """
    explanation = [target_triple.tolist()]
    explanation.extend(examples)
    explanation.extend(prototype)
    if len(prototype) > 0:
        if len(prototype[0]) == 4:
            return explanation
    explanation = np.unique(np.array(explanation), axis=0).tolist()
    return explanation


def aggregate(G, mode="NE", strategy="strict", weights=False):
    """Helper function implementing aggregation strategies.

       Parameters
       ----------
       G: list
           lists of graphs to aggregate.
       mode: str
           which elements to make operations on (N nodes, E - edges, NE - nodes and edges).
       strategy: str
           procedure to follow when aggregating neighbourhood (permissive, strict).

            - permissive: union of examples neighbourhoods intersected with target triple
              neighbourhood with weights being a number of occurences.
            - strict: intersection of examples neighbourhoods and target triple neighbourhood.
            - weights [True]: only used with permissive strategy, returns weights along
              with a subgraph.

       Returns
       -------
       prototype: np.array
           prototype graph in a form of list of triples.
    """
    nodes = []
    for graph in G:
        nodes.extend(graph.nodes())
    for graph in G:
        # nx intersection requires all graphs to have same nodes sets
        graph.add_nodes_from(nodes)
    if strategy == "permissive":
        prototype = union(G[1:])
        prototype = intersection([G[0], prototype], mode=mode)
    if strategy == "strict":
        prototype = intersection(G, mode=mode)
    if weights:
        prototype_triples = [(t[0],
                              t[2]['rel'],
                              t[1],
                              t[2]['weight']) for t in prototype.edges(data=True)]
    else:
        prototype_triples = [(t[0],
                              t[2]['rel'],
                              t[1]) for t in prototype.edges(data=True)]

    if len(prototype_triples) > 0:
        return np.unique(prototype_triples, axis=0).tolist()
    return []


def nodes_edges(graphs, prototype):
    """Helper function to keep attibutes in the intersection graph."""
    edges = copy.deepcopy(prototype.edges())
    # iterate over union of edges from neighbourhoods
    for edge in edges:
        # get corresponding relation type in target triple
        rel = graphs[0][edge[0]][edge[1]]['rel']
        # iterate over example graphs excluding target triple neighbourhood
        for graph in graphs[1:]:
            # if there isn't a relation type (predicate for given edge)
            if rel is not None:
               # if this predicate type is different from what we expected it to be
                if rel != graph[edge[0]][edge[1]]['rel']:
                    if prototype.has_edge(edge[0], edge[1]):
                        # remove this edge from prototype
                        prototype.remove_edge(edge[0], edge[1])
                    continue
            # set relation type to the one from the first graph
            rel = graph[edge[0]][edge[1]]['rel']
    return prototype

def edges_prototype(graphs):
    """Returns edges overlap prototype betwen graphs."""
    count = len(graphs) - 1
    edges = []
    edges_counts = []
    for e in range(count+1):
        edges.append(set([edge[2]['rel'] for edge in list(graphs[e].edges(data=True))]))
        edges_counts.append(Counter([edge[2]['rel'] for edge in list(graphs[e].edges(data=True))]))

    united_edges = Counter([x for d in edges for x in list(d)])
    G = graphs[0]
    ### Add counter weights:
    weighted_triples = [(edge[0], edge[1], {'rel': edge[2]['rel'], 'weight':united_edges[edge[2]['rel']]}) for edge in list(G.edges(data=True)) if edge[2]['rel'] in united_edges]

    prototype = nx.DiGraph()
    prototype.add_edges_from(weighted_triples)
    return prototype

def intersection(graphs, mode="NE"):
    """Wrapper function around networkx intersection to copy attributes.

       Parameters
       ----------
       graphs: list
           graphs to intersect.
       mode: str
           what to intersect? Edges (E), Nodes (N) or Nodes and Edges (NE).
    """
    # doesn't take into account overlap of edges types
    prototype = nx.intersection_all(graphs)
    if mode == "NE":
        prototype = nodes_edges(graphs, prototype)
        rel = nx.get_edge_attributes(graphs[0], "rel")
        nx.set_edge_attributes(prototype, rel, "rel")
    
        weight = nx.get_edge_attributes(graphs[1], "weight")
        if weight == {}:
            weight = {edge: 1 for edge in graphs[0].edges()}
        nx.set_edge_attributes(prototype, weight, "weight")

    elif mode == "E":
        # only for 1-hop
        prototype = edges_prototype(graphs)
    return prototype

def union(graphs):
    """Wrapper around networkx compose_all function, to
       include also weights in terms of occurences of edges.
       This union operation makes union on sets of nodes along with sets of edges.
       Specialization of this function to include only union of edges
       may be needed for graphs with huge hubs.

       Parameters
       ----------
       graphs: list
           graphs for which union to take.

       Returns
       -------
       prototype: nx.Graph
           union of input graphs with weights.
    """
    prototype = nx.compose_all(graphs)
    for edge in prototype.edges(data=True):
        cnt = 0
        for graph in graphs:
            if graph.has_edge(*edge[:2]):
                cnt += 1
        prototype[edge[0]][edge[1]]['weight'] = cnt
    return prototype

@register_explainer("example")
class ExamplE(AbstractExplainer):
    """ExamplE - example-based explainer for Knowledge Graph Embeddings models.

       Explains model's prediction for a target triple by finding
       similar examples and aggregating their neighbourhood into explanation graph.
       Consists of four steps: neighbourhood sampler, filter for example triples,
       prototype aggregator and explanation graph assembler.

       Example
       -------
       >>> X = {'train': np.array([['a','b','c'],...]),
                'test':np.array(['d','e','f']...]),
                'valid':...}
       >>> model = restore_model(model_name_path='./model.pkl') # needs to be calibrated
       >>> target_triple = ['d','e','f']
       >>> explainer = ExamplE(X, model)
       >>> expl = explainer.predict_explain(target_triple)
       >>> expl
       {'prediction': array([-15.234796], dtype=float32),
        'probability': array([0.66178256], dtype=float32),
        'examples': array(['d', 'e', 'h']),
        'resembles': array(['d','resembles','d']),
        'prototype': array([...]),
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
        self.__name__ = "examplE"
        try:
            self.visualizer = VISUALIZERS_REGISTRY.get("example_viz")(X)
        except:
            logger.warning("Visualizer for {} not found!".format(self.__name__))
        self.nodes = set(self.X['train'][:,0]).union(set(self.X['train'][:,2]))
        self.nodes = list(self.nodes)
        self.rels = list(set(self.X['train'][:,1]))
        self.embedded_rels = self.model.get_embeddings(self.rels, 'r')
        self.embedded_ents = self.model.get_embeddings(self.nodes)
        self.initialized_nn = False
        self.k = len(self.embedded_ents[0])
        self.nodes_emb_dict = dict(zip(self.nodes, self.embedded_ents))

    def initialize_nn(self, epsilon=None, m=25, metric='cosine'):
        neighs = NearestNeighbors(n_neighbors=m, metric=metric, radius=epsilon)
        self.ents_n_model = neighs.fit(self.embedded_ents)
        neighs = NearestNeighbors(n_neighbors=m, metric=metric, radius=epsilon)
        self.rels_n_model = neighs.fit(self.embedded_rels)
        self.initialized_nn = True

    def get_neighbourhood(self, node, epsilon=None, m=25, metric='cosine', kind='e', score=False):
        """Get neighbourhood in the embedding space.
        
           Parameters
           ----------
           node: str
              node for which neighbours will be drawn.
           epsilon: float
              radius to be used in similarity measure.
           m: int
              number of neighbours to be picked.
           metric: str
              similarity metric to be used.
           kind: str
              'e' for entity or 'r' for relation

           Return
           ------
           neighbours: set
              set of neighbours.
        """
        if not self.initialized_nn:
            self.initialize_nn(epsilon, m, metric)
        emb_node = self.model.get_embeddings([node], kind)
        if kind == 'e':
            ng = self.ents_n_model.kneighbors(emb_node.reshape(1,-1), m)
            raw_neighs = np.array(self.nodes)[ng[1]]

            neighbours = set(raw_neighs[0])
            if score:
                neighbours_dict = dict(zip(raw_neighs[0].tolist(), ng[0].tolist()[0]))
                return neighbours, neighbours_dict
        else:
            ng = self.rels_n_model.kneighbors(emb_node.reshape(1,-1), m)
            raw_neighs = np.array(self.rels)[ng[1]]
            neighbours = set(raw_neighs[0])

            if score:
                scores = ng[0]
                neighbours_dict = dict(zip(raw_neighs[0].tolist(), ng[0].tolist()[0]))
                return neighbours, neighbours_dict
        return neighbours, {}


    def batch_explain(self, X, mode="NE", strategy="strict", hop=1,
                              sampling_mode="so", m=25, epsilon=None, metric='cosine',
                              weights=False, exclude_p=True, score=False):
        """Explain given set of triples with infuential examples.

           Parameters
           ----------
           X:  np.array
               triples for which explanations were requested.
           exclude_p: bool
               Whether to search the neighbouring embeddings of predicate (False).
               If True will only use the target triples predicates.
           score: bool
               Whether to rank examples.

           Returns
           -------
           explanations: np.array
               array of influential example triples as explanations.
        """
        if not isinstance(X, type(np.array([]))):
            X = np.asarray(X)
        fc = np.frompyfunc(self.get_neighbourhood,6,2)
        fccp = np.frompyfunc(itertools.product,3,1)
        s, s_d = fc(X[:,0], epsilon, m, metric, 'e', score)
        o, o_d = fc(X[:,2], epsilon, m, metric, 'e', score)
        if exclude_p:
            p, p_d = [{e} for e in X[:,1]], [{} for i in range(len(X))]
        else:
            p, p_d = fc(X[:,1], epsilon, m, metric, 'r', score)
        out = fccp(s,p,o)
        tmp_X = set(map(tuple, self.X['train']))
        examples = np.asarray([np.asarray(list(set(map(tuple, elem)).intersection(tmp_X))) for elem in out])
        examples = self.process_examples(examples, s_d, o_d, p_d)
        return examples


    def process_examples(self, examples, s_d, o_d, p_d):
        """Post process examples"""
        logging.warning(f"Examples shape: {np.shape(examples)}, {np.shape(s_d)}, {np.shape(o_d)}, {np.shape(p_d)} ")
        examples_scored = []
        measure_relevance = lambda s,p,o: np.mean([s,p,o])

        for i, b in enumerate(examples):
            try:
                ex_score = []
                for e in b:
                    score = measure_relevance(s_d[i][e[0]] if e[0] in s_d[i] else 0, p_d[i][e[1]] if e[1] in p_d[i] else 0, o_d[i][e[2]] if e[2] in o_d[i] else 0.)
                    ex_score.append(np.array([e[0],e[1],e[2],float(score)]))
                ex_score = np.asarray(ex_score)
                df = pd.DataFrame()
                df['s'] = ex_score[:,0]
                df['p'] = ex_score[:,1]
                df['o'] = ex_score[:,2]
                df['dist'] = ex_score[:,3]
                # sort scores in ascending order as these are distances -> the smaller the better!
                df = df.sort_values(by=['dist']).values
                examples_scored.append(df)
            except Exception as e:
                logging.warning("No examples found, please try using different parameters.")
        return np.asarray(examples_scored)


    def neighbourhood_sampler(self, target_triple, mode="so", m=25, epsilon=None, metric='cosine', score=False):
        """Samples neighbourhood of target triple.

           Parameters
           ----------
           target_triple: np.array
               triple for which explanations was requested.
           mode: str
               signals what neighbours to consider when constructing examples. Has to contain
               at least 2 out of "sop" (s - subject, o - object, p - predicate).
           m: int
               limit for how many neighbours to draw.
           epsilon: float
               how similar two embeddings has to be to be considered neighbours.
           metric: str
               similarity function to be used.

           Returns
           -------
           sets: dict
               2 or 3 of neighbours sets for subject, object or predicate in a form
               ``{'subjects': {target_triple[0]}, 'objects': {target_triple[2]},
               'predicates': {target_triple[1]}}``.
        """
        sets = {}
        if "s" in mode or "S" in mode:
            sets['subjects'] = {target_triple[0]}
        if "o" in mode or "O" in mode:
            sets['objects'] = {target_triple[2]}
        if "p" in mode or "P" in mode:
            sets['predicates'] = {target_triple[1]}

        for key in sets:
            elems = self.get_neighbourhood(sets[key].pop(), epsilon=epsilon, m=m, metric=metric, score=score)
            sets[key].update(elems[0])

        if 'predicates' not in sets:
            sets['predicates'] = {target_triple[1]}
        if 'subjects' not in sets:
            sets['subjects'] = {target_triple[0]}
        if 'objects' not in sets:
            sets['objects'] = {target_triple[2]}

        return sets


    def examples_filter(self, sets, target_triple):
        """Generates potential examples and filter feasible ones.

           Parameters
           ----------
           sets: dict
               2 or 3 of neighbours sets coming from neighbourhood_sampler for subject,
               object or predicate in a form {'subjects':{target_triple[0]},
               'objects':{target_triple[2]}, 'predicates':{target_triple[1]}}.
           target_triple: np.array
               triple being queried.
           Returns
           -------
           examples: list
               examples graph in a form of list of triples belonging to the examples graph.
        """
        candidates_pool = itertools.product(sets['subjects'], sets['predicates'], sets['objects'])
        tmp_X = set(map(tuple, self.X['train']))
        examples = list(set(map(tuple, candidates_pool)).intersection(tmp_X))
        triples = []
        for example in examples:
            triples.append((target_triple[0], 'resembles', example[0]))
            triples.append((target_triple[2], 'resembles', example[2]))

        examples.extend(triples)
        if len(examples) == 0:
            return np.array([])
        return np.unique(examples, axis=0).tolist()


    def prototype_aggregator(self, examples, target_triple, mode="NE",
                                   strategy="strict", hop=1, weights=False):
        """ Aggregates examples neighbourhoods into a prototype graph.

            Parameters
            ----------
            examples: np.array
                examples graph in a form of list of triples belonging to the examples graph
                      coming from examples_filter.
            target_triple: np.array
                triple for which explanations was requested.
            mode: str
                which elements to make operations on (N nodes, E - edges, NE - nodes and edges).
            strategy: str
                procedure to follow when aggregating neighbourhood (permissive, strict).

                - permissive: union of examples neighbourhoods intersected with target triple
                  neighbourhood with weights being a number of occurences.
                - strict: intersection of examples neighbourhoods and target triple neighbourhood.

            Returns
            -------
            prototype: np.array
                prototype graph in a form of list of triples.
        """
        G = self.navigator.get_n_hop_neighbourhood_subgraph(target_triple, n_hop=hop)
        n_hood = G.edges(data=True)
        mapping = {target_triple[0]:"S-?", target_triple[2]:"O-?"} #, example[1]:"P-?"
        H = nx.relabel_nodes(G, mapping)
        graphs = [H]

        for example in examples:
            if example[1] != 'resembles':
                G = self.navigator.get_n_hop_neighbourhood_subgraph(example, n_hop=hop)
                mapping = {example[0]:"S-?", example[2]:"O-?"} #, example[1]:"P-?"
                H = nx.relabel_nodes(G, mapping)
                graphs.append(H)

        if len(graphs) == 1:
            prototype = np.array([])

        else:
            prototype =  aggregate(graphs, mode=mode, strategy=strategy, weights=weights)

            mapping = {"S-?":target_triple[0], "O-?":target_triple[2]} #, example[1]:"P-?"
            if len(prototype) > 0:
                if len(prototype[0]) == 4:
                    prototype =  [(mapping[x[0]] if x[0] in mapping else x[0], mapping[x[1]]\
                                  if x[1] in mapping else x[1], mapping[x[2]]\
                                  if x[2] in mapping else x[2], x[3]) for x in prototype]
                else:
                    prototype =  [(mapping[x[0]] if x[0] in mapping else x[0], mapping[x[1]]\
                                  if x[1] in mapping else x[1], mapping[x[2]]\
                                  if x[2] in mapping else x[2]) for x in prototype]

        return prototype, n_hood


    def predict_explain(self, target_triple, mode="NE", strategy="strict", hop=1,
                              sampling_mode="so", m=25, epsilon=None, metric='cosine',
                              weights=False):
        """Explain given triple.

           Parameters
           ----------
           target_triple:  np.array
               triple for which explanations was requested.

           Returns
           -------
           explanation: dict
               explanation dictionary containing explanation graph.
        """
        # only single triple prediction with no scores in this function, for scores and mutiple switch to batch_explain and set score=True

        sets = self.neighbourhood_sampler(target_triple, mode=sampling_mode,
                                          m=m, epsilon=epsilon, metric=metric)
        examples = self.examples_filter(sets, target_triple)
        prototype, n_hood = self.prototype_aggregator(examples, target_triple,  mode=mode,
                                                      strategy=strategy, hop=hop, weights=weights)
        self.explanation_graph = explanation_graph_assembler(examples, prototype, target_triple)
        prediction = self.model.predict([target_triple])
        proba = self.model.predict_proba([target_triple])
        if len(examples) == 0:
            logger.info("No examples found.")
            examples_resembles = np.array([]) 
            examples_triples = np.array([])
        else:
            examples_resembles = np.asarray(examples)[np.asarray(examples)[:,1]=='resembles']
            examples_triples = np.asarray(examples)[np.asarray(examples)[:,1]!='resembles']
        n_hood = np.asarray([[t[0], t[2]['rel'], t[1]] for t in n_hood])
        explanation_dictionary = {'prediction':prediction,
                                  'probability': proba,
                                  'examples': examples_triples,
                                  'resembles': examples_resembles,
                                  'prototype': prototype,
                                  'query_triple': target_triple,
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

           Returns
           -------
           viz: Source
               pygraphviz Source object.

        """
        viz = self.visualizer.visualize(explanation, fmt=fmt, fname=fname)
        return viz
