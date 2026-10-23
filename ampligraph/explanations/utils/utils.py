"""Helper functions."""


def apply_edges_attributes(G, edges, attributes):
    """Apply attribute to a list of edges from graph G.

       Parameters
       ----------
       G: nx.Graph
           networkx graph to be modified.
       edges: list
           list of edges from G to apply attributes to.
       attributes: dict
           dictionary with attributes where each attribute is
                   a list of size len(edges) with the attribute value.

       Returns
       -------
       G: nx.Graph
           original networkx graph enhanced with attributes.

       Example
       -------
       >>> G = nx.random_graphs.barabasi_albert_graph(20,19)
       >>> G1 = apply_edges_attributes(G, [[9,19],[19,2]], {'color':['yellow','red']})
       >>> G1.edges(data=True)
       EdgeDataView([(0, 19, {}), (2, 19, {'color': 'red'}),
                     (3, 19, {}), (9, 19, {'color': 'yellow'}), ...])
    """
    for i, edge in enumerate(edges):
        for attr in attributes:
            G[edge[0]][edge[1]][attr] = attributes[attr][i]

    return G

def apply_nodes_attributes(G, nodes, attributes):
    """Apply attribute to a list of nodes from graph G.

       Parameters
       ----------
       G: nx.Graph
            networkx graph to be modified.
       nodes: list
            list of nodes from G to apply attributes to.
       attributes: dict
            dictionary with attributes where each attribute is
                   a list of size len(edges) with the attribute value.

       Returns
       -------
       G: nx.Graph
           original networkx graph enhanced with attributes.
    """
    for i, node in enumerate(nodes):
        for attr in attributes:
            G.nodes[node][attr] = attributes[attr][i]

    return G


def create_cluster_with_nodes(src, nodes, name='child', color="white"):
    """Create spatial cluster for visualizing explanation graph with graphviz.

       Parameters
       ----------
       src: Source
           graph.
       nodes: list
           nodes to be clustered together.
       name: str
           name of the cluster.
       color: str
           color of the cluster, default white.

       Return
       ------
       src: Source
           updated graph with cluster.
    """
    src.add_subgraph(name='cluster_{}'.format(name), color=color)
    cluster = src.subgraphs()[-1]
    for node in nodes:
        cluster.add_node(node)
    return src


def create_cluster_with_edges(src, edges, name='child', color="white"):
    """Create spatial cluster for visualizing explanation graph with graphviz.

       Parameters
       ----------
       src: Source
           graph.
       edges: list
           edges to be clustered together.
       name: str
           name of the cluster.
       color: str
           color of the cluster, default white.

       Return
       ------
       src: Source
           updated graph with cluster.
    """

    src.add_subgraph(name='cluster_{}'.format(name), color=color)
    cluster = src.subgraphs()[-1]
    for edge in edges:
        if src.has_edge(edge[0], edge[1]):
            src.remove_edge(edge[0], edge[1])
        cluster.add_edge(edge[0], edge[1], **edge[2])

    return src


def update_attrs(which, attrs, G):
    """Helper function to update attributes of graph (graphviz)."""
    added = []
    for key, val in attrs.items():
        if key not in G.graph[which]:
            G.graph[which][key] = val
            added.append(key)
    return G

def clean_attrs(which, added, G):
    """Helper function to clean attributes of graph (graphviz)."""
    for attr in added:
        del G.graph[which][attr]
    if not G.graph[which]:
        del G.graph[which]
    return G

def format_dot_attrs(G, reset=True):
    """Helper function to apply attributes to graph (graphviz)."""
    attrs = ["edge", "node", "graph"]
    for attr in attrs:
        if attr not in G.graph or reset:
            G.graph[attr] = {}

    edge_attrs = {"fontsize": "10"}
    node_attrs = {}
    graph_attrs = {"rankdir":"LR", "splines":"true"}

    update_attrs("edge", edge_attrs, G)
    update_attrs("node", node_attrs, G)
    update_attrs("graph", graph_attrs, G)

    return G
