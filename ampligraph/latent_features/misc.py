import numpy as np
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed
import os
from ast import literal_eval
import random
from deap import base, creator, tools, algorithms
from ..evaluation import quality_loss_mse



SUBJECT = 0
PREDICATE = 1
OBJECT = 2
DEBUG = True


def get_neighbour_triples_evolutionary(triple, graph, seed=0, NGEN=50, CXPB=0.5, MUTPB=0.5, MU=10, LAMBDA=30, k=4,
                                       model_class=None, model_hyperparams=None):

    """

    Parameters
    ----------
    triple : np.ndarray, shape [1, 3], dtype=int
        The candidate triple [subject, predicate, object]
    graph : np.ndarray, shape [n, 3], dtype=int
        The graph
    seed : int
        A random seed.
    NGEN : int
        The number of generations to run the genetic algorithm. [default: 50]
    CXPB : float
        The probability that offspring is produced by crossover. CXPB and MUTPB must sum to 1. [default: 0.5]
    MUTPB : float
        The probability that offspring is produced by mutation. CXPB and MUTPB must sum to 1. [default: 0.5]
    MU : int
        The number of fit offspring to select to populate the next generation. [default: 10]
    LAMBDA : int
        The number of individuals to produce in each generation. [default: 30]
    k : int
        The number of hops for the exhaustive search. This will constrain the GA solutions.
        This is a major speed bottleneck, so there's a trade off between speed and novelty of solutions. [default: 4]
    model_class : ampligraph.latent_models.EmbeddingModel
        The KG embedding model to use.
    model_hyperparams : dict
        The hyperparameters of the KG embedding model.

    Returns
    -------
    pareto_optimal : list [subgraph : tuple (np.array, float)]
        A list of tuples containing Pareto optimal subgraphs and their corresponding MSE.

    """


    # fitness function - trains an embedding model on a subset (individual)
    # returns MSE between original model and subset model and number of triples
    def fitness_fx(individual, neighbours):
        ind = np.array(individual).astype(bool)
        subset = neighbours[ind]
        subset_model = train_model(subset)
        return ((quality_loss_mse(nbr_model, subset_model, subset), sum(ind)))

    # convert a list of triples into a binary vector
    def subset_vector(graph, subset):
        vec = np.array([int((triple == subset).all(1).any()) for triple in graph])
        return (vec)

    # create individual
    def initIndividual(icls, content):
        return icls(content)

    # intialise the population (first generation)
    # randomly sample LAMBDA elements from k-hop neighbours to generate LAMBDA individuals
    def initPopulation(pcls, ind_init, neighbours):
        return pcls(
            ind_init(subset_vector(neighbours, neighbours[sorted(random.sample(range(len(neighbours)), LAMBDA)), :]))
            for c in range(0, LAMBDA))

    # feasibility function
    # if a subset contains fewer than 5 or more than 50 triples, don't evaluate
    def feasible(individual):
        if 5 <= sum(individual) <= 50:
            return True
        else:
            return False

    # train the relational model over subset
    # since this is part of the fitness function, the priority here is speed, so I'd consider changing the params to
    # help ensure faster training
    def train_model(training_data):
        model = model_class(**model_hyperparams, batches_count=1, verbose=False)
        model.fit(training_data)
        return (model)


    # constrain solutions by performing a k-hop exhaustive search of the graph and returning the subset
    # of the graph corresponding to this neighbourhood
    nbrs = get_neighbour_triples_exhaustive(graph=graph, triple=triple, k=k)
    # train model over exhaustive neighbourhood
    nbr_model = train_model(nbrs)

    # Fitness function - weights are for the minimisation of MSE (between exhaustive neighbourhood and subset) and number of triples
    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
    # Individual inherits from "list" and has a fitness property
    creator.create("Individual", list, fitness=creator.Fitness)

    # define GA operations
    toolbox = base.Toolbox()
    # each individual is a binary vector representing LAMBDA triples sampled from the k-hop neighbourhood
    toolbox.register("individual_guess", initIndividual, creator.Individual)
    # generate LAMBDA individuals
    toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, neighbours=nbrs)
    # register fitness function
    toolbox.register("evaluate", fitness_fx, neighbours=nbrs)
    # if individual does not meet feasibility criteria, do not evaluate and return a high default value
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, (100, 100)))
    # crossover operation
    # pick two random indices and swap values between those positions
    toolbox.register("mate", tools.cxTwoPoint)
    # mutation operation
    # flip 0 / 1
    # indp = probability of each digit being flipped (0 / 1)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    # selection algorithm
    # http://ieeexplore.ieee.org/document/996017/
    toolbox.register("select", tools.selNSGA2)

    # store Pareto-optimal (non-dominated) solutions
    hof = tools.ParetoFront()

    # statistics to calculate for each generation
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # intialise population by randomly sampling from nbrs
    # the number of individuals is determined by LAMBDA
    pop = toolbox.population_guess()

    # run GA - see above for parameter explanations
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)

    # return list of explanatory subgraphs and corresponding MSE
    pareto_opt = [(nbrs[np.array(subset).astype(bool)], literal_eval(str(subset.fitness))[0]) for subset in hof]
    return (pareto_opt)


# def get_neighbour_triples(entity, graph):
#     """
#     Given an entity label e and an ndarray of triples G, returns an ndarray that includes all triples where e appears either as subject or object.
#
#        Parameters
#        ----------
#        entity : str, shape [n, 1]
#            An entity label.
#        graph : np.ndarray, shape [n, 3]
#            An ndarray of triples.
#
#         Returns
#         -------
#         neighbours : np.ndarray, shape [n, 3]
#             An ndarray of triples where e is either the subject or the object.
#     """
#
#     # Get rows and cols where entity is found in graph
#     rows, cols = np.where((entity == graph))
#
#     # In the unlikely event that entity is found in the relation column (index 1)
#     rows = rows[np.where(cols != PREDICATE)]
#
#     # Subset graph to neighbourhood of entity
#     neighbours = graph[rows, :]
#
#     return neighbours


def get_entity_triples(entity, graph):
    """
    Given an entity label e included in the graph G, returns an list of all triples where e appears either as subject or object.
    
        Parameters
        ----------
        entity : str, shape [n, 1]
            An entity label.
        graph : np.ndarray, shape [n, 3]
            An ndarray of triples.

        Returns
        -------
        neighbours : np.ndarray, shape [n, 3]
            An ndarray of triples where e is either the subject or the object.
    """

    # NOTE: The current implementation is slightly faster (~15%) than the more readable one-liner:
    #           rows, _ = np.where((entity == graph[:,[SUBJECT,OBJECT]]))

    # Get rows and cols where entity is found in graph
    rows, cols = np.where((entity == graph))

    # In the unlikely event that entity is found in the relation column (index 1)
    rows = rows[np.where(cols != PREDICATE)]

    # Subset graph to neighbourhood of entity
    neighbours = graph[rows, :]

    return neighbours


def get_neighbour_triples_weighted_walks(triple, graph, weighting='pagerank', k=2, seed=0, page_ranks=None,
                                         frequency_map=None, d=0.85, max_iter=None, **kwargs):
    """
    Given a candidate triple t [s, p, o] included in the graph G, return a weighted random walk of the graph.

    NOTE: To randomly sample the transitions from an arbitrary triple, the global probabilities of neighbouring triples
    are rescaled to a local transition probability using the probability rescale function. Alternatively the softmax
    function could be used, however this squashes the probability ratio when converting from global to local transition
    probability.

        Parameters
        ----------
        triple : np.ndarray, shape [1, 3], dtype=int
            The candidate triple [subject, predicate, object]
        graph : np.ndarray, shape [n, 3], dtype=int
            The graph
        weighting : str 
            The weighting strategy {predicate_frequency, inverse_predicate_frequency, predicate_object_frequency, 
              inverse_predicate_object_frequency, object_frequency, inverse_object_frequency, pagerank} 
               
               Unimplemented: {inverse_object_frequency_split, inverse_pagerank, pagerank_split, pagerank-split, inverse_pagerank_split} 
               [default=pagerank]
        k : int
            Number of hops [default=2]
        page_ranks : dict
            Dictionary of entity:pagerank mappings
        d : float
            Damping factor for pagerank [default=0.85]
        max_iter : int
            maximum number of pagerank iterations over an entity
        frequency_map : dict
            dict where frequency_map[predicate] is frequency of predicate in graph
        seed : int  
            Pseudo-random generator seed to preserve reproducible results [default=0]
        kwargs : 
            Additional weighting strategy specific arguments 
        
        Returns
        -------
        neighbours : np.ndarray, shape [n, 3]
            An ndarray of triples. 
    """

    # Can pre-calculate frequency maps object, predicate, and predicate-object

    # If-statement to run specified weighted walking scheme
    if weighting == 'predicate_frequency':
        return get_neighbour_triples_frequency_weight(triple, graph, type='predicate', k=k, frequency_map=frequency_map)

    elif weighting == 'inverse_predicate_frequency':
        return get_neighbour_triples_frequency_weight(triple, graph, type='predicate', k=k,  frequency_map=frequency_map, inverse_freq=True)

    elif weighting == 'object_frequency':
        return get_neighbour_triples_frequency_weight(triple, graph, type='object', k=k, frequency_map=frequency_map)

    elif weighting == 'inverse_object_frequency':
        return get_neighbour_triples_frequency_weight(triple, graph, type='object', k=k, frequency_map=frequency_map, inverse_freq=True)

    elif weighting == 'predicate_object_frequency':
        return get_neighbour_triples_frequency_weight(triple, graph, type='predicate-object', k=k, frequency_map=frequency_map)

    elif weighting == 'inverse_predicate_object_frequency':
        return get_neighbour_triples_frequency_weight(triple, graph, type='predicate-object', k=k, frequency_map=frequency_map, inverse_freq=True)

    elif weighting == 'inverse_object_frequency_split':
        raise NotImplementedError('Weighted random walk with inverse object frequency split is unimplemented.')

    elif weighting == 'pagerank':

        if page_ranks is None:
            page_ranks = calculate_graph_pagerank(graph, d, max_iter=max_iter)

        return get_neighbour_triples_pagerank_weight(triple, graph, page_ranks=page_ranks, k=k)

    elif weighting == 'inverse_pagerank':
        raise NotImplementedError('Weighted random walk with inverse pagerank is unimplemented.')

    elif weighting == 'pagerank_split':
        raise NotImplementedError('Weighted random walk with pagerank split is unimplemented.')

    elif weighting == 'inverse_pagerank_split':
        raise NotImplementedError('Weighted random walk with inverse pagerank split is unimplemented.')

    else:
        raise ValueError('Invalid weighting strategy: %s' % weighting)


def calculate_graph_pagerank(graph, d=0.85, max_iter=None):
    """
    PageRank algorithm for graphs.

        Parameters:
        -----------
        graph : np.ndarray, shape [n, 3], dtype=int
            The graph
        d : float
            Damping factor for pagerank [default=0.85]
        max_iter : int
            maximum number of iterations over an entity

        Returns
        -------
        page_ranks : dict 
            Dictionary of entity:pagerank mappings 
    """

    # Initialize all entities to uniform probability
    entities = np.unique(graph[:, [SUBJECT, OBJECT]])
    num_entities = len(entities)
    page_ranks = {x: 1 / num_entities for x in entities}

    # safety condition: must at least iterate over each entity.
    if max_iter:
        if type(max_iter) != int or max_iter < 1:
            raise ValueError('Invalid max_iter: %s' % max_iter)

    if DEBUG:
        print('Number of entities in graph: %d' % num_entities)

    # Create dictionaries of in and out links for each entity
    in_links = {entity: graph[graph[:, OBJECT] == entity, SUBJECT] for entity in entities}
    out_links = {entity: graph[graph[:, SUBJECT] == entity, OBJECT] for entity in entities}

    # PageRank algorithm
    converged = False
    iteration = 0

    if max_iter:
        print('max_iter=%d' % max_iter)
        pbar = tqdm(total=max_iter)

    while not converged:

        if max_iter:
            pbar.update(1)

        converged = True

        for entity in tqdm(entities):

            links_sum = np.sum([page_ranks[x] / len(out_links[x]) for x in in_links[entity]])

            updated_rank = (1 - d) + d * links_sum

            previous_rank = page_ranks[entity]
            page_ranks[entity] = updated_rank

            if previous_rank != updated_rank:
                converged = False

        iteration += 1

        if max_iter and (iteration == max_iter):
            print('max_iter reached')
            converged = True # strictly, not converged :d
            break

    if DEBUG:
        print('PageRank converged after %d iterations ' % iteration)

    return page_ranks

def normalize_page_rank_weights(graph, page_ranks):
    """
    Normalizes all page rank weights by outgoing links. 

    In order to form a correct probability model, the individual link weight is normalized
    in accordance to the link weights of all outgoing links of a page. Normalization is performed
    using softmax function.

        Parameters:
        -----------
        graph : np.ndarray, shape [n, 3], dtype=int
            The graph
        page_ranks : float
            Damping factor for pagerank [default=0.85]
        
        Returns
        -------
        page_ranks : dict 
            Dictionary of entity:pagerank mappings 
    """
    # List of graph entities
    entities = np.unique(graph[:, [SUBJECT, OBJECT]])

    # Transition matrix maps entity labels to probability of transitioning to a triple (defined as its index in the graph)
    transition_matrix = {}

    for entity in entities:

        # Get rows where entity is the SUBJECT
        X = np.where(graph[:, SUBJECT] == entity)[0]

        # If it's empty, ignore!
        if len(X) > 0:
            # Get the linked entities
            O = graph[X, OBJECT]

            # Gather page ranks for all outgoing links
            W = [page_ranks[x] for x in O]

            # Normalize via softmax
            WN = softmax(W)

            # Construct dictionary linking index of triple in graph to transition probability
            link_weights = {}
            for i, graph_index in enumerate(X):
                link_weights[graph_index] = WN[i]
        else:
            link_weights = {}

        # Add transition probabilities for this entity
        transition_matrix[entity] = link_weights

    return transition_matrix

def softmax(x):
    """
    Compute softmax values for each sets of scores in x

        Parameters:
        -----------
        x : list, dict
            probability values
    """

    if isinstance(x, list):
        e_x = np.exp(x - np.max(x)) # using the max softmax subtraction trick to avoid potential numerical issues with scaling summand
        return e_x / np.sum(e_x)

    elif isinstance(x, dict):
        e_tot = np.sum(np.exp([x[k] for k in x.keys()]  ))
        return {k: (np.exp(x[k]) / e_tot) for k in x.keys()}

    else:
        raise ValueError('Softmax function input must be a list or dict of key:float pairs.')

def normalize_probabilities(x):
    """
    Normalize a list or dict of probabilities using division by sum.

        Parameters:
        -----------
        x : list, dict
            probability values
    """
    if isinstance(x, list):
        total = np.sum(x)
        return x / total
    elif isinstance(x, dict):
        total = np.sum([x[k] for k in x.keys()])
        return {k: (x[k] / total) for k in x.keys()}
    else:
        raise ValueError('Softmax function input must be a list or dict of key:float pairs.')


def get_neighbour_triples_pagerank_weight(triple, graph, k=2, d=0.85, max_iter=None, seed=1337, page_ranks=None, **kwargs):
    """
    PageRank algorithm to determine important entities and follow them in a random walk. 
    
    NOTE: In contrast to other weighted random walk methods, the pagerank weight random walk looks at direction of
     triple relationship. 
    
        Parameters
        ----------
        triple : np.ndarray, shape [1, 3], dtype=int
            The candidate triple [subject, predicate, object]
        graph : np.ndarray, shape [n, 3], dtype=int
            The graph
        k : int
            Number of hops [default=2]
        d : float
            Damping factor for pagerank [default=0.85]
        max_iter : int
            maximum number of iterations over an entity
        page_ranks : dict
            Dictionary of entity:pagerank mappings
        seed : int
            Pseudo-random generator seed to preserve reproducible results [default=0]
        kwargs : 
            Additional weighting strategy specific arguments 

        Returns
        -------
        neighbours : np.ndarray, shape [n, 3]
            An ndarray of triples. 
    """

    if page_ranks is None:
        page_ranks = calculate_graph_pagerank(graph, d, max_iter=max_iter)

    # Set random seed
    np.random.seed(seed)

    # Normalize page rank weights and get transition matrix
    transition_matrix = normalize_page_rank_weights(graph, page_ranks)

    # Init neighbours
    neighbours = [triple]

    # Go FORWARD from triple
    head_of_walk = triple[OBJECT]

    i = 0
    while i < k:

        # Get entity transition matrix entries
        T = transition_matrix[head_of_walk]

        # If no transitions, end the walk
        if len(T) == 0:
            break

        # Get link transition weights (convert to inverse if specified)
        link_weights = [T[x] for x in T.keys()]

        # Choose from triples using transition matrix probability
        walk = np.random.choice(list(T.keys()), p=link_weights)

        # Randomly choose a triple
        walk_triple = graph[walk, :].tolist()

        neighbours.append(walk_triple)
        head_of_walk = walk_triple[OBJECT]
        i += 1

    neighbours = np.array(neighbours)

    return neighbours


def calculate_frequency_map(graph, type, inverse_freq=False):
    """
        Calculate a frequency map of entities or entity pairs in the graph.

        It is a dict of dicts where frequency_map[predicate][object] is frequency of predicate->object pair in graph.

            Parameters
            ----------
            graph : np.ndarray, shape [n, 3], dtype=int
                The graph
            type : str
                The graph entity type or pair on which to calculate frequency ['object', 'predicate', 'predicate-object']
            inverse_freq : bool
                Invert frequencies in graph [default=False]
        """


    num_triples = len(graph)

    if type == 'object':

        unique_objects = np.unique(graph[:, OBJECT])

        if inverse_freq:
            frequency_map = {o: 1 - np.sum(o == graph[:, OBJECT]) / num_triples for o in unique_objects}
        else:
            frequency_map = {o: np.sum(o == graph[:, OBJECT]) / num_triples for o in unique_objects}

    elif type == 'predicate':

        unique_predicates = np.unique(graph[:, PREDICATE])

        if inverse_freq:
            frequency_map = {p: 1 - np.sum(p == graph[:, PREDICATE]) / num_triples for p in unique_predicates}
        else:
            frequency_map = {p: np.sum(p == graph[:, PREDICATE]) / num_triples for p in unique_predicates}

    elif type == 'predicate-object':

        unique_objects = np.unique(graph[:, OBJECT])
        unique_predicates = np.unique(graph[:, PREDICATE])

        if inverse_freq:
            frequency_map = {
                p: {
                    o: 1 - np.sum(np.all(np.array([p, o]) == graph[:, [PREDICATE, OBJECT]], axis=1)) / num_triples
                    for o in unique_objects}
                for p in unique_predicates}
        else:
            frequency_map = {
                p: {
                    o: np.sum(np.all(np.array([p, o]) == graph[:, [PREDICATE, OBJECT]], axis=1)) / num_triples
                    for o in unique_objects}
                for p in unique_predicates}


    else:
        raise ValueError("Invalid type '%s' specified for calculating frequency map. " \
                         "Valid options are ['object', 'predicate', 'predicate-object']" % type)

    return frequency_map


def get_neighbour_triples_frequency_weight(triple, graph, type, k=2, frequency_map=None, inverse_freq=False, seed=1337):
    """
    Follow entities by their relative frequency in the graph.

        Parameters
        ----------
        triple : np.ndarray, shape [1, 3], dtype=int
            The candidate triple [subject, predicate, object]
        graph : np.ndarray, shape [n, 3], dtype=int
            The graph
        type : str
            The graph entity type or pair on which to calculate frequency ['object', 'predicate', 'predicate-object']
        k : int
            Number of hops [default=2]
        inverse_freq : bool
            Weight random walk by inverse frequency of predicate in graph [default=False]
        frequency_map : dict
            dict where frequency_map[predicate] is frequency of predicate in graph
        seed : int
            Pseudo-random generator seed to preserve reproducible results [default=0]
        kwargs :
            Additional weighting strategy specific arguments

        Returns
        -------
        neighbours : np.ndarray, shape [n, 3]
            An ndarray of triples.
    """
    import functools

    # Set random seed
    np.random.seed(seed)

    # Calculate frequency map if it's not provided. For performance over multiple triples, this should be pre-computed.
    if frequency_map is None:
        frequency_map = calculate_frequency_map(graph, type=type, inverse_freq=inverse_freq)

    # Instantiate get_neighbour_global_prob functions as lambda expressions based on type
    if type.lower() == 'object':
        get_neighbour_global_prob = lambda N: [frequency_map[n[OBJECT]] for n in N]
    elif type.lower() == 'predicate':
        get_neighbour_global_prob = lambda N: [frequency_map[n[PREDICATE]] for n in N]
    elif type.lower() == 'predicate-object':
        get_neighbour_global_prob = lambda N: [frequency_map[n[PREDICATE]][n[OBJECT]] for n in N]
    else:
        raise ValueError("Invalid type '%s' specified for getting neighbour triples by frequency. " \
                         "Valid options are ['object', 'predicate', 'predicate-object']" % type)

    # Initialize the walk, and randomly select a direction
    weighted_walk = [triple]
    head_of_walk = np.random.choice([triple[SUBJECT], triple[OBJECT]])

    # Walk, don't run!
    i = 0
    while i < k:

        # Get triples connected to current head of walk
        neighbourhood = get_entity_triples(head_of_walk, graph)

        # Remove the triple last visited (to avoid cycles)
        neighbourhood = neighbourhood[~np.all(neighbourhood == weighted_walk[i], axis=1), :]

        if len(neighbourhood) == 0:
            break

        # Probability of neighbourhood object over entire graph
        global_prob = get_neighbour_global_prob(neighbourhood)

        # Rescale global probabilities to transition probability from head of walk
        local_transition_prob = normalize_probabilities(global_prob)

        # Choose neighbour triple based on transition probability
        walk_triple_index = np.random.choice(list(range(len(neighbourhood))), p=local_transition_prob)
        walk_triple = neighbourhood[walk_triple_index, :].tolist()

        # Append it to list of neighbours, get next direction of walk
        weighted_walk.append(walk_triple)
        head_of_walk = walk_triple[SUBJECT] if walk_triple[SUBJECT] != head_of_walk else walk_triple[OBJECT]
        i += 1

    weighted_walk = np.array(weighted_walk)

    return weighted_walk


def get_neighbour_triples_uniform_walks(triple, graph, k=2, seed=0):
    """
    Given a candidate triple t [s, p, o] included in the graph G, return a uniformly weighted random walk of the graph.

        Parameters
        ----------
        triple : np.ndarray, shape [1, 3], dtype=int
            The candidate triple [subject, predicate, object]
        graph : np.ndarray, shape [n, 3], dtype=int
            The graph
        k : int
            Number of hops [default=2] 
        seed : int  
            Pseudo-random generator seed to preserve reproducible results [default=0]

        Returns
        -------
        neighbours : np.ndarray, shape [n, 3]
            An ndarray of triples. 
    """

    # Init neighbours
    neighbours = [triple]

    # Set random seed
    np.random.seed(seed)

    # Randomly select direction of walk
    head = np.random.choice([triple[SUBJECT], triple[OBJECT]])

    i = 0
    while i < k:

        # Get triples connected to current head of walk
        N = get_entity_triples(head, graph)

        # Remove the triple last visited (to avoid cycles)
        N = N[~np.all(N == neighbours[i], axis=1), :]

        if len(N) == 0:
            break

        # Randomly choose a triple
        walk = np.random.choice(range(len(N)))
        walk_triple = N[walk,:].tolist()

        neighbours.append(walk_triple)
        head = walk_triple[SUBJECT] if walk_triple[SUBJECT] != head else walk_triple[OBJECT]
        i += 1

    neighbours = np.array(neighbours)

    return neighbours


def get_neighbour_entities(entity, graph, ent_type):
    """
    Given an entity label integer encoding, ndarray of integer triples G and entity type returns a ndarray of unique neighbour entities give head or tail position.

       Parameters
       ----------
       entity : int
           An entity label.
       graph : np.ndarray, shape [n, 3]
           An ndarray of triples.
        ent_type: str
           Type of entity we want to query for, supported options are either "head" or "tail"

       Returns
       -------
       H : np.ndarray, shape [n,]
           An ndarray of int encoded unique neighbour entities.

    """

    # TODO: SEEMINLY DUPLICATE FUNCTIONALITY WITH get_neighbour_triples(e, g) and get_entity_triples(e, g)

    # Get head and tail neighbours of the entity e
    H_tr, T_tr = get_neighbour_triples_head_tail(entity, graph)

    # If required entity type is head, fetch all unique head entities
    if ent_type == 'head':
        E = np.unique(H_tr[:, 0])
        E = E[E != entity]

    # If required entity type is tail, fetch all unique tail entities
    elif ent_type == 'tail':
        E = np.unique(T_tr[:, 2])
        E = E[E != entity]
    else:
        raise ValueError('Bad ent_type argument!')

    return E


def get_neighbour_triples_exhaustive(triple, graph, k=2, limit=None, seed=1337):
    """
    Given a candidate triple t [s, p, o] included in the graph G, return a k-hop exhaustive neighbourhood of t.

        Parameters
        ----------
        triple : np.ndarray, shape [1, 3], dtype=int
            The candidate triple [subject, predicate, object]
        graph : np.ndarray, shape [n, 3], dtype=int
            The graph
        k : int
            Number of hops [default=2]
        limit : int
            Maximum size of neighbourhood [default=None]
        seed : int
            Seed for pseudo-random generator, and reproducibility of results.

        Returns
        -------
        neighbours : np.ndarray, shape [n, 3]
            An ndarray of triples.
    """

    entities_queue = set([triple[SUBJECT], triple[OBJECT]]) # entities to be processed
    entities_done = set()                                   # entities that have been processed
    entities_bag = set()                                    # entities reached by current hop

    # Set of tuples of triples (facilitates
    neighbours = set()

    for hop in range(0, k):

        for e in entities_queue:
            N = get_entity_triples(e, graph)
            entities_bag.update(set(N[:, [SUBJECT, OBJECT]].flatten().tolist()))
            neighbours.update([tuple(x) for x in N])

        entities_done.update(entities_queue)
        entities_queue.clear()
        entities_queue.update(entities_bag.difference(entities_done))
        entities_bag.clear()

    # Convert neighbours set to np.array
    neighbours = np.array([x for x in neighbours])

    # Remove candidate triple
    neighbours = neighbours[~np.all(neighbours == triple, axis=1), :]

    # Sort by object, then by subject (downside of using sets)
    neighbours = neighbours[neighbours[:, OBJECT].argsort()]
    neighbours = neighbours[neighbours[:, SUBJECT].argsort()]

    # If limiting the neighbourhood size arbitrarily, sample the rows uniformly
    if limit is not None:
        if len(neighbours) > limit:
            random.seed(seed)
            samples = sorted(random.sample(range(len(neighbours)), limit))
            neighbours = neighbours[samples, :]
        else:
            warnings.warn('Neighbourhood limit is larger than population size.')

    return neighbours


def get_neighbour_triples_head_tail(entity, graph):
    """
    Given an entity label integer encoding and an ndarray of integer triples G, returns two ndarrays representing head and tail neighbourhood of the entity e.

       Parameters
       ----------
       entity : int
           An entity label.
       graph : np.ndarray, shape [n, 3]
           An ndarray of triples.

       Returns
       -------
       H : np.ndarray, shape [n, 3]
           An ndarray of triples representing head neighbourhood of the entity e.
       T : np.ndarray, shape [n, 3]
           An ndarray of triples representing tail neighbourhood of the entity e.

    """

    #TODO: I think this can be deleted, it's a convenience function with no usages found except in test cases.
    #TODO: Furthermore, questioning the use of 'head' and 'tail' as synonyms for object and subject, when this may not be the case

    # Get all neighbourhood triples of the entity e.
    N = get_entity_triples(entity, graph)

    # Obtain head and tail neighbouring entities
    H = N[np.where(N[:, OBJECT] == entity)]
    T = N[np.where(N[:, SUBJECT] == entity)]

    return H, T



def get_neighbour_entities(entity, graph, ent_type):
    """
    Given an entity label integer encoding, ndarray of integer triples G and entity type returns a ndarray of unique neighbour entities give head or tail position.

       Parameters
       ----------
       entity : int
           An entity label.
       graph : np.ndarray, shape [n, 3]
            An ndarray of triples.
       ent_type: str
           Type of entity we want to query for, supported options are either "head" or "tail"

       Returns
       -------
       H : np.ndarray, shape [n,]
           An ndarray of int encoded unique neighbour entities.

    """

    #TODO: I think this can be deleted, it's a convenience function with no usages found except in test cases.
    #TODO: Furthermore, questioning the use of 'head' and 'tail' as synonyms for object and subject, when this may not be the case

    # If required entity type is head, fetch all unique head entities
    if ent_type == 'head':
        E = np.unique(graph[np.where((entity == graph[:, SUBJECT]))[0], SUBJECT])
        E = E[E != entity]

    # If required entity type is tail, fetch all unique tail entities
    elif ent_type == 'tail':
        E = np.unique(graph[np.where((entity == graph[:, OBJECT]))[0], OBJECT])
        E = E[E != entity]
    else:
        raise ValueError('Bad ent_type argument!')

    return E


def precompute_neighbour_entities(entities_unique, graph):
    """
    Based on unique entities of the graph, precompute neighbourhood for each entity.
    In order to improve performance, computation is performed in parallel.

    Parameters
    ----------
    entities_unique: ndarray
        Array unique graph entities
    graph: ndarray
        Graph triplets

    Returns
    -------
    full_head_map: dict of ndarray
        Dictionary with entityId as key and neighbourhood ndarray as value, contains tail neighbours
    full_tail_map: dict of ndarrays
        Dictionary with entityId as key and neighbourhood ndarray as value, contains head neighbours

    """

    n = len(entities_unique) // os.cpu_count() if len(entities_unique) > os.cpu_count() else len(entities_unique)
    batches = [entities_unique[i:i + n] for i in range(0, len(entities_unique), n)]

    dict_tuple_list = Parallel(n_jobs=os.cpu_count())(delayed(process_entity_batch)(batch, graph) for batch in batches)

    full_head_map = {}
    full_tail_map = {}

    for h,t in dict_tuple_list:
        full_head_map.update(h)
        full_tail_map.update(t)

    return full_head_map, full_tail_map


def process_entity_batch(entities, graph):

    head_neighbour_map = {}
    tail_neighbour_map = {}

    for entity in entities:
        H_tr, T_tr = get_neighbour_triples_head_tail(entity, graph)

        # If required entity type is head, fetch all unique head entities
        E = np.unique(H_tr[:, 0])
        head_neighbour_map[entity] = E[E != entity]

        E = np.unique(T_tr[:, 2])
        tail_neighbour_map[entity] = E[E != entity]

    return (head_neighbour_map, tail_neighbour_map)


def get_neighbour_array_for_entities_precomputed(entities, head_map, tail_map):
    """
    Given an ndarray of integer encoded entities and ndarray of triples G return 3-dimensional ndarray of neighbour
    entities given input entities and another array stating number of neighbours per entity..

    Returned neighbour array has dimensions N x 2 x man_num_rel
        N - length of input entities, returns one neighbourhood per input entity
        2 - dimension of neighbourhood, [head, tail]
        max_num_len - number of maximum relations per entity in the given graph, arrays are padded to this length


       Parameters
       ----------
       entities : np.ndarray
           Array of integer encoded entities.
       graph : np.ndarray, shape [n, 3]
           An ndarray of triples.
       max_num_rel: int
           Maximum number of relations per entity in the graph.
       pad_value: int
           Value used to pad neighbour entity dimension till max_num_rel


       Returns
       -------
       merged_array : np.ndarray, shape [n, 2, max_num_rel]
           Neighbouring entities of the input entities organized in 3D ndarray
       merged_num : np.ndarra, shape [n, 2]
           Number of neighbour entities per head and tail dimension
    """

    def mapping(entity, mapper_dict):
        if entity in mapper_dict:
            return mapper_dict[entity]
        else:
            print('Entity not found in precomputed dict')
            return np.array([], dtype=np.int32)

    # Get list of ndarrays representing head neighbours of every entity
    head_list = [mapping(entity, head_map) for entity in entities]
    #n = len(entities) // os.cpu_count()
    #batches = [entities[i:i + n] for i in range(0, len(entities), n)]
    #head_list = Parallel(n_jobs=os.cpu_count())(delayed(get_neighbour_entities_cached)(batch, head_map) for batch in batches)

    # Number of head neighbours per entity
    head_num_list = [el.shape[0] for el in head_list]
    #head_num_list = Parallel(n_jobs=os.cpu_count())(delayed(get_shape)(el, head_map) for el in head_list)

    # Get list of ndarrays representing tail neighbours of every entity
    tail_list = [mapping(entity, tail_map) for entity in entities]
    #tail_list = Parallel(n_jobs=os.cpu_count())(delayed(get_neighbour_entities_cached)(entity, tail_map) for entity in entities)

    # Number of tail neighbours per entity
    tail_num_list = [el.shape[0] for el in tail_list]
    #tail_num_list = Parallel(n_jobs=os.cpu_count())(delayed(get_shape)(el) for el in tail_list)

    # Stack together head and tail arrays into a single 3D array
    merged_list = [head_list, tail_list]

    # Stack together head and tail neigbour count lists into a single 2D array
    merged_num_list = [head_num_list, tail_num_list]


    return merged_list, merged_num_list


def get_neighbour_array_for_entities(entities, graph, max_num_rel, pad_value):
    """
    Given an ndarray of integer encoded entities and ndarray of triples G return 3-dimensional ndarray of neighbour 
    entities given input entities and another array stating number of neighbours per entity..
    
    Returned neighbour array has dimensions N x 2 x man_num_rel
        N - length of input entities, returns one neighbourhood per input entity
        2 - dimension of neighbourhood, [head, tail]
        max_num_len - number of maximum relations per entity in the given graph, arrays are padded to this length


       Parameters
       ----------
       entities : np.ndarray
           Array of integer encoded entities.
       graph : np.ndarray, shape [n, 3]
           An ndarray of triples.
       max_num_rel: int
           Maximum number of relations per entity in the graph.
       pad_value: int
           Value used to pad neighbour entity dimension till max_num_rel


       Returns
       -------
       merged_array : np.ndarray, shape [n, 2, max_num_rel]
           Neighbouring entities of the input entities organized in 3D ndarray
       merged_num : np.ndarra, shape [n, 2]
           Number of neighbour entities per head and tail dimension
    """

    # Get list of ndarrays representing head neighbours of every entity
    head_list = [get_neighbour_entities(entity, graph, 'head') for entity in entities]
    #head_list = Parallel(n_jobs=os.cpu_count())(delayed(get_neighbour_entities_cached)(entity, 'head') for entity in entities)
    # Get list of ndarrays representing tail neighbours of every entity
    tail_list = [get_neighbour_entities(entity, graph, 'tail') for entity in entities]
    #tail_list = Parallel(n_jobs=os.cpu_count())(delayed(get_neighbour_entities_cached)(entity, 'tail') for entity in entities)

    # Number of head neighbours per entity
    head_num_list = [el.shape[0] for el in head_list]
    # Number of tail neighbours per entity
    tail_num_list = [el.shape[0] for el in tail_list]

    # Padding of head neighbour dimension to max_num_rel length
    head_array = np.full((len(head_list), max_num_rel), pad_value)
    for i, el in enumerate(head_list):
        head_array[i, :el.shape[0]] = el

    # Padding of tail neighbour dimension to max_num_rel length
    tail_array = np.full((len(tail_list), max_num_rel), pad_value)
    for i, el in enumerate(tail_list):
        tail_array[i, :el.shape[0]] = el

    # Stack together head and tail arrays into a single 3D array
    merged_array = np.dstack((head_array, tail_array))
    merged_array = np.transpose(merged_array, (0, 2, 1))

    # Stack together head and tail neigbour count lists into a single 2D array
    merged_num = np.array((head_num_list, tail_num_list))
    merged_num = np.transpose(merged_num, (1, 0))

    return merged_array, merged_num


def get_neighbour_matrix_for_triplets(graph, max_num_rel=None, head_map = None, tail_map = None):
    """
    Given an input ndarray of triples G returns list of list of all possible neighbour entities computed for both head and tail entities od the G.
    Returned neighbour list of lists has dimensions N x 4
        N - number of input triplets, returns a neighbourhood per triplet
        4 - dimension of neighbourhood, [object_head, object_tail_tail, subject_head, subject_tail]
        max_num_len - number of maximum relations per entity in the given graph defines maximum number of neighbours
    Returns also list representing count of neighbour entities per dimensions of neighbourhood

       Parameters
       ----------
       graph : np.ndarray, shape [n, 3]
           An ndarray of triples.
       pad_value: int
           Value used to pad neighbour entity dimension till max_num_rel

       Returns
       -------
       full_head_tail : list of lists
           Neighbourhood list for input graph G
       full_nums : list
           Number of neighbour entities per object/subject head and tail dimension
       max_num_len : int
           maximum number of neighbours per any entity
    """

    #TODO: Could be re-written to make use of similar functionality in methods above.

    if max_num_rel is None:
        # Counts number of relations per any entity in the graph
        _, counts_head = np.unique(graph[:, 0], return_counts=True)
        _, counts_tail = np.unique(graph[:, 2], return_counts=True)

        # Computes maximum number of relations in the graph
        max_num_rel = max(np.max(counts_head), np.max(counts_tail))
        print(max_num_rel)
        del counts_head
        del counts_tail

    if head_map == None and tail_map == None:
        unique_entities = np.unique(np.concatenate([graph[:, 0], graph[:, 2]]))
        #unique_entities = unique_entities[counts > precompute_filter]
        head_map, tail_map = precompute_neighbour_entities(unique_entities, graph)

    # Returns neighbour and count arrays for subject entities
    subject_merged_list, subject_nums = get_neighbour_array_for_entities_precomputed(graph[:, 0], head_map, tail_map)
    # Returns neighbour and count arrays for object entities
    object_merged_list, object_nums = get_neighbour_array_for_entities_precomputed(graph[:, 2], head_map, tail_map)

    # Merge subject and object arrays
    subject_merged_list.extend(object_merged_list)
    subject_merged_list = [list(i) for i in zip(*subject_merged_list)]

    subject_nums.extend(object_nums)
    subject_nums = [list(i) for i in zip(*subject_nums)]

    return subject_merged_list, subject_nums, max_num_rel, head_map, tail_map


def get_neighbour_matrix_for_triplets_predict(graph, pad_id, max_num_rel=None, head_map = None, tail_map = None):
    """
    Given an input ndarray of triples G returns 3-dimensional ndarray of all possible neighbour entities computed for both head and tail entities od the G.
    Returned neighbour list of list has dimensions N x 4
        N - number of input triplets, returns a neighbourhood per triplet
        4 - dimension of neighbourhood, [object_head, object_tail_tail, subject_head, subject_tail]
        max_num_len - number of maximum relations per entity in the given graph defines maximum number of neighbours
    Returns also list representing count of neighbour entities per dimensions of neighbourhood


       Parameters
       ----------
       graph : np.ndarray, shape [n, 3]
           An ndarray of triples.
       pad_value: int
           Value used to pad neighbour entity dimension till max_num_rel


       Returns
       -------
       full_head_tail : list of lists
           Neighbourhood array for input graph G
       full_nums : list
           Number of neighbour entities per object/subject head and tail dimension
       max_num_len : int
           maximum number of neighbours per any entity
    """

    # Returns neighbour and count arrays for subject entities
    subject_merged_list, subject_nums = get_neighbour_array_for_entities_precomputed(graph[:, 0], head_map, tail_map)
    # Returns neighbour and count arrays for object entities
    object_merged_list, object_nums = get_neighbour_array_for_entities_precomputed(graph[:, 2], head_map, tail_map)

    # Merge subject and object arrays
    subject_merged_list.extend(object_merged_list)
    subject_merged_list = [list(i) for i in zip(*subject_merged_list)]

    subject_nums.extend(object_nums)
    subject_nums = [list(i) for i in zip(*subject_nums)]

    return subject_merged_list, subject_nums, max_num_rel


def get_classes_given_entities(E, S, max_num_rel, pad_value=100):
    """
    Given an input ndarray of triples S returns 2-dimensional ndarray of schema classes computed given entities E.
    Returned schema classes array has dimensions N x man_num_rel
        N - number of input triplets, returns a neighbourhood per triplet
        max_num_len - number of maximum relations per entity in the schema graph
    Returns also 2D ndarray representing count of schema classes


       Parameters
       ----------
       E : np.ndarray, shape [n, ]
           An ndarray of triples.
       S : np.ndarray, shape [n, 3]
           An ndarray of triples.
       pad_value: int
           Value used to pad class dimension till max_num_rel


       Returns
       -------
       res_array : np.ndarray, shape [n, max_num_rel]
           Neighbourhood array for input graph G
       cnt_array : np.ndarra, shape [n,]
           Number of classes  per input entity
    """

    res_array = np.full((E.shape[0], max_num_rel), pad_value)
    cnt_array = np.zeros((E.shape[0]))

    for i, e in enumerate(E):
        filtered_triplets = S[S[:, 0] == e]
        unique_cls = np.unique(filtered_triplets[:, 2])
        count = unique_cls.shape[0]
        cnt_array[i] = count
        res_array[i, :count] = unique_cls

    return res_array, cnt_array


def get_schema_matrix_for_triplets(X, S, pad_value=100):
    """
    Given an input ndarray of triples X returns 3-dimensional ndarray of schema classes.
    Returned schema classes array has dimensions N x 2x man_num_rel
        N - number of input triplets, returns a neighbourhood per triplet
        2 - dimensions for subject and object of the input triple
        max_num_len - number of maximum relations per entity in the schema graph
    Returns also 3D ndarray representing count of schema classes per subject and object dimension


       Parameters
       ----------
       E : np.ndarray, shape [n, ]
           An ndarray of triples.
       S : np.ndarray, shape [n, 3]
           An ndarray of triples.
       max_num_rel: int
           Maximum number of schema classes relationships, to be used for padding
       pad_value: int
           Numerical value used for padding

       Returns
       -------
       cls_array : np.ndarray, shape [n, 2,  max_num_rel]
           Schema neighbours array for input graph G
       num_array : np.ndarra, shape [n, 2]
           Number of classes  per input entity for objects and subjects
    """

    # count number of relationship in schema
    _, counts = np.unique(S[:, 0], return_counts=True)
    # find maximum number of relationships in schema
    max_num_rel = max(counts)

    # get classes for subject entities
    s_cls, s_nums = get_classes_given_entities(X[:, 0], S, max_num_rel, pad_value)
    # get classes for object entities
    o_cls, o_nums = get_classes_given_entities(X[:, 2], S, max_num_rel, pad_value)

    s_cls = np.expand_dims(s_cls, axis=1)
    o_cls = np.expand_dims(o_cls, axis=1)

    s_nums = np.expand_dims(s_nums, axis=1)
    o_nums = np.expand_dims(o_nums, axis=1)

    cls_array = np.concatenate((s_cls, o_cls), axis=1)
    num_array = np.concatenate((s_nums, o_nums), axis=1)

    return cls_array, num_array, max_num_rel


def extract_schema_triplets(X, relationId = None):
    """
    Extracts schema triplets from graph X and stores them in separate triplet array schema

    Parameters
    ----------
    X: ndarray
        Array of graph triplets
    relationId: str
        String defining schema relationship

    Returns
    -------
    X: ndarray
        Array of graph triplets without schema entries
    schema: ndarray
        Schema triplets extracted from the input graph

    """

    schema = X[X[:, 1] == relationId]
    X = X[X[:, 1] != relationId]

    return X, schema
