# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
r"""This module includes a number of functions to perform knowledge discovery in graph embeddings.

Functions provided include ``discover_facts`` which will generate candidate statements using one of several
defined strategies and return triples that perform well when evaluated against corruptions, ``find_clusters`` which
will perform link-based cluster analysis on a knowledge graph, ``find_duplicates`` which will find duplicate entities
in a graph based on their embeddings, and ``query_topn`` which when given two elements of a triple will return
the top_n results of all possible completions ordered by predicted score.

"""

from .discovery import discover_facts, find_clusters, find_duplicates, query_topn, find_nearest_neighbours

__all__ = ['discover_facts', 'find_clusters', 'find_duplicates', 'query_topn', 'find_nearest_neighbours']
