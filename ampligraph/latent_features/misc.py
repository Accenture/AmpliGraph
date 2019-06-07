# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
import logging

SUBJECT = 0
PREDICATE = 1
OBJECT = 2
DEBUG = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    logger.debug('Return a list of all triples where {} appears as subject or object.'.format(entity))
    # NOTE: The current implementation is slightly faster (~15%) than the more readable one-liner:
    #           rows, _ = np.where((entity == graph[:,[SUBJECT,OBJECT]]))

    # Get rows and cols where entity is found in graph
    rows, cols = np.where((entity == graph))

    # In the unlikely event that entity is found in the relation column (index 1)
    rows = rows[np.where(cols != PREDICATE)]

    # Subset graph to neighbourhood of entity
    neighbours = graph[rows, :]

    return neighbours
