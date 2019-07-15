# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
from ampligraph.latent_features.misc import get_entity_triples

def test_get_entity_triples():

    # Graph
    X = np.array([['a', 'y', 'b'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['d', 'y', 'e'],

                  ['e', 'y', 'f'],
                  ['f', 'y', 'c']])

    # Entity of interest
    u = 'c'

    # Neighbours of u
    XN = np.array([['a', 'y', 'c'],
                   ['c', 'y', 'a'],
                   ['f', 'y', 'c']])

    # Call function
    N = get_entity_triples(u, X)

    assert(np.all(N == XN))

