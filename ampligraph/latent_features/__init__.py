# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
r"""This module includes neural graph embedding models and support functions.

Knowledge graph embedding models are neural architectures that encode concepts from a knowledge graph
(i.e. entities :math:`\mathcal{E}` and relation types :math:`\mathcal{R}`) into low-dimensional, continuous vectors
:math:`\in \mathcal{R}^k`. Such *knowledge graph embeddings* have applications in knowledge graph completion,
entity resolution, and link-based clustering, just to cite a few :cite:`nickel2016review`.

"""

from .models.EmbeddingModel import EmbeddingModel, MODEL_REGISTRY, set_entity_threshold, reset_entity_threshold
from .models.TransE import TransE
from .models.DistMult import DistMult
from .models.ComplEx import ComplEx
from .models.HolE import HolE
from .models.RandomBaseline import RandomBaseline
from .models.ConvKB import ConvKB
from .models.ConvE import ConvE

from .loss_functions import Loss, AbsoluteMarginLoss, SelfAdversarialLoss, NLLLoss, PairwiseLoss,\
    NLLMulticlass, BCELoss, LOSS_REGISTRY
from .regularizers import Regularizer, LPRegularizer, REGULARIZER_REGISTRY
from .optimizers import Optimizer, AdagradOptimizer, AdamOptimizer, MomentumOptimizer, SGDOptimizer, OPTIMIZER_REGISTRY
from .initializers import Initializer, RandomNormal, RandomUniform, Xavier, INITIALIZER_REGISTRY
from .misc import get_entity_triples
from ..utils import save_model, restore_model

__all__ = ['LOSS_REGISTRY', 'REGULARIZER_REGISTRY', 'MODEL_REGISTRY', 'OPTIMIZER_REGISTRY', 'INITIALIZER_REGISTRY',
           'set_entity_threshold', 'reset_entity_threshold',
           'EmbeddingModel', 'TransE', 'DistMult', 'ComplEx', 'HolE', 'ConvKB', 'ConvE', 'RandomBaseline',
           'Loss', 'AbsoluteMarginLoss', 'SelfAdversarialLoss', 'NLLLoss', 'PairwiseLoss', 'BCELoss', 'NLLMulticlass',
           'Regularizer', 'LPRegularizer', 'Optimizer', 'AdagradOptimizer', 'AdamOptimizer', 'MomentumOptimizer',
           'SGDOptimizer', 'Initializer', 'RandomNormal', 'RandomUniform', 'Xavier', 'get_entity_triples',
           'save_model', 'restore_model']
