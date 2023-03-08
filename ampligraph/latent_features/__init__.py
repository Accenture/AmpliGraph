# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
r"""This module includes neural graph embedding models and support functions.

Knowledge graph embedding models are neural architectures that encode concepts
from a knowledge graph (i.e., entities :math:`\mathcal{E}` and relation types
:math:`\mathcal{R}`) into low-dimensional, continuous vectors :math:`\in
\mathcal{R}^k`. Such *knowledge graph embeddings* have applications in
knowledge graph completion, entity resolution, and link-based clustering,
just to cite a few :cite:`nickel2016review`.

"""
from .loss_functions import (
    AbsoluteMarginLoss,
    NLLLoss,
    NLLMulticlass,
    PairwiseLoss,
    SelfAdversarialLoss,
)
from .models import ScoringBasedEmbeddingModel
from .regularizers import LP_regularizer

__all__ = [
    "layers",
    "models",
    "ScoringBasedEmbeddingModel",
    "PairwiseLoss",
    "NLLLoss",
    "AbsoluteMarginLoss",
    "SelfAdversarialLoss",
    "NLLMulticlass",
    "LP_regularizer",
]
