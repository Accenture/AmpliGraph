"""This module includes neural graph embedding models and support functions.

Knowledge graph embedding models are neural architectures that encode concepts from a knowledge graph (i.e. entities :math:`\mathcal{E}` and relation types :math:`\mathcal{R}`) into low-dimensional, continuous vectors :math:`\in \mathcal{R}^k`. Such \textit{knowledge graph embeddings} have applications in knowledge graph completion, entity resolution, and link-based clustering, just to cite a few :cite:`nickel2016review`.

"""

from .loss_functions import LOSS_REGISTRY
from .regularizers import REGULARIZER_REGISTRY
from .models import MODEL_REGISTRY
from .models import EmbeddingModel, TransE, DistMult, ComplEx, HolE, RandomBaseline
from .loss_functions import Loss, AbsoluteMarginLoss, SelfAdversarialLoss, NLLLoss, PairwiseLoss, NLLAdversarialLoss
from .regularizers import Regularizer, L1Regularizer, L2Regularizer, L3Regularizer

from .misc import get_entity_triples
from .model_utils import save_model, restore_model


__all__ = ['LOSS_REGISTRY', 'REGULARIZER_REGISTRY', 'MODEL_REGISTRY', 
           'EmbeddingModel', 'TransE', 'DistMult', 'ComplEx', 'HolE', 'RandomBaseline',
           'Loss', 'AbsoluteMarginLoss', 'SelfAdversarialLoss', 'NLLLoss', 'PairwiseLoss', 'NLLAdversarialLoss',
           'Regularizer', 'L1Regularizer', 'L2Regularizer', 'L3Regularizer',
            'get_entity_triples','save_model', 'restore_model']


