Latent Features Models
======================

.. currentmodule:: ampligraph.latent_features

.. automodule:: ampligraph.latent_features


Embedding Models
----------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    EmbeddingModel
    RandomBaseline
    TransE
    DistMult
    ComplEx
    HolE


Loss Functions
--------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    Loss
    PairwiseLoss
    NLLLoss
    AbsoluteMarginLoss
    SelfAdversarialLoss
    
Regularizers
--------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    Regularizer
    NoRegularizer
    L1Regularizer
    L2Regularizer


Utils Functions
---------------

.. autosummary::
    :toctree: generated
    :template: function.rst

    save_model
    restore_model
