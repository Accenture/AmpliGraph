Pre-Trained Models
===================

.. currentmodule:: ampligraph.pretrained_models

.. automodule:: ampligraph.pretrained_models

This module provides an API to download and have ready to use pre-trained
:class:`~ampligraph.latent_features.ScoringBasedEmbeddingModel`.

.. autosummary::
    :toctree:
    :template: function.rst

    load_pretrained_model

Currently the available models are trained on "FB15K-237", "WN18RR", "YAGO310", "FB15K" and "WN18" and have as
scoring function "TransE", "DistMult", "ComplEx", "HolE" and "RotatE".

