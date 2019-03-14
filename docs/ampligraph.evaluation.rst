Evaluation
==========
.. currentmodule:: ampligraph.evaluation

.. automodule:: ampligraph.evaluation

Metrics
-------

Learning-to-rank metrics to evaluate the performance of neural graph embedding models.

.. autosummary::
    :toctree: generated
    :template: function.rst

    rank_score
    mrr_score
    mr_score
    hits_at_n_score


.. _negatives:

Negatives Generation
--------------------

Negatives generation routines. These are corruption strategies based on the Local Closed-World Assumption (LCWA).

.. autosummary::
    :toctree: generated
    :template: function.rst

    generate_corruptions_for_eval
    generate_corruptions_for_fit


.. _eval:

Evaluation & Model Selection
-----------------------------

Functions to evaluate the predictive power of knowledge graph embedding models, and routines for model selection.

.. autosummary::
    :toctree: generated
    :template: function.rst

    evaluate_performance
    select_best_model_ranking


Helper Functions
----------------

Utilities and support functions for evaluation procedures.

.. autosummary::
    :toctree: generated
    :template: function.rst

    train_test_split_no_unseen
    create_mappings
    to_idx

    

