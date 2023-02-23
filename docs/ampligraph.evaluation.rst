Evaluation
==========
.. currentmodule:: ampligraph.evaluation

.. automodule:: ampligraph.evaluation

After the training is complete, the model is ready to perform predictions and to be evaluated on unseen data. Given a
triple, the model can score it and quantify its plausibility. Importantly, the entities and relations of new triples
must have been seen during training, otherwise no embedding for them is available. Future extensions of the code base
will introduce inductive methods as well.

The standard evaluation of a test triples is achieved by comparing the score assigned by the model to that triple with
those assigned to the same triple where we corrupted either the object or the subject. From this comparison we
extract some metrics. By aggregating the metrics obtained for all triples in the test set, we finally obtain a "thorough"
(depending on the quality of the test set and of the corruptions) evaluation of the model.

Metrics
^^^^^^^

The available metrics implemented in AmpliGraph to rank a triple against its corruptions are listed in the table below.

.. autosummary::
    :toctree:
    :template: function.rst

    rank_score
    mr_score
    mrr_score
    hits_at_n_score

Model Selection
^^^^^^^^^^^^^^^

AmpliGraph implements a model selection routine for KGE models via either a grid search or a random search.
Random search is typically more efficient, but grid search, on the other hand, can provide a more controlled selection framework.

.. autosummary::
    :toctree:
    :template: function.rst

    select_best_model_ranking

Helper Functions
^^^^^^^^^^^^^^^^

Utilities and support functions for evaluation procedures.

.. autosummary::
    :toctree:
    :template: function.rst

    train_test_split_no_unseen
    filter_unseen_entities
