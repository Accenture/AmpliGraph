Explanations
============
.. currentmodule:: ampligraph.explanations

.. automodule:: ampligraph.explanations

The explanations module is a framework to explain the link predictions of a trained knowledge graph embedding model.
An **explainer** takes a triple ``(subject, predicate, object)`` and a model, and returns a dictionary containing the
prediction of the model together with the supporting evidence found in the knowledge graph. A matching **visualizer**
can then draw that evidence as a graph.

Explainers and visualizers are plugged into the framework through registries, so that different explanation methods
share the same interface, data handling and utilities. New methods can be added by extending the abstract classes
(see `Adding a New Explainer`_).

Explainers
----------

Every explainer is built from the data and the model to explain:

.. code-block:: python

    explainer = ExamplE(X, model)

* ``X`` is a dictionary of triples ``{'train': ..., 'valid': ..., 'test': ...}``. Only ``'train'`` is required, and a
  plain ``numpy.ndarray`` is treated as the training set. The evidence of an explanation is always drawn from the
  ``'train'`` triples.
* ``model`` is a trained AmpliGraph model, which must provide ``get_embeddings``, ``predict`` and ``predict_proba``
  (e.g. :class:`~ampligraph.latent_features.ScoringBasedEmbeddingModel`). The model should be calibrated
  (see :meth:`~ampligraph.latent_features.ScoringBasedEmbeddingModel.calibrate`) so that the returned probabilities
  are meaningful.

All explainers extend :class:`AbstractExplainer` and implement two methods:

* ``predict_explain(target_triple)``: returns the explanation dictionary. All explainers include the keys
  ``prediction``, ``probability``, ``query_triple`` and ``N-hood`` (the neighbourhood of the target triple in the
  training graph), plus keys that are specific to the explanation method.
* ``visualize(explanation, fmt='pdf', fname='graph')``: draws the explanation with the visualizer of the explainer.

The following explainers are available:

.. list-table::
    :header-rows: 1
    :widths: 20 20 60

    * - Explainer
      - Registry name
      - Description
    * - :class:`ExamplE`
      - ``example``
      - Example-based explanations: finds training triples similar to the target triple in the embedding space, and
        aggregates their neighbourhoods into a prototype graph.
    * - :class:`RandomExplainer`
      - ``random_explainer``
      - Baseline: randomly samples triples from the neighbourhood of the target triple. Useful for comparison with
        other explainers.

.. autosummary::
    :toctree: generated
    :template: class.rst

    ExamplE
    RandomExplainer
    AbstractExplainer

.. autosummary::
    :toctree: generated
    :template: function.rst

    register_explainer


ExamplE
^^^^^^^

:class:`ExamplE` explains a prediction with *examples*: triples of the training set that are similar to the target
triple. :meth:`~ExamplE.predict_explain` runs four steps, each of them available also as a method:

1. :meth:`~ExamplE.neighbourhood_sampler` finds the ``m`` nearest entities (and optionally relations) to the subject,
   object and predicate of the target triple in the embedding space.
2. :meth:`~ExamplE.examples_filter` builds candidate triples from these neighbours and keeps the ones that exist in the
   training set. ``resembles`` links record which entity of the target triple each example resembles.
3. :meth:`~ExamplE.prototype_aggregator` aggregates the neighbourhoods of the target triple and of the examples into a
   prototype graph. With ``strategy="strict"`` the prototype is the intersection of all neighbourhoods, with
   ``strategy="permissive"`` it is the union of the examples neighbourhoods intersected with the neighbourhood of the
   target triple, weighted by the number of occurrences.
4. :func:`explanation_graph_assembler` merges target triple, examples and prototype into the final explanation graph,
   which is stored in ``explainer.explanation_graph``.

The main arguments of :meth:`~ExamplE.predict_explain` are ``mode`` (``"N"`` nodes, ``"E"`` edges or ``"NE"`` both),
``strategy``, ``hop``, ``sampling_mode`` (any combination of at least two of ``"s"``, ``"p"``, ``"o"``), ``m``,
``epsilon``, ``metric`` (default ``"cosine"``) and ``weights``.

The returned dictionary has the keys ``prediction``, ``probability``, ``examples``, ``resembles``, ``prototype``,
``query_triple`` and ``N-hood``.

:meth:`~ExamplE.batch_explain` explains many triples at once and, with ``score=True``, ranks the examples of each
triple by their distance in the embedding space (smaller is more similar).

.. autosummary::
    :toctree: generated
    :template: function.rst

    explanation_graph_assembler
    aggregate
    intersection
    union
    nodes_edges
    get_explaining_subgraph


RandomExplainer
^^^^^^^^^^^^^^^

:class:`RandomExplainer` draws a random fraction (``ratio``) of the triples in the ``n_hop`` neighbourhood of the
target triple. It has no notion of similarity, which makes it a useful reference point when evaluating other
explainers. The returned dictionary has the keys ``prediction``, ``probability``, ``query_triple``,
``relevant_triples`` and ``N-hood``.


Visualizers
-----------

Visualizers extend :class:`AbstractGraphVisualizer` and implement ``visualize(explanation, fname, fmt)``. They are
registered in the visualizers registry as ``<explainer_name>_viz``, and each explainer looks up its own visualizer when
it is created.

.. note::
    Visualizations require packages that are not installed together with AmpliGraph::

        pip install pygraphviz graphviz pydot

    The Graphviz system package must also be installed. Without these packages the explainers still work, but the
    visualizers are not available.

.. autosummary::
    :toctree: generated
    :template: class.rst

    GraphVisualizerExamplE
    GraphVisualizerRandom
    AbstractGraphVisualizer

.. autosummary::
    :toctree: generated
    :template: function.rst

    register_visualizer


Utilities
---------

Helpers shared by explainers and visualizers.

:class:`GraphNavigator` retrieves the neighbours of a node and builds the n-hop neighbourhood subgraph of a triple as a
``networkx`` graph. :class:`Decoder` translates entity identifiers into human-readable labels, using a JSON file that
maps identifiers to labels, either for a set of triples or for a whole explanation dictionary.

.. autosummary::
    :toctree: generated
    :template: class.rst

    GraphNavigator
    Decoder

The functions below set styling attributes on ``networkx`` graphs and group nodes and edges for Graphviz rendering.

.. autosummary::
    :toctree: generated
    :template: function.rst

    apply_edges_attributes
    apply_nodes_attributes
    create_cluster_with_nodes
    create_cluster_with_edges
    update_attrs
    clean_attrs
    format_dot_attrs


Adding a New Explainer
----------------------

1. Extend :class:`AbstractExplainer` and implement ``predict_explain`` and ``visualize``. The base class provides
   ``self.X`` (the data), ``self.model`` and ``self.navigator`` (a :class:`GraphNavigator` for the data).
2. Register the class with :func:`register_explainer`. Names must be unique.
3. Optionally, extend :class:`AbstractGraphVisualizer` and register it with :func:`register_visualizer`, using the
   name ``<explainer_name>_viz``.
4. Export the new classes from the relevant ``__init__.py``.

.. code-block:: python

    from ampligraph.explanations.explainers import AbstractExplainer, register_explainer

    @register_explainer("my_explainer")
    class MyExplainer(AbstractExplainer):
        def predict_explain(self, target_triple):
            ...

        def visualize(self, explanation, fmt='pdf', fname='graph'):
            ...


Example
-------

.. code-block:: python

    from ampligraph.datasets import load_fb15k_237
    from ampligraph.utils import restore_model
    from ampligraph.explanations import ExamplE

    dataset = load_fb15k_237(return_mapper=True)
    model = restore_model('saved_model_transE')

    explainer = ExamplE(dataset, model)
    explanation = explainer.predict_explain(dataset['test'][35])
    explainer.visualize(explanation)

A full walk-through, including the translation of identifiers into labels and batch explanations, is available in the
`Explanations tutorial notebook <https://github.com/Accenture/AmpliGraph/blob/master/notebooks/AmpliGraph-Tutorials/Explanations%201%20-%20Example-based%20Explanations.ipynb>`_.


Notes
-----

* Explanations are built from triples of the training set: they describe the data surrounding a prediction rather than
  the internal workings of the model.
* :class:`ExamplE` returns no examples when nothing similar to the target triple exists in the training set. Try a
  larger ``m``, or a different ``sampling_mode`` or ``metric``.
* :class:`RandomExplainer` writes the sampled triples to a ``triples`` folder in the current directory.
* The visualizer of :class:`ExamplE` writes ``graph.dot`` in the current directory and opens it with the system viewer,
  ignoring the ``fmt`` and ``fname`` arguments.
