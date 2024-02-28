Models
======

.. currentmodule:: ampligraph.latent_features

.. automodule:: ampligraph.latent_features

Knowledge Graph Embedding models (KGE) are neural architectures that encode concepts from a knowledge graph
(i.e., entities :math:`\mathcal{E}` and relation types :math:`\mathcal{R}`) into low-dimensional,
continuous vectors living in :math:`\mathbb{R}^k`, where :math:`k` can be specified by the user.

Knowledge Graph Embeddings have applications in knowledge graph completion, entity resolution, and link-based
clustering, just to cite a few :cite:`nickel2016review`.

In Ampligraph 2, KGE models are implemented in the :class:`ScoringBasedEmbeddingModel`
class, that inherits from `Keras Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model/>`_:

.. autosummary::
    :toctree:
    :template: class.rst

    ScoringBasedEmbeddingModel

The advantage of inheriting from Keras models are many. We can use most of Keras initializers (HeNormal, GlorotNormal...),
regularizers (:math:`L^1`, :math:`L^2`...), optimizers (Adam, AdaGrad...) and callbacks (early stopping, model
checkpointing...), all without having to reimplement them. From a user perspective, people already acquainted to Keras
can seemlessly work with AmpliGraph due to the similarity of the APIs.

We also provide backward compatibility with the APIs of Ampligraph 1, by wrapping the older APIs around the newer ones.

Anatomy of a Model
^^^^^^^^^^^^^^^^^^

Knowledge Graph Embeddings are learned by training a neural architecture over a graph. Although such architecture can be
of many different kinds, the training phase always consists in minimizing a :ref:`loss function <loss>`
:math:`\mathcal{L}` that optimizes the scores output by a :ref:`scoring function <scoring>` :math:`f_{m}(t)`,
i.e., a model-specific function that assigns a score to a triple :math:`t=(sub,pred,obj)`.

+ :ref:`Embedding Generation Layer <embedding>`
+ :ref:`Negatives Generation Layer <negatives>`
+ :ref:`Scoring Layer <scoring>`
+ :ref:`Loss function <loss>`
+ :ref:`Optimizer <optimizer>`
+ :ref:`Regularizer <ref-reg>`
+ :ref:`Initializer <ref-init>`

The first three elements are included in the :class:`ScoringBasedEmbeddingModel` class and they inherit from
`Keras Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer/>`_.

Further, for the scoring layer and the loss function, AmpliGraph features abstract classes that can be extended to
design new models:

.. autosummary::
    :toctree:
    :template: class.rst

    ~layers.scoring.AbstractScoringLayer.AbstractScoringLayer
    ~loss_functions.Loss

.. _embedding:

Embedding Generation Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~
The embedding generation layer generates the embeddings of the concepts present in the triples. It may be as simple as
a *shallow* encoding (i.e., a lookup of the embedding of an input node or edge type), or it can be as complex as a
neural network, which tokenizes nodes and generates embeddings for nodes using a neural encoder (e.g., NodePiece).
Currently, AmpliGraph implements the shallow look-up strategy but will be expanded soon to include other efficient approaches.

.. autosummary::
    :toctree:
    :template: class.rst

    ~layers.encoding.EmbeddingLookupLayer

.. _negatives:

Negatives Generation Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~
This layer is responsible for generation of synthetic negatives. The strategies to generate negatives can be multiple.
In our case, we assume a local close world assumption, and implement a simple negative generation strategy,
where we randomly corrupt either the subject, the object or both the subject and the object of a triple, to generate a
synthetic negative. Further, we allow filtering the true positives out of the generated negatives.

.. autosummary::
    :toctree:
    :template: class.rst

    ~layers.corruption_generation.CorruptionGenerationLayerTrain

.. _scoring:

Scoring Layer
~~~~~~~~~~~~~

The scoring layer applies a scoring function :math:`f` to a triple :math:`t=(s,p,o)`. This function combines the embeddings
:math:`\mathbf{e}_{s},\mathbf{r}_{p}, \mathbf{e}_{o} \in \mathbb{R}^k` (or :math:`\in \mathbb{C}^k`) of the subject, predicate,
and object of :math:`t` into a score representing the plausibility of the triple.

.. autosummary::
    :toctree:
    :template: class.rst

    ~ampligraph.latent_features.layers.scoring.TransE
    ~ampligraph.latent_features.layers.scoring.DistMult
    ~ampligraph.latent_features.layers.scoring.ComplEx
    ~ampligraph.latent_features.layers.scoring.RotatE
    ~ampligraph.latent_features.layers.scoring.HolE

Different scoring functions are designed according to different intuitions:

+ :class:`~layers.scoring.TransE` :cite:`bordes2013translating` relies on distances. The scoring function computes a similarity between
the embedding of the subject translated by the embedding of the predicate  and the embedding of the object, using the
:math:`L^1` or :math:`L^2` norm :math:`||\cdot||`:
    .. math::
        f_{TransE}=-||\mathbf{e}_{s} + \mathbf{r}_{p} - \mathbf{e}_{o}||

+ :class:`~layers.scoring.DistMult` :cite:`yang2014embedding` uses the trilinear dot product:
    .. math::
        f_{DistMult}=\langle \mathbf{r}_p, \mathbf{e}_s, \mathbf{e}_o \rangle

+ :class:`~layers.scoring.ComplEx` :cite:`trouillon2016complex` extends DistMult with the Hermitian dot product:
    .. math::
        f_{ComplEx}=Re(\langle \mathbf{r}_p, \mathbf{e}_s, \overline{\mathbf{e}_o}  \rangle)

+ :class:`~layers.scoring.RotatE` :cite:`sun2018rotate` models relations as rotations in the Complex space:
    .. math::
        f_{RotatE}=||\mathbf{e}_{s} \circ \mathbf{r}_{p} - \mathbf{e}_{o}||

+ :class:`~layers.scoring.HolE` :cite:`nickel2016holographic` uses circular correlation (denoted by :math:`\otimes`):
    .. math::
        f_{HolE}=\mathbf{w}_r \cdot (\mathbf{e}_s \otimes \mathbf{e}_o) = \frac{1}{k}\mathcal{F}(\mathbf{w}_r)\cdot( \overline{\mathcal{F}(\mathbf{e}_s)} \odot \mathcal{F}(\mathbf{e}_o))


.. _loss:

Loss Functions
~~~~~~~~~~~~~~
AmpliGraph includes a number of loss functions commonly used in literature.
Each function can be used with any of the implemented models. Loss functions are passed to models at the compilation
stage as the ``loss`` parameter to the :meth:`~models.ScoringBasedEmbeddingModel.compile` method. Below are the loss functions available in AmpliGraph.

.. autosummary::
    :toctree:
    :template: class.rst

    PairwiseLoss
    AbsoluteMarginLoss
    SelfAdversarialLoss
    NLLLoss
    NLLMulticlass

.. _ref-reg:

Regularizers
~~~~~~~~~~~~

AmpliGraph includes a number of regularizers that can be used with the :ref:`loss function <loss>`.
Regularizers can be passed to the ``entity_relation_regularizer`` parameter of :meth:`~models.ScoringBasedEmbeddingModel.compile` method.

:meth:`LP_regularizer` supports :math:`L^1, L^2` and :math:`L^3` regularization.
Ampligraph also supports the `regularizers <https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/>`_
available in TensorFlow.

.. autosummary::
    :toctree:
    :template: class.rst

    LP_regularizer


.. _ref-init:

Initializers
~~~~~~~~~~~~

To initialize embeddings, AmpliGraph supports all the `initializers <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/>`_
available in TensorFlow.
Initializers can be passed to the ``entity_relation_initializer`` parameter of :meth:`~models.ScoringBasedEmbeddingModel.compile` method.

.. _optimizer:

Optimizers
~~~~~~~~~~

The goal of the optimization procedure is learning optimal embeddings, such that the scoring function is able to
assign high scores to positive statements and low scores to statements unlikely to be true.

We support `optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ available in TensorFlow.
They can be specified as the ``optimizer`` argument of the :meth:`~ScoringBasedEmbeddingModel.compile` method.

.. _ref_training:

Training
^^^^^^^^

The training procedure follows that of Keras models:

+   The model is initialised as an instance of the :class:`ScoringBasedEmbeddingModel` class. During its initialisation,
    we can specify, among the other hyper-parameters of the model: the size of the embedding (argument ``k``); the scoring
    function applied by the model (argument ``scoring_type``); the number of synthetic negatives generated for each triple
    in the training set (argument ``eta``).

+   The model needs to be compiled through the :meth:`~ScoringBasedEmbeddingModel.compile` method. At this stage we define, among the others,
    the optimizer and the objective functions. These are passed as arguments to the aforementioned method.

+   The model is fitted to the training data using the :meth:`~.ScoringBasedEmbeddingModel.fit` method. Next to the usual parameters that can be
    specified at this stage, AmpliGraph allows to also specify:

        * A ``validation_filter`` that contains the true positives to be removed from the synthetically corrupted \
          triples used during validation.
        * A ``focusE`` option, which enables the FocusE layer :cite:`pai2021learning`: this allows to handle datasets with
          a numeric value associated to the edges, which can signify importance, uncertainty, significance, confidence...
        * A ``partitioning_k`` argument that specifies whether the data needs to be partitioned in order to make training
          with datasets not fitting in memory more efficient.

    For more details and options, check the :meth:`~ScoringBasedEmbeddingModel.fit` method.

Calibration
^^^^^^^^^^^

Another important feature implemented in AmpliGraph is calibration :cite:`calibration`.
Such a method leverages a heuristics that significantly enhance the performance of the models. Further, it bounds the
score of the model in the range :math:`[0,1]`, making the score of the prediction more meaningful and interpretable.

.. autosummary::
    :toctree:
    :template: class.rst

    ~ampligraph.latent_features.layers.calibration.CalibrationLayer


.. _ref_focusE:

Numeric Values on Edges
^^^^^^^^^^^^^^^^^^^^^^^

Numeric values associated to edges of a knowledge graph have been used to represent uncertainty, edge importance, and
even out-of-band knowledge in a growing number of scenarios, ranging from genetic data to social networks.
Nevertheless, traditional KGE models (TransE, DistMult, ComplEx, RotatE, HolE) are not designed to capture such
information, to the detriment of predictive power.

AmpliGraph includes FocusE :cite:`pai2021learning`, a method to inject numeric edge attributes into the scoring
layer of a traditional KGE architecture, thus accounting for the precious information embedded in the edge weights.
In order to add the FocusE layer, set ``focusE=True`` and specify the hyperparameters dictionary ``focusE_params`` in
the :meth:`~ScoringBasedEmbeddingModel.fit()` method.

It is possible to load some benchmark knowledge graphs with numeric-enriched edges through Ampligraph
`dataset loaders <ampligraph.datasets.html#_numeric-enriched-edges-loaders>`__.

Saving/Restoring Models
^^^^^^^^^^^^^^^^^^^^^^^

The weights of a trained model can be saved and restored from disk. This is useful to avoid re-training a model.
In order to save and restore the weights of a model, we can use the :meth:`~ScoringBasedEmbeddingModel.save_weights`
and :meth:`~ScoringBasedEmbeddingModel.load_weights` methods. When the model is saved and loaded with these methods,
however, it is not possible to restart the training from where it stopped. AmpliGraph gives the possibility of doing
that using :meth:`~ampligraph.utils.save_model` and :meth:`~ampligraph.utils.restore_model` available in
the :mod:`.utils` module.

Compatibility Ampligraph 1.x
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: ampligraph.compat
.. automodule:: ampligraph.compat

For those familiar with versions of AmpliGraph 1.x, we have created backward compatible APIs under the
:mod:`ampligraph.compat` module.

These APIs act as wrappers around the newer Keras style APIs and provide seamless experience for our existing user base.

The first group of APIs defines the classes that wraps around the ScoringBasedEmbeddingModel with a specific scoring function.

.. autosummary::
    :toctree:
    :template: class.rst

    TransE
    ComplEx
    DistMult
    HolE

When it comes to evaluation, on the other hand, the following API wraps around the new evaluation process of Ampligraph 2.

.. autosummary::
    :toctree:
    :template: function.rst

    evaluate_performance




