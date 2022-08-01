Models
======

Knowledge Graph Embedding models (KGE) are neural architectures that encode concepts from a knowledge graph
(i.e. entities :math:`\mathcal{E}` and relation types :math:`\mathcal{R}`) into low-dimensional,
continuous vectors :math:`\in \mathbb{R}^k`.

Such *knowledge graph embeddings* have applications
in knowledge graph completion, entity resolution, and link-based clustering, just to cite a few :cite:`nickel2016review`.

Anatomy of AmpliGraph 2 Models
--------------------------------

Knowledge graph embeddings are learned by training a neural architecture over a graph. Although such architectures vary,
the training phase always consists in minimizing a :ref:`loss function <loss>` :math:`\mathcal{L}` that includes a
*scoring function* :math:`f_{m}(t)`, i.e. a model-specific function that assigns a score to a triple :math:`t=(sub,pred,obj)`.

In Ampligraph 2, knowledge graph embedding models are implemented as the ``ScoringBasedEmbeddingModel`` class, that
inherits from `Keras' Model <https://keras.io/api/models/model/>`_:

.. currentmodule:: ampligraph.latent_features.models

.. automodule:: ampligraph.latent_features.models

.. autosummary::
    :toctree:
    :template: class.rst

    ScoringBasedEmbeddingModel

AmpliGraph's ``ScoringBasedEmbeddingModel`` includes the following layers
(all inherit from `Keras' Layer <https://keras.io/api/layers/>`_):

+ :ref:`Encoding Layer <encoding>`
+ :ref:`Scoring Layer <scoring>` :math:`f(t)`
+ :ref:`Negatives Generation Layer <negatives>`

Neural layers aside, a model also requires:

+ :ref:`Loss function <loss>` :math:`\mathcal{L}`
+ :ref:`Optimization algorithm <optimizer>`
+ :ref:`Regularizer <ref-reg>`
+ :ref:`Initializer <ref-init>`

.. _encoding:

Encoding Layer
^^^^^^^^^^^^^^
The encoding layer consists in a 'shallow' encoding, i.e. a lookup of each the embedding of an input node or edge type
from the embedding matrix stored within the model.


.. _scoring:

Scoring Layer
^^^^^^^^^^^^^

.. currentmodule:: ampligraph.latent_features.layers.scoring

.. automodule:: ampligraph.latent_features.layers.scoring

.. autosummary::
    :toctree:
    :template: class.rst

    TransE
    DistMult
    ComplEx
    HolE

Existing models propose scoring functions that combine the embeddings
:math:`\mathbf{e}_{s},\mathbf{r}_{p}, \mathbf{e}_{o} \in \mathcal{R}^k` of the subject, predicate,
and object of a triple :math:`t=(s,p,o)` according to different intuitions:


+ :class:`TransE` :cite:`bordes2013translating` relies on distances. The scoring function computes a similarity between the embedding of the subject translated by the embedding of the predicate  and the embedding of the object, using the :math:`L_1` or :math:`L_2` norm :math:`||\cdot||`:

.. math::
    f_{TransE}=-||\mathbf{e}_{s} + \mathbf{r}_{p} - \mathbf{e}_{o}||_n

+ :class:`DistMult` :cite:`yang2014embedding` uses the trilinear dot product:

.. math::
    f_{DistMult}=\langle \mathbf{r}_p, \mathbf{e}_s, \mathbf{e}_o \rangle

+ :class:`ComplEx` :cite:`trouillon2016complex` extends DistMult with the Hermitian dot product:

.. math::
    f_{ComplEx}=Re(\langle \mathbf{r}_p, \mathbf{e}_s, \overline{\mathbf{e}_o}  \rangle)

+ :class:`HolE` :cite:`nickel2016holographic` uses circular correlation (denoted by :math:`\otimes`):

.. math::
    f_{HolE}=\mathbf{w}_r \cdot (\mathbf{e}_s \otimes \mathbf{e}_o) = \frac{1}{k}\mathcal{F}(\mathbf{w}_r)\cdot( \overline{\mathcal{F}(\mathbf{e}_s)} \odot \mathcal{F}(\mathbf{e}_o))


.. _negatives:

Negatives Generation Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

.. _loss:

Loss Functions
^^^^^^^^^^^^^^

AmpliGraph includes a number of loss functions commonly used in literature.
Each function can be used with any of the implemented models. Loss functions are passed to models as hyperparameter,
and they can be thus used :ref:`during model selection <eval>`.

.. currentmodule:: ampligraph.latent_features

.. automodule:: ampligraph.latent_features
    :noindex:

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
^^^^^^^^^^^^

AmpliGraph includes a number of regularizers that can be used with the :ref:`loss function <loss>`.
`LP_regularizer` supports L1, L2, and L3. We also support regularizers defined in tensorflow.

.. currentmodule:: ampligraph.latent_features

.. automodule:: ampligraph.latent_features

.. autosummary::
    :toctree:
    :template: class.rst

    LP_regularizer


.. _ref-init:

Initializers
^^^^^^^^^^^^
To initialize embeddings, AmpliGraph supports all the initializers defined in TensorFlow.
The initializer is passed as hyperparameter of the ``model.compile`` method.

.. _optimizer:

Optimizers
^^^^^^^^^^

The goal of the optimization procedure is learning optimal embeddings, such that the scoring function is able to
assign high scores to positive statements and low scores to statements unlikely to be true.

We support SGD-based optimizers provided by TensorFlow, by setting the ``optimizer`` argument in a model initializer.

Best results are obtained with Adam.


Saving/Restoring Models
-----------------------

Models can be saved and restored from disk. This is useful to avoid re-training a model.

More details in the :mod:`.utils` module.







