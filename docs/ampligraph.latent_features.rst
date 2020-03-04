Models
======

.. currentmodule:: ampligraph.latent_features

.. automodule:: ampligraph.latent_features


Knowledge Graph Embedding Models
--------------------------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    RandomBaseline
    TransE
    DistMult
    ComplEx
    HolE
    ConvE
    ConvKB

Anatomy of a Model
^^^^^^^^^^^^^^^^^^

Knowledge graph embeddings are learned by training a neural architecture over a graph. Although such architectures vary,
the training phase always consists in minimizing a :ref:`loss function <loss>` :math:`\mathcal{L}` that includes a
*scoring function* :math:`f_{m}(t)`, i.e. a model-specific function that assigns a score to a triple :math:`t=(sub,pred,obj)`.


AmpliGraph models include the following components:

+ :ref:`Scoring function <scoring>` :math:`f(t)`
+ :ref:`Loss function <loss>` :math:`\mathcal{L}`
+ :ref:`Optimization algorithm <optimizer>`
+ :ref:`Negatives generation strategy <negatives>`

AmpliGraph comes with a number of such components. They can be used in any combination to come up with a model that
performs sufficiently well for the dataset of choice.

AmpliGraph features a number of abstract classes that can be extended to design new models:

.. autosummary::
    :toctree: generated
    :template: class.rst

    EmbeddingModel
    Loss
    Regularizer
    Initializer


.. _scoring:

Scoring functions
-----------------

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

+ :class:`ConvE` :cite:`DettmersMS018` uses convolutional layers (:math:`g` is a non-linear activation function, :math:`\ast` is the linear convolution operator, :math:`vec` indicates 2D reshaping):

.. math::

    f_{ConvE} =  \langle \sigma \, (vec \, ( g \, ([ \overline{\mathbf{e}_s} ; \overline{\mathbf{r}_p} ] \ast \Omega )) \, \mathbf{W} )) \, \mathbf{e}_o\rangle


+ :class:`ConvKB` :cite:`Nguyen2018` uses convolutional layers and a dot product:

.. math::

    f_{ConvKB}= concat \,(g \, ([\mathbf{e}_s, \mathbf{r}_p, \mathbf{e}_o]) * \Omega)) \cdot W


.. _loss:

Loss Functions
--------------

AmpliGraph includes a number of loss functions commonly used in literature.
Each function can be used with any of the implemented models. Loss functions are passed to models as hyperparameter,
and they can be thus used :ref:`during model selection <eval>`.



.. autosummary::
    :toctree: generated
    :template: class.rst

    PairwiseLoss
    AbsoluteMarginLoss
    SelfAdversarialLoss
    NLLLoss
    NLLMulticlass
    BCELoss
    
.. _ref-reg:

Regularizers
--------------

AmpliGraph includes a number of regularizers that can be used with the :ref:`loss function <loss>`.
:class:`LPRegularizer` supports L1, L2, and L3.

.. autosummary::
    :toctree: generated
    :template: class.rst

    LPRegularizer


.. _ref-init:

Initializers
--------------

AmpliGraph includes a number of initializers that can be used to initialize the embeddings. They can be passed as hyperparameter,
and they can be thus used :ref:`during model selection <eval>`.

.. autosummary::
    :toctree: generated
    :template: class.rst

    RandomNormal
    RandomUniform
    Xavier


.. _optimizer:

Optimizers
----------

The goal of the optimization procedure is learning optimal embeddings, such that the scoring function is able to
assign high scores to positive statements and low scores to statements unlikely to be true.

We support SGD-based optimizers provided by TensorFlow, by setting the ``optimizer`` argument in a model initializer.
Best results are currently obtained with Adam.


Saving/Restoring Models
-----------------------

Models can be saved and restored from disk. This is useful to avoid re-training a model.

More details in the :mod:`.utils` module.
