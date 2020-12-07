Models
======

Knowledge Graph Embedding Models
--------------------------------
In Ampligraph library, we categorize the models into two types:
.. currentmodule:: ampligraph.latent_features.models

.. automodule:: ampligraph.latent_features.models

.. autosummary::
    :toctree: generated
    :template: class.rst
    
+ :class:`ScoringBasedEmbeddingModel` : These are the type of models which follows the ranking protocol during training. For each training sample, the model generates eta corruptions and scores them. The model then tries to maximize the margin between the positive triple and the generated corruption using any one of the distance based losses mentioned below.

+ :class:`TargetBasedEmbeddingModel` : These type of models take as input subject and predicate embeddings of the input triple and try to predict the object. This is the standard protocol that is followed by models such as ConvE and typically use target based losses such as BCE loss.
    

Anatomy of a Model
^^^^^^^^^^^^^^^^^^

Knowledge graph embeddings are learned by training a neural architecture over a graph. Although such architectures vary,
the training phase always consists in minimizing a :ref:`loss function <loss>` :math:`\mathcal{L}` that includes a
*scoring function* :math:`f_{m}(t)`, i.e. a model-specific function that assigns a score to a triple :math:`t=(sub,pred,obj)`.


AmpliGraph models include the following components:

+ :ref:`Scoring function <scoring>` :math:`f(t)`
+ :ref:`Loss function <loss>` :math:`\mathcal{L}`
+ :ref:`Optimization algorithm <optimizer>`
+ :ref:`Regularizer <ref-reg>`
+ :ref:`Initializer <ref-init>`
+ :ref:`Negatives generation strategy <negatives>`

AmpliGraph comes with a number of such components. They can be used in any combination to come up with a model that
performs sufficiently well for the dataset of choice.

AmpliGraph features a number of abstract classes that can be extended to design new models:

.. currentmodule:: ampligraph.latent_features.layers.scoring

.. automodule:: ampligraph.latent_features.layers.scoring

.. autosummary::
    :toctree: generated
    :template: class.rst

    AbstractScoringLayer

.. currentmodule:: ampligraph.datasets

.. automodule:: ampligraph.datasets

.. autosummary::
    :toctree: generated
    :template: class.rst

    AbstractGraphPartitioner
    PartitionDataManager
    
.. currentmodule:: ampligraph.latent_features.loss_functions

.. automodule:: ampligraph.latent_features.loss_functions

.. autosummary::
    :toctree: generated
    :template: class.rst
    
    Loss

.. _scoring:

Scoring functions
-----------------

.. currentmodule:: ampligraph.latent_features.layers.scoring

.. automodule:: ampligraph.latent_features.layers.scoring

.. autosummary::
    :toctree: generated
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


.. _loss:

Loss Functions
--------------

.. currentmodule:: ampligraph.latent_features.loss_functions

.. automodule:: ampligraph.latent_features.loss_functions

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
    LossFunctionWrapper
    
.. autosummary::
    :toctree: generated
    :template: function.rst
    
    get

.. _ref-reg:

Regularizers
--------------

AmpliGraph includes a number of regularizers that can be used with the :ref:`loss function <loss>`.
Ampligraph supports any regularizer defined in tensorflow.

.. _ref-init:

Initializers
--------------

AmpliGraph embeddings can be initialized using any types of initializers provided by tensorflow. This can be passed as a hyperparameter to the compile function as ``entity_relation_initializer`` argument. It can either be a list of 2 initializers for entity and relations or a single initializer for both the embedding matrices.

.. _optimizer:

Optimizers
----------

.. currentmodule:: ampligraph.latent_features.optimizers

.. automodule:: ampligraph.latent_features.optimizers

The goal of the optimization procedure is learning optimal embeddings, such that the scoring function is able to
assign high scores to positive statements and low scores to statements unlikely to be true.

We support SGD-based optimizers provided by TensorFlow, by setting the ``optimizer`` argument in compile function.
Best results are currently obtained with Adam.

.. autosummary::
    :toctree: generated
    :template: class.rst

    OptimizerWrapper

.. autosummary::
    :toctree: generated
    :template: function.rst
    
    get
