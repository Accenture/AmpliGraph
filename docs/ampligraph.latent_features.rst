Models
======

.. currentmodule:: ampligraph.latent_features

.. automodule:: ampligraph.latent_features


Knowledge Graph Embedding Models
--------------------------------

Existing models propose scoring functions that combine the embeddings :math:`\mathbf{e}_{sub},\mathbf{e}_{pred}, \mathbf{e}_{obj} \in \mathcal{R}^k` of the subject, predicate, and object of triple :math:`t=(sub,pred,obj)` using different intuitions: TransE :cite:`bordes2013translating` relies on distances, DistMult :cite:`yang2014embedding` and ComplEx :cite:`trouillon2016complex` are bilinear-diagonal models, HolE :cite:`nickel2016holographic` uses circular correlation. While the above models can be interpreted as multilayer perceptrons, others such as ConvE include convolutional layers :cite:`DettmersMS018`.

As example, the scoring function of TransE computes a similarity between the embedding of the subject :math:`\mathbf{e}_{sub}` translated by the embedding of the predicate :math:`\mathbf{e}_{pred}` and the embedding of the object :math:`\mathbf{e}_{obj}`, using the :math:`L_1` or :math:`L_2` norm :math:`||\cdot||`:

.. math::

    f_{TransE}=-||\mathbf{e}_{sub} + \mathbf{e}_{pred} - \mathbf{e}_{obj}||_n


Such scoring function is then used on positive and negative triples :math:`t^+, t^-` in the loss function.


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


Knowledge graph embeddings are learned by training a neural architecture over a graph. Although such architectures vary, the training phase always consists in minimizing a loss function :math:`\mathcal{L}` that includes a *scoring function* :math:`f_{m}(t)`, i.e. a model-specific function that assigns a score to a triple :math:`t=(sub,pred,obj)`.  

The goal of the optimization procedure is learning optimal embeddings, such that the scoring function is able to assign high scores to positive statements and low scores to statements unlikely to be true.

 This can be for example a pairwise margin-based loss, as shown in the equation below:

.. math::
    \mathcal{L}(\Theta) = \sum_{t^+ \in \mathcal{G}}\sum_{t^- \in \mathcal{N}}max(0, [\gamma + f_{m}(t^-;\Theta) - f_{m}(t^+;\Theta)])

where :math:`\Theta` are the embeddings learned by the model, :math:`f_{m}` is the model-specific scoring function, :math:`\gamma \in \mathcal{R}` is the margin and :math:`\mathcal{N}` is a set of negative triples generated with a corruption heuristic :cite:`bordes2013translating`.





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

AmpliGraph includes a number of regularizers that can be used with the loss functions defined above.

.. autosummary::
    :toctree: generated
    :template: class.rst

    Regularizer
    L1Regularizer
    L2Regularizer


Utils Functions
---------------

Models can be saved and restored from disk. This is useful to avoid re-training a model.


.. autosummary::
    :toctree: generated
    :template: function.rst

    save_model
    restore_model
