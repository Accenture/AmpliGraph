Models
======

In Ampligraph 2, the models are inherited from the keras `Model` class and is implemented in the `ScoringBasedEmbeddingModel` class. 

.. currentmodule:: ampligraph.latent_features.models

.. automodule:: ampligraph.latent_features.models

.. autosummary::
    :toctree: generated
    :template: class.rst

    ScoringBasedEmbeddingModel
    
The model consists of following layers:
    - Encoding Layer
    - Corruption Generation Layer
    - Scoring Layer
    
The above layers are inherited from keras `Layer` class. The encoding layer looks up the embeddings of input triples, the corruption generation layer generates the corruptions and the Scoring layer uses one of the scoring function described below, to compute the scores of positive triples and their corruptions. The loss is computed using one of the loss function described later.

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

