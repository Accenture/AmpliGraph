.. _eval_experiments:

Performance
===========


Predictive Performance
----------------------

We report the filtered MR, MRR, Hits@1,3,10 for the most common datasets used in literature.


.. note:: **On ConvE Evaluation**.
    Results reported in the literature for ConvE are based on the alternative *1-N* evaluation protocol which requires
    that reciprocal relations are added to the dataset :cite:`DettmersMS018`:

    .. math::
        D \leftarrow (D, D_{recip})

    .. math::
        D_{recip} \leftarrow \{ \, (o, p_r, s) \,|\, \forall x \in D, x = (s, p, o) \}

    During training each unique pair of subject and predicate can predict all possible object scores for that pairs, and
    therefore object corruptions evaluation can be performed with a single forward pass:

    .. math::
        ConvE(s, p, o)

    In the standard corruption procedure the subject entity is replaced by corruptions:

    .. math::
        ConvE(s_{corr}, p, o),

    However in the `1-N` protocol subject corruptions are interpreted as object corruptions of the reciprocal relation:

    .. math::
        ConvE(o, p_r, s_{corr})


    To reproduce the results reported in the literature using the `1-N` evaluation protocol, add reciprocal relations by
    specifying ``add_reciprocal_rels`` in the dataset loader function, e.g. ``load_fb15k(add_reciprocal_rels=True)``,
    and run the evaluation protocol with object corruptions by specifying ``corrupt_sides='o'``.

    Results obtained with the standard evaluation protocol are labeled *ConvE*, while those obtained with the *1-N*
    protocol are marked *ConvE(1-N)*.



FB15K-237
---------

========== ======== ====== ======== ======== ========== ========================
  Model       MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ======== ====== ======== ======== ========== ========================
  TransE    208     0.31    0.22     0.35      0.50      k: 400;
                                                         epochs: 4000;
                                                         eta: 30;
                                                         loss: multiclass_nll;
                                                         regularizer: LP;
                                                         regularizer_params:
                                                         lambda: 0.0001;
                                                         p: 2;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         embedding_model_params:
                                                         norm: 1;
                                                         normalize_ent_emb: false;
                                                         seed: 0;
                                                         batches_count: 64;

  DistMult  199     0.31      0.22     0.35      0.49    k: 300;
                                                         epochs: 4000;
                                                         eta: 50;
                                                         loss: multiclass_nll;
                                                         regularizer: LP;
                                                         regularizer_params:
                                                         lambda: 0.0001;
                                                         p: 3;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.00005;
                                                         seed: 0;
                                                         batches_count: 50;
                                                         normalize_ent_emb: false;

  ComplEx   184     0.32      0.23     0.35      0.50    k: 350;
                                                         epochs: 4000;
                                                         eta: 30;
                                                         loss: multiclass_nll;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.00005;
                                                         seed: 0;
                                                         regularizer: LP;
                                                         regularizer_params:
                                                         lambda: 0.0001;
                                                         p: 3;
                                                         batches_count: 64;

  HolE      184     0.31       0.22     0.34     0.49    k: 350;
                                                         epochs: 4000;
                                                         eta: 50;
                                                         loss: multiclass_nll;
                                                         regularizer: LP;
                                                         regularizer_params:
                                                         lambda: 0.0001;
                                                         p: 2;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         seed: 0;
                                                         batches_count: 64;

  ConvKB    327     0.23       0.15     0.25     0.40    k: 200;
                                                         epochs: 500;
                                                         eta: 10;
                                                         loss: multiclass_nll;
                                                         loss_params: {}
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         embedding_model_params:{
                                                         num_filters: 32,
                                                         filter_sizes: 1,
                                                         dropout: 0.1};
                                                         seed: 0;
                                                         batches_count: 300;

  ConvE    1060     0.26   0.19     0.28     0.38        k: 200;
                                                         epochs: 4000;
                                                         loss: bce;
                                                         loss_params: {label_smoothing=0.1}
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         embedding_model_params:{
                                                         conv_filters: 32,
                                                         conv_kernel_size: 3,
                                                         dropout_embed: 0.2,
                                                         dropout_conv: 0.1,
                                                         dropout_dense: 0.3,
                                                         use_batchnorm: True,
                                                         use_bias: True};
                                                         seed: 0;
                                                         batches_count: 100;

ConvE(1-N)   234     0.32   0.23      0.35     0.50      k: 200;
                                                         epochs: 4000;
                                                         loss: bce;
                                                         loss_params: {label_smoothing=0.1}
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         embedding_model_params:{
                                                         conv_filters: 32,
                                                         conv_kernel_size: 3,
                                                         dropout_embed: 0.2,
                                                         dropout_conv: 0.1,
                                                         dropout_dense: 0.3,
                                                         use_batchnorm: True,
                                                         use_bias: True};
                                                         seed: 0;
                                                         batches_count: 100;

========== ======== ====== ======== ======== ========== ========================

.. note:: FB15K-237 validation and test sets include triples with entities that do not occur
    in the training set. We found 8 unseen entities in the validation set and 29 in the test set.
    In the experiments we excluded the triples where such entities appear (9 triples in from the validation
    set and 28 from the test set).


WN18RR
------

============ =========== ======== ========== ========== ============ =========================
 Model        MR          MRR      Hits@1     Hits@3     Hits\@10     Hyperparameters
============ =========== ======== ========== ========== ============ =========================
  TransE      2692        0.22     0.03       0.37       0.54         k: 350;
                                                                      epochs: 4000;
                                                                      eta: 30;
                                                                      loss: multiclass_nll;
                                                                      optimizer: adam;
                                                                      optimizer_params:
                                                                      lr: 0.0001;
                                                                      regularizer: LP;
                                                                      regularizer_params:
                                                                      lambda: 0.0001;
                                                                      p: 2;
                                                                      seed: 0;
                                                                      normalize_ent_emb: false;
                                                                      embedding_model_params:
                                                                      norm: 1;
                                                                      batches_count: 150;

 DistMult     5531        0.47     0.43       0.48       0.53         k: 350;
                                                                      epochs: 4000;
                                                                      eta: 30;
                                                                      loss: multiclass_nll;
                                                                      optimizer: adam;
                                                                      optimizer_params:
                                                                      lr: 0.0001;
                                                                      regularizer: LP;
                                                                      regularizer_params:
                                                                      lambda: 0.0001;
                                                                      p: 2;
                                                                      seed: 0;
                                                                      normalize_ent_emb: false;
                                                                      batches_count: 100;

 ComplEx      4177        0.51     0.46       0.53       0.58         k: 200;
                                                                      epochs: 4000;
                                                                      eta: 20;
                                                                      loss: multiclass_nll;
                                                                      loss_params:
                                                                      margin: 1;
                                                                      optimizer: adam;
                                                                      optimizer_params:
                                                                      lr: 0.0005;
                                                                      seed: 0;
                                                                      regularizer: LP;
                                                                      regularizer_params:
                                                                      lambda: 0.05;
                                                                      p: 3;
                                                                      batches_count: 10;

 HolE         7028        0.47     0.44       0.48       0.53         k: 200;
                                                                      epochs: 4000;
                                                                      eta: 20;
                                                                      loss: self_adversarial;
                                                                      loss_params:
                                                                      margin: 1;
                                                                      optimizer: adam;
                                                                      optimizer_params:
                                                                      lr: 0.0005;
                                                                      seed: 0;
                                                                      batches_count: 50;

 ConvKB       3652        0.39     0.33       0.42       0.48         k: 200;
                                                                      epochs: 500;
                                                                      eta: 10;
                                                                      loss: multiclass_nll;
                                                                      loss_params: {}
                                                                      optimizer: adam;
                                                                      optimizer_params:
                                                                      lr: 0.0001;
                                                                      embedding_model_params:{
                                                                      num_filters: 32,
                                                                      filter_sizes: 1,
                                                                      dropout: 0.1};
                                                                      seed: 0;
                                                                      batches_count: 300;

 ConvE        5346        0.45     0.42       0.47       0.52         k: 200;
                                                                      epochs: 4000;
                                                                      loss: bce;
                                                                      loss_params: {label_smoothing=0.1}
                                                                      optimizer: adam;
                                                                      optimizer_params:
                                                                      lr: 0.0001;
                                                                      embedding_model_params:{
                                                                      conv_filters: 32,
                                                                      conv_kernel_size: 3,
                                                                      dropout_embed: 0.2,
                                                                      dropout_conv: 0.1,
                                                                      dropout_dense: 0.3,
                                                                      use_batchnorm: True,
                                                                      use_bias: True};
                                                                      seed: 0;
                                                                      batches_count: 100;

 ConvE(1-N)   4842        0.48     0.45       0.49       0.54         k: 200;
                                                                      epochs: 4000;
                                                                      loss: bce;
                                                                      loss_params: {label_smoothing=0.1}
                                                                      optimizer: adam;
                                                                      optimizer_params:
                                                                      lr: 0.0001;
                                                                      embedding_model_params:{
                                                                      conv_filters: 32,
                                                                      conv_kernel_size: 3,
                                                                      dropout_embed: 0.2,
                                                                      dropout_conv: 0.1,
                                                                      dropout_dense: 0.3,
                                                                      use_batchnorm: True,
                                                                      use_bias: True};
                                                                      seed: 0;
                                                                      batches_count: 100;

============ =========== ======== ========== ========== ============ =========================

.. note:: WN18RR validation and test sets include triples with entities that do not occur
    in the training set. We found 198 unseen entities in the validation set and 209 in the test set.
    In the experiments we excluded the triples where such entities appear (210 triples in from the validation
    set and 210 from the test set).


YAGO3-10
--------

========== ========== ======== ========== ========== =========== ===========================
 Model      MR         MRR      Hits@1     Hits@3     Hits\@10    Hyperparameters
========== ========== ======== ========== ========== =========== ===========================
TransE      1264       0.51     0.41       0.57       0.67        k: 350;
                                                                  epochs: 4000;
                                                                  eta: 30;
                                                                  loss: multiclass_nll;
                                                                  optimizer: adam;
                                                                  optimizer_params:
                                                                  lr: 0.0001;
                                                                  regularizer: LP;
                                                                  regularizer_params:
                                                                  lambda: 0.0001;
                                                                  p: 2;
                                                                  embedding_model_params:
                                                                  norm: 1;
                                                                  normalize_ent_emb: false;
                                                                  seed: 0;
                                                                  batches_count: 100;

DistMult    1107       0.50     0.41       0.55       0.66        k: 350;
                                                                  epochs: 4000;
                                                                  eta: 50;
                                                                  loss: multiclass_nll;
                                                                  optimizer: adam;
                                                                  optimizer_params:
                                                                  lr: 5e-05;
                                                                  regularizer: LP;
                                                                  regularizer_params:
                                                                  lambda: 0.0001;
                                                                  p: 3;
                                                                  seed: 0;
                                                                  normalize_ent_emb: false;
                                                                  batches_count: 100;

ComplEx     1227       0.49     0.40       0.54       0.66        k: 350;
                                                                  epochs: 4000;
                                                                  eta: 30;
                                                                  loss: multiclass_nll;
                                                                  optimizer: adam;
                                                                  optimizer_params:
                                                                  lr: 5e-05;
                                                                  regularizer: LP;
                                                                  regularizer_params:
                                                                  lambda: 0.0001;
                                                                  p: 3;
                                                                  seed: 0;
                                                                  batches_count: 100

HolE        6776       0.50     0.42       0.56       0.65        k: 350;
                                                                  epochs: 4000;
                                                                  eta: 30;
                                                                  loss: self_adversarial;
                                                                  loss_params:
                                                                  alpha: 1;
                                                                  margin: 0.5;
                                                                  optimizer: adam;
                                                                  optimizer_params:
                                                                  lr: 0.0001;
                                                                  seed: 0;
                                                                  batches_count: 100

ConvKB      2820       0.30     0.21       0.34       0.50        k: 200;
                                                                  epochs: 500;
                                                                  eta: 10;
                                                                  loss: multiclass_nll;
                                                                  loss_params: {}
                                                                  optimizer: adam;
                                                                  optimizer_params:
                                                                  lr: 0.0001;
                                                                  embedding_model_params:{
                                                                  num_filters: 32,
                                                                  filter_sizes: 1,
                                                                  dropout: 0.1};
                                                                  seed: 0;
                                                                  batches_count: 3000;

 ConvE      6063       0.40     0.33       0.42       0.53        k: 300;
                                                                  epochs: 4000;
                                                                  loss: bce;
                                                                  loss_params: {label_smoothing=0.1}
                                                                  optimizer: adam;
                                                                  optimizer_params:
                                                                  lr: 0.0001;
                                                                  embedding_model_params:{
                                                                  conv_filters: 32,
                                                                  conv_kernel_size: 3,
                                                                  dropout_embed: 0.2,
                                                                  dropout_conv: 0.1,
                                                                  dropout_dense: 0.3,
                                                                  use_batchnorm: True,
                                                                  use_bias: True};
                                                                  seed: 0;
                                                                  batches_count: 300;

ConvE(1-N)  2741       0.55     0.48       0.60       0.69        k: 300;
                                                                  epochs: 4000;
                                                                  loss: bce;
                                                                  loss_params: {label_smoothing=0.1}
                                                                  optimizer: adam;
                                                                  optimizer_params:
                                                                  lr: 0.0001;
                                                                  embedding_model_params:{
                                                                  conv_filters: 32,
                                                                  conv_kernel_size: 3,
                                                                  dropout_embed: 0.2,
                                                                  dropout_conv: 0.1,
                                                                  dropout_dense: 0.3,
                                                                  use_batchnorm: True,
                                                                  use_bias: True};
                                                                  seed: 0;
                                                                  batches_count: 300;

========== ========== ======== ========== ========== =========== ===========================


.. note:: YAGO3-10 validation and test sets include triples with entities that do not occur
    in the training set. We found 22 unseen entities in the validation set and 18 in the test set.
    In the experiments we excluded the triples where such entities appear (22 triples in from the validation
    set and 18 from the test set).


FB15K
-----


.. warning::
    The dataset includes a large number of inverse relations, and its use in experiments has been deprecated.
    Use FB15k-237 instead.


========== ======== ====== ======== ======== ========== ========================
  Model       MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ======== ====== ======== ======== ========== ========================
  TransE    44      0.63    0.50     0.73      0.85     k: 150;
                                                        epochs: 4000;
                                                        eta: 10;
                                                        loss: multiclass_nll;
                                                        optimizer: adam;
                                                        optimizer_params:
                                                        lr: 5e-5;
                                                        regularizer: LP;
                                                        regularizer_params:
                                                        lambda: 0.0001;
                                                        p: 3;
                                                        embedding_model_params:
                                                        norm: 1;
                                                        normalize_ent_emb: false;
                                                        seed: 0;
                                                        batches_count: 100;

 DistMult   179      0.78    0.74     0.82      0.86     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         normalize_ent_emb: false;
                                                         batches_count: 50;

 ComplEx    184      0.80    0.76     0.82      0.86     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         batches_count: 100;

   HolE     216      0.80    0.76     0.83      0.87     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         batches_count: 50;

  ConvKB    331      0.65    0.55     0.71      0.82     k: 200;
                                                         epochs: 500;
                                                         eta: 10;
                                                         loss: multiclass_nll;
                                                         loss_params: {}
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         embedding_model_params:{
                                                         num_filters: 32,
                                                         filter_sizes: 1,
                                                         dropout: 0.1};
                                                         seed: 0;
                                                         batches_count: 300;

  ConvE     385      0.50    0.42     0.52     0.66      k: 300;
                                                         epochs: 4000;
                                                         loss: bce;
                                                         loss_params: {label_smoothing=0.1}
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         embedding_model_params:{
                                                         conv_filters: 32,
                                                         conv_kernel_size: 3,
                                                         dropout_embed: 0.2,
                                                         dropout_conv: 0.1,
                                                         dropout_dense: 0.3,
                                                         use_batchnorm: True,
                                                         use_bias: True};
                                                         seed: 0;
                                                         batches_count: 100;

ConvE(1-N)    55     0.80     0.74    0.84     0.89      k: 300;
                                                         epochs: 4000;
                                                         loss: bce;
                                                         loss_params: {label_smoothing=0.1}
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         embedding_model_params:{
                                                         conv_filters: 32,
                                                         conv_kernel_size: 3,
                                                         dropout_embed: 0.2,
                                                         dropout_conv: 0.1,
                                                         dropout_dense: 0.3,
                                                         use_batchnorm: True,
                                                         use_bias: True};
                                                         seed: 0;
                                                         batches_count: 100;

========== ======== ====== ======== ======== ========== ========================

WN18
----

.. warning::
    The dataset includes a large number of inverse relations, and its use in experiments has been deprecated.
    Use WN18RR instead.


========== ======== ====== ======== ======== ========== ========================
  Model       MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ======== ====== ======== ======== ========== ========================
TransE     260      0.66    0.44     0.88      0.95     k: 150;
                                                        epochs: 4000;
                                                        eta: 10;
                                                        loss: multiclass_nll;
                                                        optimizer: adam;
                                                        optimizer_params:
                                                        lr: 5e-5;
                                                        regularizer: LP;
                                                        regularizer_params:
                                                        lambda: 0.0001;
                                                        p: 3;
                                                        embedding_model_params:
                                                        norm: 1;
                                                        normalize_ent_emb: false;
                                                        seed: 0;
                                                        batches_count: 100;

 DistMult   675      0.82    0.73     0.92      0.95     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: nll;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         normalize_ent_emb: false;
                                                         batches_count: 50;

 ComplEx    726      0.94    0.94     0.95      0.95     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: nll;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         batches_count: 50;

  HolE     665      0.94    0.93     0.94       0.95     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         batches_count: 50;

  ConvKB     331      0.80    0.69     0.90       0.94   k: 200;
                                                         epochs: 500;
                                                         eta: 10;
                                                         loss: multiclass_nll;
                                                         loss_params: {}
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         embedding_model_params:{
                                                         num_filters: 32,
                                                         filter_sizes: 1,
                                                         dropout: 0.1};
                                                         seed: 0;
                                                         batches_count: 300;

  ConvE     492      0.93   0.91     0.94     0.95       k: 300;
                                                         epochs: 4000;
                                                         loss: bce;
                                                         loss_params: {label_smoothing=0.1}
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         embedding_model_params:{
                                                         conv_filters: 32,
                                                         conv_kernel_size: 3,
                                                         dropout_embed: 0.2,
                                                         dropout_conv: 0.1,
                                                         dropout_dense: 0.3,
                                                         use_batchnorm: True,
                                                         use_bias: True};
                                                         seed: 0;
                                                         batches_count: 100;

ConvE(1-N)    436    0.95    0.93     0.95     0.95      k: 300;
                                                         epochs: 4000;
                                                         loss: bce;
                                                         loss_params: {label_smoothing=0.1}
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         embedding_model_params:{
                                                         conv_filters: 32,
                                                         conv_kernel_size: 3,
                                                         dropout_embed: 0.2,
                                                         dropout_conv: 0.1,
                                                         dropout_dense: 0.3,
                                                         use_batchnorm: True,
                                                         use_bias: True};
                                                         seed: 0;
                                                         batches_count: 100;

========== ======== ====== ======== ======== ========== ========================


To reproduce the above results: ::

    $ cd experiments
    $ python predictive_performance.py


.. note:: Running ``predictive_performance.py`` on all datasets, for all models takes ~115 hours on
    an Intel Xeon Gold 6142, 64 GB Ubuntu 16.04 box equipped with a Tesla V100 16GB.
    The long running time is mostly due to the early stopping configuration (see section below).

.. note:: All of the experiments above were conducted with early stopping on half the validation set.
    Typically, the validation set can be found in ``X['valid']``.
    We only used half the validation set so the other half is available for hyperparameter tuning.

    The exact early stopping configuration is as follows:

      * x_valid: validation[::2]
      * criteria: mrr
      * x_filter: train + validation + test
      * stop_interval: 4
      * burn_in: 0
      * check_interval: 50

    Note that early stopping adds a significant computational burden to the learning procedure.
    To lessen it, you may either decrease the validation set, the stop interval, the check interval,
    or increase the burn in.


.. note:: Due to a combination of model and dataset size it is not possible to evaluate Yago3-10 with ConvKB on the
    GPU. The fastest way to replicate the results above is to train ConvKB with Yago3-10 on a GPU using the hyper-
    parameters described above (~15hrs on GTX 1080Ti), and then evaluate the model in CPU only mode (~15 hours on
    Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz).

.. note:: ConvKB with early-stopping evaluation protocol does not fit into GPU memory, so instead is just
    trained for a set number of epochs.

Experiments can be limited to specific models-dataset combinations as follows: ::

    $ python predictive_performance.py -h
    usage: predictive_performance.py [-h] [-d {fb15k,fb15k-237,wn18,wn18rr,yago310}]
                                     [-m {complex,transe,distmult,hole,convkb,conve}]

    optional arguments:
      -h, --help            show this help message and exit
      -d {fb15k,fb15k-237,wn18,wn18rr,yago310}, --dataset {fb15k,fb15k-237,wn18,wn18rr,yago310}
      -m {complex,transe,distmult,hole,convkb,conve}, --model {complex,transe,distmult,hole,convkb,conve}


Runtime Performance
-------------------

Training the models on FB15K-237 (``k=100, eta=10, batches_count=100, loss=multiclass_nll``), on an Intel Xeon Gold 6142, 64 GB
Ubuntu 16.04 box equipped with a Tesla V100 16GB gives the following runtime report:

======== ==============
model     seconds/epoch
======== ==============
ComplEx     1.33
TransE      1.22
DistMult    1.20
HolE        1.30
ConvKB      2.83
ConvE       1.13
======== ==============

.. note::

    ConvE is trained with ``bce`` loss instead of ``multiclass_nll``.