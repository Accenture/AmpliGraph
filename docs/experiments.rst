.. _eval_experiments:

Performance
===========


Predictive Performance
----------------------

We report AmpliGraph 2 filtered MR, MRR, Hits@1,3,10 results for the most common datasets used in literature.


.. note:: **AmpliGraph 1.x Benchmarks**.
    AmpliGraph 1.x predictive power report is available `here <https://docs.ampligraph.org/en/1.4.0/experiments.html>`_.


FB15K-237
---------

========== ======== ====== ======== ======== ========== ========================
  Model       MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ======== ====== ======== ======== ========== ========================
  TransE    211     0.31    0.22     0.34     0.48       k: 400;
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
                                                         seed: 0;
                                                         batches_count: 64;

  DistMult  211     0.30      0.21     0.33      0.48    k: 300;
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

  ComplEx   197     0.31      0.21     0.34      0.49    k: 350;
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

  HolE      190     0.30       0.21     0.33     0.48    k: 350;
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
  TransE      3143        0.22     0.03       0.38       0.52         k: 350;
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
                                                                      embedding_model_params:
                                                                      norm: 1;
                                                                      batches_count: 150;

 DistMult     4832        0.47     0.43       0.48       0.54         k: 350;
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
                                                                      batches_count: 100;

 ComplEx      4229        0.50     0.47       0.52       0.58         k: 200;
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

 HolE         7072        0.47     0.44       0.49       0.54         k: 200;
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
TransE      1210       0.50     0.41       0.56       0.67        k: 350;
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
                                                                  seed: 0;
                                                                  batches_count: 100;

DistMult    2301       0.48     0.39       0.53       0.64        k: 350;
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
                                                                  batches_count: 100;

ComplEx     3153       0.49     0.40       0.54       0.65        k: 350;
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

HolE        7525       0.47     0.38       0.52       0.62        k: 350;
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
  TransE    45      0.62    0.48     0.72      0.84     k: 150;
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
                                                        seed: 0;
                                                        batches_count: 100;

 DistMult   227      0.71    0.66     0.75      0.80     k: 200;
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

 ComplEx    199      0.73    0.67     0.77      0.82     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         regularizer: LP;
                                                         regularizer_params:
                                                         lambda: 0.0001;
                                                         p: 3;
                                                         seed: 0;
                                                         batches_count: 100;

   HolE     216      0.00    0.00     0.00      0.00     k: 200;
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

========== ======== ====== ======== ======== ========== ========================

WN18
----

.. warning::
    The dataset includes a large number of inverse relations, and its use in experiments has been deprecated.
    Use WN18RR instead.


========== ======== ====== ======== ======== ========== ========================
  Model       MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ======== ====== ======== ======== ========== ========================
TransE     278      0.64    0.39     0.87      0.95     k: 150;
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
                                                        seed: 0;
                                                        batches_count: 100;

 DistMult   729      0.82    0.72     0.92      0.95     k: 200;
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

 ComplEx    758      0.94    0.94     0.95      0.95     k: 200;
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

  HolE     676      0.94    0.93     0.94       0.95     k: 200;
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

========== ======== ====== ======== ======== ========== ========================


To reproduce the above results: ::

    $ cd experiments
    $ python predictive_performance.py


.. note:: Running ``predictive_performance.py`` on all datasets, for all models takes ~40 hours on
    an an Intel Xeon Gold 6226R, 256 GB, equipped with Tesla A100 40GB GPUs and  Ubuntu 20.04.

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

    Note that early stopping can save a lot of training time, but it also adds some computational cost to the
    learning procedure. To lessen it, you may either decrease the validation set, the stop interval, the check interval,
    or increase the burn in.


Experiments can be limited to specific models-dataset combinations as follows: ::

    $ python predictive_performance.py -h
    usage: predictive_performance.py [-h] [-d {fb15k,fb15k-237,wn18,wn18rr,yago310}]
                                     [-m {complex,transe,distmult,hole}]

    optional arguments:
      -h, --help            show this help message and exit
      -d {fb15k,fb15k-237,wn18,wn18rr,yago310}, --dataset {fb15k,fb15k-237,wn18,wn18rr,yago310}
      -m {complex,transe,distmult,hole}, --model {complex,transe,distmult,hole}


Runtime Performance
-------------------

Training the models on FB15K-237 (``k=100, eta=10, batches_count=10, loss=multiclass_nll``), on an Intel Xeon
Gold 6226R, 256 GB, equipped with Tesla A100 40GB GPUs and Ubuntu 20.04 gives the following runtime report:

======== ==============
model     seconds/epoch
======== ==============
ComplEx     0.18
TransE      0.09
DistMult    0.10
HolE        0.18
======== ==============
