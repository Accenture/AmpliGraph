Performance
===========


Predictive Performance
----------------------

We report the filtered MR, MRR, Hits@1,3,10 for the most common datasets used in literature.


FB15K-237 
---------

========== ======== ====== ======== ======== ========== ========================
  Model       MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ======== ====== ======== ======== ========== ========================
  TransE    199     0.32    0.23     0.36      0.50      k: 400;
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
                                                         early_stopping:{
                                                         x_valid: validation[::10],
                                                         criteria: mrr,
                                                         x_filter: train + validation + test,
                                                         stop_interval: 2,
                                                         burn_in: 0,
                                                         check_interval: 100
                                                         };

 DistMult   194     0.32      0.23     0.35      0.49    k: 300;
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
                                                         early_stopping:{
                                                         x_valid: validation[::10],
                                                         criteria: mrr,
                                                         x_filter: train + validation + test,
                                                         stop_interval: 2,
                                                         burn_in: 0,
                                                         check_interval: 100
                                                         };

   ComplEx  158     0.33      0.23     0.36      0.51    k: 350;
                                                         epochs: 4000;
                                                         eta: 30;
                                                         loss: multiclass_nll;
                                                         loss_params:
                                                         alpha: 1;
                                                         margin: 0.5;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         seed: 0;
                                                         regularizer: LP;
                                                         regularizer_params:
                                                         lambda: 0.0001;
                                                         p: 2;
                                                         batches_count: 50;
                                                         early_stopping:{
                                                         x_valid: validation[::10],
                                                         criteria: mrr,
                                                         x_filter: train + validation + test,
                                                         stop_interval: 2,
                                                         burn_in: 0,
                                                         check_interval: 100
                                                         };
                                                         

   HolE     175     0.32       0.22     0.35       0.49  k: 350;
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
                                                         early_stopping:{
                                                         x_valid: validation[::10],
                                                         criteria: mrr,
                                                         x_filter: train + validation + test,
                                                         stop_interval: 2,
                                                         burn_in: 0,
                                                         check_interval: 100
                                                         };
                                                         

========== ======== ====== ======== ======== ========== ========================

.. note:: FB15K-237 validation and test sets include triples with entities that do not occur 
    in the training set. We found 8 unseen entities in the validation set and 29 in the test set.
    In the experiments we excluded the triples where such entities appear (9 triples in from the validation
    set and 28 from the test set).



WN18RR 
------

========== ========= ====== ======== ======== ========== =======================
  Model       MR      MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ========= ====== ======== ======== ========== =======================
  TransE    2929      0.23    0.03     0.39      0.54     k: 350;
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
                                                          early_stopping:{
                                                          x_valid: validation[::10],
                                                          criteria: mrr,
                                                          x_filter: train + validation + test,
                                                          stop_interval: 2,
                                                          burn_in: 0,
                                                          check_interval: 100
                                                          };

 DistMult   5186      0.48    0.45     0.49      0.54     k: 350;
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
                                                          early_stopping:{
                                                          x_valid: validation[::10],
                                                          criteria: mrr,
                                                          x_filter: train + validation + test,
                                                          stop_interval: 2,
                                                          burn_in: 0,
                                                          check_interval: 100
                                                          };

 ComplEx    4550      0.50    0.47     0.52      0.57     k: 200;
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
                                                          early_stopping:{
                                                          x_valid: validation[::10],
                                                          criteria: mrr,
                                                          x_filter: train + validation + test,
                                                          stop_interval: 2,
                                                          burn_in: 0,
                                                          check_interval: 100
                                                          };
                                                          
   HolE     7236      0.47    0.43     0.48      0.53     k: 200;
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
                                                          early_stopping:{
                                                          x_valid: validation[::10],
                                                          criteria: mrr,
                                                          x_filter: train + validation + test,
                                                          stop_interval: 2,
                                                          burn_in: 0,
                                                          check_interval: 100
                                                          };

========== ========= ====== ======== ======== ========== =======================

.. note:: WN18RR validation and test sets include triples with entities that do not occur
    in the training set. We found 198 unseen entities in the validation set and 209 in the test set.
    In the experiments we excluded the triples where such entities appear (210 triples in from the validation
    set and 210 from the test set).


YAGO3-10
--------

======== ======== ====== ======== ======== ========= =========================
 Model      MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
======== ======== ====== ======== ======== ========= =========================
TransE   1119      0.50   0.40     0.57     0.68      k: 350;
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
                                                      early_stopping:{
                                                      x_valid: validation[::10],
                                                      criteria: mrr,
                                                      x_filter: train + validation + test,
                                                      stop_interval: 2,
                                                      burn_in: 0,
                                                      check_interval: 100
                                                      };
                                                      
DistMult 1348     0.50   0.41     0.55     0.67       k: 350;
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
                                                      early_stopping:{
                                                      x_valid: validation[::10],
                                                      criteria: mrr,
                                                      x_filter: train + validation + test,
                                                      stop_interval: 2,
                                                      burn_in: 0,
                                                      check_interval: 100
                                                      };

ComplEx  1473     0.51   0.42     0.56     0.67       k: 350;
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
                                                      early_stopping:{
                                                      x_valid: validation[::10],
                                                      criteria: mrr,
                                                      x_filter: train + validation + test,
                                                      stop_interval: 2,
                                                      burn_in: 0,
                                                      check_interval: 100
                                                      };

HolE     6365     0.50   0.42     0.55     0.65       k: 350;
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
                                                      early_stopping:{
                                                      x_valid: validation[::10],
                                                      criteria: mrr,
                                                      x_filter: train + validation + test,
                                                      stop_interval: 2,
                                                      burn_in: 0,
                                                      check_interval: 100
                                                      };
======== ======== ====== ======== ======== ========= =========================                                                        



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
  TransE    105      0.55    0.38     0.68      0.79     k: 150;
                                                         epochs: 4000;
                                                         eta: 5;
                                                         loss: pairwise;
                                                         loss_params:
                                                         margin: 0.5;
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
                                                         normalize_ent_emb: false;
                                                         batches_count: 10;
                                                         early_stopping: None;

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
                                                         early_stopping:{
                                                         x_valid: validation[::10],
                                                         criteria: mrr,
                                                         x_filter: train + validation + test,
                                                         stop_interval: 2,
                                                         burn_in: 0,
                                                         check_interval: 100
                                                         };

 ComplEx    183      0.80    0.75     0.82      0.87     k: 200;
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
                                                         early_stopping:{
                                                         x_valid: validation[::10],
                                                         criteria: mrr,
                                                         x_filter: train + validation + test,
                                                         stop_interval: 2,
                                                         burn_in: 0,
                                                         check_interval: 100
                                                         };

   HolE     215      0.80    0.76     0.83      0.87     k: 200;
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
                                                         early_stopping:{
                                                         x_valid: validation[::10],
                                                         criteria: mrr,
                                                         x_filter: train + validation + test,
                                                         stop_interval: 2,
                                                         burn_in: 0,
                                                         check_interval: 100
                                                         };

========== ======== ====== ======== ======== ========== ========================

WN18
----

.. warning::
    The dataset includes a large number of inverse relations, and its use in experiments has been deprecated.
    Use WN18RR instead.


========== ======== ====== ======== ======== ========== ========================
  Model       MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ======== ====== ======== ======== ========== ========================
TransE     477      0.51    0.20     0.81      0.89     k: 150;
                                                        epochs: 4000;
                                                        eta: 5;
                                                        loss: pairwise;
                                                        loss_params:
                                                        margin: 0.5;
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
                                                        batches_count: 10;
                                                        early_stopping: None;
                                                        
 DistMult   755      0.82    0.72     0.92      0.94     k: 200;
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
                                                         early_stopping:{
                                                         x_valid: validation[::10],
                                                         criteria: mrr,
                                                         x_filter: train + validation + test,
                                                         stop_interval: 2,
                                                         burn_in: 0,
                                                         check_interval: 100
                                                         };

 ComplEx    749      0.94    0.94     0.95      0.95     k: 200;
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
                                                         early_stopping:{
                                                         x_valid: validation[::10],
                                                         criteria: mrr,
                                                         x_filter: train + validation + test,
                                                         stop_interval: 2,
                                                         burn_in: 0,
                                                         check_interval: 100
                                                         };
                                                         
   HolE     641      0.93    0.93     0.94      0.95     k: 200;
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
                                                         early_stopping:{
                                                         x_valid: validation[::10],
                                                         criteria: mrr,
                                                         x_filter: train + validation + test,
                                                         stop_interval: 2,
                                                         burn_in: 0,
                                                         check_interval: 100
                                                         };
                                                         
========== ======== ====== ======== ======== ========== ========================

To reproduce the above results: ::
    
    $ cd experiments
    $ python predictive_performance.py


.. note:: Running ``predictive_performance.py`` on all datasets, for all models takes ~43 hours on
    an Intel Xeon Gold 6142, 64 GB Ubuntu 16.04 box equipped with a Tesla V100 16GB.



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

Training the models on FB15K-237 (``k=200, eta=2, batches_count=100, loss=nll``), on an Intel Xeon Gold 6142, 64 GB
Ubuntu 16.04 box equipped with a Tesla V100 16GB gives the following runtime report:

======== ==============
model     seconds/epoch
======== ==============
ComplEx     3.19
TransE      3.26
DistMult    2.61
HolE        3.21
======== ==============
