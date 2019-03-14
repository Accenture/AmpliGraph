Performance
===========


Predictive Performance
----------------------

We report the filtered MR, MRR, Hits@1,3,10 for the most common datasets used in literature.


FB15K-237 
---------

========== ======= ======== ======== ======== ======== ==========================
  Model      MR     MRR     Hits@1    Hits@3  Hits\@10      Hyperparameters
========== ======= ======== ======== ======== ======== ==========================
TransE     153      0.32     0.22     0.35     0.51      batches_count: 60;
                                                         embedding_model_params:
                                                         norm: 1;
                                                         epochs: 4000;
                                                         eta: 50;
                                                         k: 1000;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 5;
                                                         alpha: 0.5;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         seed: 0
                                                  
 DistMult   441      0.29    0.20      0.32      0.48    batches_count: 50;
                                                         embedding_model_params:
                                                         norm: 1;
                                                         epochs: 4000;
                                                         eta: 50;
                                                         k: 400;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         alpha: 1;
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         regularizer: LP;
                                                         regularizer_params:
                                                         lambda: 1.0e-05;
                                                         p: 2;
                                                         seed: 0
                                                 
 ComplEx    513     0.30    0.20      0.33      0.48      batches_count: 50;
                                                          embedding_model_params:
                                                          norm: 1;
                                                          epochs: 4000;
                                                          eta: 30;
                                                          k: 350;
                                                          loss: self_adversarial;
                                                          loss_params:
                                                          alpha: 1;
                                                          margin: 0.5;
                                                          optimizer: adam;
                                                          optimizer_params:
                                                          lr: 0.0001;
                                                          regularizer: LP;
                                                          regularizer_params:
                                                          lambda: 0.0001;
                                                          p: 2;
                                                          seed: 0
                                                   
 HolE       296      0.28   0.19     0.31      0.46       batches_count: 50;
                                                          epochs: 4000;
                                                          eta: 30;
                                                          k: 350;
                                                          loss: self_adversarial;
                                                          loss_params:
                                                          alpha: 1;
                                                          margin: 0.5;
                                                          optimizer: adam;
                                                          optimizer_params:
                                                          lr: 0.0001;
                                                          seed: 0

========== ======= ======== ======== ======== ======== ==========================

.. note:: FB15K-237 validation and test sets include triples with entities that do not occur 
    in the training set. We removed (8, 29) unseen entities inside (9, 28) triples in (validation, test) sets in this experiment.



WN18RR 
------

========== ======= ======== ======== ======== ======== ==========================
  Model      MR     MRR     Hits@1    Hits@3  Hits\@10      Hyperparameters
========== ======= ======== ======== ======== ======== ==========================
TransE     1532    0.23     0.07     0.34      0.50       batches_count: 100;
                                                          embedding_model_params:
                                                          norm: 1;
                                                          epochs: 4000;
                                                          eta: 20;
                                                          k: 200;
                                                          loss: self_adversarial;
                                                          loss_params:
                                                          margin: 1;
                                                          optimizer: adam;
                                                          optimizer_params:
                                                          lr: 0.0001;
                                                          regularizer: LP;
                                                          regularizer_params:
                                                          lambda: 1.0e-05;
                                                          p: 1;
                                                          seed: 0
                                                 
 DistMult  6853     0.44      0.42    0.45     0.50      batches_count: 25;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         k: 200;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0
                                                 
 ComplEx    8213    0.44      0.41     0.45     0.50     batches_count: 10;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         k: 200;
                                                         loss: nll;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0
                                                 
   HolE     7304   0.47       0.43     0.48     0.53     batches_count: 50;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         k: 200;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0
========== ======= ======== ======== ======== ======== ==========================

.. note:: We removed (198, 209) unseen entities inside (210, 210) triples in (validation, test) sets in this experiment.


FB15K
-----

========== ======= ======== ======== ======== ======== ==========================
  Model      MR     MRR     Hits@1    Hits@3  Hits\@10      Hyperparameters
========== ======= ======== ======== ======== ======== ==========================
  TransE    105    0.55      0.39     0.68     0.79      batches_count: 10;
                                                         embedding_model_params:
                                                         norm: 1;
                                                         epochs: 4000;
                                                         eta: 5;
                                                         k: 150;
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
                                                         seed: 0
                                                 

 DistMult   177    0.79      0.74     0.82     0.86      batches_count: 50;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         k: 200;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0
                                                         

 ComplEx    188    0.79      0.76     0.82     0.86      batches_count: 100;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         k: 200;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0
                                                         

   HolE     212    0.80       0.76     0.83     0.87     batches_count: 50;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         k: 200;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0
========== ======= ======== ======== ======== ======== ==========================


WN18
----

========== ======= ======== ======== ======== ======== ==========================
  Model      MR     MRR     Hits@1    Hits@3  Hits\@10      Hyperparameters
========== ======= ======== ======== ======== ======== ==========================
 TransE    445      0.50     0.16     0.82     0.90      batches_count: 10;
                                                         embedding_model_params:
                                                         norm: 1;
                                                         epochs: 4000;
                                                         eta: 5;
                                                         k: 150;
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
                                                         seed: 0
                                                

 DistMult   746    0.83      0.73     0.92     0.95      batches_count: 50;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         k: 200;
                                                         loss: nll;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0
                                                                                                
 ComplEx    715    0.94      0.94     0.95     0.95      batches_count: 50;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         k: 200;
                                                         loss: nll;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0

   HolE     658     0.94     0.93      0.94     0.95     batches_count: 50;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         k: 200;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0
========== ======= ======== ======== ======== ======== ==========================





To reproduce the above results: ::
    
    $ cd experiments
    $ predictive_performance.py


.. note:: Running ``predictive_performance.py`` on all datasets, for all models takes ~xxx hours on 
    an Intel Xeon Gold 6142, 64 GB Ubuntu 16.04 box equipped with a Tesla V100 16GB.



Experiments can be limited to specific models-dataset combinations as follows: ::

    $ python predictive_performance.py -h
    usage: predictive_performance.py [-h] [-d DATASET] [-m MODEL]

    optional arguments:
      -h, --help            show this help message and exit
      -d DATASET, --dataset DATASET
      -m MODEL, --model MODEL



Runtime Performance
-------------------

//TODO
see issue #49
