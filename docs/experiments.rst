Performance
===========


Predictive Performance
----------------------

We report the filtered MR, MRR, Hit at 1, 3 and 10 on the most common datasets used in literature.


FB15K
-----

========== ======== ====== ====== ====== ====== =========================
  Model       MR     MRR   H @ 1  H @ 3  H @ 10       Hyperparameters
========== ======== ====== ====== ====== ====== =========================
  TransE    105.45   0.55   0.39   0.68   0.79   batches_count: 10;
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
                                                 regularizer: L2;
                                                 regularizer_params:
                                                 lambda: 0.0001;
                                                 seed: 0
                                                 

 DistMult   177.23   0.79   0.74   0.82   0.86   batches_count: 50;
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
                                                 

 ComplEx    188.43   0.79   0.76   0.82   0.86   batches_count: 100;
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
                                                 

   HolE     212.62   0.80    0.76   0.83   0.87  batches_count: 50;
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
========== ======== ====== ====== ====== ====== =========================


WN18
----

========== ======== ====== ====== ====== ====== =========================
  Model       MR     MRR   H @ 1  H @ 3  H @ 10      Hyperparameters
========== ======== ====== ====== ====== ====== =========================
 TransE    445.28    0.50   0.16   0.82   0.90   batches_count: 10;
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
                                                 regularizer: L2;
                                                 regularizer_params:
                                                 lambda: 0.0001;
                                                 seed: 0
                                                

 DistMult  746.44    0.83   0.73   0.92   0.95   batches_count: 50;
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
                                                
 ComplEx   715.09    0.94   0.94   0.95   0.95   batches_count: 50;
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

   HolE    658.85    0.94   0.93   0.94   0.95   batches_count: 50;
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
========== ======== ====== ====== ====== ====== =========================

WN18RR 
------

.. warning:: We removed (198,209) unseen entities in (validation, test) sets in this experiment.

========== ========= ====== ====== ====== ====== ========================
  Model       MR      MRR   H @ 1  H @ 3  H @ 10      Hyperparameters
========== ========= ====== ====== ====== ====== ========================
TransE     1532.28   0.23   0.07   0.34   0.50    batches_count: 100;
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
                                                  regularizer: L1;
                                                  regularizer_params:
                                                  lambda: 1.0e-05;
                                                  seed: 0
                                                 
 DistMult   6853.22   0.44   0.42   0.45   0.50  batches_count: 25;
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
                                                 
 ComplEx    8213.51   0.44   0.41   0.45   0.50  batches_count: 10;
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
                                                 
   HolE     7304.87   0.47   0.43   0.48   0.53  batches_count: 50;
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
========== ========= ====== ====== ====== ====== ========================


FB15K-237 
---------

.. warning:: We removed (8,29) unseen entities in (validation, test) sets in this experiment.

========= ======== ====== ====== ====== ====== ==========================
  Model      MR     MRR    H @ 1 H @ 3  H @ 10      Hyperparameters
========= ======== ====== ====== ====== ====== ==========================
TransE     373.63   0.27   0.18   0.30   0.44    batches_count: 10;
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
                                                 regularizer: L2;
                                                 regularizer_params:
                                                 lambda: 0.0001;
                                                 seed: 0
                                                  
 DistMult   441.22   0.29   0.20   0.32   0.48   batches_count: 50;
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
                                                 regularizer: L2;
                                                 regularizer_params:
                                                 lambda: 1.0e-05;
                                                 seed: 0
                                                 
ComplEx   606.17   0.27   0.18   0.29   0.45    batches_count: 100;
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

 HolE      408.71   0.20  0.12   0.22   0.38    batches_count: 100;
                                                epochs: 4000;
                                                eta: 20;
                                                k: 200;
                                                loss: nll;
                                                optimizer: adam;
                                                optimizer_params:
                                                lr: 0.0005;
                                                regularizer: L2;
                                                regularizer_params:
                                                lambda: 1.0e-05;
                                                seed: 0
========= ======== ====== ====== ====== ====== ==========================

Results in the table above can be reproduced by running as below:

`$ cd experiments`

For all experiments: 

`$ python predictive_performance.py`

For single dataset:

`$ python predictive_performance.py -d dataset`

For single model:

`$ python predictive_performance.py -m model`

For single model with single dataset:

`$ python predictive_performance.py -m model -d dataset`


Runtime Performance
-------------------

//TODO
