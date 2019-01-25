#!/bin/bash
# 1923
parallel -j1 --verbose 'python no_grid_exp.py --hyperparams no_grid_1923.json --dataset {1} --model {2} --gpu 1 &> {1}_{2}.console.log' ::: fb15k ::: ComplEx TransE DistMult

# 1925
parallel -j1 --verbose 'python no_grid_exp.py --hyperparams no_grid_1923.json --dataset {1} --model {2} --gpu 1 &> {1}_{2}.console.log' ::: wn18 ::: TransE DistMult ComplEx

# New Hyperparameters
#Only for TransE
parallel -j1 --verbose 'python exp_new_hp.py --dataset fb15k --model TransE --gpu 1 --hyperparams {1} &> fb15k_TransE_{1}.console.log' ::: 1923_self_adverserial.json 1923_regularizer.json 

#All
parallel -j1 --verbose 'python exp_new_hp.py --dataset {1} --model {2} --gpu 1  --hyperparams {3} &> {1}_{2}_{3}.console.log' ::: fb15k ::: TransE DistMult ComplEx  ::: 1923_regularizer.json 