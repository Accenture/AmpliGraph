#!/bin/bash
# 1923
parallel -j1 --verbose 'python exp.py --hyperparams hp_1923.json --dataset {1} --model {2} --gpu 1 &> {1}_{2}.console.log' ::: fb15k ::: TransE DistMult ComplEx

# 1925
parallel -j1 --verbose 'python exp.py --hyperparams hp_1925.json --dataset {1} --model {2} --gpu 1 &> {1}_{2}.console.log' ::: wn18 ::: TransE DistMult ComplEx
