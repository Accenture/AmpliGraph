#!/bin/bash
<!-- fb15k_237 wn18rr -->
 parallel -j1 --verbose 'PYTHONUNBUFFERED=1 python grid_search_exp.py  --dataset {1} --model {2} --gpu 1 --clean_unseen True --hyperparams {3} > {1}_{2}_{3}.console.log' ::: fb15k_237 wn18rr ::: TransE ::: grid_TransE.json


<!-- fb15k -->
screen parallel -j1 --verbose 'PYTHONUNBUFFERED=1 python grid_search_exp.py  --dataset {1} --model {2} --gpu 1 --clean_unseen True --hyperparams {3} > {1}_{2}_{3}.console.log' ::: fb15k ::: DistMult ComplEx ::: grid_ComplEx_DistMult_adam.json

parallel -j1 --verbose 'PYTHONUNBUFFERED=1 python single_exp.py  --dataset {1} --model {2} --gpu 0  --hyperparams {3} --clean_unseen True > {1}_{2}_{3}.console.log' ::: fb15k_237 ::: ComplEx ::: input.json

parallel -j1 --verbose 'PYTHONUNBUFFERED=1 python single_exp.py  --dataset {1} --model {2} --gpu 0  --hyperparams {3} --clean_unseen True > {1}_{2}_{3}.console.log' ::: wn18rr ::: DistMult ::: input.json