#!/bin/bash
cd ..
pip install .
cd experiments
python grid_search_exp.py --dataset fb15k --model ComplEx --hyperparams grid.json  --gpu 0