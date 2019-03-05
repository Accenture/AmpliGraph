# Performance


## Predictive Performance

We report the filtered MRR on the most common datasets used in literature.

|          | WN18 | FB15k | FB15k | WN18RR |
|----------|------|-------|-------|--------|
| TransE   | .??  | .??   | .??   | .??    |
| DistMult | .??  | .??   | .??   | .??    |
| ComplEx  | .??  | .??   | .??   | .??    |
| HolE     | .??  | .??   | .??   | .??    |
| RotatE   | .??  | .??   | .??   | .??    |

The above results have been obtained with the following hyperparameters values:

|          | Hyperparams |
|----------|------|
| TransE   | `...`  |
| DistMult | `...`  |
| ComplEx  | `...`  |
| HolE     | `...`  |
| RotatE   | `...`  |


Results in the table above can be reproduced by running the script below:
`$ ./predictive_performance.py -i dataset -m model`



## Runtime Performance

//TODO

