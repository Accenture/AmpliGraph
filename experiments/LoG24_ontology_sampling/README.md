In this folder you can find the code to set up the experiments to reproduce the results reported on the extended abstract 
*Domain and Range Aware Synthetic Negatives Generation for Knowledge Graph Embedding Models* submitted to the Learning
on Graph Conference 2024.

In order to reproduce the results, run `experiments.py` specifying the dataset and the scoring function of interest,
the code will take care of loading the specific configuration file.
An example of usage is the following:

```shell
python experiments.py --dataset fb15k_237 --scoring-type ComplEx
```

The scoring functions used in the paper are `TransE`, `DistMult`, `RotatE`, `ComplEx`, while the datasets are
`fb15k_237`, `wn18rr`, `hetionet`.


 