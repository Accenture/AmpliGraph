import sys
import time

from ampligraph.evaluation.protocol import select_best_model_ranking
from ampligraph.latent_features import ComplEx, DistMult, TransE
from ampligraph.datasets import load_wn11, load_fb13, load_yago39k

model_arg = sys.argv[1]
dataset_arg = sys.argv[2]
max_combinations = int(sys.argv[3])
sleep = int(sys.argv[4]) if len(sys.argv) >= 5 else 0

print("sleeping {} seconds".format(sleep))
time.sleep(sleep)
print("starting {} on {}".format(model_arg, dataset_arg))

model_map = {
    "transe": TransE,
    "distmult": DistMult,
    "complex": ComplEx
}

dataset_map = {
    "fb13": load_fb13,
    "wn11": load_wn11,
    "yago39k": load_yago39k
}

X = dataset_map[dataset_arg]()
model = model_map[model_arg]

param_grid = {
    "batches_count": [50, 100, 150],
    "seed": 0,
    "epochs": 2000,
    "k": [50, 100, 150, 200, 250],
    "eta": [5, 15, 25, 50, 75],
    "loss":  ["pairwise", "nll", "multiclass_nll", "self_adversarial"],
    "loss_params": {
        "margin": [0, 1, 3, 10],
        "alpha": [0.0, 0.5, 1.0]
    },
    "embedding_model_params": {
    },
    "regularizer": [None, "LP"],
    "regularizer_params": {
        "lambda": [1e-5, 1e-4, 1e-3, 1e-2],
        "p": [2, 3]
    },
    "optimizer": ["adam"],
    "optimizer_params": {
        "lr": [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    },
    "verbose": False
}

best_model1, best_params1, best_mrr_train1, _, mrr_test1, results = select_best_model_ranking(
    model,
    X['train'],
    X['valid'][X['valid_labels']][::2],
    X['test'][X['test_labels']],
    param_grid,
    max_combinations=max_combinations,
    verbose=True,
    early_stopping=True,
    early_stopping_params={
      'x_valid': X['valid'][X['valid_labels']][1::2],
      'criteria': 'mrr',
      'stop_interval': 4,
      'burn_in': 100,
      'check_interval': 25
    }
)


print(best_model1)
print(best_params1)
print(best_mrr_train1)
print(mrr_test1)

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                      np.int16, np.int32, np.int64, np.uint8,
                      np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


with open('results_{}_{}.json'.format(model_arg, dataset_arg), 'w') as outfile:
    json.dump(results, outfile, ensure_ascii=False, cls=NumpyEncoder, indent=2, default=str)
