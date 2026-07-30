"""
One-time utility to cache the AmpliGraph datasets locally so the main pipeline
never needs to import ampligraph (and therefore TensorFlow) at runtime.

Run this once in an environment that has ampligraph installed:
    python cache_datasets.py

It writes tab-separated `train.txt`, `valid.txt`, `test.txt` (columns: s, p, o)
under <dataset>/, which is exactly what main.py reads via load_dataset().
"""

import os

# ampligraph pulls in TensorFlow, which grabs the whole GPU on init by default.
# The loaders are pure CPU (they just download/read triples), so hide the GPU.
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import numpy as np
from ampligraph.datasets import load_fb15k_237, load_wn18rr, load_yago3_10

DATASET_LOADERS = {
    "fb15k-237": load_fb15k_237,
    "wn18rr": load_wn18rr,
    "yago3-10": load_yago3_10,
}


def main():
    for dataset, loader in DATASET_LOADERS.items():
        print(f"Caching {dataset} ...")
        splits = loader()
        out_dir = dataset
        os.makedirs(out_dir, exist_ok=True)

        for split in ("train", "valid", "test"):
            path = os.path.join(out_dir, f"{split}.txt")
            np.savetxt(path, splits[split], fmt="%s", delimiter="\t")
            print(f"  {split}: {len(splits[split])} triples -> {path}")


if __name__ == "__main__":
    main()
