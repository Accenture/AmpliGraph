from ampligraph.datasets import load_wn18, load_fb15k, load_fb15k_237, load_wn18rr

from ampligraph.latent_features import TransE, DistMult, ComplEx

from ampligraph.evaluation import select_best_model_ranking, hits_at_n_score, mar_score, evaluate_performance, mrr_score

import ampligraph.datasets
import ampligraph.latent_features

import argparse, os, json 

import numpy as np
import pandas as pd

from utils import clean_data




def main():
    X = load_fb15k_237()
    train = X["train"]
    valid = X["valid"]
    test = X["test"]
    
    train_ent = set(train.flatten())
    valid_ent = set(valid.flatten())
    test_ent = set(test.flatten())
       
    # filter test 
    train_valid_ent = set(train.flatten()) | set(valid.flatten())
    ent_test_diff_train_valid = test_ent - train_valid_ent
    
    idxs_test = []

    unseen_triples = []

    if len(ent_test_diff_train_valid) > 0:
        count_test = 0
        c_if=0
        for row in test:
            tmp = set(row)
            if len(tmp & ent_test_diff_train_valid) != 0:
                unseen_triples.append(row)
    unseen_triples = np.asarray(unseen_triples)
    df = pd.DataFrame({"s": unseen_triples[:,0], "p": unseen_triples[:,1], "o": unseen_triples[:,2]}, dtype=str)
    df.to_csv("unseen_fb15k_237.csv", header=False, sep="\t", index=False)
if __name__ == "__main__":
    main()
