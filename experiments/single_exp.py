from ampligraph.datasets import load_wn18, load_fb15k, load_fb15k_237

from ampligraph.latent_features import TransE, DistMult, ComplEx

from ampligraph.evaluation import select_best_model_ranking, hits_at_n_score, mar_score, evaluate_performance, mrr_score

import ampligraph.datasets
import ampligraph.latent_features

import argparse, os, json 

import numpy as np

from utils import clean_data

f_map = {
    "wn18": "load_wn18",
    "fb15k": "load_fb15k",
    "fb15k_237": "load_fb15k_237",
    "wn18rr": "load_wn18rr"
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--hyperparams", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--clean_unseen", type=bool)
    args = parser.parse_args()
    
    print("Will use gpu number: ", args.gpu, "...")

    from os import path
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    # load dataset
    load_func = getattr(ampligraph.datasets, f_map[args.dataset])

    X = load_func()

    if args.clean_unseen:
        X["valid"], X["test"] = clean_data(X["train"], X["valid"], X["test"], keep_valid=True)

    print("loaded...{0}".format(args.dataset))
    
    # load model
    model_class = getattr(ampligraph.latent_features, args.model)
    # Init a ComplEx neural embedding model with pairwise loss function:
    # The model will be trained on 30 epochs.
    # Turn stdout messages off with verbose=False
    with open(args.hyperparams, "r") as fi:
        hyperparams = json.load(fi)

    print("input hyperparameters: ", hyperparams)

    model = model_class(**hyperparams)
    # Fit the model on training and validation set


    # The entire dataset will be used to filter out false positives statements
    # created by the corruption procedure:
    filter = np.concatenate((X['train'], X['valid'], X['test']))
    
    print("Start fitting...with early stopping")
    
    model.fit(np.concatenate((X['train'], X['valid'])), True, 
    {
        'x_valid':X['test'][:1000], 
        'criteria':'mrr', 'x_filter':filter,
        'stop_interval': 2, 
        'burn_in':0, 
        'check_interval':100
    })
    # model.fit(np.concatenate((X['train'], X['valid'])))

    # Run the evaluation procedure on the test set. Will create filtered rankings.
    # To disable filtering: filter_triples=None
    ranks = evaluate_performance(X['test'], model=model, filter_triples=filter,
                                verbose=True)

    # compute and print metrics:
    mr = mar_score(ranks)
    mrr = mrr_score(ranks)
    hits_1 = hits_at_n_score(ranks, n=1)
    hits_3 = hits_at_n_score(ranks, n=3)
    hits_10 = hits_at_n_score(ranks, n=10)

    with open("result_{0}_{1}.txt".format(args.dataset, args.model), "w") as fo:
        fo.write("mr(test): {0} mrr(test): {1} hits 1: {2} hits 3: {3} hits 10: {4}".format(mr, mrr, hits_1, hits_3, hits_10))

if __name__ == "__main__":
    main()
