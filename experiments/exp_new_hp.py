from ampligraph.datasets import load_wn18, load_fb15k, load_fb15k_237

from ampligraph.latent_features import TransE, DistMult, ComplEx

from ampligraph.evaluation import select_best_model_ranking, hits_at_n_score, mar_score, evaluate_performance, mrr_score

import ampligraph.datasets
import ampligraph.latent_features

import argparse, os, json 

import numpy as np

f_map = {
    "wn18": "load_wn18",
    "fb15k": "load_fb15k",
    "fb15k_237": "load_fb15k_237"
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--hyperparams", type=str)
    parser.add_argument("--gpu", type=int)

    args = parser.parse_args()
    
    print("Will use gpu number: ", args.gpu, "...")

    from os import path
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    # load dataset
    load_func = getattr(ampligraph.datasets, f_map[args.dataset])
    X = load_func()
    print("loaded...{0}".format(args.dataset))
    
    # load model
    model_class = getattr(ampligraph.latent_features, args.model)
    # Init a ComplEx neural embedding model with pairwise loss function:
    # The model will be trained on 30 epochs.
    # Turn stdout messages off with verbose=False
    with open(args.hyperparams, "r") as fi:
        hyperparams = json.load(fi)

    print("input hyperparameters: ", hyperparams)

    model = model_class(**hyperparams, verbose=True)
    # Fit the model on training and validation set
    # print("k: ", model.k, "loss: ", model.loss, "batch_count: ", model.batches_count, "seed: ", model.seed, "epochs: ", model.epochs)
    print("Start fitting...")
    model.fit(np.concatenate((X['train'], X['valid'])))

    # The entire dataset will be used to filter out false positives statements
    # created by the corruption procedure:
    filter = np.concatenate((X['train'], X['valid'], X['test']))

    # Run the evaluation procedure on the test set. Will create filtered rankings.
    # To disable filtering: filter_triples=None
    ranks = evaluate_performance(X['test'], model=model, filter_triples=filter,
                                verbose=True, splits=50)

    # compute and print metrics:
    mr = mar_score(ranks)
    mrr = mrr_score(ranks)
    hits_1 = hits_at_n_score(ranks, n=1)
    hits_3 = hits_at_n_score(ranks, n=3)
    hits_10 = hits_at_n_score(ranks, n=10)

    with open("result_{0}_{1}_{2}.txt".format(args.dataset, args.model, args.hyperparams), "w") as fo:
        fo.write("mr(test): {0} mrr(test): {1} hits 1: {2} hits 3: {3} hits 10: {4}".format(mr, mrr, hits_1, hits_3, hits_10))

if __name__ == "__main__":
    main()