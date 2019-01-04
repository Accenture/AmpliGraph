from ampligraph.datasets import load_wn18, load_fb15k, load_fb15k_237

from ampligraph.latent_features import TransE, DistMult, ComplEx

from ampligraph.evaluation import select_best_model_ranking, hits_at_n_score, mar_score

import ampligraph.datasets
import ampligraph.latent_features

import argparse, os, json 

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

    load_func = getattr(ampligraph.datasets, f_map[args.dataset])

    # load Wordnet18 dataset:
    X_dict = load_func()
    print("loaded...{0}".format(args.dataset))

    model_class = getattr(ampligraph.latent_features, args.model)

    # Put here the desired hyperparameter values that will be evaluated in the grid search:
    
    with open(args.hyperparams, "r") as fi:
        param_grid = json.load(fi)
        print("input param: ", param_grid)

    # Train the model on all possibile combinations of hyperparameters.
    # Models are validated on the validation set.
    # It returnes a model re-trained on training and validation sets.
    print("start executing to find the best...")
    best_model, best_params, best_mrr_train, \
    ranks_test, mrr_test = select_best_model_ranking(model_class, X_dict,
                                                      param_grid,
                                                      filter_retrain=True,
                                                      eval_splits=100,
                                                      verbose=True)

    mr_test = mar_score(ranks_test)
    hits_1 = hits_at_n_score(ranks_test, n=1)
    hits_3 = hits_at_n_score(ranks_test, n=3)
    hits_10 = hits_at_n_score(ranks_test, n=10)

    with open("result_{0}_{1}.txt".format(args.dataset, args.model), "w") as fo:
        fo.write("type(best_model).__name__: {0}\n".format(type(best_model).__name__))
        fo.write("best_params: {0}\n".format(best_params))
        fo.write("mr(test): {0} mrr(test): {1} hits 1: {2} hits 3: {3} hits 10: {4}".format(mr_test, mrr_test, hits_1, hits_3, hits_10))

if __name__ == "__main__":
    main()
