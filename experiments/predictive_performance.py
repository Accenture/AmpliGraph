import ampligraph.datasets
import ampligraph.latent_features
from ampligraph.datasets import load_wn18, load_fb15k, load_fb15k_237
from ampligraph.latent_features import TransE, DistMult, ComplEx
from ampligraph.evaluation import select_best_model_ranking, hits_at_n_score, mar_score, evaluate_performance, mrr_score

import argparse, os, json, sys 
import numpy as np

from os import path
from beautifultable import BeautifulTable
from tqdm import tqdm

def display_scores(scores):
    output_readme = {}
    output_rst = {}

    for obj in scores:
        output_readme[obj["dataset"]] = []
        output_rst[obj["dataset"]] = BeautifulTable()
        output_rst[obj["dataset"]].column_headers = ["Model", "MR", "MRR", "H@1", "H@3", "H@10", "Hyperparameters"]
    
    for obj in scores:
        try:
            output_readme[obj["dataset"]].append("|{0}|{1:.2f}|{2:.2f}|{3:.2f}|{4:.2f}|{5:.2f}|{6}|".format(
                obj["model"],
                obj["mr"],
                obj["mrr"],
                obj["H@1"],
                obj["H@3"],
                obj["H@10"],
                obj["hyperparams"]))
            output_rst[obj["dataset"]].append_row([obj["model"],  
                                                   "{0:.2f}".format(obj["mr"]),  
                                                   "{0:.2f}".format(obj["mrr"]),  
                                                   "{0:.2f}".format(obj["H@1"]),  
                                                   "{0:.2f}".format(obj["H@3"]),  
                                                   "{0:.2f}".format(obj["H@10"]),
                                                   obj["hyperparams"]])
        except:
                output_readme[obj["dataset"]].append("|{0}|.??|.??|.??|.??|.??|.??|".format(obj["model"]))
                output_rst[obj["dataset"]].append_row([obj["model"],  
                                                   ".??",  
                                                   ".??",  
                                                   ".??",  
                                                   ".??",  
                                                   ".??",
                                                   ".??"])  
                
    # for key, value in output_readme.items():
    #     print("####",key)
    #     print("|Model|MR|MRR|H@1|H@3|H@10|Hyperparameters|\n|-----|--|---|---|---|----|----|")
    #     for elm in value:
    #         print(elm)
    
    for key, value in output_rst.items():
        print(key)
        print(value)

# clean datasets with unseen entities
def clean_data(train, valid, test, keep_valid = False):
    train_ent = set(train.flatten())
    valid_ent = set(valid.flatten())
    test_ent = set(test.flatten())
    
    if not keep_valid:
        # filter valid
        valid_diff_train_ent = valid_ent - train_ent
        idxs_valid = []
        if len(valid_diff_train_ent) > 0:
            count_valid = 0
            c_if = 0
            for row in valid:
                tmp = set(row)
                if len(tmp & valid_diff_train_ent) != 0:
                    idxs_valid.append(count_valid)
                    c_if+=1
                count_valid = count_valid + 1
        filtered_valid = np.delete(valid, idxs_valid, axis=0)
        # logging.debug("shape valid: {0} shape - filtered valid: {1}: {2}".format(valid.shape, filtered_valid.shape, c_if))
        valid = filtered_valid
       
    # filter test 
    train_valid_ent = set(train.flatten()) | set(valid.flatten())
    ent_test_diff_train_valid = test_ent - train_valid_ent
    
    idxs_test = []

    if len(ent_test_diff_train_valid) > 0:
        count_test = 0
        c_if=0
        for row in test:
            tmp = set(row)
            if len(tmp & ent_test_diff_train_valid) != 0:
                idxs_test.append(count_test)
                c_if+=1
            count_test = count_test + 1
    filtered_test = np.delete(test, idxs_test, axis=0)
    # print("shape test: {0}  -  filtered test: {1}: {2}".format(test.shape, filtered_test.shape, c_if))

    return valid, filtered_test

def run_single_exp(config, dataset, model):
    hyperparams = config["hyperparams"][dataset][model]
    if hyperparams is None:
        print("dataset {0}...model {1} experiment is not conducted yet...".format(dataset, config["model_name_map"][model]))
        return {
            "hyperparams": ".??"
        }
    print("dataset {0}...model {1}...best hyperparameter:...{2}".format(dataset, config["model_name_map"][model], hyperparams))
    es_code = "{0}_{1}".format(dataset, model)

    load_func = getattr(ampligraph.datasets, config["load_function_map"][dataset])
    X = load_func()
    # print("Loaded...{0}...".format(dataset))

    if dataset in config["UNSEEN_DATASETS"]:
        print("{0} contains unseen entities in test dataset, we cleaned them...".format(dataset))
        X["valid"], X["test"] = clean_data(X["train"], X["valid"], X["test"], keep_valid=True)

    # load model
    model_class = getattr(ampligraph.latent_features, config["model_name_map"][model])
    model = model_class(**hyperparams)
    # Fit the model on training and validation set
    # The entire dataset will be used to filter out false positives statements
    # created by the corruption procedure:
    filter = np.concatenate((X['train'], X['valid'], X['test']))

    if es_code in config["no_early_stopping"]:
        # print("Fit without early stopping...")
        model.fit(np.concatenate((X['train'], X['valid'])))
    else:
        # print("Fit with early stopping...")
        model.fit(np.concatenate((X['train'], X['valid'])), True, 
        {
            'x_valid':X['test'][:1000], 
            'criteria':'mrr', 
            'x_filter':filter,
            'stop_interval': 2, 
            'burn_in':0, 
            'check_interval':100
        })

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

    return {
        "mr": mr,
        "mrr": mrr,
        "H@1": hits_1,
        "H@3": hits_3,
        "H@10": hits_10,
        "hyperparams": hyperparams
    }    

def run_all(config):
    obj = []
    for dataset in tqdm(config["hyperparams"].keys(), desc = "evaluation done: "):
        for model in config["hyperparams"][dataset].keys():
            result = run_single_exp(config, dataset, model)
            obj.append({
                **result,
                "dataset": dataset,
                "model": config["model_name_map"][model]
            })
    return obj

def run_single_dataset(config, dataset):
    obj = []
    for model in tqdm(config["hyperparams"][dataset].keys(), desc = "evaluation done: "):
        result = run_single_exp(config, dataset, model)
        obj.append({
            **result,
            "dataset": dataset,
            "model": config["model_name_map"][model]
        })
    return obj

def run_single_model(config, model):
    obj = []
    for dataset in tqdm(config["hyperparams"].keys(), desc = "evaluation done: "):
        result = run_single_exp(config, dataset, model)
        obj.append({
            **result,
            "dataset": dataset,
            "model": config["model_name_map"][model]
        })
    return obj

def main():
    with open("config.json", "r") as fi:
        config = json.load(fi)

    # set GPU id to run
    os.environ["CUDA_VISIBLE_DEVICES"]=config["CUDA_VISIBLE_DEVICES"]
    # print("Will use gpu number...",config["CUDA_VISIBLE_DEVICES"])

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-m", "--model", type=str)

    args = parser.parse_args()
    # print("Input dataset...{0}...input model...{1}...".format(args.dataset, args.model))

    if args.dataset is None:
        if args.model is None:
            display_scores(run_all(config))
        else:
            if args.model.upper() not in config["model_name_map"]:
                sys.exit("Input model is not valid...")            
            display_scores(run_single_model(config, args.model.upper()))
    else:
        if args.model is not None:
            if args.model.upper() not in config["model_name_map"] or args.dataset.upper() not in config["load_function_map"]:
                sys.exit("Input model or dataset is not valid...")        
                    
            result = run_single_exp(config, args.dataset.upper(), args.model.upper())
            display_scores([{
                **result, 
                "dataset": args.dataset,
                "model": config["model_name_map"][args.model.upper()]
            }])
        else:
            if args.dataset.upper() not in config["load_function_map"]:
                sys.exit("Input dataset is not supported yet...")                        
            display_scores(run_single_dataset(config, args.dataset.upper()))

if __name__ == "__main__":
    main()
