# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import ampligraph.datasets
import ampligraph.latent_features
from ampligraph.evaluation import hits_at_n_score, mr_score, evaluate_performance, mrr_score

import argparse
import json
import sys
import yaml
import logging
import time

import numpy as np
from beautifultable import BeautifulTable
from tqdm import tqdm
import warnings

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

SUPPORT_DATASETS = ["fb15k", "fb15k-237", "wn18", "wn18rr", "yago310"]
SUPPORT_MODELS = ["complex", "transe", "distmult", "hole", "convkb", "conve"]


def display_scores(scores):
    output_readme = {}
    output_rst = {}

    for obj in scores:
        output_rst[obj["dataset"]] = BeautifulTable()
        output_rst[obj["dataset"]].set_style(BeautifulTable.STYLE_RST)
        output_rst[obj["dataset"]].column_headers = \
            ["Model", "MR", "MRR", "Hits@1", \
             "Hits@3", "Hits@10", "Time (s)", "ES epochs", "Hyperparameters"]

    for obj in scores:
        try:
            output_rst[obj["dataset"]] \
                .append_row([obj["model"],
                             "{0:.1f}".format(obj["mr"]),
                             "{0:.4f}".format(obj["mrr"]),
                             "{0:.3f}".format(obj["H@1"]),
                             "{0:.3f}".format(obj["H@3"]),
                             "{0:.3f}".format(obj["H@10"]),
                             "{0:.1f}".format(obj["time"]),
                             "{}".format(obj["early_stopping_epoch"]),
                             yaml.dump(obj["hyperparams"],
                                       default_flow_style=False)])
        except:
            output_rst[obj["dataset"]] \
                .append_row([obj["model"],
                             ".??",
                             ".??",
                             ".??",
                             ".??",
                             ".??",
                             ".??",
                             ".??",
                             ".??"])

    for key, value in output_rst.items():
        print(key)
        print(value)


def run_single_exp(config, dataset, model):
    start_time = time.time()

    hyperparams = config["hyperparams"][dataset][model]
    if hyperparams is None:
        print("dataset {0}...model {1} \
                      experiment is not conducted yet..." \
                     .format(dataset, config["model_name_map"][model]))
        return {
            "hyperparams": ".??"
        }
    print("dataset {0}...model {1}...\
                  hyperparameter:...{2}" \
                 .format(dataset,
                         config["model_name_map"][model],
                         hyperparams))

    es_code = "{0}_{1}".format(dataset, model)

    load_func = getattr(ampligraph.datasets,
                        config["load_function_map"][dataset])
    X = load_func()
    # logging.debug("Loaded...{0}...".format(dataset))

    # load model
    model_class = getattr(ampligraph.latent_features,
                          config["model_name_map"][model])
    model = model_class(**hyperparams)
    # Fit the model on training and validation set
    # The entire dataset will be used to filter out false positives statements
    # created by the corruption procedure:
    filter = np.concatenate((X['train'], X['valid'], X['test']))

    if es_code in config["no_early_stopping"]:
        logging.debug("Fit without early stopping...")
        model.fit(X["train"])
    else:
        logging.debug("Fit with early stopping...")
        model.fit(X["train"], True,
                  {
                      'x_valid': X['valid'][::2],
                      'criteria': 'mrr',
                      'x_filter': filter,
                      'stop_interval': 4,
                      'burn_in': 0,
                      'check_interval': 50
                  })

    if not hasattr(model, 'early_stopping_epoch') or model.early_stopping_epoch is None:
        early_stopping_epoch = np.nan
    else:
        early_stopping_epoch = model.early_stopping_epoch

    # Run the evaluation procedure on the test set. Will create filtered rankings.
    # To disable filtering: filter_triples=None
    ranks = evaluate_performance(X['test'],
                                 model,
                                 filter,
                                 verbose=False)

    # compute and print metrics:
    mr = mr_score(ranks)
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
        "hyperparams": hyperparams,
        "time": time.time() - start_time,
        "early_stopping_epoch": early_stopping_epoch
    }


def run_all(config):
    obj = []
    for dataset in tqdm(config["hyperparams"].keys(),
                        desc="evaluation done: "):
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
    for model in tqdm(config["hyperparams"][dataset].keys(),
                      desc="evaluation done: "):
        result = run_single_exp(config, dataset, model)
        obj.append({
            **result,
            "dataset": dataset,
            "model": config["model_name_map"][model]
        })
    return obj


def run_single_model(config, model):
    obj = []
    for dataset in tqdm(config["hyperparams"].keys(),
                        desc="evaluation done: "):
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        type=str.lower,
                        choices=SUPPORT_DATASETS)
    parser.add_argument("-m", "--model",
                        type=str.lower,
                        choices=SUPPORT_MODELS)
    parser.add_argument("--gpu",  type=str)

    args = parser.parse_args()

    if args.gpu is not None:
        # set GPU id to run
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logging.debug("Will use gpu number...{0}".format(args.gpu))

    logging.debug("Input dataset...{0}...input model...{1}..." \
                  .format(args.dataset, args.model))

    if args.dataset is None:
        if args.model is None:
            display_scores(run_all(config))
        else:
            if args.model.upper() not in config["model_name_map"]:
                sys.exit("Input model is not valid...")
            display_scores(run_single_model(config, args.model.upper()))
    else:
        if args.model is not None:
            if args.model.upper() not in config["model_name_map"] \
                    or args.dataset.upper() \
                            not in config["load_function_map"]:
                sys.exit("Input model or dataset is not valid...")

            result = run_single_exp(config,
                                    args.dataset.upper(),
                                    args.model.upper())
            display_scores([{
                **result,
                "dataset": args.dataset,
                "model": config["model_name_map"][args.model.upper()]
            }])
        else:
            if args.dataset.upper() not in config["load_function_map"]:
                sys.exit("Input dataset is not supported yet...")
            display_scores(run_single_dataset(config,
                                              args.dataset.upper()))


if __name__ == "__main__":
    main()
