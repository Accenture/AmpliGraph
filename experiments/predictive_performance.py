"""
Train KGE model on benchmark datasets:
Usage:
  predictive_performance.py <fb15k,fb15k-237,wn18,wn18rr,yago310>  <complex,transe,distmult,hole,convkb,conve> <config> <gpu> --save <root>
  predictive_performance.py -h | --help
  predictive_performance.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
"""
# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from docopt import docopt
import ampligraph.datasets
from ampligraph.evaluation import hits_at_n_score, mr_score, mrr_score
from ampligraph.compat import evaluate_performance
from schema import Schema, And, Use
import argparse
import json
import sys
import yaml
import logging
import time
import datetime
import numpy as np
from beautifultable import BeautifulTable
from tqdm import tqdm
import warnings

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

SUPPORT_DATASETS = ["fb15k", "fb15k-237", "wn18", "wn18rr", "yago310"]
SUPPORT_MODELS = ["complex", "transe", "distmult", "hole", "convkb", "conve"]


def display_scores(scores, root=None):
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

    if root is not None:
        fmt='%Y-%m-%d-%H-%M-%S'
        date = datetime.datetime.now().strftime(fmt)
        name = os.path.join(root, "results_{}.json".format(date))
        out = {str(x):str(y) for x, y in output_rst.items()}
        with open(os.path.join("output.rst"), "w") as f:
            f.write(str(output_rst))
        with open(name, "w") as f:
            f.write(json.dumps(out))
        print("Experiments' results saved in {}.".format(name))


    for key, value in output_rst.items():
        print(key)
        print(value)


def run_single_exp(config, dataset, model, root=None):
    print("Run single experiment for {} and {}".format(dataset, model))
    start_time = time.time()
    print("Started: ",start_time)

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
    model_name = model
    # load model
    model_class = getattr(ampligraph.compat,
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

    if root is not None: 
        fmt='%Y-%m-%d-%H-%M-%S'
        date = datetime.datetime.now().strftime(fmt)
        name = "{}/{}-{}-{}".format(root, model_name, dataset, date)
        save_model(model, name)
        print("Model saved in {}.".format(name))


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

    result = {
        "mr": mr,
        "mrr": mrr,
        "H@1": hits_1,
        "H@3": hits_3,
        "H@10": hits_10,
        "hyperparams": hyperparams,
        "time": time.time() - start_time,
        "early_stopping_epoch": early_stopping_epoch
    }
    if root is not None:
        name = "{}/result_{}_{}_{}.json".format(root, model_name, dataset, date)
        with open(name, "w") as f:
            f.write(json.dumps(result))
        print("Results saved in the file: {}".format(name))

    return result 


def run_all(config, root=None):
    obj = []
    for dataset in tqdm(config["hyperparams"].keys(),
                        desc="evaluation done: "):
        for model in config["hyperparams"][dataset].keys():
            result = run_single_exp(config, dataset, model, root=root)
            obj.append({
                **result,
                "dataset": dataset,
                "model": config["model_name_map"][model]
            })
    return obj


def run_single_dataset(config, dataset, root=None):
    obj = []
    for model in tqdm(config["hyperparams"][dataset].keys(),
                      desc="evaluation done: "):
        result = run_single_exp(config, dataset, model, root=root)
        obj.append({
            **result,
            "dataset": dataset,
            "model": config["model_name_map"][model]
        })
    return obj


def run_single_model(config, model, root=None):
    obj = []
    for dataset in tqdm(config["hyperparams"].keys(),
                        desc="evaluation done: "):
        result = run_single_exp(config, dataset, model, root=root)
        obj.append({
            **result,
            "dataset": dataset,
            "model": config["model_name_map"][model]
        })
    return obj


def train_model(dataset, model, config, gpu, save, root):
    if gpu is not None:
        # set GPU id to run
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        logging.debug("Will use gpu number...{0}".format(gpu))

    logging.debug("Input dataset...{0}...input model...{1}..." \
                  .format(dataset, model))

    if dataset is None:
        if model is None:
            display_scores(run_all(config, save=save, root=root), root=root)
        else:
            if model.upper() not in config["model_name_map"]:
                sys.exit("Input model is not valid...")
            display_scores(run_single_model(config, model.upper(), root=root), root=root)
    else:
        if model is not None:
            if model.upper() not in config["model_name_map"] \
                    or dataset.upper() \
                            not in config["load_function_map"]:
                sys.exit("Input model or dataset is not valid...")

            result = run_single_exp(config,
                                    dataset.upper(),
                                    model.upper(),
                                    root=root)
            display_scores([{
                **result,
                "dataset": dataset,
                "model": config["model_name_map"][model.upper()]
            }], root=root)
        else:
            if dataset.upper() not in config["load_function_map"]:
                sys.exit("Input dataset is not supported yet...")
            display_scores(run_single_dataset(config,
                                              dataset.upper(), root=root),
                                              root=root)


if __name__ == "__main__":
    arguments = docopt(__doc__, version='Train KGE model on benchamark data') 
    schema = Schema({'<fb15k,fb15k-237,wn18,wn18rr,yago310>': And(Use(lambda s: s.split(',')),
                                                              lambda l: all([1 if elem.lower() in SUPPORT_DATASETS else 0 for elem in l])),
                     '<complex,transe,distmult,hole,convkb,conve>': And(Use(lambda s: s.split(',')),
                                                              lambda l: all([1 if elem.lower() in SUPPORT_MODELS else 0 for elem in l])),
                         object: object})  # don't validate other keys
    
    schema.validate(arguments)


    dataset = arguments['<fb15k,fb15k-237,wn18,wn18rr,yago310>'] 
    config = arguments['<config>'] 
    model = arguments['<complex,transe,distmult,hole,convkb,conve>']
    gpu = arguments['<gpu>'] 
    save = arguments['--save'] 
    root = arguments['<root>']
    if not os.path.exists(root):
        os.mkdir(root)
    with open(config) as f:
        conf = json.loads(f.read())
    train_model(dataset, model, conf, gpu, save, root)
