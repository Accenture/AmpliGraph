import os
import sys
import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import pickle
import numpy as np
import pandas as pd
import datetime
from ampligraph.datasets import load_fb15k_237, load_wn18rr
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.latent_features.regularizers import get as get_regularizer
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.utils import save_model
from ampligraph.evaluation import hits_at_n_score, mr_score, mrr_score, train_test_split_no_unseen
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_hetionet():
    """
    Function to load Hetionet. Make sure you have downloaded the data from the
    official repository.
    """
    triples = pd.read_csv(
        "./data/hetionet/hetionet-v1.0-edges.sif.gz",
        sep="\t", names=['s', 'p', 'o'], header=0
    )

    train, test = train_test_split_no_unseen(triples.values, 10000, seed=0)
    train, valid = train_test_split_no_unseen(train, 5000, seed=0)

    dataset = {"train": train, "valid": valid, "test": test}

    return dataset

def extract_ontology(dataset_name, train):
    """
    Function to extract the ontology of a certain dataset from the training set.

    If the ``dataset_name == "wn18rr"`` or ``dataset_name == "fb15k_237"``, we define
    two classes for each relation in the dataset, and we make all the subjects
    of that relation belonging to one of these two classes. Obviously, it might be that
    some entities belong to more than one class.
    If instead ``dataset_name == "hetionet"``, we infer the ontology from the training
    triples by parsing the entity names.
    """

    if dataset_name == "wn18rr" or dataset_name == "fb15k_237":
        unique_relations = np.unique(train[:, 1]).tolist()
        ontology_classes = dict()
        ontology_domain_range = dict()
        for i, relation in enumerate(unique_relations):
            ontology_classes[str(i + 1)] = \
                np.unique(train[train[:, 1] == relation, 0]).tolist()
            ontology_classes[str(i + 1 + len(unique_relations))] = \
                np.unique(train[train[:, 1] == relation, 2]).tolist()
            ontology_domain_range[relation] = \
                (str(i + 1), str(i + 1 + len(unique_relations)))

    elif dataset_name == "hetionet":
        types = pd.read_csv(
            "./data/hetionet/hetionet-v1.0-nodes.tsv",
            sep="\t"
        )
        ontology_classes = {
            group: types.loc[types.kind == group, "id"].unique() \
            for group in types.kind.unique()
        }

        df_train = pd.DataFrame(train, columns=["s", "p", "o"])
        ontology_domain = df_train.groupby("p")['s'].apply(
            lambda x: list(set([el.split("::")[0] for el in x.tolist()]))[0]
        )
        ontology_range = df_train.groupby("p")['o'].apply(
            lambda x: list(set([el.split("::")[0] for el in x.tolist()]))[0]
        )
        ontology_domain_range = pd.merge(
            ontology_domain, ontology_range,
            left_index=True, right_index=True
        )
        ontology_domain_range = ontology_domain_range.apply(
            lambda row: (row['s'], row['o']), axis=1
        ).to_dict()

    return ontology_classes, ontology_domain_range

def load_data_and_ontology(dataset_name):
    """
    Function to load the datasets, split them in training, validation and test sets,
    and extract the ontology.
    """

    assert dataset_name in ["wn18rr", "fb15k_237", "hetionet"],\
        "Dataset not supported yet! Specify one of 'wn18rr', 'fb15k_237', 'hetionet'."

    if dataset_name == "wn18rr":
        dataset = load_wn18rr()
    elif dataset_name == "fb15k_237":
        dataset = load_fb15k_237()
    elif dataset_name == 'hetionet':
        dataset = load_hetionet()

    train, valid, test = dataset["train"], dataset["valid"], dataset["test"]

    ontology_classes, ontology_domain_range = extract_ontology(dataset_name, train)

    return train, valid, test, ontology_classes, ontology_domain_range

def setup_output_dir(args):
    """
    Create the folder where the output is going to be saved.
    """
    output_folder = args.output_folder
    dataset_name = args.dataset
    scoring_type = args.scoring_type

    fmt = '%Y-%m-%d-%H-%M-%S'
    date = datetime.datetime.now().strftime(fmt)
    output_folder = os.path.join(output_folder, dataset_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    experiment_name = "{}/{}-{}".format(output_folder, scoring_type, date)
    assert not os.path.exists(experiment_name), 'Experiment Folder already exist!'
    print(f"Output folder for the experiment: {experiment_name}")
    os.mkdir(experiment_name)

    name_config_file = os.path.join(experiment_name, 'config.json')
    with open(name_config_file, 'w') as f:
        json.dump(vars(args), f)

    return experiment_name



def run_model(args, hyperparams):

    dataset_name = args.dataset
    scoring_type = args.scoring_type
    regularizer = hyperparams.get("regularizer", args.regularizer)

    k = hyperparams["k"]
    eta = hyperparams["eta"]
    epochs = hyperparams["epochs"]
    batch_size = hyperparams["batch_size"]
    loss = hyperparams["loss"]
    loss_hyperparams = hyperparams.get("loss_hyperparams", {})
    reg_lambda = hyperparams["regularizer_lambda"]
    lr = hyperparams["optimizer_lr"]
    ontology_sampling = hyperparams["ontology_sampling"]

    experiment_name = setup_output_dir(args)
    stdout = os.path.join(experiment_name, "output.txt")

    if stdout != sys.stdout:
        stdout = open(stdout, 'w')
    print(f"Experiment Folder: {experiment_name}", file=stdout, flush=True)
    print(f"Experiment Configuration: {str(args)}", file=stdout, flush=True)
    print(f"Experiment Hyperparams: {hyperparams}", file=stdout, flush=True)

    train, valid, test, ontology_classes, ontology_domain_range = load_data_and_ontology(dataset_name)

    start_time = time.time()
    model = ScoringBasedEmbeddingModel(
        k=k, eta=eta, scoring_type=scoring_type
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = get_loss(loss, hyperparams=loss_hyperparams)
    regularizer = get_regularizer('LP', {'p': regularizer, 'lambda': reg_lambda})
    initializer = 'glorot_uniform'

    model.compile(
        optimizer=optimizer, loss=loss, entity_relation_regularizer=regularizer,
        entity_relation_initializer=initializer, run_eagerly=False
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_{}'.format('mrr'), min_delta=0, patience=10,
        verbose=1, mode='max', restore_best_weights=True
    )
    callbacks = [early_stop]

    if dataset_name == 'wn18rr':
        validation_burn_in = 500
    elif dataset_name == 'fb15k_237':
        validation_burn_in = 100
    else:
        validation_burn_in = 0


    history = model.fit(
        train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=valid,
        validation_freq=25,
        validation_batch_size=100,
        validation_burn_in=validation_burn_in,
        validation_corrupt_side='s',
        validation_filter={'train': train, 'valid': valid},
        callbacks=callbacks,
        verbose=False,
        ontology_sampling=ontology_sampling,
        ontology_classes=ontology_classes,
        ontology_domain_range=ontology_domain_range
    )
    print(f"Training completed! \nEarly stopping epoch: {early_stop.best_epoch + 1}", file=stdout, flush=True)

    name_history = os.path.join(experiment_name, "history.pkl")
    with open(name_history, 'wb') as history_file:
        pickle.dump(history.history, history_file)
    print(f"History of training saved at {name_history}", file=stdout, flush=True)

    if args.save_model:
        save_model(model, experiment_name)
        print("Trained model successfully saved!", file=stdout, flush=True)

    ranks = model.evaluate(
        test, batch_size=10,
        use_filter={'train': train, 'valid': valid, 'test': test}
    )

    # compute and print metrics:
    mr = mr_score(ranks)
    mrr = mrr_score(ranks)
    hits_1 = hits_at_n_score(ranks, n=1)
    hits_3 = hits_at_n_score(ranks, n=3)
    hits_10 = hits_at_n_score(ranks, n=10)
    hits_100 = hits_at_n_score(ranks, n=100)

    print(
        f'mr: {mr} \t mrr: {mrr} \t hits_1: {hits_1} \t hits_3: {hits_3} \t '
        f'hits_10: {hits_10} \t hits_100: {hits_100}', file=stdout, flush=True
    )

    results = {
        "mr": mr, "mrr": mrr, "H@1": hits_1, "H@3": hits_3,
        "H@10": hits_10, "hits@100": hits_100
    }
    name_results = os.path.join(experiment_name, "results.json")
    with open(name_results, "w") as f:
        f.write(json.dumps(results))
    print("Results saved in the file: {}".format(name_results), file=stdout, flush=True)

    print(
        f"Experiment complete. Total duration: {time.time() - start_time}",
        file=stdout, flush=True
    )
    stdout.close()

    print(f"Experiment successfully completed. Find output at {experiment_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pipeline to control the experiments for the evaluation of ontology-based negative sampling.'
    )

    parser.add_argument('--dataset', default='fb15k_237', type=str, help='Dataset to process')
    parser.add_argument('--output-folder', default="./",
                        type=str, help='Base folder for the output of the experiment.')
    parser.add_argument('--save-model', default=None, action="store_true",
                        help="Whether to save or not the model trained.")

    parser.add_argument('--scoring-type', '-score', default="ComplEx", type=str,
                        help="KGE Scoring function to use for the experiment.")
    parser.add_argument('--regularizer', default=3, type=int, help="Norm to use to regularize the loss.")

    args = parser.parse_args()
    print(f"experiment arguments: {args}")


    with open(f"config_{args.dataset}.json") as config:
        config_file = json.loads(config.read())

    model_config = config_file[args.scoring_type]
    for experiment_type in model_config:
        print(f"Running experiment '{experiment_type}' on {args.dataset} using {args.scoring_type}.")
        hyperparams = model_config[experiment_type]
        run_model(args, hyperparams)
