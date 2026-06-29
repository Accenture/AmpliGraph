"""
This script is meant to postprocess the ranks of some experiments run with the first version of
the early stopping, that are now converted to an updated version (v2).

More in detail, in v1 the early stopper would look at all the corruption triples at test time and
split them in bins (high-high, high-mid2...) depending on the degree of the target entity. For each
corrupted triple, we would save the scores at each early stopping epoch for the bin it fell in.
This strategy failed, as the early stopping scores were not consistent across different runs.

Early stopping v2, instead, estimates the bin in which the target triple falls and then stops the
latent search according to that. This is more subject to the quality of the estimate, which is not
always perfect.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import pickle
from experiments.LatentSearch.code.utils_KGE_latent_search import *
from experiments.LatentSearch.code.main_kge_latent import EarlyStopper


def convert_ranks(ranks, train_df, valid_df, test_df, early_stop, group='sp'):
    """
    Convert ranks from early stopping v1 to early stopping v2.
    """
    degrees = pd.concat([train_df.s, train_df.o]).value_counts()

    train_grouped, valid_grouped, test_grouped = group_by_pair(
        train_df, valid_df, test_df, group=group
    )

    for pair in tqdm(test_grouped.groups, total=len(test_grouped)):

        try:
            train_targets = train_grouped.get_group(pair)["o" if group == 'sp' else 's'].values
        except KeyError:
            train_targets = []
        try:
            valid_targets = valid_grouped.get_group(pair)["o" if group == 'sp' else 's'].values
        except KeyError:
            valid_targets = []

        observed_targets = np.concatenate([train_targets, valid_targets])

        if len(observed_targets) == 0:
            continue

        anchor_degree = degrees[pair[0]]
        estimated_target_degree = np.quantile(
            degrees[observed_targets.tolist()],
            0.2
        )
        estimated_type = map_degree_to_bin(
            anchor_degree if group == 'sp' else estimated_target_degree,
            anchor_degree if group == 'po' else estimated_target_degree,
            thresholds=early_stop.thresholds
        )
        early_stop_epoch = early_stop.epochs[group][estimated_type] - 1

        if early_stop_epoch == -1:
            ranks[pair]["early_stop_new_ranks"] = ranks[pair]["pre_ranks"]
        else:
            ranks[pair]["early_stop_new_ranks"] = ranks[pair]["new_ranks"][early_stop_epoch]

    return ranks


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Post processing script to change early stopping ranks.')

    parser.add_argument('--output-folder', type=str,
                        help='Base folder for the output of the experiment.')

    args = parser.parse_args()

    path = args.output_folder
    print(path)
    if 'fb15k-237' in path:
        dataset = 'fb15k-237'
        X = load_fb15k_237()
        # thresholds = [10, 30, 100]
    elif 'wn18rr' in path:
        dataset = 'wn18rr'
        X = load_wn18rr()
        # thresholds = [5, 10, 20]
    elif 'yago310' in path:
        dataset = 'yago310'
        X = load_yago3_10()

    train, valid, test = X["train"], X["valid"], X["test"]

    train_df = pd.DataFrame(train, columns=['s', 'p', 'o'])
    valid_df = pd.DataFrame(valid, columns=['s', 'p', 'o'])
    test_df = pd.DataFrame(test, columns=['s', 'p', 'o'])

    degrees = pd.concat([train_df.s, train_df.o]).value_counts()

    # We set thresholds with a hard constraint on the low degree in case the .25 quantile is below 5.
    # Indeed, we don't want to have a low degree threshold that is too low.
    thresholds = [np.max([5, np.quantile(degrees, 0.25)])]
    if thresholds[0] == 5:
        thresholds += np.quantile(degrees[degrees > 5], [0.33, 0.66]).tolist()
    else:
        thresholds = np.quantile(degrees, [0.25, 0.5, 0.75]).tolist()

    # Load the validation ranks and define the early stopper
    with open(os.path.join(path, "valid_ranks_po.pkl"), "rb") as f:
        valid_ranks_po = pickle.load(f)

    with open(os.path.join(path, "valid_ranks_sp.pkl"), "rb") as f:
        valid_ranks_sp = pickle.load(f)

    early_stopper = EarlyStopper(
        valid_ranks_sp, valid_ranks_po, valid_df, thresholds=thresholds,
        degrees=pd.concat([train_df.s, train_df.o]).value_counts()
    )

    with open(os.path.join(path, "ranks_po.pkl"), "rb") as f:
        ranks_po = pickle.load(f)

    with open(os.path.join(path, "ranks_sp.pkl"), "rb") as f:
        ranks_sp = pickle.load(f)

    new_ranks_sp = convert_ranks(ranks_sp, train_df, valid_df, test_df, early_stopper, group='sp')
    with open(os.path.join(path, "ranks_sp.pkl"), "wb") as f:
        pickle.dump(new_ranks_sp, f)

    new_ranks_po = convert_ranks(ranks_po, train_df, valid_df, test_df, early_stopper, group='po')
    with open(os.path.join(path, "ranks_po.pkl"), "wb") as f:
        pickle.dump(new_ranks_po, f)
