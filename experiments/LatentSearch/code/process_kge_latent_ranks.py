import argparse

import pickle

from experiments.LatentSearch.code.utils_KGE_latent_search import *
from ampligraph.latent_features.models.ImbalancE import EarlyStopper


def load_pretrained_model(dataset, scoring_type):
    """Load the pretrained KGE model"""
    from ampligraph.utils import restore_model

    return restore_model(f"/home/alberto/ampligraph_models/{dataset.upper()}_{scoring_type.upper()}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Pipeline to generate plots for the KGE inference latent search.')

    parser.add_argument('--output-folder', type=str,
                        help='Base folder for the output of the experiment.')
    parser.add_argument('--valid', action='store_true', help="Whether to add validation plots.")

    args = parser.parse_args()

    path = args.output_folder
    print(path)
    scoring_type = path.split("/")[-1].split("_")[0]
    if 'fb15k-237' in path:
        dataset = 'fb15k-237'
        X = load_fb15k_237()
    elif 'wn18rr' in path:
        dataset = 'wn18rr'
        X = load_wn18rr()
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

    kge_model = load_pretrained_model(dataset=dataset, scoring_type=scoring_type)

    ent_dict = kge_model.data_indexer.backend.entities_dict
    rel_dict = kge_model.data_indexer.backend.relations_dict

    idx_to_raw_ents = pd.DataFrame.from_dict(
        ent_dict, orient="index", columns=["raw"]
    )

    early_stopper = None

    if args.valid:
        with open(os.path.join(path, "valid_ranks_po.pkl"), "rb") as f:
            valid_ranks_po = pickle.load(f)

        with open(os.path.join(path, "valid_ranks_sp.pkl"), "rb") as f:
            valid_ranks_sp = pickle.load(f)

        valid_dict_splits = split_triples_by_degree_4way(valid_df, degrees, thresholds=thresholds)

        aux_valid_df = valid_df.copy()
        aux_valid_df["triple_type"] = None
        for split, indices in valid_dict_splits.items():
            aux_valid_df.loc[indices, "triple_type"] = split

        all_triples_sp = ranks_to_triples(valid_ranks_sp, 'sp')
        all_triples_po = ranks_to_triples(valid_ranks_po, 'po')

        all_triples_sp = all_triples_sp.merge(aux_valid_df, on=['s', 'p', 'o'])
        all_triples_po = all_triples_po.merge(aux_valid_df, on=['s', 'p', 'o'])

        metrics = compute_metrics_across_epochs(all_triples_sp, all_triples_po)
        metrics_by_split = compute_metrics_across_epochs(
            all_triples_sp, all_triples_po, dict_splits=valid_dict_splits
        )

        early_stopper = EarlyStopper(
            valid_ranks_sp, valid_ranks_po, valid_df, thresholds=thresholds,
            degrees=pd.concat([train_df.s, train_df.o]).value_counts()
        )


    with open(os.path.join(path, "ranks_po.pkl"), "rb") as f:
        ranks_po = pickle.load(f)

    with open(os.path.join(path, "ranks_sp.pkl"), "rb") as f:
        ranks_sp = pickle.load(f)

    dict_splits = split_triples_by_degree_4way(test_df, degrees, thresholds=thresholds)

    aux_test_df = test_df.copy()
    aux_test_df["triple_type"] = None
    for split, indices in dict_splits.items():
        aux_test_df.loc[indices, "triple_type"] = split

    all_triples_sp = ranks_to_triples(ranks_sp, 'sp')
    all_triples_po = ranks_to_triples(ranks_po, 'po')

    all_triples_sp = all_triples_sp.merge(aux_test_df, on=['s', 'p', 'o'])
    all_triples_po = all_triples_po.merge(aux_test_df, on=['s', 'p', 'o'])

    metrics = compute_metrics_across_epochs(all_triples_sp, all_triples_po)
    metrics_by_split = compute_metrics_across_epochs(
        all_triples_sp, all_triples_po, dict_splits=dict_splits, aggregated=args.valid
    )
    if args.valid:
        metrics_by_split, aggregated_metrics = metrics_by_split
        aggregated_to_tables(aggregated_metrics, path=path)
