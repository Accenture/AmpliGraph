import os
import argparse
import pickle
import json

from tensorflow_estimator.python.estimator import early_stopping
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf

from ampligraph.datasets import load_fb15k_237, load_wn18rr, load_yago3_10
from ampligraph.pretrained_models import load_pretrained_model
from ampligraph.utils import restore_model

from experiments.LatentSearch.code.utils_KGE_latent_search import *


def kge_latent_search(kge_model, train_df, valid_df, test_df, ent_dict, group='sp', args=None, early_stop=None):
    """
    This function implements the latent search. From the definition of the targets
    observed during training and the optimization of (subject, predicate) or
    (predicate, object), to recompute the rank for the target entities after each
    epoch of the latent search.
    """
    assert group in ['sp', 'po'], "Group must be 'sp' or 'po'!"

    idx_to_raw_ents = pd.DataFrame.from_dict(
        ent_dict, orient="index", columns=["raw"]
    )

    train_grouped, valid_grouped, test_grouped = group_by_pair(
        train_df, valid_df, test_df, group=group
    )

    degrees = pd.concat([train_df.s, train_df.o]).value_counts()

    if args.llm_integration:
        if len(valid_df) > 0:
            path_closest = args.llm_integration
        else:
            # If we are considering the validation set, we need to select the file
            # that contains the neighbours for the pairs extracted from the validation set.
            folder_neighbors = os.path.dirname(args.llm_integration)
            file_neighbors = os.path.basename(args.llm_integration)
            path_closest = os.path.join(
                folder_neighbors, "validation_" + file_neighbors
            )

        with open(path_closest, "rb") as f:
            closest_ents = pickle.load(f)[group]

    ranks = dict()

    for pair in tqdm(test_grouped.groups, total=len(test_grouped)):
        ranks[pair] = dict()

        try:
            train_targets = train_grouped.get_group(pair)["o" if group == 'sp' else 's'].values
        except KeyError:
            train_targets = []
        try:
            valid_targets = valid_grouped.get_group(pair)["o" if group == 'sp' else 's'].values
        except KeyError:
            valid_targets = []

        test_targets = test_grouped.get_group(pair)["o" if group == 'sp' else 's'].values

        observed_targets = np.concatenate([train_targets, valid_targets])

        all_ents_to_filter = set(train_targets).union(valid_targets).union(test_targets)

        all_corruption_entities = kge_model.encoding_layer.ent_emb

        # Let's compute the scores before applying the latent search
        if group == 'sp':
            pre_scores = - kge_model.scoring_layer([
                kge_model.get_embeddings([pair[0]], embedding_type="e"),
                kge_model.get_embeddings([pair[1]], embedding_type="r"),
                all_corruption_entities
            ])
        else:
            pre_scores = - kge_model.scoring_layer([
                all_corruption_entities,
                kge_model.get_embeddings([pair[1]], embedding_type="r"),
                kge_model.get_embeddings([pair[0]], embedding_type="e")
            ])

        filtered_pre_ranks = compute_filtered_ranking(
            pre_scores, idx_to_raw_ents, test_targets, all_ents_to_filter
        )

        ranks[pair]["targets"] = test_targets
        ranks[pair]["pre_ranks"] = filtered_pre_ranks
        ranks[pair]["new_ranks"] = dict() # key: epoch; value: new_ranks for the target objects

        #### Actual latent search
        if len(observed_targets) > 0:

            if early_stop:
                anchor_degree = degrees[pair[0]]
                estimated_target_degree = np.quantile(
                    degrees[train_targets].tolist() + degrees[valid_targets].tolist(),
                    0.2
                )
                estimated_type = map_degree_to_bin(
                    anchor_degree if group == 'sp' else estimated_target_degree,
                    anchor_degree if group == 'po' else estimated_target_degree,
                    thresholds=early_stop.thresholds
                )
                early_stop_epoch = early_stop.epochs[estimated_type]
                ranks[pair]["early_stop_epoch"] = early_stop_epoch

                if early_stop_epoch == 0:
                    ranks[pair]["early_stop_new_ranks"] = filtered_pre_ranks

            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

            all_inputs = set_trainable_variables(
                kge_model, pair, observed_targets,
                group=group, freeze_rels=args.freeze_relations,
                additional_ents=closest_ents[pair],
                train_additional_ents=args.tune_llm_ents
            )
            input_triples, trainable_params, pair_idx = all_inputs[0], all_inputs[1], all_inputs[2]
            if args.tune_llm_ents:
                additional_triples = all_inputs[3]

            for i in range(0, args.epochs):
                with tf.GradientTape() as tape:
                    inp_score = - kge_model.scoring_layer(input_triples)

                    if args.tune_llm_ents:
                        inp_score = tf.concat([inp_score, - kge_model.scoring_layer(additional_triples)], axis=0)

                grads = tape.gradient(inp_score, trainable_params)
                optimizer.apply_gradients(zip(grads, trainable_params))

                updated_ent, updated_pred = get_updated_embeddings(
                    trainable_params, pair, pair_idx, kge_model,
                    freeze_rels=args.freeze_relations,
                )


                all_corruption_entities = tf.tensor_scatter_nd_update(
                    all_corruption_entities,
                    kge_model.get_indexes([pair[0]], 'e', 'raw2ind').reshape(-1, 1),
                    updated_ent
                )
                if args.tune_llm_ents:
                    # As we updated the embedding of some entities, we need to update them.
                    idx_additional_ents = kge_model.get_indexes(closest_ents[pair], 'e', 'raw2ind')
                    all_corruption_entities = tf.tensor_scatter_nd_update(
                        all_corruption_entities,
                        idx_additional_ents.reshape(-1, 1),
                        trainable_params[1] if args.freeze_relations else trainable_params[2]
                    )

                # Evaluate the updated subj and pred against all the possible object corruptions
                scores = - kge_model.scoring_layer(
                    [updated_ent, updated_pred, all_corruption_entities] if group == 'sp' \
                    else [all_corruption_entities, updated_pred, updated_ent]
                )

                filtered_new_ranks = compute_filtered_ranking(
                    scores, idx_to_raw_ents, test_targets, all_ents_to_filter
                )
                ranks[pair]["new_ranks"][i] = filtered_new_ranks

            if early_stop and (i + 1) == early_stop_epoch:
                ranks[pair]["early_stop_new_ranks"] = filtered_new_ranks

    return ranks

def extract_kge_embeddings(kge_model, pair, observed_targets, group='sp'):
    """
    Extract the subject, predicate and object embeddings from the pretrained KGE. Importantly,
    depending on whether we group by 'sp' or 'po' we are going to retrieve target embeddings
    for the subject or the object.
    """
    if group == 'sp':
        subjects = kge_model.get_embeddings([pair[0]], embedding_type="e")
        objects = kge_model.get_embeddings(
            observed_targets, embedding_type="e"
        )
    else:
        subjects = kge_model.get_embeddings(
            observed_targets, embedding_type="e"
        )
        objects = kge_model.get_embeddings([pair[0]], embedding_type="e")

    predicates = kge_model.get_embeddings([pair[1]], embedding_type="r")

    return tf.constant(subjects), tf.constant(predicates), tf.constant(objects)

def compute_filtered_ranking(scores, df_idx_to_raw_ents, test_targets, all_ents_to_filter):
    """
    Function that, given the scores for the triples, extracts the ranking filtering
    triples that were observed in the training, validation and test sets.
    """
    sorted_ents = np.argsort(scores)
    sorted_scores = scores.numpy()[sorted_ents]
    ranking = pd.DataFrame({'score': sorted_scores, "idx_pre": sorted_ents}).merge(
        df_idx_to_raw_ents, left_on="idx_pre", right_index=True
    )

    filtered_ranks = []
    for el in test_targets:
        ents_to_filter = all_ents_to_filter.difference([el])
        filtered_ranking = ranking[~ranking.raw.isin(ents_to_filter)]
        target_score = filtered_ranking.loc[filtered_ranking.raw == el, "score"].item()
        filtered_ranks.append((filtered_ranking.score <= target_score).sum())

    return np.array(filtered_ranks)

def set_trainable_variables(
        model, pair, targets, group='sp', freeze_rels=None,
        additional_ents=None, train_additional_ents=None):
    """
    Extract the input triples and the trainable parameters after setting the state
    of the correct variables to trainable.
    """
    if additional_ents is not None and not train_additional_ents:
        targets = np.concatenate([targets, additional_ents])

    subjects, predicates, objects = extract_kge_embeddings(model, pair, targets, group)
    updated_pred = tf.Variable(predicates) if not freeze_rels else predicates
    if group == 'sp':
        updated_subj = tf.Variable(subjects)
        input_triples = [updated_subj, updated_pred, objects]
        trainable_params = [updated_subj, updated_pred] if not freeze_rels else [updated_subj]
    else:
        updated_obj = tf.Variable(objects)
        input_triples = [subjects, updated_pred, updated_obj]
        trainable_params = [updated_obj, updated_pred] if not freeze_rels else [updated_obj]

    # v0.7 - If we want to train the additional entities added with the LLM integration
    if train_additional_ents:
        assert additional_ents is not None, "additional_ents must be provided if you want to optimize them!"
        updated_additional_ents = tf.Variable(
            model.get_embeddings(additional_ents, embedding_type="e")
        )

        additional_triples = [
            updated_subj if group == 'sp' else updated_additional_ents,
            updated_pred,
            updated_obj if group == 'po' else updated_additional_ents
        ]
        trainable_params += [updated_additional_ents]

    pair_idx = None

    if train_additional_ents:
        return input_triples, trainable_params, pair_idx, additional_triples

    return input_triples, trainable_params, pair_idx


def get_updated_embeddings(
        trainable_params, pair, pair_idx, model, freeze_rels=None
):
    """
    Extract, from all the trainable variables, just those for the anchor embeddings that we
    want to update. Typically, when group=='sp', the anchor embeddings are that of the subject
    and that of the relation.
    """
    updated_ent = trainable_params[0]
    updated_pred = trainable_params[1] if not freeze_rels \
        else model.get_embeddings([pair[1]], embedding_type="r")

    return updated_ent, updated_pred

def create_early_stopping_epochs(sp_ranks, po_ranks, valid, thresholds, degrees, metric="Hits@10"):
    """
    Function to create a dictionary that splits validation triples based on the degree
    of the subject and object and it associates, to each of these group the optimal
    number of epochs for which to carry out the latent search.
    This information is going to be leveraged at test time in order to early stop the
    latent search and get the best results.
    """
    from experiments.TrackEmbeddings.analysis_utils import split_triples_by_degree_4way

    dict_splits = split_triples_by_degree_4way(
        valid, degrees, thresholds=thresholds
    )
    valid['triple_type'] = None
    for split, indices in dict_splits.items():
        valid.loc[indices, "triple_type"] = split

    triples_sp = ranks_to_triples(sp_ranks, 'sp')
    triples_po = ranks_to_triples(po_ranks, 'po')

    all_triples_sp = triples_sp.merge(valid, on=['s', 'p', 'o'])
    all_triples_po = triples_po.merge(valid, on=['s', 'p', 'o'])

    metrics_by_split_affected = compute_metrics_across_epochs(
        all_triples_sp, all_triples_po, dict_splits=dict_splits, only_affected=True
    )
    early_stopping_epochs = {'sp': {}, 'po': {}}
    for split in dict_splits:
        # Extract the epoch where the metric of interest is maximised.
        sp_metric_of_interest = pd.DataFrame(metrics_by_split_affected[split]['corr_o']).loc[metric]
        po_metric_of_interest = pd.DataFrame(metrics_by_split_affected[split]['corr_s']).loc[metric]

        early_stopping_epochs['sp'][split] = sp_metric_of_interest.argmax()
        early_stopping_epochs['po'][split] = po_metric_of_interest.argmax()

    return early_stopping_epochs

class EarlyStopper():
    """
    Class to model our Early Stopper.
    """
    def __init__(
            self, valid_ranks_sp, valid_ranks_po,
            valid_df, thresholds, degrees
    ):
        self.thresholds = sorted(thresholds)
        self.epochs = create_early_stopping_epochs(
            valid_ranks_sp, valid_ranks_po, valid_df, thresholds=thresholds,
            degrees=degrees
        )

    def extract_anchor_bins(self, anchor_deg, group='sp'):
        """
        Given the anchor entity degree, it returns the bin to which the target
        object/subject could belong to.
        """
        if group == 'sp':
            possible_bins = {
                0: ["low-low", "low-mid1", "low-mid2", "low-high"],
                1: ["mid1-low", "mid1-mid1", "mid1-mid2", "mid1-high"],
                2: ["mid2-low", "mid2-mid1", "mid2-mid2", "mid2-high"],
                3: ["high-low", "high-mid1", "high-mid2", "high-high"],
            }
        elif group == 'po':
            possible_bins = {
                0: ["low-low", "mid1-low", "mid2-low", "high-low"],
                1: ["low-mid1", "mid1-mid1", "mid2-mid1", "high-mid1"],
                2: ["low-mid2", "mid1-mid2", "mid2-mid2", "high-mid2"],
                3: ["low-high", "mid1-high", "mid2-high", "high-high"]
            }
        for j in range(len(self.thresholds)):
            if anchor_deg <= self.thresholds[j]:
                return possible_bins[j]
        return possible_bins[j + 1]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Pipeline to control the experiments for the KGE inference latent search.')

    parser.add_argument('--dataset', default='fb15k-237', type=str, help='Dataset to process')
    parser.add_argument('--output-folder', type=str,
                        default="/data/alberto/ampligraph_results/latent_search/KGE_inference_search",
                        help='Base folder for the output of the experiment.')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--scoring-type', '-score', default="RotatE", type=str,
                        help="KGE pretrained model to use for the experiment.")
    parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float,
                        help="Learning rate of the optimizer of the latent search.")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs for the latent search.")
    parser.add_argument('--regularization', default=None, type=float,
                        help="L3 Regularization coefficient to apply. Defaults to `None` (no regularization).")

    parser.add_argument('--freeze-relations', action='store_true', default=None,
                        help="Whether to fine-tune the relations as well as the entities or just the entities")

    parser.add_argument('--run-on-validation', '-valid', action='store_true',
                        help="Whether to run run the experiment on the test set instead of the test.")
    parser.add_argument('--no-run-on-test', action='store_false', default=True, dest='run_on_test',
                        help='No execution on the test set.')

    parser.add_argument('--llm-integration', type=str, default=None,
                        help="Path to the file with the closest entities for each pair as determined with an LLM."
                             "This new feature will be identified as v0.6.")
    parser.add_argument('--tune-llm-ents', action='store_true', default=None,
                        help='Whether to optimize also the target entity we include with LLM integration - v0.7')

    parser.add_argument('--debug-valid', action='store_true', default=False)


    args = parser.parse_args()

    tf.keras.utils.set_random_seed(seed=args.seed)

    train_df, valid_df, test_df = load_dataset(args.dataset)

    if args.dataset == "fb15k-237":
        thresholds = [10, 30, 100]
    else:
        thresholds = np.quantile(pd.concat([train_df.s, train_df.o]).value_counts(), q=[0.25, 0.5, 0.75]).tolist()

    if args.dataset != "biokg":
        kge_model = load_pretrained_model(dataset=args.dataset, scoring_type=args.scoring_type)
    else:
        if args.scoring_type == "ComplEx":
            path_to_experiment = "/data/alberto/ampligraph_results/results_grid_search/biokg/ComplEx-2025-07-29-21-36-53"
        elif args.scoring_type == "RotatE":
            path_to_experiment = "/data/alberto/ampligraph_results/results_grid_search/biokg/RotatE-2025-08-01-15-57-17"
        kge_model = restore_model(os.path.join(path_to_experiment, "model"))

    ent_dict = kge_model.data_indexer.backend.entities_dict
    rel_dict = kge_model.data_indexer.backend.relations_dict

    output_path = os.path.join(
        args.output_folder, args.dataset,
        f"{args.scoring_type}_lr{args.learning_rate}"
    )
    if args.seed != 0:
        output_path += f"_seed{args.seed}"

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    print(f"Saving output at {output_path}")

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(vars(args), f)

    early_stopper = None
    if args.run_on_validation:
        # We use an empty df as a placeholder for the missing
        # validation set as we are using it as test here.
        valid_ranks_sp = kge_latent_search(
            kge_model, train_df=train_df,
            valid_df=pd.DataFrame([], columns=['s', 'p', 'o']),
            test_df=valid_df, ent_dict=ent_dict,
            group='sp', args=args
        )
        with open(os.path.join(output_path, "valid_ranks_sp.pkl"), "wb") as f:
            pickle.dump(valid_ranks_sp, f)

        # We use an empty df as a placeholder for the missing validation set as we are using it as test here.
        valid_ranks_po = kge_latent_search(
            kge_model, train_df=train_df,
            valid_df=pd.DataFrame([], columns=['s', 'p', 'o']),
            test_df=valid_df, ent_dict=ent_dict,
            group='po', args=args
        )
        with open(os.path.join(output_path, "valid_ranks_po.pkl"), "wb") as f:
            pickle.dump(valid_ranks_po, f)

        early_stopper = EarlyStopper(
            valid_ranks_sp, valid_ranks_po, valid_df, thresholds=thresholds,
            degrees=pd.concat([train_df.s, train_df.o]).value_counts()
        )

    if args.debug_valid:
        with open(os.path.join(output_path, "valid_ranks_po.pkl"), "rb") as f:
            valid_ranks_po = pickle.load(f)

        with open(os.path.join(output_path, "valid_ranks_sp.pkl"), "rb") as f:
            valid_ranks_sp = pickle.load(f)

        early_stopper = EarlyStopper(
            valid_ranks_sp, valid_ranks_po, valid_df, thresholds=thresholds,
            degrees=pd.concat([train_df.s, train_df.o]).value_counts()
        )

    if args.run_on_test:
        ranks_sp = kge_latent_search(
            kge_model, train_df, valid_df, test_df, ent_dict, group='sp', args=args,
            early_stop=early_stopper
        )
        with open(os.path.join(output_path, "ranks_sp.pkl"), "wb") as f:
            pickle.dump(ranks_sp, f)

        ranks_po = kge_latent_search(
            kge_model, train_df, valid_df, test_df, ent_dict, group='po', args=args,
            early_stop=early_stopper
        )
        with open(os.path.join(output_path, "ranks_po.pkl"), "wb") as f:
            pickle.dump(ranks_po, f)
