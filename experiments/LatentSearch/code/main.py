"""
This script can be used to run the experiments found in the paper
"ImbalancE: Inference-Time Latent Search Against Degree Imbalance in Link Prediction"
accepted at ISWC 2026.
"""

import os

# Silence TensorFlow / CUDA startup logging. These MUST be set before `import tensorflow`.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # silences the oneDNN notice
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")

import argparse
import pickle
import json
import logging

import tensorflow as tf

# Silence Python-level and absl-level TF warnings (e.g. deprecation notices).
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

from ampligraph.utils.imbalancE_utils import *
from ampligraph.latent_features.models import ImbalancE
from ampligraph.latent_features.models.ImbalancE import EarlyStopper
from ampligraph.pretrained_models import load_pretrained_model

def map_entities_dataset(train, valid, test, entity_mapper, relation_mapper):
    """
    Map entities to indices.
    """
    ent_mapper = pd.DataFrame({"raw": list(entity_mapper.keys()), "idx": list(entity_mapper.values())})
    rel_mapper = pd.DataFrame({"raw": list(relation_mapper.keys()), "idx": list(relation_mapper.values())})

    train['s'] = train.merge(ent_mapper, left_on='s', right_on="raw", how='left')['idx']
    train['o'] = train.merge(ent_mapper, left_on='o', right_on="raw", how='left')['idx']
    train['p'] = train.merge(rel_mapper, left_on='p', right_on="raw", how='left')['idx']

    valid['s'] = valid.merge(ent_mapper, left_on='s', right_on="raw", how='left')['idx']
    valid['o'] = valid.merge(ent_mapper, left_on='o', right_on="raw", how='left')['idx']
    valid['p'] = valid.merge(rel_mapper, left_on='p', right_on="raw", how='left')['idx']

    test['s'] = test.merge(ent_mapper, left_on='s', right_on="raw", how='left')['idx']
    test['o'] = test.merge(ent_mapper, left_on='o', right_on="raw", how='left')['idx']
    test['p'] = test.merge(rel_mapper, left_on='p', right_on="raw", how='left')['idx']

    return train, valid, test

def setup_output_folder(config):
    """
    Function to set up the output folder for the experiment.

    The results are going to be output in the following structure:
    PATH_TO_OUTPUT_FOLDER/DATASET/SCORING_TYPE_LR[LEARNING_RATE].
    """
    output_path = os.path.join(
        config.output_folder, config.dataset,
        f"{config.scoring_type}_lr{config.learning_rate}"
    )
    if config.seed != 0:
        output_path += f"_seed{config.seed}"
    if config.only_llm:
        output_path += "_only_llm"

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    print(f"Saving output at {output_path}")

    return output_path


def from_batches_to_results(batches, no_support_batch, rev_ent_dict, rev_rel_dict, group='sp'):
    """
    Function to unpack the content of batches and restructure it into a suitable format
    """
    num_epochs = len(batches[0]['new_ranks'])
    ls_test_triples = np.vstack([batch['test_triples'] for batch in batches])
    ls_pre_ranks = np.concatenate([batch['pre_ranks'] for batch in batches])
    ls_new_ranks = {
        epoch: np.concatenate([batch['new_ranks'][epoch] for batch in batches])
        for epoch in range(num_epochs)
    }
    # The batches carry the early-stopping ranks only when a latent search was run with an
    # early stopper (i.e. on the test set). When present, they are propagated to the dict below.
    has_early_stop = 'early_stop_ranks' in batches[0]
    if has_early_stop:
        ls_early_stop_ranks = np.concatenate([batch['early_stop_ranks'] for batch in batches])
    all_test_triples = np.vstack([
        ls_test_triples, no_support_batch['test_triples']
    ])
    all_pre_ranks = np.concatenate([
        ls_pre_ranks, no_support_batch['pre_ranks']
    ])
    df_ranks = pd.DataFrame(
        np.concatenate([all_test_triples, all_pre_ranks.reshape(-1, 1)], axis=1),
        columns=['s', 'p', 'o', 'pre_ranks']
    )
    array_new_ranks = np.vstack(list(ls_new_ranks.values())).transpose()
    extra_columns = [f'ranks_epoch{i}' for i in range(num_epochs)]
    extra_values = array_new_ranks
    if has_early_stop:
        extra_columns = extra_columns + ['early_stop_new_ranks']
        extra_values = np.concatenate([array_new_ranks, ls_early_stop_ranks.reshape(-1, 1)], axis=1)
    ls_df_new_ranks = pd.DataFrame(
        np.concatenate([ls_test_triples, extra_values], axis=1),
        columns=['s', 'p', 'o'] + extra_columns
    )

    ranks_pair = df_ranks.merge(ls_df_new_ranks, on=['s', 'p', 'o'], how='left')

    # Perform mapping back to entities and relations
    ranks_pair[['s', 'o']] = ranks_pair[['s', 'o']].replace(rev_ent_dict)
    ranks_pair[['p']] = ranks_pair[['p']].replace(rev_rel_dict)

    if group == 'sp':
        grouped_ranks = ranks_pair.groupby(['s', 'p'])
    else:
        grouped_ranks = ranks_pair.groupby(['o', 'p'])

    ranks_pair_dict = dict()
    for pair in grouped_ranks.groups:
        ranks_pair_dict[pair] = {
            'targets': grouped_ranks.get_group(pair)['o' if group == 'sp' else 's'].values,
            'pre_ranks': grouped_ranks.get_group(pair)['pre_ranks'].values,
        }
        if grouped_ranks.get_group(pair)['ranks_epoch0'].isna().sum():
            ranks_pair_dict[pair]['new_ranks'] = dict()
        else:
            ranks_pair_dict[pair]['new_ranks'] = {
                i: grouped_ranks.get_group(pair)[f'ranks_epoch{i}'].values for i in range(args.epochs)
            }
            if has_early_stop:
                ranks_pair_dict[pair]['early_stop_new_ranks'] = \
                    grouped_ranks.get_group(pair)['early_stop_new_ranks'].values

    return ranks_pair_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Pipeline to control the experiments for the KGE inference latent search.')

    parser.add_argument('--dataset', default='fb15k-237', type=str, help='Dataset to process')
    parser.add_argument('--output-folder', type=str,
                        default="../results",
                        help='Base folder for the output of the experiment.')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--scoring-type', '-score', default="RotatE", type=str,
                        help="KGE pretrained model to use for the experiment.")
    parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float,
                        help="Learning rate of the optimizer of the latent search.")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs for the latent search.")
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--only-llm', action='store_true', default=None)
    parser.add_argument('--regularization', '-reg', type=float, default=None)

    parser.add_argument('--freeze-relations', action='store_true', default=None,
                        help="Whether to fine-tune the relations as well as the entities or just the entities")

    parser.add_argument('--run-on-validation', '-valid', action='store_true',
                        help="Whether to run run the experiment on the valid set instead of the test.")
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

    # Keep raw (unmapped) copies of train and valid. The early stopper operates on raw entity
    # ids, since the validation ranks dictionaries are keyed by raw entities (see
    # from_batches_to_results), whereas the dataframes below get mapped to integer indices.
    raw_train_df, raw_valid_df = train_df.copy(), valid_df.copy()

    kge_model = load_pretrained_model(dataset=args.dataset, scoring_type=args.scoring_type)

    # Extract mappers from KGE models to map entities and relations to integers and vice versa
    ent_dict = kge_model.data_indexer.backend.reversed_entities_dict
    rel_dict = kge_model.data_indexer.backend.reversed_relations_dict
    rev_ent_dict = kge_model.data_indexer.backend.entities_dict
    rev_rel_dict = kge_model.data_indexer.backend.relations_dict

    # Map dataset to integers
    train_df, valid_df, test_df = map_entities_dataset(train_df, valid_df, test_df, ent_dict, rel_dict)

    # Setup output folder
    output_path = setup_output_folder(args)

    # Save configuration of the experiment
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(vars(args), f)

    early_stopper = None

    #################################################################################
    # We first run the experiment on the validation set. This is going to be useful
    # to extract the early stopping epochs later on when we post process the results.
    # We first run ImbalancE on (subject, predicate) pairs and then, separately,
    # on (predicate, object) pairs.
    #################################################################################
    if args.run_on_validation:
        print("Running ImbalancE on the validation set...")
        ############### Validation (subject, predicate) pairs ###############
        # We use an empty df as a placeholder for the missing
        # validation set as we are using it as test here.
        empty_df = pd.DataFrame([], columns=['s', 'p', 'o'])
        imbalancE_valid_sp = ImbalancE(
            kge_model, train_df=train_df,
            valid_df=empty_df, test_df=valid_df,
            group='sp', args=args
        )
        valid_batches_sp, valid_batches_sp_no_support = imbalancE_valid_sp.perform_latent_search()

        # We unpack the results into a compact form
        valid_ranks_sp_dict = from_batches_to_results(
            valid_batches_sp, valid_batches_sp_no_support,
            rev_ent_dict, rev_rel_dict, group='sp'
        )
        # Save output
        with open(os.path.join(output_path, "valid_ranks_sp.pkl"), "wb") as f:
            pickle.dump(valid_ranks_sp_dict, f)

        ############### Validation (object, predicate) pairs ###############
        empty_df = pd.DataFrame([], columns=['s', 'p', 'o'])
        imbalancE_valid_po = ImbalancE(
            kge_model, train_df=train_df,
            valid_df=empty_df, test_df=valid_df,
            group='po', args=args
        )
        valid_batches_po, valid_batches_po_no_support = imbalancE_valid_po.perform_latent_search()

        # We unpack the results into a compact form
        valid_ranks_po_dict = from_batches_to_results(
            valid_batches_po, valid_batches_po_no_support,
            rev_ent_dict, rev_rel_dict, group='po'
        )
        # Save output
        with open(os.path.join(output_path, "valid_ranks_po.pkl"), "wb") as f:
            pickle.dump(valid_ranks_po_dict, f)

        #####################################################
        # Build the early stopper from the validation ranks #
        #####################################################
        degrees = pd.concat([raw_train_df.s, raw_train_df.o]).value_counts()

        # We set thresholds with a hard constraint on the low degree in case the .25 quantile is
        # below 5. Indeed, we don't want a low degree threshold that is too low.
        thresholds = [np.max([5, np.quantile(degrees, 0.25)])]
        if thresholds[0] == 5:
            thresholds += np.quantile(degrees[degrees > 5], [0.33, 0.66]).tolist()
        else:
            thresholds = np.quantile(degrees, [0.25, 0.5, 0.75]).tolist()

        early_stopper = EarlyStopper(
            valid_ranks_sp_dict, valid_ranks_po_dict, raw_valid_df,
            thresholds=thresholds, degrees=degrees
        )

    #####################################################
    ### We now run the latent search on the test set. ###
    #####################################################
    if args.run_on_test:
        print("Running ImbalancE on the test set...")
        ############### Test (object, predicate) pairs ###############
        imbalance_test_sp = ImbalancE(
            kge_model, train_df, valid_df, test_df,
            group='sp', args=args, early_stop=early_stopper
        )
        test_batches_sp, test_batches_sp_no_support = imbalance_test_sp.perform_latent_search()

        # We unpack the results into a compact form
        test_ranks_sp_dict = from_batches_to_results(
            test_batches_sp, test_batches_sp_no_support,
            rev_ent_dict, rev_rel_dict, group='sp'
        )

        with open(os.path.join(output_path, "ranks_sp.pkl"), "wb") as f:
            pickle.dump(test_ranks_sp_dict, f)

        ############### Test (object, predicate) pairs ###############
        imbalancE_test_po = ImbalancE(
            kge_model, train_df, valid_df, test_df,
            group='po', args=args, early_stop=early_stopper
        )
        test_batches_po, test_batches_po_no_support = imbalancE_test_po.perform_latent_search()

        # We unpack the results into a compact form
        test_ranks_po_dict = from_batches_to_results(
            test_batches_po, test_batches_po_no_support,
            rev_ent_dict, rev_rel_dict, group='po'
        )

        # Save output
        with open(os.path.join(output_path, "ranks_po.pkl"), "wb") as f:
            pickle.dump(test_ranks_po_dict, f)

    print(f"Experiment completed. Find output at: {output_path}")
