import os
import argparse
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf

from ampligraph.datasets import load_fb15k_237, load_wn18rr
from ampligraph.pretrained_models import load_pretrained_model
from ampligraph.utils import restore_model


def kge_latent_search(kge_model, train_df, valid_df, test_df, ent_dict, group='sp', args=None):
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
    if args.dataset == "loreal":
        gene_indices = get_loreal_gene_indices(train_df, idx_to_raw_ents)
        # We redefine the mapper to only account for genes as we are going to
        # corrupt only using them.
        idx_to_raw_ents = idx_to_raw_ents.loc[gene_indices].reset_index()

    train_grouped, valid_grouped, test_grouped = group_by_pair(
        train_df, valid_df, test_df, group=group
    )

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

        all_ents_to_filter = set(train_targets).union(valid_targets).union(test_targets)

        subjects, predicates, objects = extract_kge_embeddings(
            kge_model, pair, train_targets, valid_targets, group
        )

        if args.dataset != "loreal":
            all_corruption_entities = kge_model.encoding_layer.ent_emb
        else:
            all_corruption_entities = tf.gather(
                kge_model.encoding_layer.ent_emb,
                gene_indices
            )

        # Let's compute the scores before applying the latent search
        if group == 'sp':
            pre_scores = - kge_model.scoring_layer([subjects, predicates, all_corruption_entities])
        else:
            pre_scores = - kge_model.scoring_layer([all_corruption_entities, predicates, objects])
        filtered_pre_ranks = compute_filtered_ranking(
            pre_scores, idx_to_raw_ents, test_targets, all_ents_to_filter
        )

        ranks[pair]["targets"] = test_targets
        ranks[pair]["pre_ranks"] = filtered_pre_ranks
        ranks[pair]["new_ranks"] = dict() # key: epoch; value: new_ranks for the target objects

        #### Actual latent search
        if len(train_targets) + len(valid_targets) > 0:
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

            # Let's make predicates and objects/subjects trainable
            updated_pred = tf.Variable(predicates)
            if group == 'sp':
                updated_subj = tf.Variable(subjects)
                input_triples = [updated_subj, updated_pred, objects]
                trainable_params = [updated_subj, updated_pred] if not args.freeze_relations else [updated_subj]
            else:
                updated_obj = tf.Variable(objects)
                input_triples = [subjects, updated_pred, updated_obj]
                trainable_params = [updated_pred, updated_obj] if not args.freeze_relations else [updated_obj]

            for i in range(0, args.epochs):
                with tf.GradientTape() as tape:
                    inp_score = - kge_model.scoring_layer(input_triples)

                grads = tape.gradient(inp_score, trainable_params)
                optimizer.apply_gradients(zip(grads, trainable_params))

                if group == 'sp':
                    # Evaluate the updated subj and pred against all the possible object corruptions
                    scores = - kge_model.scoring_layer(
                        [updated_subj, updated_pred, kge_model.encoding_layer.ent_emb]
                    )
                else:
                    # Evaluate the updated obj and pred against all the possible subject corruptions
                    scores = - kge_model.scoring_layer(
                        [kge_model.encoding_layer.ent_emb, updated_pred, updated_obj]
                    )

                filtered_new_ranks = compute_filtered_ranking(
                    scores, idx_to_raw_ents, test_targets, all_ents_to_filter
                )
                ranks[pair]["new_ranks"][i] = filtered_new_ranks

    return ranks

def group_by_pair(train, valid, test, group='sp'):
    """
    Function to group train, valid, and test based on (subject, predicate) or (predicate, object).
    """
    if group == 'sp':
        group_by = ['s', 'p']
    else:
        group_by = ['o', 'p']

    return train.groupby(group_by), valid.groupby(group_by), test.groupby(group_by)

def extract_kge_embeddings(kge_model, pair, train_targets, valid_targets, group='sp'):
    """
    Extract the subject, predicate and object embeddings from the pretrained KGE. Importantly,
    depending on whether we group by 'sp' or 'po' we are going to retrieve target embeddings
    for the subject or the object.
    """
    if group == 'sp':
        subjects = kge_model.get_embeddings([pair[0]], embedding_type="e")
        objects = kge_model.get_embeddings(
            np.concatenate([train_targets, valid_targets]),
            embedding_type="e"
        )
    else:
        subjects = kge_model.get_embeddings(
            np.concatenate([train_targets, valid_targets]),
            embedding_type="e"
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

def get_loreal_gene_indices(train, idx_to_raw):
    """
    Extract the embeddings of all the genes in the dataset, so
    to extract the subset of entities we want to use to corrupt
    the subject of our triples in the test set.
    """
    gene_raw = pd.DataFrame(
        {"raw": train.loc[train.p == '<urn:loreal:isExpressedBy>', "o"].unique()}
    )

    gene_indices = gene_raw.merge(idx_to_raw.reset_index(), how="left", on="raw")[["index"]]
    return gene_indices.values.reshape(-1)

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

    parser.add_argument('--freeze-relations', action='store_true', default=None,
                        help="Whether to fine-tune the relations as well as the entities or just the entities")

    args = parser.parse_args()

    tf.keras.utils.set_random_seed(seed=args.seed)

    if args.dataset == "fb15k-237":
        X = load_fb15k_237()
    elif args.dataset == "wn18rr":
        X = load_wn18rr()
    elif args.dataset == "loreal":
        # path_to_experiment = "/data/alberto/loreal/output/2.0.0+dev18.82c30fb/background_ontology_sampling_uniform/experiment_1"
        path_to_experiment = "/Users/alberto.bernardi/Documents/Projects/Data L'Oreal/2.0.0+dev18.82c30fb/best_experiment_background/experiment_1"
        X = {
            "train": pd.read_csv(os.path.join(path_to_experiment, "eval_data", "train.csv"), index_col=0).values,
            "valid": pd.DataFrame({'s': [], 'p': [], 'o': []}).values,
            "test": pd.read_csv(os.path.join(path_to_experiment, "eval_data", "test.csv"), index_col=0).values
        }

        kge_model = restore_model(os.path.join(path_to_experiment, "model"))

    train, valid, test = X["train"], X["valid"], X["test"]

    train_df = pd.DataFrame(train, columns=['s', 'p', 'o'])
    valid_df = pd.DataFrame(valid, columns=['s', 'p', 'o'])
    test_df = pd.DataFrame(test, columns=['s', 'p', 'o'])

    if args.dataset != "loreal":
        kge_model = load_pretrained_model(dataset=args.dataset, scoring_type=args.scoring_type)


    ent_dict = kge_model.data_indexer.backend.entities_dict
    rel_dict = kge_model.data_indexer.backend.relations_dict

    output_path = os.path.join(args.output_folder, args.dataset, f"{args.scoring_type}_lr{args.learning_rate}")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    print(f"Saving output at {output_path}")

    if args.dataset != "loreal":
        ranks_sp = kge_latent_search(kge_model, train_df, valid_df, test_df, ent_dict, group='sp', args=args)
        with open(os.path.join(output_path, "ranks_sp.pkl"), "wb") as f:
            pickle.dump(ranks_sp, f)

    ranks_po = kge_latent_search(kge_model, train_df, valid_df, test_df, ent_dict, group='po', args=args)
    with open(os.path.join(output_path, "ranks_po.pkl"), "wb") as f:
        pickle.dump(ranks_po, f)
