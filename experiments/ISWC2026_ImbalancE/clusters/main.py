""""
This file is meant to generate (and eventually investigate) the clustering of
FB15k-237 entities using the embeddings extracted via an LLM model.
We are going to group test triples based on whether they share subject and predicte
(or object and predicate). For each of such batches, we are going to compute the
baricenter of the group and then try to find the top-k nearest neighbours.

Example of command to generate description for the test set:
CUDA_VISIBLE_DEVICES=1 python main.py --dataset fb15k-237 --k 10 --hf-model NovaSearch/stella_en_400M_v5 --save
Example of command to generate description for the validation set:
CUDA_VISIBLE_DEVICES=1 python main.py --dataset fb15k-237 --k 10 --hf-model NovaSearch/stella_en_400M_v5 --save -valid
"""

import os
import argparse
import numpy as np
import random
import pandas as pd
import json
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import time

# Supported datasets. Their splits are fetched via the AmpliGraph API and cached
# locally under <dataset>/ by cache_datasets.py, so the runtime here stays
# pure-PyTorch and never needs to import ampligraph / TensorFlow.
SUPPORTED_DATASETS = ["fb15k-237", "wn18rr", "yago3-10"]

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)


def load_dataset(dataset):
    # Read the splits cached by cache_datasets.py (tab-separated, columns s, p, o).
    data_dir = dataset
    train = pd.read_csv(os.path.join(data_dir, "train.txt"), names=['s', 'p', 'o'], sep='\t', header=None, dtype=str)
    valid = pd.read_csv(os.path.join(data_dir, "valid.txt"), names=['s', 'p', 'o'], sep='\t', header=None, dtype=str)
    test = pd.read_csv(os.path.join(data_dir, "test.txt"), names=['s', 'p', 'o'], sep='\t', header=None, dtype=str)

    train_ents = pd.concat([train.s, train.o]).unique()
    train_rels = train.p.unique()

    valid = valid[valid.s.isin(train_ents) & valid.o.isin(train_ents) & valid.p.isin(train_rels)]
    test = test[test.s.isin(train_ents) & test.o.isin(train_ents) & test.p.isin(train_rels)]

    return train, valid, test

def get_textual_descriptions(dataset):
    if dataset == "fb15k-237":
        with open("fb15k-237/entity2wikidata.json", "r") as f:
            entity_desc_dict = json.load(f)
            entity_desc_dict = {
                entity: entity_desc_dict[entity].get("description", "") for entity in entity_desc_dict
            }

        label_mapper = pd.read_csv(
            "fb15k-237/mapper.csv",
            index_col=0, header=None
        ).to_dict()[1]

    elif dataset == "wn18rr":
        mapper = pd.read_csv("wn18rr/wordnet-mlj12-definitions.txt",
                             sep='\t', header=None, names=['id', 'label', 'description'], dtype={"id": str})

        mapper["label"] = mapper["label"].str.replace(r"^__", "", regex=True)
        mapper["label"] = mapper["label"].str.replace(r"_(NN|VB|JJ|RB)_[0-9]+$", "", regex=True)
        mapper["label"] = mapper["label"].str.replace("_", " ")

        label_mapper = mapper.set_index('id')["label"].to_dict()
        entity_desc_dict = mapper.set_index("id")["description"].to_dict()

    elif dataset == "yago3-10":
        train, _, _ = load_dataset("yago3-10")

        mapper = pd.concat([train.s, train.o]).drop_duplicates().to_frame("id")
        mapper["label"] = mapper["id"].str.split("_").str.join(" ")

        label_mapper = mapper.set_index("id")["label"].to_dict()
        entity_desc_dict = dict()

    entity_descriptions = dict()
    for entity, label in label_mapper.items():
        entity_descriptions[entity] = label
        if entity_desc_dict.get(entity, '') and entity_desc_dict[entity] != label:
            entity_descriptions[entity] += f" - {entity_desc_dict[entity]}"

    return entity_descriptions

def match_descriptions(entities, dataset):
    """
    For each element among the entities, it extracts the corresponding description.

    :param entities: list with the entities for which we want to get the textual description.
    :return: list of descriptions where the indices of entities in `entities` match the indices
    of entities in `descriptions`
    """
    entity_descriptions_dict = get_textual_descriptions(dataset)

    descriptions = {entity: entity_descriptions_dict[entity] for entity in entities}

    return descriptions

def get_targets(pair, train_grouped, valid_grouped, test_grouped, group='sp'):

    try:
        train_targets = train_grouped.get_group(pair)["o" if group == 'sp' else 's'].values.tolist()
    except KeyError:
        train_targets = []
    try:
        valid_targets = valid_grouped.get_group(pair)["o" if group == 'sp' else 's'].values.tolist()
    except KeyError:
        valid_targets = []

    test_targets = test_grouped.get_group(pair)["o" if group == 'sp' else 's'].values
    observed_targets = train_targets + valid_targets

    return test_targets, observed_targets

def top_k_closest(subset_embeddings, all_embeddings, k=10):
    """
    Find the `k` most similar items to the centroid of a subset of embeddings within all embeddings.

    This function calculates the centroid of a subset of embeddings, which represents the "query
    concept". It then computes the cosine similarity of all embeddings against this centroid and
    retrieves the indices of the most similar embeddings. The indices of the subset embeddings are
    excluded from the results to ensure that the output contains only new embeddings.

    :param subset_embeddings: A tensor containing the embeddings of the selected subset.
    :param all_embeddings: A tensor containing all available embeddings, including the subset.
    :param k: An integer specifying the number of items to return. Defaults to 10.
    :type subset_embeddings: Tensor
    :type all_embeddings: Tensor
    :type k: int
    :return: A list of the top `k` closest indices from `all_embeddings`, excluding indices belonging
        to the subset embeddings.
    :rtype: list[int]
    """
    # Compute centroid of subset embeddings (represents the "query concept")
    subset_center = torch.mean(subset_embeddings, dim=0, keepdim=True)

    # Compute cosine similarity of all texts against the subset centroid
    similarities = util.cos_sim(subset_center, all_embeddings).squeeze(0)

    # Get top-k most related items (excluding the subset itself if desired)
    top_k_indices = torch.topk(similarities, k + len(observed_targets_idx)).indices.tolist()

    # Filter out subset indices
    top_k_indices = [i for i in top_k_indices if i not in observed_targets_idx][:k]

    return top_k_indices

def top_k_closest_and_lowest(subset_embeddings, all_embeddings, degrees, all_ents, k=10):
    """
    Find the `2k` most similar items to the centroid of a subset of embeddings within all embeddings
    and opt for the `k` with the lowest degree.
    """
    # Compute centroid of subset embeddings (represents the "query concept")
    subset_center = torch.mean(subset_embeddings, dim=0, keepdim=True)

    # Compute cosine similarity of all texts against the subset centroid
    similarities = util.cos_sim(subset_center, all_embeddings).squeeze(0)

    # Get top-k most related items (excluding the subset itself if desired)
    top_2k_indices = torch.topk(similarities, 2 * k + len(observed_targets_idx)).indices.tolist()

    # Filter out subset indices
    top_2k_indices = [i for i in top_2k_indices if i not in observed_targets_idx][:2*k]
    # Find the positions of the k with the lowest degree among the 2k
    lowest_k_positions = np.argsort([degrees[all_ents[idx]] for idx in top_2k_indices], stable=True)[:k]
    # Extract the indices of the final top-k
    top_k_indices = [top_2k_indices[pos] for pos in lowest_k_positions]

    return top_k_indices

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fb15k-237', choices=SUPPORTED_DATASETS,
                        help='Knowledge Graph considered')
    parser.add_argument(
        '--hf-model', type=str, default="NovaSearch/stella_en_400M_v5", help="LM to extract embeddings."
    )
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbours to consider.')
    parser.add_argument('--validation', '-valid', action='store_true',
                        help='Whether to extract closest neighbors for the validation set. In that case, no test set considered.')
    parser.add_argument('--save', action='store_true', default=None, help='Whether to save final results or not.')

    args = parser.parse_args()
    start = time.time()

    train, valid, test = load_dataset(args.dataset)
    if args.validation:
        test = valid.copy()
        valid = pd.DataFrame({'s': [], 'p': [], 'o': []})

    entities_lst = pd.concat([train.s, train.o]).unique().tolist()
    descriptions_dict = match_descriptions(entities_lst, args.dataset)

    entities_txt = list(descriptions_dict.values())
    degrees_dict = pd.concat([train.s, train.o]).value_counts().to_dict()

    # Load the embedding model
    model = SentenceTransformer(args.hf_model, trust_remote_code=True)

    # Encode all texts
    txt_embeddings = model.encode(entities_txt, convert_to_tensor=True, normalize_embeddings=True)

    sp_train_grouped, po_train_grouped = train.groupby(["s", "p"]), train.groupby(["o", "p"])
    sp_valid_grouped, po_valid_grouped = valid.groupby(["s", "p"]), valid.groupby(["o", "p"])
    sp_test_grouped, po_test_grouped = test.groupby(["s", "p"]), test.groupby(["o", "p"])

    print(f"k = {args.k}")
    sp_overlap_stats = dict()
    sp_closest_ents = dict()

    for pair in tqdm(sp_test_grouped.groups, total=len(sp_test_grouped)):

        test_targets, observed_targets = get_targets(
            pair, sp_train_grouped, sp_valid_grouped, sp_test_grouped, group='sp'
        )
        if len(observed_targets):
            observed_targets_idx = [entities_lst.index(el) for el in observed_targets]
            test_targets_idx = [entities_lst.index(el) for el in test_targets]

            observed_txt_embeddings = txt_embeddings[observed_targets_idx]

            closest_ents_idx = top_k_closest(observed_txt_embeddings, txt_embeddings, k=args.k)

            # Look into the overlap
            overlap = set(closest_ents_idx).intersection(test_targets_idx)
            sp_overlap_stats[pair] = {
                "ents": overlap, "prop": len(overlap) / len(test_targets_idx), "len_targets": len(test_targets_idx)
            }

            sp_closest_ents[pair] = [entities_lst[idx] for idx in closest_ents_idx]
        else:
            sp_closest_ents[pair] = []

    po_overlap_stats = dict()
    po_closest_ents = dict()

    for pair in tqdm(po_test_grouped.groups, total=len(po_test_grouped)):
        test_targets, observed_targets = get_targets(
            pair, po_train_grouped, po_valid_grouped, po_test_grouped, group='po'
        )
        if len(observed_targets):
            observed_targets_idx = [entities_lst.index(el) for el in observed_targets]
            test_targets_idx = [entities_lst.index(el) for el in test_targets]

            observed_txt_embeddings = txt_embeddings[observed_targets_idx]

            closest_ents_idx = top_k_closest(observed_txt_embeddings, txt_embeddings, k=args.k)

            overlap = set(closest_ents_idx).intersection(test_targets_idx)
            po_overlap_stats[pair] = {
                "ents": overlap, "prop": len(overlap) / len(test_targets_idx), "len_targets": len(test_targets_idx)
            }

            po_closest_ents[pair] = [entities_lst[idx] for idx in closest_ents_idx]
        else:
            po_closest_ents[pair] = []

    closest_ents = {'sp': sp_closest_ents, 'po': po_closest_ents}

    print('Time taken: ', time.time() - start)

    if args.save:
        file_name = f"{'validation_' if args.validation else ''}{args.dataset}_top{args.k}_{args.hf_model.replace('/', '.')}.pkl"
        folder_name = args.dataset
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)

        with open(os.path.join(folder_name, file_name), "wb") as f:
            pickle.dump(closest_ents, f)

        print(f"Experiment run successfully. Output at: {os.path.join(folder_name, file_name)}")


# Reproduce (from this folder, using the isolated uv env — see README.md):
#   uv sync
#   uv run python main.py --dataset fb15k-237 --k 10 --save
#   uv run python main.py --dataset fb15k-237 --k 10 --save --valid
