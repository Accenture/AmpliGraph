import argparse
import pickle
import json
from tqdm import tqdm
import tensorflow as tf
from ampligraph.pretrained_models import load_pretrained_model
from ampligraph.utils import restore_model

from experiments.LatentSearch.code.utils_KGE_latent_search import *


def generate_batch_of_batches(
        train_grouped, valid_grouped, test_grouped,
        group='sp', llm_pairs_and_ents=None, max_pairs_per_group=10
    ):
    """
    Function that builds batches used during the latent search.

    In particular, each batch includes (subject, predicate) pairs from the test set
    (analogously, (predicate, object) if `group='po'`), with the corresponding training context
    extracted from train + valid (if any) and the corresponding llm_triples. In particular, within
    each batch, there can be no overlap between the objects (or subjects, if `group='po'`) of the
    llm_triples. This is made to avoid that the latent search optimization for one triple is influenced
    by another one.

    For each pair in the test set:
      - Extract test targets (objects for 'sp', subjects for 'po') and build test_triples
      - Extract observed targets from train + valid (if any) and build observed_triples
      - Extract llm entities (if any) for the pair and build llm_triples
    Build grouped batches of per-pair training triples derived from the pairs present in the test set.

    Grouping uses ONLY the entities from llm_integration to check disjointness between groups as
    the observed targets are not updated during the optimization, so, even if they overlap, the optimization
    process remains disjoint. The function also returns a list of empty batches (i.e., groups with no
    observed triples) that can be used to fill empty batches in the final evaluation.

    Grouping is performed per-pair greedily so that:
      - No overlap between the anchor entity and llm entity sets of any two units in the same group
      - Each grouped batch contains at most `max_pairs_per_group` observed_triples.

    Returns:
        Lists[Dict] with entries:
            {
              'group_id': int,
              'pairs': List[Tuple],
              'counts': Dict[pair, int], # number of observed_triples contributed by each pair
              'test_triples': np.ndarray (Nt, 3),
              'observed_triples': np.ndarray (No, 3),
              'llm_triples': np.ndarray (Nl, 3),
              'pre_ranks': np.ndarray (1, Nt), # ranks before the latent search
              'new_ranks': Dict[epoch, np.ndarray (1, Nt)], # ranks after each epoch of the latent search
            }

    """
    assert group in ('sp', 'po')

    target_col = 'o' if group == 'sp' else 's'

    units = []
    empty_batches = []
    for pair in test_grouped.groups:
        # Observed targets from train + valid
        try:
            train_targets = train_grouped.get_group(pair)[target_col].values
        except KeyError:
            train_targets = np.array([], dtype=np.int32)
        try:
            valid_targets = valid_grouped.get_group(pair)[target_col].values
        except KeyError:
            valid_targets = np.array([], dtype=np.int32)
        observed_targets = np.unique(np.concatenate([train_targets, valid_targets]))

        # Test targets (from test set)
        test_targets = np.sort(test_grouped.get_group(pair)[target_col].values)

        # Skip if no training context
        if len(observed_targets) == 0:
            if group == 'sp':
                s, p = pair[0], pair[1]
                test_triples = np.column_stack([
                    np.full(len(test_targets), s, dtype=np.int32),
                    np.full(len(test_targets), p, dtype=np.int32),
                    test_targets.astype(np.int32),
                ])
            else:
                o, p = pair[0], pair[1]
                test_triples = np.column_stack([
                    test_targets.astype(np.int32),
                    np.full(len(test_targets), p, dtype=np.int32),
                    np.full(len(test_targets), o, dtype=np.int32),
                ])
            empty_batches.append({
                'group_id': None,
                'pairs': [pair],
                'test_counts': {pair: test_triples.shape[0]},
                'test_triples': test_triples,
                'observed_triples': None,
                'llm_triples': None,
                'pre_ranks': None,
                'new_ranks': None
            })
            continue

        # Extract llm entities for each pair
        if llm_pairs_and_ents is None:
            llm_entities = []
        else:
            llm_entities = llm_pairs_and_ents[pair]
            assert len(set(llm_entities)) == len(llm_entities), \
                "LLM entities should be unique for the same pair. Did you mess up?"

        # Build triples
        if group == 'sp':
            s, p = pair[0], pair[1]
            observed_triples = np.column_stack([
                np.full(len(observed_targets), s),
                np.full(len(observed_targets), p),
                observed_targets
            ])

            test_triples = np.column_stack([
                np.full([tf.shape(test_targets)[0]], s),
                np.full([tf.shape(test_targets)[0]], p),
                test_targets
            ])
            if llm_pairs_and_ents is not None:
                llm_triples = np.column_stack([
                    np.full([tf.shape(llm_entities)[0]], s),
                    np.full([tf.shape(llm_entities)[0]], p),
                    llm_entities
                ])
        else:
            o, p = pair[0], pair[1]
            observed_triples = np.column_stack([
                observed_targets,
                np.full([tf.shape(observed_targets)[0]], p),
                np.full([tf.shape(observed_targets)[0]], o),
            ])
            test_triples = np.column_stack([
                test_targets,
                np.full([tf.shape(test_targets)[0]], p),
                np.full([tf.shape(test_targets)[0]], o),
            ])
            if llm_pairs_and_ents is not None:
                llm_triples = np.column_stack([
                    llm_entities,
                    np.full([tf.shape(llm_entities)[0]], p),
                    np.full([tf.shape(llm_entities)[0]], o),
                ])

        # Entity set to check whether groups are disjoint: anchor entity and llm entities
        entity_set = set(llm_entities + [pair[0]])

        units.append({
            'pair': pair,
            'observed_triples': tf.constant(observed_triples, tf.int32),
            'test_triples': tf.constant(test_triples, tf.int32),
            'llm_triples': tf.constant(llm_triples, tf.int32) if llm_pairs_and_ents is not None else [],
            'entity_set': entity_set # used for disjointness
        })

    grouped = []
    for unit in units:
        placed = False
        for g in grouped:
            if g['entity_set'].isdisjoint(unit['entity_set']) and (g['size'] + 1 <= max_pairs_per_group):
                # Place unit
                g['pairs'].append(unit['pair'])
                g['observed_counts'][unit['pair']] = unit['observed_triples'].shape[0]
                g['test_counts'][unit['pair']] = unit['test_triples'].shape[0]
                g['entity_set'].update(unit['entity_set'])
                g['observed_list'].append(unit['observed_triples'])
                g['test_list'].append(unit['test_triples'])
                g['llm_list'].append(unit['llm_triples'])
                g['size'] += 1
                placed = True
                break
        if placed:
            continue

        # New group
        grouped.append({
            'group_id': len(grouped),
            'pairs': [unit['pair']],
            'observed_counts': {unit['pair']: unit['observed_triples'].shape[0]},
            'test_counts': {unit['pair']: unit['test_triples'].shape[0]},
            'entity_set': set(unit['entity_set']),
            'observed_list': [unit['observed_triples']],
            'test_list': [unit['test_triples']],
            'llm_list': [unit['llm_triples']],
            'size': 1,
        })

    # Finalize structure
    finalized = []
    for g in grouped:
        observed_triples = tf.concat(g['observed_list'], axis=0)
        test_triples = tf.concat(g['test_list'], axis=0)
        llm_triples = tf.concat(g['llm_list'], axis=0)
        finalized.append({
            'group_id': g['group_id'],
            'pairs': g['pairs'],
            'observed_counts': g['observed_counts'],
            'test_counts': g['test_counts'],
            'size': g['size'],
            'test_triples': test_triples,
            'observed_triples': observed_triples,
            'llm_triples': llm_triples,
            'pre_ranks': None,
            'new_ranks': dict(),
        })

    # Let's deal with triples without observed triples
    no_support_batch = {
        'group_id': None,
        'pairs': np.concatenate([empty_batch['pairs'] for empty_batch in empty_batches]).tolist(),
        'test_counts': {k: v for empty_batch in empty_batches for k, v in empty_batch['test_counts'].items()},
        'test_triples': np.vstack([empty_batch['test_triples'] for empty_batch in empty_batches]),
        'observed_triples': None,
        'llm_triples': None,
        'pre_ranks': None,
    }

    return finalized, no_support_batch


def kge_latent_search(kge_model, train_df, valid_df, test_df, group='sp', args=None, early_stop=None):
    """
    This function implements the latent search. From the definition of the targets
    observed during training and the optimization of (subject, predicate) or
    (predicate, object), to recompute the rank for the target entities after each
    epoch of the latent search.
    """
    assert group in ['sp', 'po'], "Group must be 'sp' or 'po'!"

    train_grouped, valid_grouped, test_grouped = group_by_pair(
        train_df, valid_df, test_df, group=group
    )

    ent_dict = kge_model.data_indexer.backend.reversed_entities_dict
    rel_dict = kge_model.data_indexer.backend.reversed_relations_dict

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
        closest_ents = {
            (ent_dict[pair[0]], rel_dict[pair[1]]): [ent_dict[raw_ent] for raw_ent in llm_entities] \
            for pair, llm_entities in closest_ents.items()
        }

    batches, no_support_batch = generate_batch_of_batches(
        train_grouped, valid_grouped, test_grouped, group=group,
        llm_pairs_and_ents=closest_ents if args.llm_integration else None,
        max_pairs_per_group=args.batch_size
    )

    # Let's compute the ranks of those target triples for which we have no observed triples
    empty_batch_mask_test = build_filter(kge_model, no_support_batch, group, no_support=True)

    no_support_ranks = []
    for idx_subset in range(0, empty_batch_mask_test.shape[0], args.batch_size):
        if idx_subset + args.batch_size <= empty_batch_mask_test.shape[0]:
            subset_pairs = [pair for pair in no_support_batch['pairs'][idx_subset: idx_subset + args.batch_size]]
            subset_counts = {pair: counts for pair, counts in no_support_batch['test_counts'].items() if
                             list(pair) in subset_pairs}
            subset_triples = no_support_batch['test_triples'][idx_subset: idx_subset + args.batch_size]
            subset_mask = empty_batch_mask_test[idx_subset: idx_subset + args.batch_size]
        else:
            subset_pairs = [pair for pair in no_support_batch['pairs'][idx_subset:]]
            subset_counts = {pair: counts for pair, counts in no_support_batch['test_counts'].items() if
                             list(pair) in subset_pairs}
            subset_triples = no_support_batch['test_triples'][idx_subset:]
            subset_mask = empty_batch_mask_test[idx_subset:]
        subset_batch = {
            'pairs': subset_pairs, 'test_counts': subset_counts, 'test_triples': subset_triples
        }

        anchor_embs = kge_model.encoding_layer([pair[0] for pair in subset_batch['pairs']], type_of="e")
        relation_embs = kge_model.encoding_layer([pair[1] for pair in subset_batch['pairs']], type_of="r")

        subset_ranks = batched_evaluation(
            kge_model, subset_batch, anchor_embs, relation_embs,
            all_emb_matrices=tf.expand_dims(kge_model.encoding_layer.ent_emb, axis=0),
            full_mask=subset_mask, test_mask=subset_mask, group=group
        )
        no_support_ranks.append(subset_ranks)

    no_support_batch['pre_ranks'] = np.concatenate(no_support_ranks)

    # Create a copy of the embeddings before starting the optimization.
    # We will use this copy to reset the embeddings when switching batch
    initial_kge_embeddings = tf.identity(kge_model.encoding_layer.ent_emb)

    #TODO: put together the content of empty_batches and run one single evaluation for all of them.

    for batch in tqdm(batches, total=len(batches)):
        kge_model.encoding_layer.ent_emb = tf.identity(initial_kge_embeddings)

        # Define the filter we will use at test time
        batch_mask_full, batch_mask_test = build_filter(kge_model, batch, group)

        anchor_embs = kge_model.encoding_layer([pair[0] for pair in batch['pairs']], type_of="e")
        relation_embs = kge_model.encoding_layer([pair[1] for pair in batch['pairs']], type_of="r")

        filtered_pre_ranks = batched_evaluation(
            kge_model, batch, anchor_embs, relation_embs,
            all_emb_matrices=tf.expand_dims(kge_model.encoding_layer.ent_emb, axis=0),
            full_mask=batch_mask_full, test_mask=batch_mask_test, group=group
        )
        batch['pre_ranks'] = filtered_pre_ranks

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

        all_inputs = set_trainable_variables(
            kge_model, batch,
            group=group, freeze_rels=args.freeze_relations,
            train_additional_ents=args.tune_llm_ents
        )
        input_triples, trainable_params = all_inputs[0], all_inputs[1]
        if args.tune_llm_ents:
            additional_triples = all_inputs[2]

        for i in range(0, args.epochs):
            with tf.GradientTape() as tape:

                input_triples = repeat_and_gather(input_triples, batch, group=group, model="kge")
                inp_score = - kge_model.scoring_layer(input_triples)

                if args.tune_llm_ents:
                    if args.only_llm:
                        inp_score *= 0.0
                    additional_triples = repeat_and_gather(additional_triples, batch, group=group, model="llm")
                    inp_score = tf.concat([inp_score, - kge_model.scoring_layer(additional_triples)], axis=0)

            grads = tape.gradient(inp_score, trainable_params)
            optimizer.apply_gradients(zip(grads, trainable_params))

            updated_ents, updated_preds = get_updated_embeddings(
                trainable_params, batch, kge_model,
                freeze_rels=args.freeze_relations,
            )

            if group == 'sp':
                input_triples = [updated_ents, updated_preds, input_triples[2]]
                if args.tune_llm_ents:
                    additional_triples = [updated_ents, updated_preds, additional_triples[2]]
            else:
                input_triples = [input_triples[0], updated_preds, updated_ents]
                if args.tune_llm_ents:
                    additional_triples = [additional_triples[0], updated_preds, updated_ents]

            all_emb_matrices = get_embedding_matrices(
                kge_model, batch['pairs'], trainable_params, freeze_rels=args.freeze_relations,
                llm_ents=batch['llm_triples'][:, 2 if group == 'sp' else 0] if args.tune_llm_ents else None
            )

            filtered_new_ranks = batched_evaluation(
                kge_model, batch, updated_ents, updated_preds,
                all_emb_matrices=all_emb_matrices,
                full_mask=batch_mask_full, test_mask=batch_mask_test, group=group
            )

            batch['new_ranks'][i] = filtered_new_ranks

            # if early_stop and (i + 1) == early_stop_epoch:
            #     ranks[pair]["early_stop_new_ranks"] = filtered_new_ranks

    ls_test_triples = np.vstack([batch['test_triples'] for batch in batches])
    ls_pre_ranks = np.concatenate([batch['pre_ranks'] for batch in batches])
    ls_new_ranks = {epoch: np.concatenate([batch['new_ranks'][epoch] for batch in batches])
                    for epoch in batch['new_ranks']}

    all_test_triples = np.vstack([
        ls_test_triples, no_support_batch['test_triples']
    ])
    all_pre_ranks = np.concatenate([
        ls_pre_ranks, no_support_batch['pre_ranks']
    ])

    df_ranks = pd.DataFrame(
        np.concatenate([all_test_triples, all_pre_ranks.reshape(-1, 1)], axis=1),
        columns=['s', 'p', 'o','pre_ranks']
    )

    array_new_ranks = np.vstack(list(ls_new_ranks.values())).transpose()
    ls_df_new_ranks = pd.DataFrame(
        np.concatenate([ls_test_triples, array_new_ranks], axis=1),
        columns=['s', 'p', 'o'] + [f'ranks_epoch{i}' for i in range(args.epochs)]
    )

    df_results = df_ranks.merge(ls_df_new_ranks, on=['s', 'p', 'o'], how='left')

    return df_results

def build_filter(model, batch, group, no_support=None):
    """
    Function to build the filter for the batch.
    """
    test_targets = batch['test_triples'][:, 2 if group == 'sp' else 0]
    test_targets_per_pair = tf.split(test_targets, list(batch['test_counts'].values()))

    # Build a mask to filter the test targets.
    row_len_test = tf.constant(
        [len(pair_filter) for pair_filter in test_targets_per_pair],
        dtype=tf.int32
    )
    flat_rows_test = tf.repeat(tf.range(len(row_len_test), dtype=tf.int32), row_len_test)
    scatter_indices_test = tf.stack([flat_rows_test, test_targets], axis=1)
    updates_test = tf.ones_like(test_targets, dtype=tf.bool)
    mask_test = tf.scatter_nd(scatter_indices_test, updates_test, shape=(len(row_len_test), model.num_ents))

    if no_support:
        return mask_test

    # Build a mask for all entities to filter at evaluation (train, valid and test ones)
    observed_targets = batch['observed_triples'][:, 2 if group == 'sp' else 0]
    obs_targets_per_pair = tf.split(observed_targets, list(batch['observed_counts'].values()))

    to_filter_per_pair = [tf.concat([test_targets_per_pair[i], obs_targets_per_pair[i]], axis=0) \
                          for i in range(len(test_targets_per_pair))]

    row_lengths = tf.constant([len(pair_filter) for pair_filter in to_filter_per_pair], dtype=tf.int32)

    to_filter_per_pair = tf.concat(to_filter_per_pair, axis=0)
    flat_rows = tf.repeat(tf.range(len(test_targets_per_pair), dtype=tf.int32), row_lengths)

    scatter_indices = tf.stack([flat_rows, to_filter_per_pair], axis=1)

    updates = tf.ones_like(to_filter_per_pair, dtype=tf.bool)
    mask = tf.scatter_nd(scatter_indices, updates, shape=(len(test_targets_per_pair), model.num_ents))

    return mask, mask_test

def batched_evaluation(
        model, batch, anchor_embs, relation_embs, all_emb_matrices, full_mask, test_mask, group='sp'):
    """
    Function to implement the evaluation of the model on a batch. Importantly, we can't simply
    use the `ScoringBasedEmbeddingModel.evaluate` method as we need a different embedding matrix
    for each pair of the batch.
    """

    scores = model.scoring_layer([
        tf.expand_dims(anchor_embs, axis=1) if group == 'sp' else all_emb_matrices,
        tf.expand_dims(relation_embs, axis=1),
        tf.expand_dims(anchor_embs, axis=1) if group == 'po' else all_emb_matrices
    ]) # (#pairs, #ents)

    # Extract filtered ranks
    test_scores = tf.split(
        tf.boolean_mask(scores, test_mask), list(batch['test_counts'].values())
    )

    masked_scores = scores - tf.cast(full_mask, tf.float32) * 1e9

    filtered_ranks = []
    for i, pair_target_scores in enumerate(test_scores):
        filtered_rank = tf.reduce_sum(tf.cast(
            tf.less_equal(tf.expand_dims(pair_target_scores, axis=1), masked_scores[i]),
            tf.int32), axis=1
        ) + 1
        filtered_ranks.append(filtered_rank)

    return np.concatenate(filtered_ranks)


def build_masks(index_lists, length):
    """Return a tensor [num_lists, length] with 1s and inf at given indices."""
    masks = []
    for idxs in index_lists:
        mask = tf.ones([length], dtype=tf.float32)
        infs = tf.constant(np.inf, dtype=tf.float32, shape=[len(idxs)])
        mask = tf.tensor_scatter_nd_update(
            mask,
            indices=tf.expand_dims(tf.constant(idxs, dtype=tf.int32), axis=1),
            updates=infs
        )
        masks.append(mask)
    return tf.stack(masks, axis=1)

def get_embedding_matrices(model, pairs, trainable_params=None, freeze_rels=False, llm_ents=None):
    """
    Function to generate multiple embedding matrices, one for each test sample
    in the batch.
    """

    num_pairs = len(pairs)

    all_corruption_entities = tf.stack([tf.identity(model.encoding_layer.ent_emb)] * num_pairs, axis=0)
    replacements = trainable_params[0] # updated embeddings of the anchor
    # Indices to for the rows to replace in each of the slices of all_corruption_entities
    scatter_idxs = tf.stack([tf.range(num_pairs), [pair[0] for pair in pairs]], axis=1)

    all_corruption_entities = tf.tensor_scatter_nd_update(
        all_corruption_entities, scatter_idxs, replacements
    )

    if llm_ents is not None:
        num_ents_per_pair = len(llm_ents) // num_pairs

        llm_replacements = trainable_params[1] if freeze_rels else trainable_params[2]
        llm_scatter_idxs = tf.stack(
            [tf.repeat(tf.range(num_pairs), repeats=num_ents_per_pair), llm_ents],
            axis=1
        )
        all_corruption_entities = tf.tensor_scatter_nd_update(
            all_corruption_entities, llm_scatter_idxs, llm_replacements
        )

    return all_corruption_entities


def extract_kge_embeddings(kge_model, pairs, observed_triples, group='sp'):
    """
    Extract the subject, predicate and object embeddings from the pretrained KGE. Importantly,
    depending on whether we group by 'sp' or 'po' we are going to retrieve target embeddings
    for the subject or the object.
    """
    observed_targets = observed_triples[:, 2 if group == 'sp' else 0]
    if group == 'sp':
        subjects = kge_model.encoding_layer([pair[0] for pair in pairs], type_of="e")
        objects = kge_model.encoding_layer(
            observed_targets, type_of="e"
        )
    else:
        subjects = kge_model.encoding_layer(
            observed_targets, type_of="e"
        )
        objects = kge_model.encoding_layer([pair[0] for pair in pairs], type_of="e")

    predicates = kge_model.encoding_layer([pair[1] for pair in pairs], type_of="r")

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
        model, batch, group='sp', freeze_rels=None,
        train_additional_ents=None):
    """
    Extract the input triples and the trainable parameters after setting the state
    of the correct variables to trainable.
    """
    # if batch['llm_triples'] is not None and not train_additional_ents:
    #     targets = np.concatenate([targets, additional_ents])

    subjects, predicates, objects = extract_kge_embeddings(model, batch['pairs'], batch['observed_triples'], group)

    updated_pred = tf.Variable(predicates) if not freeze_rels else predicates

    if group == 'sp':
        updated_subj = tf.Variable(subjects)
        input_triples = [
            updated_subj,
            updated_pred,
            objects
        ]
        trainable_params = [updated_subj, updated_pred] if not freeze_rels else [updated_subj]
    else:
        updated_obj = tf.Variable(objects)
        input_triples = [
            subjects,
            updated_pred,
            updated_obj
        ]
        trainable_params = [updated_obj, updated_pred] if not freeze_rels else [updated_obj]

    # v0.7 - If we want to train the additional entities added with the LLM integration
    if train_additional_ents:
        assert batch['llm_triples'] is not None, "llm_triples must be provided if you want to optimize the target!"
        # We can make all the embeddings trainable as we know the LLM entities are unique.
        updated_additional_ents = tf.Variable(
            model.encoding_layer(batch['llm_triples'][:, 2 if group=='sp' else 0], type_of="e")
        )
        additional_triples = [
            updated_subj if group == 'sp' else updated_additional_ents,
            updated_pred,
            updated_obj if group == 'po' else updated_additional_ents
        ]
        trainable_params += [updated_additional_ents]

    if train_additional_ents:
        return input_triples, trainable_params, additional_triples

    return input_triples, trainable_params

def repeat_and_gather(triples, batch, group='sp', model="kge"):
    """

    """
    if model == "kge":
        repeating_idxs = tf.repeat(
            tf.range(len(batch['pairs'])),
            tf.constant([count for count in batch['observed_counts'].values()], tf.int32)
        )
    elif model == "llm":
        repeating_idxs = tf.repeat(
            tf.range(len(batch['pairs'])),
            tf.fill([len(batch['pairs'])], batch['llm_triples'].shape[0] // len(batch['pairs']))
        )
    subjects, predicates, objects = triples
    triples = [
        tf.gather(subjects, repeating_idxs) if group == 'sp' else subjects,
        tf.gather(predicates, repeating_idxs),
        tf.gather(objects, repeating_idxs) if group == 'po' else objects
    ]
    return triples


def get_updated_embeddings(
        trainable_params, batch, model, freeze_rels=None
):
    """
    Extract, from all the trainable variables, just those for the anchor embeddings that we
    want to update. Typically, when group=='sp', the anchor embeddings are that of the subject
    and that of the relation.
    """
    updated_ent = trainable_params[0]
    updated_pred = trainable_params[1] if not freeze_rels \
        else model.encoding_layer([pair[1] for pair in batch['pairs']], type_of="r")

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
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--only-llm', action='store_true', default=None)

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

    ent_dict = kge_model.data_indexer.backend.reversed_entities_dict
    rel_dict = kge_model.data_indexer.backend.reversed_relations_dict

    rev_ent_dict = kge_model.data_indexer.backend.entities_dict
    rev_rel_dict = kge_model.data_indexer.backend.relations_dict

    train_df, valid_df, test_df = map_entities_dataset(train_df, valid_df, test_df, ent_dict, rel_dict)

    output_path = os.path.join(
        args.output_folder, args.dataset,
        f"{args.scoring_type}_lr{args.learning_rate}"
    )
    if args.seed != 0:
        output_path += f"_seed{args.seed}"
    if args.only_llm:
        output_path += "_only_llm"

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
            test_df=valid_df, group='sp', args=args
        )
        # valid_ranks_sp.to_csv(os.path.join(output_path, "valid_ranks_sp.csv"), index=False)

        valid_ranks_sp[['s', 'o']] = valid_ranks_sp[['s', 'o']].replace(rev_ent_dict)
        valid_ranks_sp[['p']] = valid_ranks_sp[['p']].replace(rev_rel_dict)

        grouped_ranks = valid_ranks_sp.groupby(['s', 'p'])
        valid_ranks_sp_dict = dict()
        for pair in grouped_ranks.groups:
            valid_ranks_sp_dict[pair] = {
                'targets': grouped_ranks.get_group(pair)['o'].values,
                'pre_ranks': grouped_ranks.get_group(pair)['pre_ranks'].values,
            }
            if grouped_ranks.get_group(pair)['ranks_epoch0'].isna().sum():
                valid_ranks_sp_dict[pair]['new_ranks'] = dict()
            else:
                valid_ranks_sp_dict[pair]['new_ranks'] = {
                    i: grouped_ranks.get_group(pair)[f'ranks_epoch{i}'].values for i in range(args.epochs)
                }

        with open(os.path.join(output_path, "valid_ranks_sp.pkl"), "wb") as f:
            pickle.dump(valid_ranks_sp_dict, f)

        # We use an empty df as a placeholder for the missing validation set as we are using it as test here.
        valid_ranks_po = kge_latent_search(
            kge_model, train_df=train_df,
            valid_df=pd.DataFrame([], columns=['s', 'p', 'o']),
            test_df=valid_df, group='po', args=args
        )

        # valid_ranks_po.to_csv(os.path.join(output_path, "valid_ranks_po.csv"), index=False)

        valid_ranks_po[['s', 'o']] = valid_ranks_po[['s', 'o']].replace(rev_ent_dict)
        valid_ranks_po[['p']] = valid_ranks_po[['p']].replace(rev_rel_dict)

        grouped_ranks = valid_ranks_po.groupby(['o', 'p'])
        valid_ranks_po_dict = dict()
        for pair in grouped_ranks.groups:
            valid_ranks_po_dict[pair] = {
                'targets': grouped_ranks.get_group(pair)['s'].values,
                'pre_ranks': grouped_ranks.get_group(pair)['pre_ranks'].values,
            }
            if grouped_ranks.get_group(pair)['ranks_epoch0'].isna().sum():
                valid_ranks_po_dict[pair]['new_ranks'] = dict()
            else:
                valid_ranks_po_dict[pair]['new_ranks'] = {
                    i: grouped_ranks.get_group(pair)[f'ranks_epoch{i}'].values for i in range(args.epochs)
                }

        with open(os.path.join(output_path, "valid_ranks_po.pkl"), "wb") as f:
            pickle.dump(valid_ranks_po_dict, f)

        # early_stopper = EarlyStopper(
        #     valid_ranks_sp_dict, valid_ranks_po_dict, valid_df, thresholds=thresholds,
        #     degrees=pd.concat([train_df.s, train_df.o]).value_counts()
        # )

    # if args.debug_valid:
    #     with open(os.path.join(output_path, "valid_ranks_po.pkl"), "rb") as f:
    #         valid_ranks_po = pickle.load(f)
    #
    #     with open(os.path.join(output_path, "valid_ranks_sp.pkl"), "rb") as f:
    #         valid_ranks_sp = pickle.load(f)
    #
    #     early_stopper = EarlyStopper(
    #         valid_ranks_sp, valid_ranks_po, valid_df, thresholds=thresholds,
    #         degrees=pd.concat([train_df.s, train_df.o]).value_counts()
    #     )

    if args.run_on_test:
        ranks_sp = kge_latent_search(
            kge_model, train_df, valid_df, test_df, group='sp', args=args,
            early_stop=early_stopper
        )
        # ranks_sp.to_csv(os.path.join(output_path, "ranks_sp.csv"), index=False)

        ranks_sp[['s', 'o']] = ranks_sp[['s', 'o']].replace(rev_ent_dict)
        ranks_sp[['p']] = ranks_sp[['p']].replace(rev_rel_dict)

        grouped_ranks = ranks_sp.groupby(['s', 'p'])
        ranks_sp_dict = dict()
        for pair in grouped_ranks.groups:
            ranks_sp_dict[pair] = {
                'targets': grouped_ranks.get_group(pair)['o'].values,
                'pre_ranks': grouped_ranks.get_group(pair)['pre_ranks'].values,
            }
            if grouped_ranks.get_group(pair)['ranks_epoch0'].isna().sum():
                ranks_sp_dict[pair]['new_ranks'] = dict()
            else:
                ranks_sp_dict[pair]['new_ranks'] =  {
                    i: grouped_ranks.get_group(pair)[f'ranks_epoch{i}'].values for i in range(args.epochs)
                }

        with open(os.path.join(output_path, "ranks_sp.pkl"), "wb") as f:
            pickle.dump(ranks_sp_dict, f)

        ranks_po = kge_latent_search(
            kge_model, train_df, valid_df, test_df, group='po', args=args,
            early_stop=early_stopper
        )
        ranks_po.to_csv(os.path.join(output_path, "ranks_po.csv"), index=False)

        ranks_po[['s', 'o']] = ranks_po[['s', 'o']].replace(rev_ent_dict)
        ranks_po[['p']] = ranks_po[['p']].replace(rev_rel_dict)

        grouped_ranks = ranks_po.groupby(['o', 'p'])
        ranks_po_dict = dict()
        for pair in grouped_ranks.groups:
            ranks_po_dict[pair] = {
                'targets': grouped_ranks.get_group(pair)['s'].values,
                'pre_ranks': grouped_ranks.get_group(pair)['pre_ranks'].values,
            }
            if grouped_ranks.get_group(pair)['ranks_epoch0'].isna().sum():
                ranks_po_dict[pair]['new_ranks'] = dict()
            else:
                ranks_po_dict[pair]['new_ranks'] = {
                    i: grouped_ranks.get_group(pair)[f'ranks_epoch{i}'].values for i in range(args.epochs)
                }

        with open(os.path.join(output_path, "ranks_po.pkl"), "wb") as f:
            pickle.dump(ranks_po_dict, f)
