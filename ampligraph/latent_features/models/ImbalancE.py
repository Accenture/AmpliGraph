"""
This module includes the implementation of ImbalancE, an inference time latent search system
designed to extend the evaluation of KGE models on imbalanced triples.
"""
import pickle
from tqdm import tqdm
import tensorflow as tf

from experiments.LatentSearch.code.utils_KGE_latent_search import *

class ImbalancE():

    def __init__(
            self,
            kge_model,
            train_df,
            valid_df,
            test_df,
            group='sp',
            args=None,
            early_stop=None,
            regularization=None
    ):
        train_grouped, valid_grouped, test_grouped = group_by_pair(
            train_df, valid_df, test_df, group=group
        )
        self.train_grouped = train_grouped
        self.valid_grouped = valid_grouped
        self.test_grouped = test_grouped

        self.kge_model = kge_model

        assert group in ['sp', 'po'], "Group must be 'sp' or 'po'!"
        self.group = group

        self.args = args
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs

        self.early_stop = early_stop

        self.regularization = regularization

        self.tune_llm_ents = args.tune_llm_ents
        if self.args.llm_integration:
            if len(valid_df) > 0:
                path_closest = args.llm_integration
            else:
                # If we are considering evaluating on the validation set instead of the test set,
                # we need to select the file that contains the neighbours for the pairs
                # extracted from the validation set.
                folder_neighbors = os.path.dirname(args.llm_integration)
                file_neighbors = os.path.basename(args.llm_integration)
                path_closest = os.path.join(
                    folder_neighbors, "validation_" + file_neighbors
                )

            with open(path_closest, "rb") as f:
                closest_ents = pickle.load(f)[group]

            ent_dict = self.kge_model.data_indexer.backend.reversed_entities_dict
            rel_dict = self.kge_model.data_indexer.backend.reversed_relations_dict

            self.closest_ents = {
                (ent_dict[pair[0]], rel_dict[pair[1]]): [ent_dict[raw_ent] for raw_ent in llm_entities] \
                for pair, llm_entities in closest_ents.items()
            }

    def generate_batch_of_batches(
            self, max_pairs_per_group=10
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
        target_col = 'o' if self.group == 'sp' else 's'

        units = []
        empty_batches = []
        for pair in self.test_grouped.groups:
            # Observed targets from train + valid
            try:
                train_targets = self.train_grouped.get_group(pair)[target_col].values
            except KeyError:
                train_targets = np.array([], dtype=np.int32)
            try:
                valid_targets = self.valid_grouped.get_group(pair)[target_col].values
            except KeyError:
                valid_targets = np.array([], dtype=np.int32)
            observed_targets = np.unique(np.concatenate([train_targets, valid_targets]))

            # Test targets (from test set)
            test_targets = np.sort(self.test_grouped.get_group(pair)[target_col].values)

            # Skip if no training context
            if len(observed_targets) == 0:
                if self.group == 'sp':
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
            if self.args.llm_integration is None:
                llm_entities = []
            else:
                llm_entities = self.closest_ents[pair]
                assert len(set(llm_entities)) == len(llm_entities), \
                    "LLM entities should be unique for the same pair. Did you mess up?"

            # Build triples to associate with each pair
            if self.group == 'sp':
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
                if self.args.llm_integration is not None:
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
                if self.args.llm_integration is not None:
                    llm_triples = np.column_stack([
                        llm_entities,
                        np.full([tf.shape(llm_entities)[0]], p),
                        np.full([tf.shape(llm_entities)[0]], o),
                    ])

            # Entity set to check whether groups are disjoint: anchor entity and llm entities.
            # Indeed, these are the entities that are going to be optimized, so they need to be
            # unique across different pairs to avoid the optimization of one pair interfering with another
            entity_set = set(llm_entities + [pair[0]])

            units.append({
                'pair': pair,
                'observed_triples': tf.constant(observed_triples, tf.int32),
                'test_triples': tf.constant(test_triples, tf.int32),
                'llm_triples': tf.constant(llm_triples, tf.int32) if self.args.llm_integration is not None else [],
                'entity_set': entity_set  # used for disjointness
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

    def build_filter(self, batch, no_support=None):
        """
        Function to build a mask that is used to filter test targets for the batch when
        computing the ranking, so that we exclude train, valid and test triples (so called
        filtered evaluation setting).
        Parameters
        ----------
        batch : Dict
            Dictionary storing, for each batch of pairs, the training context, test targets...
        no_support : bool, optionally
            Whether the batch we are considering contains triples without training context.
        Returns
        -------
        mask : tf.Tensor
            Mask to filter only test targets.
        mask_test : tf.Tensor
            Mask to filter train, valid and test targets when computing the ranking.
        """
        test_targets = batch['test_triples'][:, 2 if self.group == 'sp' else 0]
        test_targets_per_pair = tf.split(test_targets, list(batch['test_counts'].values()))

        # Build a mask to filter the test targets.
        row_len_test = tf.constant(
            [len(pair_filter) for pair_filter in test_targets_per_pair],
            dtype=tf.int32
        )
        flat_rows_test = tf.repeat(tf.range(len(row_len_test), dtype=tf.int32), row_len_test)
        scatter_indices_test = tf.stack([flat_rows_test, test_targets], axis=1)
        updates_test = tf.ones_like(test_targets, dtype=tf.bool)
        mask_test = tf.scatter_nd(
            scatter_indices_test, updates_test,
            shape=(len(row_len_test), self.kge_model.num_ents)
        )

        if no_support:
            return mask_test

        # Build a mask for all entities to filter at evaluation (train, valid and test ones)
        observed_targets = batch['observed_triples'][:, 2 if self.group == 'sp' else 0]
        obs_targets_per_pair = tf.split(observed_targets, list(batch['observed_counts'].values()))

        to_filter_per_pair = [tf.concat([test_targets_per_pair[i], obs_targets_per_pair[i]], axis=0) \
                              for i in range(len(test_targets_per_pair))]

        row_lengths = tf.constant([len(pair_filter) for pair_filter in to_filter_per_pair], dtype=tf.int32)

        to_filter_per_pair = tf.concat(to_filter_per_pair, axis=0)
        flat_rows = tf.repeat(tf.range(len(test_targets_per_pair), dtype=tf.int32), row_lengths)

        scatter_indices = tf.stack([flat_rows, to_filter_per_pair], axis=1)

        updates = tf.ones_like(to_filter_per_pair, dtype=tf.bool)
        mask = tf.scatter_nd(
            scatter_indices, updates,
            shape=(len(test_targets_per_pair), self.kge_model.num_ents)
        )

        return mask, mask_test

    def compute_no_support_ranks(self, no_support_batch, empty_batch_mask_test):
        """
        Method to compute the ranks for triples without training context. Since we group them
        together, they might be bigger than the batch size, so we need to batch the evaluation
        manually.
        """

        no_support_ranks = []
        # We need to batch evaluation since we might have more triples than the batch size.
        for idx_subset in range(0, empty_batch_mask_test.shape[0], self.batch_size):
            if idx_subset + self.batch_size <= empty_batch_mask_test.shape[0]:
                subset_pairs = [pair for pair in no_support_batch['pairs'][idx_subset: idx_subset + self.batch_size]]
                subset_counts = {pair: counts for pair, counts in no_support_batch['test_counts'].items() if
                                 list(pair) in subset_pairs}
                subset_triples = no_support_batch['test_triples'][idx_subset: idx_subset + self.batch_size]
                subset_mask = empty_batch_mask_test[idx_subset: idx_subset + self.batch_size]
            else:
                subset_pairs = [pair for pair in no_support_batch['pairs'][idx_subset:]]
                subset_counts = {pair: counts for pair, counts in no_support_batch['test_counts'].items() if
                                 list(pair) in subset_pairs}
                subset_triples = no_support_batch['test_triples'][idx_subset:]
                subset_mask = empty_batch_mask_test[idx_subset:]
            subset_batch = {
                'pairs': subset_pairs, 'test_counts': subset_counts, 'test_triples': subset_triples
            }

            subset_ranks = self.compute_pretrained_ranks(
                subset_batch, full_mask=subset_mask, test_mask=subset_mask
            )

            no_support_ranks.append(subset_ranks)

        return np.concatenate(no_support_ranks)


    def compute_pretrained_ranks(self, batch, full_mask, test_mask):
        """
        Compute the test ranks using the pretrained KGE embeddings.
        """
        # We consider the embeddings of the pretrained KGE model as no latent search is going
        # to be performed since we have no training context.
        anchor_embs = self.kge_model.get_embeddings(
            [pair[0] for pair in batch['pairs']], embedding_type="e", from_idx=True
        )
        relation_embs = self.kge_model.get_embeddings(
            [pair[1] for pair in batch['pairs']], embedding_type="e", from_idx=True
        )

        ranks = self.batched_evaluation(
            batch, anchor_embs, relation_embs,
            all_emb_matrices=tf.expand_dims(self.kge_model.encoding_layer.ent_emb, axis=0),
            full_mask=full_mask, test_mask=test_mask
        )
        return ranks

    def batched_evaluation(
            self, batch, anchor_embs, relation_embs, all_emb_matrices, full_mask, test_mask):
        """
        Function to implement the evaluation of the model on a batch. Importantly, we can't simply
        use the `~ScoringBasedEmbeddingModel.evaluate` method as we need a different embedding matrix
        for each pair of the batch, since the latent search alters embeddings for each pair
        that are left untouched for all the other pairs in the batch.
        """

        scores = self.kge_model.scoring_layer([
            tf.expand_dims(anchor_embs, axis=1) if self.group == 'sp' else all_emb_matrices,
            tf.expand_dims(relation_embs, axis=1),
            tf.expand_dims(anchor_embs, axis=1) if self.group == 'po' else all_emb_matrices
        ])  # (#pairs, #ents)

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

    def extract_kge_embeddings(self, pairs, observed_triples):
        """
        Extract the subject, predicate and object embeddings from the pretrained KGE. Importantly,
        depending on whether we group by 'sp' or 'po' we are going to retrieve target embeddings
        for the subject or the object.
        """
        observed_targets = observed_triples[:, 2 if self.group == 'sp' else 0]
        if self.group == 'sp':
            subjects = self.kge_model.get_embeddings([pair[0] for pair in pairs], embedding_type="e", from_idx=True)
            objects = self.kge_model.get_embeddings(
                observed_targets, embedding_type="e", from_idx=True
            )
        else:
            subjects = self.kge_model.get_embeddings(
                observed_targets, embedding_type="e", from_idx=True
            )
            objects = self.kge_model.get_embeddings([pair[0] for pair in pairs], embedding_type="e", from_idx=True)

        predicates = self.kge_model.get_embeddings([pair[1] for pair in pairs], embedding_type="r", from_idx=True)

        return tf.constant(subjects), tf.constant(predicates), tf.constant(objects)

    def set_trainable_variables(
            self, batch, freeze_rels=None,
        ):
        """
        Extract the input triples and the trainable parameters after setting the state
        of the correct variables to trainable.
        """
        subjects, predicates, objects = self.extract_kge_embeddings(
            batch['pairs'], batch['observed_triples']
        )

        updated_pred = tf.Variable(predicates) if not freeze_rels else predicates

        if self.group == 'sp':
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

        # If we want to train the additional entities added with the LLM integration
        if self.tune_llm_ents:
            assert batch['llm_triples'] is not None, "llm_triples must be provided if you want to optimize the target!"
            # We can make all the embeddings trainable as we know the LLM entities are unique.
            updated_additional_ents = tf.Variable(
                self.kge_model.get_embeddings(
                    batch['llm_triples'][:, 2 if self.group == 'sp' else 0],
                    embedding_type="e", from_idx=True
                )
            )
            additional_triples = [
                updated_subj if self.group == 'sp' else updated_additional_ents,
                updated_pred,
                updated_obj if self.group == 'po' else updated_additional_ents
            ]
            trainable_params += [updated_additional_ents]

            return input_triples, trainable_params, additional_triples

        return input_triples, trainable_params

    def repeat_and_gather(self, triples, batch, model="kge"):
        """
        Method that generates a full batch of triples by repeating the anchor and relation of
        the pair and stacking on top the targets of interest, being them those coming from
        the training context (`model="kge"`) or the LLM-generated triples (`model="llm"`).
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
            tf.gather(subjects, repeating_idxs) if self.group == 'sp' else subjects,
            tf.gather(predicates, repeating_idxs),
            tf.gather(objects, repeating_idxs) if self.group == 'po' else objects
        ]
        return triples

    def get_embedding_matrices(self, batch, trainable_params=None, freeze_rels=False):
        """
        Function to generate multiple embedding matrices, one for each test sample
        in the batch.
        """
        pairs = batch['pairs']

        num_pairs = len(pairs)

        all_corruption_entities = tf.stack([tf.identity(self.kge_model.encoding_layer.ent_emb)] * num_pairs, axis=0)
        replacements = trainable_params[0]  # updated embeddings of the anchor
        # Indices to for the rows to replace in each of the slices of all_corruption_entities
        scatter_idxs = tf.stack([tf.range(num_pairs), [pair[0] for pair in pairs]], axis=1)

        all_corruption_entities = tf.tensor_scatter_nd_update(
            all_corruption_entities, scatter_idxs, replacements
        )

        if self.tune_llm_ents is not None:
            llm_ents = batch['llm_triples'][:, 2 if self.group == 'sp' else 0]
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

    def get_updated_embeddings(
            self, trainable_params, batch, freeze_rels=None
    ):
        """
        Extract, from all the trainable variables, just those for the anchor embeddings that we
        want to update. Typically, when group=='sp', the anchor embeddings are that of the subject
        and that of the relation.
        """
        updated_ent = trainable_params[0]
        updated_pred = trainable_params[1] if not freeze_rels \
            else self.kge_model.get_embeddings(
            [pair[1] for pair in batch['pairs']],
            embedding_type="e", from_idx=True
        )

        return updated_ent, updated_pred

    def compute_regularization_term(
            self, batch, anchor_embs, initial_kge_embeddings, additional_triples=None
    ):
        """
        Method to compute the regularization term.

        The idea of the regularization term is to keep the updated embeddings somewhat
        close to the original ones, so to prevent degenerate behaviours in which the
        embeddings collapse.
        """
        anchor_initial_embs = tf.gather(
            initial_kge_embeddings, [pair[0] for pair in batch['pairs']]
        )
        reg_score = tf.reduce_mean(tf.square(anchor_embs - anchor_initial_embs), axis=1)

        if additional_triples is not None:
            # We extract the embeddings of the target-variable
            oracle_embs = additional_triples[2] if self.group == 'sp' else additional_triples[0]
            oracle_initial_embs = tf.gather(
                initial_kge_embeddings, batch['llm_triples'][:, 2 if self.group == 'sp' else 0]
            )

            oracle_reg = tf.reduce_mean(tf.square(oracle_embs - oracle_initial_embs), axis=1)

            reg_score = tf.concat([reg_score, oracle_reg], axis=0)

        return reg_score

    def perform_latent_search(self):
        """
        Method to perform the latent search.
        """
        batches, no_support_batch = self.generate_batch_of_batches(
            max_pairs_per_group=self.batch_size
        )

        # Let's compute the ranks of those target triples for which we have no observed triples
        empty_batch_mask_test = self.build_filter(batch=no_support_batch, no_support=True)
        no_support_batch['pre_ranks'] = self.compute_no_support_ranks(no_support_batch, empty_batch_mask_test)

        # Create a copy of the embeddings before starting the optimization.
        # We will use this copy to reset the embeddings when switching batch
        initial_kge_embeddings = tf.identity(self.kge_model.encoding_layer.ent_emb)

        for batch in tqdm(batches, total=len(batches)):
            # We reset the embeddings to the pretrained ones before starting the optimization.
            self.kge_model.encoding_layer.ent_emb = tf.identity(initial_kge_embeddings)

            # Define the filter we will use at test time
            batch_mask_full, batch_mask_test = self.build_filter(batch)
            # Compute the ranks of test triples before (pre) optimization
            batch['pre_ranks'] = self.compute_pretrained_ranks(
                batch, full_mask=batch_mask_full, test_mask=batch_mask_test
            )

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            all_inputs = self.set_trainable_variables(
                batch, freeze_rels=self.args.freeze_relations,
            )
            input_triples, trainable_params = all_inputs[0], all_inputs[1]
            if self.tune_llm_ents:
                additional_triples = all_inputs[2]

            for i in range(0, self.epochs):
                with tf.GradientTape() as tape:
                    # Score the triples
                    input_triples = self.repeat_and_gather(
                        input_triples, batch, model="kge"
                    )
                    inp_score = - self.kge_model.scoring_layer(input_triples)

                    if self.tune_llm_ents:
                        if self.args.only_llm:
                            # We only consider the oracle triples for the optimization,
                            # neglecting the training context
                            inp_score *= 0.0
                        additional_triples = self.repeat_and_gather(additional_triples, batch, model="llm")
                        inp_score = tf.concat(
                            [inp_score, - self.kge_model.scoring_layer(additional_triples)], axis=0
                        )

                    if self.regularization is not None:
                        reg_scores = self.compute_regularization_term(
                            batch, anchor_embs=trainable_params[0],
                            initial_kge_embeddings=initial_kge_embeddings,
                            additional_triples=None if not self.tune_llm_ents else additional_triples
                        )
                        inp_score = tf.concat(
                            [inp_score, self.regularization * reg_scores], axis=0
                        )

                grads = tape.gradient(inp_score, trainable_params)
                optimizer.apply_gradients(zip(grads, trainable_params))

                updated_ents, updated_preds = self.get_updated_embeddings(
                    trainable_params, batch,
                    freeze_rels=self.args.freeze_relations,
                )

                # We update the embeddings of the input triples after the optimization
                if self.group == 'sp':
                    input_triples = [updated_ents, updated_preds, input_triples[2]]
                    if self.tune_llm_ents:
                        additional_triples = [updated_ents, updated_preds, additional_triples[2]]
                else:
                    input_triples = [input_triples[0], updated_preds, updated_ents]
                    if self.args.tune_llm_ents:
                        additional_triples = [additional_triples[0], updated_preds, updated_ents]

                # We extract embedding matrices with the updated embeddings
                all_emb_matrices = self.get_embedding_matrices(
                    batch, trainable_params, freeze_rels=self.args.freeze_relations,
                )

                # We compute the new ranks for test triples after this epoch of optimization.
                # IMPORTANT: this is the main bottleneck of the optimization, as, for convenience,
                # we compute the new ranks at all epochs. However, we could better integrate the
                # early stopper in the implementation so that such evaluation is required only
                # for the early stopping epoch.
                filtered_new_ranks = self.batched_evaluation(
                    batch, updated_ents, updated_preds,
                    all_emb_matrices=all_emb_matrices,
                    full_mask=batch_mask_full, test_mask=batch_mask_test
                )

                batch['new_ranks'][i] = filtered_new_ranks

                # if early_stop and (i + 1) == early_stop_epoch:
                #     ranks[pair]["early_stop_new_ranks"] = filtered_new_ranks

        return batches, no_support_batch


class EarlyStopper():
    """
    Class to model our Early Stopper.
    """
    def __init__(
            self, valid_ranks_sp, valid_ranks_po,
            valid_df, thresholds, degrees
    ):
        self.thresholds = sorted(thresholds)
        self.epochs = self.create_early_stopping_epochs(
            valid_ranks_sp, valid_ranks_po, valid_df,
            thresholds=thresholds, degrees=degrees
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


    def create_early_stopping_epochs(self, sp_ranks, po_ranks, valid, thresholds, degrees, metric="Hits@10"):
        """
        Function to create a dictionary that splits validation triples based on the degree
        of the subject and object and it associates, to each of these groups the optimal
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
