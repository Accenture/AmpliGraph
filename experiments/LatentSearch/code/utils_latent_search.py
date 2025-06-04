import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf


def build_training_set(train_df, neighbourhood=False):
    """
    Given a dataframe with columns ['s', 'p', 'o'], generate training examples:
    - Each example is a set of (s,p,o) triples excluding one object
    - The label is the excluded object
    """
    training_data = []

    if neighbourhood:

        s_group = train_df.groupby("s")
        o_group = train_df.groupby("o")

        entities = pd.concat([train_df.s, train_df.o]).unique()
        for entity in tqdm(entities, total=len(entities), desc="Creating Training Set..."):
            ent_triples_s = s_group.get_group(entity) if entity in s_group.groups \
                else pd.DataFrame(columns=train_df.columns)
            ent_triples_o = o_group.get_group(entity) if entity in o_group.groups \
                else pd.DataFrame(columns=train_df.columns)

            # If we have an entity of degree one, then we cannot create a training instance for it.
            # However, the entity is going to show up in the input_triples of the entity it is linked to.
            if len(ent_triples_s) + len(ent_triples_o) < 2:
                continue

            for idx, (s, p, o) in ent_triples_s.iterrows():
                input_triples_s = ent_triples_s.drop(index=idx, inplace=False).values
                if len(input_triples_s) < 1:
                    continue
                input_triples = np.concatenate(
                    [input_triples_s, ent_triples_o.values], axis=0
                )
                aux_targets = np.concatenate(
                    [input_triples_s[:, 2], ent_triples_o.values[:, 0]], axis=-1
                )
                training_data.append((input_triples, o, aux_targets, [s, p, o]))
            for idx, (s, p, o) in ent_triples_o.iterrows():
                input_triples_o = ent_triples_o.drop(index=idx, inplace=False).values
                if len(input_triples_o) < 1:
                    continue
                input_triples = np.concatenate(
                    [ent_triples_s.values, input_triples_o], axis=0
                )
                aux_targets = np.concatenate(
                    [ent_triples_s.values[:, 2], input_triples_o[:, 0]], axis=-1
                )
                training_data.append((input_triples, s, aux_targets, [s, p, o]))

        return training_data

    grouped_sp = train_df.groupby(['s', 'p'])['o'].apply(list)
    grouped_po = train_df.groupby(['o', 'p'])['s'].apply(list)

    for (s, p), objects in grouped_sp.items():
        if len(objects) < 2:
            continue  # Need at least 2 to hold out one as label

        for i in range(len(objects)):
            label = objects[i]
            input_triples = np.array(
                [(s, p, o) for j, o in enumerate(objects) if j != i], dtype=np.int32
            )
            training_data.append((input_triples, label, input_triples[:, 2], [s, p, label]))

    for (o, p), subjects in grouped_po.items():
        if len(subjects) < 2:
            continue  # Need at least 2 to hold out one as label

        for i in range(len(subjects)):
            label = subjects[i]
            input_triples = np.array(
                [(s, p, o) for j, o in enumerate(subjects) if j != i],
                dtype=np.int32
            )
            training_data.append((input_triples, label, input_triples[:, 0], [label, p, o]))

    return training_data

def build_test_set(test_df, train_df, corrupt="o", neighbourhood=False):
    """
    Given a dataframe with columns ['s', 'p', 'o'], generate training examples:
    - Each example is a set of (s,p,o) triples excluding one object
    - The label is the excluded object
    """
    testing_data = []

    if neighbourhood:

        s_group = train_df.groupby("s")
        o_group = train_df.groupby("o")

        entities = pd.concat([test_df.s, test_df.o]).unique()

        for entity in tqdm(entities, total=len(entities),
           desc=f"Creating Test Set Corrupting {'object' if corrupt == 'o' else 'subject'}..."
        ):
            ent_triples_s = s_group.get_group(entity) if entity in s_group.groups \
                else pd.DataFrame(columns=train_df.columns)
            ent_triples_o = o_group.get_group(entity) if entity in o_group.groups \
                else pd.DataFrame(columns=train_df.columns)

            input_triples = np.concatenate(
                [ent_triples_s.values, ent_triples_o.values], axis=0
            )
            aux_targets = np.concatenate(
                [ent_triples_s.values[:, 2], ent_triples_o.values[:, 0]], axis=-1
            )
            if corrupt == "o": # and (test_df.s == entity).sum():
                for s, p, o in test_df[test_df.s == entity].values:
                    testing_data.append((input_triples, o, aux_targets, [s, p, o]))

            if corrupt == "s":
                for s, p, o in test_df[test_df.o == entity].values:
                    testing_data.append((input_triples, s, aux_targets, [s, p, o]))

        return testing_data

    if corrupt == "o":
        grouped = train_df.groupby(['s', 'p'])['o'].apply(list).to_frame()
    elif corrupt == "s":
        grouped = train_df.groupby(['o', 'p'])['s'].apply(list).to_frame()

    missing_triples = 0
    for s, p, o in test_df.values:
        if corrupt == "o":
            try:
                objects = grouped.loc[(s, p), "o"]
                input_triples = np.array(
                    [(s, p, obj) for obj in objects], dtype=np.int32
                )
                testing_data.append((input_triples, o, input_triples[:, 2], [s, p, o]))
            except:
                # print(f"No examples of (subject, predicate) = {(s, p)} in the provided test set!")
                missing_triples += 1
        elif corrupt == "s":
            try:
                subjects = grouped.loc[(o, p), "s"]
                input_triples = np.array(
                    [(subj, p, o) for subj in subjects], dtype=np.int32
                )
                testing_data.append((input_triples, s, input_triples[:, 0], [s, p, o]))
            except:
                # print(f"No examples of (predicate, object) = {(p, o)} in the provided test set!")
                missing_triples += 1
    print(f"Couldn't find common {'(s, p)' if corrupt == 'o' else '(p, o)'} in the training set for {missing_triples} triples!")
    return testing_data

def build_tf_dataset(dataset):
    """
    Parameters
    -----------
    dataset: list
    """
    return tf.data.Dataset.from_generator(
        lambda: iter(dataset),
        output_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.int32),  # triples
            tf.TensorSpec(shape=(), dtype=tf.int32),  # target idx
            tf.TensorSpec(shape=(None), dtype=tf.int32),
            tf.TensorSpec(shape=(3,), dtype=tf.int32)
        )
    )

def truncate_triples(triples, target, aux_target, target_triple, pad_shape):
    """
    Function to truncate triples as `padded_batch` does not support truncation.
    """
    shuffled_idx = tf.random.shuffle(tf.range(tf.shape(triples)[0]))

    triples = tf.gather(triples, shuffled_idx)
    aux_target = tf.gather(aux_target, shuffled_idx)

    return triples[:pad_shape], target, aux_target[:pad_shape], target_triple

def build_datasets(df_train, df_valid, df_test, batch_size=1, quantile=0.9, evaluate=True, neighbourhood=False):
    """
    Build the training set for the latent search feature.

    The dataset will include, for every triple in the original training set,
    a batch of triples sharing (subject, predicate) with the triple and having
    as label the object or, conversely, sharing (predicate, object) and having
    as label the subject.

    Parameters
    ----------
    df_train: pd.DataFrame
        Training triples in the Knowledge Graph.
    df_valid: pd.DataFrame
        Validation triples in the Knowledge Graph.
    df_test: pd.DataFrame
        Test triples in the Knowledge Graph.
    batch_size: int
        Number of triples we want to process in each batch, i.e., number of pairs
        (triples, label) in each batch.
    quantile: float
        As we might have a different number of triples that share (subject, predicate)
        or (predicate, object) for every training triple, we pad triples so to have
        the same size of the arrays generated. This parameter specifies the number of
        triples to consider based on the distribution of the length of common triples.

    """
    train = build_training_set(df_train, neighbourhood=neighbourhood)
    pad_shape = int(np.quantile([el[0].shape[0] for el in train], quantile))

    train_ds = build_tf_dataset(train).shuffle(
        buffer_size=10000
    ).map(
        lambda x, y, w, z: truncate_triples(x, y, w, z, pad_shape)
    ).padded_batch(
        batch_size=batch_size, padded_shapes=([pad_shape, 3], [], [pad_shape], [3]),
        padding_values=-1
    )

    if evaluate:
        valid_ds_sp = build_tf_dataset(
            build_test_set(df_valid, df_train, corrupt="o", neighbourhood=neighbourhood),
        )
        valid_ds_po = build_tf_dataset(
            build_test_set(df_valid, df_train, corrupt="s", neighbourhood=neighbourhood),
        )

        test_ds_sp = build_tf_dataset(
            build_test_set(df_test, df_train, corrupt="o", neighbourhood=neighbourhood),
        )
        test_ds_po = build_tf_dataset(
            build_test_set(df_test, df_train, corrupt="s", neighbourhood=neighbourhood),
        )
        return train_ds, (valid_ds_sp, valid_ds_po), (test_ds_sp, test_ds_po)

    return train_ds, None, None
def map_raw2index(df_train, df_valid, df_test, dict_raw2idx_ents, dict_raw2idx_rels):
    """
    Function to map the raw entities and relations in the dataset to the indexing of the model.
    """
    df_reverse_ent = pd.DataFrame.from_dict(
        dict_raw2idx_ents, orient="index", columns=["idx"]
    )
    df_reverse_rel = pd.DataFrame.from_dict(
        dict_raw2idx_rels, orient="index", columns=["idx"]
    )

    # We map the dataset to the indexed version
    df_train = df_train.merge(
        df_reverse_ent, how="left", left_on="s", right_index=True)[['idx', 'p', 'o']]
    df_train = df_train.merge(
        df_reverse_rel, how="left", left_on="p", right_index=True)[['idx_x', 'idx_y', 'o']]
    df_train = df_train.merge(
        df_reverse_ent, how="left", left_on="o", right_index=True)[['idx_x', 'idx_y', 'idx']]
    df_train.columns = ["s", "p", "o"]

    df_valid = df_valid.merge(
        df_reverse_ent, how="left", left_on="s", right_index=True)[['idx', 'p', 'o']]
    df_valid = df_valid.merge(
        df_reverse_rel, how="left", left_on="p", right_index=True)[['idx_x', 'idx_y', 'o']]
    df_valid = df_valid.merge(
        df_reverse_ent, how="left", left_on="o", right_index=True)[['idx_x', 'idx_y', 'idx']]
    df_valid.columns = ["s", "p", "o"]

    df_test = df_test.merge(
        df_reverse_ent, how="left", left_on="s", right_index=True)[['idx', 'p', 'o']]
    df_test = df_test.merge(
        df_reverse_rel, how="left", left_on="p", right_index=True)[['idx_x', 'idx_y', 'o']]
    df_test = df_test.merge(
        df_reverse_ent, how="left", left_on="o", right_index=True)[['idx_x', 'idx_y', 'idx']]
    df_test.columns = ["s", "p", "o"]

    return df_train, df_valid, df_test

class StructureAndLatent(tf.keras.losses.Loss):
    """
    Loss function that has two terms: one to enforce the KG structure and
    the latter to reconstruct properly the latent representation.
    """
    def __init__(self, alpha=0.5, name="structure_and_latent"):
        super().__init__(name=name)
        self.alpha = alpha
        self.structure_term = tf.keras.metrics.Mean(name="structure_term")
        self.latent_term = tf.keras.metrics.Mean(name="latent_term")
        self.total_tracker = tf.keras.metrics.Mean(name="total_loss")

    def call(self, target_emb, pred_emb):
        # Unpack embeddings to separate the decoded latent from the pre-encoder-decoder representation
        latent_emb, target_triple_emb = pred_emb

        # We enforce the structure by enforcing s + p ~ o
        subjects, predicates, objects = target_triple_emb
        structure_term = tf.norm(
            subjects + predicates - objects, axis=-1
        )

        # We consider the euclidean distance
        latent_term = tf.norm(latent_emb - target_emb, axis=-1)

        total_loss = self.alpha * structure_term + latent_term

        # Store values for logging
        self.structure_term.update_state(structure_term)
        self.latent_term.update_state(latent_term)
        self.total_tracker.update_state(total_loss)

        return total_loss

    def reset_states(self):
        self.structure_term.reset_states()
        self.latent_term.reset_states()
        self.total_tracker.reset_states()

    def get_metrics(self):
        return [self.structure_term, self.latent_term, self.total_tracker]

class CosineSimilarity(tf.keras.losses.Loss):
    """
    Loss function that has two terms: one to enforce the KG structure and
    the latter to reconstruct properly the latent representation.
    """
    def __init__(self, name="cosine_similarity"):
        super().__init__(name=name)
        self.total_tracker = tf.keras.metrics.Mean(name="total_loss")

    def call(self, target_emb, output_emb):
        # Normalize both for cosine similarity
        output_emb_norm = tf.math.l2_normalize(output_emb, axis=-1)
        target_emb_norm = tf.math.l2_normalize(target_emb, axis=-1)

        cos_sim = tf.reduce_mean(output_emb_norm * target_emb_norm, axis=-1)
        total_loss = 1.0 - cos_sim

        # Store values for logging
        self.total_tracker.update_state(total_loss)

        return total_loss

    def reset_states(self):
        self.total_tracker.reset_states()

    def get_metrics(self):
        return [self.total_tracker]


class EuclideanLoss(tf.keras.losses.Loss):
    """
    Loss function that has two terms: one to enforce the KG structure and
    the latter to reconstruct properly the latent representation.
    """
    def __init__(self, name="euclidean_distance"):
        super().__init__(name=name)
        self.total_tracker = tf.keras.metrics.Mean(name="total_loss")

    def call(self, target_emb, output_emb):
        total_loss = tf.norm(output_emb - target_emb, axis=-1)

        # Store values for logging
        self.total_tracker.update_state(total_loss)

        return total_loss

    def reset_states(self):
        self.total_tracker.reset_states()

    def get_metrics(self):
        return [self.total_tracker]