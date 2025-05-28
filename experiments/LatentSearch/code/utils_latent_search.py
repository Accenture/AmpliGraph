import os
import numpy as np
import pandas as pd
import tensorflow as tf


def build_training_set(train_df):
    """
    Given a dataframe with columns ['s', 'p', 'o'], generate training examples:
    - Each example is a set of (s,p,o) triples excluding one object
    - The label is the excluded object
    """
    training_data = []

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
            training_data.append((input_triples, label, input_triples[:, 2]))

    for (o, p), subjects in grouped_po.items():
        if len(subjects) < 2:
            continue  # Need at least 2 to hold out one as label

        for i in range(len(subjects)):
            label = subjects[i]
            input_triples = np.array(
                [(s, p, o) for j, o in enumerate(subjects) if j != i],
                dtype=np.int32
            )
            training_data.append((input_triples, label, input_triples[:, 0]))

    return training_data

def build_test_set(test_df, train_df, corrupt="o"):
    """
    Given a dataframe with columns ['s', 'p', 'o'], generate training examples:
    - Each example is a set of (s,p,o) triples excluding one object
    - The label is the excluded object
    """
    testing_data = []

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
                testing_data.append((input_triples, o, input_triples[:, 2]))
            except:
                # print(f"No examples of (subject, predicate) = {(s, p)} in the provided test set!")
                missing_triples += 1
        elif corrupt == "s":
            try:
                subjects = grouped.loc[(o, p), "s"]
                input_triples = np.array(
                    [(subj, p, o) for subj in subjects], dtype=np.int32
                )
                testing_data.append((input_triples, s, input_triples[:, 0]))
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
            tf.TensorSpec(shape=(None), dtype=tf.int32)
        )
    )

def truncate_triples(triples, target, aux_target, pad_shape):
    """
    Function to truncate triples as `padded_batch` does not support truncation.
    """
    shuffled_idx = tf.random.shuffle(tf.range(tf.shape(triples)[0]))

    triples = tf.gather(triples, shuffled_idx)
    aux_target = tf.gather(aux_target, shuffled_idx)

    return triples[:pad_shape], target, aux_target[:pad_shape]

def build_datasets(df_train, df_valid, df_test, batch_size=1, quantile=0.9, evaluate=True):
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
    train = build_training_set(df_train)
    pad_shape = int(np.quantile([el[0].shape[0] for el in train], quantile))

    train_ds = build_tf_dataset(train).shuffle(
        buffer_size=10000
    ).map(
        lambda x, y, z: truncate_triples(x, y, z, pad_shape)
    ).padded_batch(
        batch_size=batch_size, padded_shapes=([pad_shape, 3], [], [pad_shape]),
        padding_values=-1
    )

    if evaluate:
        valid_ds_sp = build_tf_dataset(
            build_test_set(df_valid, df_train, corrupt="o")
        )
        valid_ds_po = build_tf_dataset(
            build_test_set(df_valid, df_train, corrupt="s")
        )

        test_ds_sp = build_tf_dataset(
            build_test_set(df_test, df_train, corrupt="o")
        )
        test_ds_po = build_tf_dataset(
            build_test_set(df_test, df_train, corrupt="s")
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