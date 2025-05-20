import os
import sys
import argparse
import time
import datetime
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
from ampligraph.datasets import load_fb15k_237
from ampligraph.pretrained_models import load_pretrained_model
from ampligraph.utils import save_model, restore_model
from ampligraph.evaluation import hits_at_n_score, mr_score, mrr_score

from encoder_decoder import *
from utils_latent_search import *

def load_model_and_data(dataset="fb15k_237", scoring_type="RotatE"):
    """
    We import the data and the information relative to the pretrained KGE we are going to exploit.
    """
    dataset = "fb15k-237" if dataset == "fb15k_237" else dataset

    pretrained_kge = load_pretrained_model(dataset=dataset, scoring_type=scoring_type)

    if dataset == "fb15k-237":
        dataset = load_fb15k_237()

    train, valid, test = dataset["train"], dataset["valid"], dataset["test"]
    train = pd.DataFrame(train, columns=['s', 'p', 'o'])
    valid = pd.DataFrame(valid, columns=['s', 'p', 'o'])
    test = pd.DataFrame(test, columns=['s', 'p', 'o'])

    # We load the dict to map raw entities to their index
    reverse_ent_dict = pretrained_kge.data_indexer.backend.reversed_entities_dict
    reverse_rel_dict = pretrained_kge.data_indexer.backend.reversed_relations_dict

    train, valid, test = map_raw2index(train, valid, test, reverse_ent_dict, reverse_rel_dict)

    # We also return the embeddings of the pretrained KGE
    ent_emb = pretrained_kge.encoding_layer.ent_emb
    rel_emb = pretrained_kge.encoding_layer.rel_emb

    return train, valid, test, ent_emb, rel_emb

def drop_ranks(test_triples, ranks_cosine, ranks_euclidean, name):
    """
    Function to drop the ranks into .csv files.

    Parameters
    ----------
    test_triples: tuple
        Tuple of length two, where the first element contains the test triples
        used to predict the object corruption, the second the triples used
        to predict the subject corruption.
    ranks_cosine: tuple
        Tuple of length two, where the first element contains the ranks assigned
        computing the cosine similarity between the predicted entity and the other
        entity embeddings.
    ranks_euclidean: tuple
        Tuple of length two, where the first element contains the ranks assigned
        computing the euclidean distance between the predicted entity and the other
        entity embeddings.
    name: str
        The path to the output folder.
    """
    # predictions_sp = tf.concat(predictions_sp, axis=0)
    # predictions_po = tf.concat(predictions_po, axis=0)

    # labels_sp = tf.concat(labels_sp, axis=0)
    # labels_po = tf.concat(labels_po, axis=0)

    test_triples_sp, test_triples_po = test_triples
    test_triples_sp = tf.constant(test_triples_sp)
    test_triples_po = tf.constant(test_triples_po)

    ranks_sp_cosine, ranks_po_cosine = ranks_cosine
    ranks_sp_cosine = tf.cast(tf.concat(ranks_sp_cosine, axis=0), dtype=tf.int32)
    ranks_po_cosine = tf.cast(tf.concat(ranks_po_cosine, axis=0), dtype=tf.int32)

    ranks_sp_euclidean, ranks_po_euclidean = ranks_euclidean
    ranks_sp_euclidean = tf.cast(tf.concat(ranks_sp_euclidean, axis=0), dtype=tf.int32)
    ranks_po_euclidean = tf.cast(tf.concat(ranks_po_euclidean, axis=0), dtype=tf.int32)

    df_ranks_sp_cos = pd.DataFrame(
        tf.concat([test_triples_sp, ranks_sp_cosine], axis=1).numpy(),
        columns=['s', 'p', 'o', 'rank_sp_cos']
    )
    df_ranks_po_cos = pd.DataFrame(
        tf.concat([test_triples_po, ranks_po_cosine], axis=1).numpy(),
        columns=['s', 'p', 'o', 'rank_po_cos']
    )

    df_ranks_sp_cos.to_csv(os.path.join(name, "df_ranks_sp_cosine.csv"), index=False)
    df_ranks_po_cos.to_csv(os.path.join(name, "df_ranks_po_cosine.csv"), index=False)
    print(f"Successfully saved cosine ranks in {name}!")

    df_ranks_sp_euclidean = pd.DataFrame(
        tf.concat([test_triples_sp, ranks_sp_euclidean], axis=1).numpy(),
        columns=['s', 'p', 'o', 'rank_sp_eucl']
    )
    df_ranks_po_euclidean = pd.DataFrame(
        tf.concat([test_triples_po, ranks_po_euclidean], axis=1).numpy(),
        columns=['s', 'p', 'o', 'rank_po_eucl']
    )

    df_ranks_sp_euclidean.to_csv(os.path.join(name, "df_ranks_sp_euclidean.csv"), index=False)
    df_ranks_po_euclidean.to_csv(os.path.join(name, "df_ranks_po_euclidean.csv"), index=False)
    print(f"Successfully saved euclidean ranks in {name}!")

def main(args):
    print(args)
    dataset_name = args.dataset
    scoring_type = args.scoring_type
    if args.load_model:
        output_folder = os.path.dirname(args.load_model)
    else:
        output_folder = args.output_folder

    fmt = '%Y-%m-%d-%H-%M-%S'
    date = datetime.datetime.now().strftime(fmt)
    if args.load_model:
        name = f"{output_folder}/evaluation-{date}"
    else:
        name = f"{output_folder}/{scoring_type}-{dataset_name}-{date}"

    if not os.path.exists(name):
        os.makedirs(name, exist_ok=True)

    stdout = os.path.join(name, "output.txt")

    if stdout != sys.stdout:
        stdout = open(stdout, 'w')

    hyperparams = {'batch_size': args.batch_size, 'seed': 0,
                   'epochs': 10, 'num_heads': 3,
                   'encoder_hidden_dim': 400, 'encoder_latent_dim': 200,
                   'decoder_hidden_dim': 1000,
                   'learning_rate': args.learning_rate,
                   'quantile': args.quantile}

    print(f"Experiment started. Find output at {name}")
    print(f"Experiment Folder: {name}", file=stdout, flush=True)
    print(f"Experiment Configuration: {str(args)}", file=stdout, flush=True)
    print(f"Experiment Hyperparams: {hyperparams}", file=stdout, flush=True)
    print("Dataset: {}, Scoring Type: {}".format(dataset_name, scoring_type))

    start_time = time.time()

    # Load the indexed dataset and the embeddings for entities and relations.
    train, valid, test, ent_emb, rel_emb = load_model_and_data(
        dataset=dataset_name, scoring_type=scoring_type
    )

    # We build the training, validation and test sets
    train_ds, valid_ds, test_ds = build_datasets(
        train, valid, test, batch_size=hyperparams["batch_size"],
        quantile=hyperparams["quantile"]
    )

    # Dimension of the pretrained embeddings
    k = ent_emb.shape[1]

    if args.load_model is None:
        model = TripleModel(
            d_model=k, num_heads=hyperparams["num_heads"],
            ff_dim=hyperparams["encoder_hidden_dim"], latent_dim=hyperparams["encoder_latent_dim"],
            decoder_hidden_dim=hyperparams['decoder_hidden_dim'], embedding_dim=k,
            ent_emb=ent_emb, rel_emb=rel_emb
        )
    else:
        model = tf.keras.models.load_model(args.load_model)
        print(
            f"Model checkpoint successfully loaded from {args.load_model}",
            file=stdout, flush=True
        )

    if args.train:
        assert args.load_model is None, "No training when you load the model, please."

        optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"])
        model.compile(optimizer=optimizer, run_eagerly=False)

        print("Training...")
        model.fit(train_ds, epochs=10)

        if args.save_model:
            model.save(os.path.join(name, "model_checkpoint_epoch5"))
            print(
                f"Model saved in {os.path.join(name, 'model_checkpoint_epoch5')} after 5 epochs.",
                file=stdout, flush=True
            )

        model.fit(train_ds, epochs=10)

        if args.save_model:
            model.save(os.path.join(name, "model_checkpoint_epoch10"))
            print(
                f"Model saved in {os.path.join(name, 'model_checkpoint_epoch10')} after 10 epochs.",
                file=stdout, flush=True
            )

        model.fit(train_ds, epochs=10)

        if args.save_model:
            model.save(os.path.join(name, "model_checkpoint_epoch15"))
            print(
                f"Model saved in {os.path.join(name, 'model_checkpoint_epoch15')} after 15 epochs.",
                file=stdout, flush=True
            )

    if args.evaluate:
        sp_filter = pd.concat([train, valid, test]).groupby(["s", "p"])["o"].apply(list).to_frame()
        po_filter = pd.concat([train, valid, test]).groupby(["p", "o"])["s"].apply(list).to_frame()

        test_sp, test_po = test_ds

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.int32)
        ])
        def my_inference(x):
            return model(x, training=False)


        test_triples_sp = []
        predictions_sp = []
        labels_sp = []
        ranks_sp_cosine = []
        ranks_sp_euclidean = []
        for triples, label in tqdm(test_sp.batch(1), desc="Evaluating"):
            s, p, _ = tf.squeeze(triples, axis=0)[0, :].numpy()
            o = label.numpy().item()
            test_triples_sp.append([s, p, o])

            predict_sp = my_inference(triples)
            predictions_sp.append(predict_sp)
            cos_sim = tf.reduce_mean(
                tf.math.l2_normalize(predict_sp, axis=-1) * tf.math.l2_normalize(model.ent_emb, axis=-1),
                axis=1
            )
            rankings_cos = tf.argsort(cos_sim, direction="ASCENDING")
            filter = ~(np.isin(rankings_cos, sp_filter.loc[(s, p)].item()) & (rankings_cos != label))
            rank_cos = tf.where(rankings_cos[filter] == label) + 1

            eucl_dist = tf.norm(predict_sp - model.ent_emb, axis=-1)
            rankings_eucl = tf.argsort(eucl_dist, direction="ASCENDING")
            filter = ~(np.isin(rankings_eucl, sp_filter.loc[(s, p)].item()) & (rankings_eucl != label))
            rank_eucl = tf.where(rankings_eucl[filter] == label) + 1

            labels_sp.append(label)
            ranks_sp_cosine.append(rank_cos)
            ranks_sp_euclidean.append(rank_eucl)

        test_triples_po = []
        predictions_po = []
        labels_po = []
        ranks_po_cosine = []
        ranks_po_euclidean = []
        for triples, label in tqdm(test_po.batch(1), desc="Evaluating"):
            _, p, o = tf.squeeze(triples, axis=0)[0, :].numpy()
            s = label.numpy().item()
            test_triples_po.append([s, p, o])

            predict_po = my_inference(triples)
            predictions_po.append(predict_po)

            cos_sim = tf.reduce_mean(
                tf.math.l2_normalize(predict_po, axis=-1) * tf.math.l2_normalize(model.ent_emb, axis=-1),
                axis=1
            )
            rankings_cos = tf.argsort(cos_sim, direction="ASCENDING")
            filter = ~(np.isin(rankings_cos, po_filter.loc[(p, o)].item()) & (rankings_cos != label))
            rank_cos = tf.where(rankings_cos[filter] == label) + 1

            eucl_dist = tf.norm(predict_po - model.ent_emb, axis=-1)
            rankings_eucl = tf.argsort(eucl_dist, direction="ASCENDING")
            filter = ~(np.isin(rankings_eucl, po_filter.loc[(p, o)].item()) & (rankings_eucl != label))
            rank_eucl = tf.where(rankings_eucl[filter] == label) + 1


            labels_po.append(label)
            ranks_po_cosine.append(rank_cos)
            ranks_po_euclidean.append(rank_eucl)

        drop_ranks(
            (test_triples_sp, test_triples_po),
            (ranks_sp_cosine, ranks_po_cosine),
            (ranks_sp_euclidean, ranks_po_euclidean),
            name=name
        )

        # ranks = None
        #
        # # compute and print metrics:
        # mr, mrr = mr_score(ranks), mrr_score(ranks)
        # hits_1 = hits_at_n_score(ranks, n=1)
        # hits_3 = hits_at_n_score(ranks, n=3)
        # hits_10 = hits_at_n_score(ranks, n=10)
        #
        # result = {
        #     "mr": mr, "mrr": mrr, "H@1": hits_1, "H@3": hits_3, "H@10": hits_10,
        #     "time": time.time() - start_time
        # }
        #
        # name_results = os.path.join(name, 'results.json')
        # with open(name_results, "w") as f:
        #     f.write(json.dumps(result))
        #
        # print(f'mr: {mr} \nmrr: {mrr} \nhits_1: {hits_1} \nhits_3: {hits_3} \nhits_10: {hits_10}')
        # print(f"Results: {result}", file=stdout, flush=True)
        # print(f"Results saved in the file: {name_results}", file=stdout, flush=True)

    print(f"Experiment complete. Total duration: {time.time() - start_time}", file=stdout, flush=True)
    print(f"Experiment successfully completed. Find output at {name}")

    stdout.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Pipeline to control the experiments for the evaluation of ontology-based negative sampling.')

    parser.add_argument('--dataset', default='fb15k_237', type=str, help='Dataset to process')
    parser.add_argument('--output-folder', default="/data/alberto/ampligraph_results/latent_search/v0.1",
                        type=str, help='Base folder for the output of the experiment.')
    parser.add_argument('--save-model', default=None, action="store_true",
                        help="Whether to save or not the model trained.")

    parser.add_argument('--no-train', action="store_false", dest='train',
                        help="Whether to train a model or not.")
    parser.add_argument('--no-evaluate', action="store_false", dest='evaluate',
                        help="Whether to evaluate the model.")
    parser.add_argument('--load-model', default=None, type=str,
                        help="Path to model checkpoint.")

    parser.add_argument('--batch-size', '-batch', default=1, type=int)
    parser.add_argument('--scoring-type', '-score', default="ComplEx", type=str,
                        help="KGE Scoring function to use for the experiment.")
    parser.add_argument('--optimizer', default='adam', type=str, help="Optimizer to be used through training.")
    parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float,
                        help="Learning rate of the optimizer.")
    parser.add_argument('-reg', '--regularization', default=1e-4, type=float)
    parser.add_argument('--loss-function', default="multiclass_nll", type=str,
                        help="Loss function to be used during training.")
    parser.add_argument('--padding-quantile', default=0.9, type=float, dest='quantile',
                        help="Quantile to compute the padding shape (i.e., for what proportion"
                             "of the triples do we want to cover the entirety of their neighbourhood)")

    args = parser.parse_args()
    main(args)