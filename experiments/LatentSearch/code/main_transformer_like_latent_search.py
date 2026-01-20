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
from ampligraph.evaluation.protocol import train_test_split_no_unseen

from encoder_decoder import *
from utils_latent_search import *

def load_model_and_data(dataset="fb15k_237", scoring_type="RotatE", debug=False):
    """
    We import the data and the information relative to the pretrained KGE we are going to exploit.
    """
    dataset = "fb15k-237" if dataset == "fb15k_237" else dataset

    pretrained_kge = load_pretrained_model(dataset=dataset, scoring_type=scoring_type)

    if dataset == "fb15k-237":
        dataset = load_fb15k_237()

    train, valid, test = dataset["train"], dataset["valid"], dataset["test"]

    if debug:
        train = train[:10000, :]
        train, test = train_test_split_no_unseen(train, test_size=500)
        test, valid = train_test_split_no_unseen(test, test_size=10)

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

    return train, valid, test, ent_emb, rel_emb, reverse_ent_dict, reverse_rel_dict

def load_llm_embeddings(path_ent_emb, reverse_ent_dict, reverse_rel_dict):
    """
    Function to load the LLM embeddings pre-computed and saved in `path_ent_emb`.

    One of the key part of this function is sorting the embeddings so to reflect the
    indexing of the entities used in the KG.
    """
    parent_folder = os.path.dirname(path_ent_emb)
    path_rel_emb = os.path.join(parent_folder, f"relation2{path_ent_emb.split(str(2))[-1]}")

    unsorted_ent_emb, unsorted_rel_emb = np.load(path_ent_emb), np.load(path_rel_emb)

    # As sometimes keys have no label, we create an additional
    # random embedding to represent such entities
    random_embedding = np.random.uniform(low=0.0, high=1.0, size=(1, 768))
    unsorted_ent_emb = np.vstack([unsorted_ent_emb, random_embedding])
    rand_emb_idx = unsorted_ent_emb.shape[0] - 1

    with open(os.path.join(parent_folder, "entity2label.json"), "r") as f:
        ent_labels = json.load(f)

    with open(os.path.join(parent_folder, "relation2label.json"), "r") as f:
        rel_labels = json.load(f)

    # We extract the indices to sort ent_labels.keys() as reverse_ent_dict.keys().
    ent_sorted_indices = [
        list(ent_labels).index(ent) if ent in ent_labels else rand_emb_idx for ent in reverse_ent_dict
    ]
    # Same for relations
    rel_index_map = {item: i for i, item in enumerate(rel_labels)}
    rel_sorted_indices = [rel_index_map[item] for item in reverse_rel_dict]

    # Now we sort ent_emb and rel_emb accordingly
    ent_emb = tf.gather(
        tf.constant(unsorted_ent_emb, dtype=tf.float32),
        ent_sorted_indices
    )
    rel_emb = tf.gather(
        tf.constant(unsorted_rel_emb, dtype=tf.float32),
        rel_sorted_indices
    )

    return ent_emb, rel_emb

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

class CustomEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, eval_fn, every_n_epochs, val_data=None, use_filter=None):
        super().__init__()
        self.eval_fn = eval_fn
        self.every_n_epochs = every_n_epochs
        self.val_data = val_data
        self.use_filter = use_filter

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            print(f"\nRunning custom evaluation at epoch {epoch + 1}")
            self.eval_fn(self.model, self.val_data)

def full_evaluation_protocol(model, test_ds, use_filter=None):

    sp_filter = pd.concat(
        [use_filter["train"], use_filter["valid"], use_filter["test"]]
    ).groupby(["s", "p"])["o"].apply(list).to_frame()
    po_filter = pd.concat(
        [use_filter["train"], use_filter["valid"], use_filter["test"]]
    ).groupby(["p", "o"])["s"].apply(list).to_frame()

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
    for triples, label, aux, _ in tqdm(test_sp.batch(1), desc="Evaluating"):
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
    for triples, label, aux, _ in tqdm(test_po.batch(1), desc="Evaluating"):
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

    return \
        (test_triples_sp, test_triples_po),\
        (ranks_sp_cosine, ranks_po_cosine),\
        (ranks_sp_euclidean, ranks_po_euclidean)

def generate_grid_search_combinations(path_to_grid):
    """
    This function loads a grid and specifies all possible combinations.
    """
    from itertools import product

    with open(path_to_grid, "r") as f:
        grid = json.load(f)

    keys = list(grid.keys())
    values = list(grid.values())

    combinations = product(*values)

    return grid, [dict(zip(keys, combo)) for combo in combinations]

def main(args, hyperparams):
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

    print(f"\nExperiment started. Find output at {name}")
    print(f"Experiment Folder: {name}", file=stdout, flush=True)
    print(f"Experiment Configuration: {str(args)}", file=stdout, flush=True)
    print(f"Experiment Hyperparams: {hyperparams}", file=stdout, flush=True)
    print(f"Experiment Hyperparams: {hyperparams}")
    print("Dataset: {}, Scoring Type: {}".format(dataset_name, scoring_type))

    start_time = time.time()

    # Load the indexed dataset and the embeddings for entities and relations.
    train, valid, test, ent_emb, rel_emb, reverse_ent_dict, reverse_rel_dict = load_model_and_data(
        dataset=dataset_name, scoring_type=scoring_type, debug=args.debug
    )
    if args.train_embeddings:
        # Glorot Uniform initialization
        max_val = tf.sqrt(6 / (2 * ent_emb.shape[1]))
        ent_emb = tf.random.uniform(shape=ent_emb.shape, minval=-max_val, maxval=max_val)
        rel_emb = tf.random.uniform(shape=rel_emb.shape, minval=-max_val, maxval=max_val)

    if args.llm_embeddings:
        ent_emb, rel_emb = load_llm_embeddings(args.llm_embeddings, reverse_ent_dict, reverse_rel_dict)

    # We build the training, validation and test sets
    train_ds, valid_ds, test_ds = build_datasets(
        train, valid, test, batch_size=hyperparams["batch_size"],
        quantile=hyperparams["quantile"], evaluate=args.evaluate,
        neighbourhood=args.neighbourhood
    )

    # Dimension of the pretrained embeddings
    k = ent_emb.shape[1]

    if args.load_model is None:
        model = TripleModel(
            input_dim=k, num_heads=hyperparams["num_heads"],
            ff_dim=hyperparams["encoder_hidden_dim"], latent_dim=hyperparams["encoder_latent_dim"],
            decoder_hidden_dim=hyperparams['decoder_hidden_dim'], embedding_dim=k,
            ent_emb=ent_emb, rel_emb=rel_emb, loss=args.loss,
            latent_aggregator=hyperparams["latent_aggregator"],
            latent_search=args.latent_search,
            train_embeddings=args.train_embeddings
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
        if args.loss == "cosine":
            loss = CosineSimilarity()
        elif args.loss == "euclidean":
            loss = EuclideanLoss()
        elif args.loss == "structure_and_latent":
            loss = StructureAndLatent(alpha=hyperparams.get("alpha_loss", 1))
        model.compile(optimizer=optimizer, loss=loss, run_eagerly=args.debug)

        # callbacks = None
        # if args.validation_every:
        #     callbacks = [
        #         CustomEvaluationCallback(
        #             eval_fn=full_evaluation_protocol,
        #             every_n_epochs=args.validation_every,
        #             val_data=valid_ds,
        #             use_filter={"train": train, "valid": valid, "test": test}
        #         )
        #     ]

        print("Training...")
        for i in range(1, 7):
            model.fit(
                train_ds, epochs=10 if not args.debug else 1
            )

            if args.save_model:
                model.save(os.path.join(name, f"model_checkpoint_epoch{i*10}"))
                print(
                    f"Model saved in {os.path.join(name, f'model_checkpoint_epoch{i*10}')} after {i*10} epochs.",
                    file=stdout, flush=True
                )

            if args.evaluate:
                eval_date = datetime.datetime.now().strftime(fmt)
                eval_name = os.path.join(name, f"evaluation-epoch{i*10}-{eval_date}")
                if not os.path.exists(eval_name):
                    os.makedirs(eval_name, exist_ok=True)

                print(f"\nCreated folder for validation evaluation: {eval_name}", file=stdout, flush=True)

                test_triples, ranks_cosine, ranks_euclidean = full_evaluation_protocol(
                    model=model, test_ds=test_ds, use_filter={"train": train, "valid": valid, "test": test}
                )

                test_triples_sp, test_triples_po = test_triples
                ranks_sp_cosine, ranks_po_cosine = ranks_cosine
                ranks_sp_euclidean, ranks_po_euclidean = ranks_euclidean

                drop_ranks(
                    (test_triples_sp, test_triples_po),
                    (ranks_sp_cosine, ranks_po_cosine),
                    (ranks_sp_euclidean, ranks_po_euclidean),
                    name=eval_name
                )

                # compute and print metrics:
                results = compute_metrics(
                    euclidean_ranks=(ranks_sp_euclidean, ranks_po_euclidean),
                    cosine_ranks=(ranks_sp_cosine, ranks_po_cosine)
                )

                for ranks_type, metrics in results.items():
                    print(f"Results on {ranks_type}: {metrics}", file=stdout, flush=True)

                name_results = os.path.join(eval_name, 'results.json')
                with open(name_results, "w") as f:
                    f.write(json.dumps(results))
                print(f"Results saved in the file: {name_results}", file=stdout, flush=True)

    if args.evaluate:
        eval_date = datetime.datetime.now().strftime(fmt)
        eval_name = os.path.join(name, f"final-evaluation-{eval_date}")
        if not os.path.exists(eval_name):
            os.makedirs(eval_name, exist_ok=True)

        print(f"\nCreated folder for final evaluation: {eval_name}", file=stdout, flush=True)

        test_triples, ranks_cosine, ranks_euclidean = full_evaluation_protocol(
            model=model, test_ds=test_ds, use_filter={"train": train, "valid": valid, "test": test}
        )

        test_triples_sp, test_triples_po = test_triples
        ranks_sp_cosine, ranks_po_cosine = ranks_cosine
        ranks_sp_euclidean, ranks_po_euclidean = ranks_euclidean

        drop_ranks(
            (test_triples_sp, test_triples_po),
            (ranks_sp_cosine, ranks_po_cosine),
            (ranks_sp_euclidean, ranks_po_euclidean),
            name=eval_name
        )

        # compute and print metrics:
        results = compute_metrics(
            euclidean_ranks=(ranks_sp_euclidean, ranks_po_euclidean),
            cosine_ranks=(ranks_sp_cosine, ranks_po_cosine)
        )

        for ranks_type, metrics in results.items():
            print(f"Results on {ranks_type}: {metrics}", file=stdout, flush=True)

        name_results = os.path.join(name, 'results.json')
        with open(name_results, "w") as f:
            f.write(json.dumps(results))
        print(f"Results saved in the file: {name_results}", file=stdout, flush=True)

    print(f"Experiment complete. Total duration: {time.time() - start_time}", file=stdout, flush=True)
    print(f"Experiment successfully completed. Find output at {name}")

    stdout.close()


def compute_metrics(euclidean_ranks, cosine_ranks):
    """
    This function computes the usual metrics given euclidean distance- and
    cosine similarity-based ranks.
    """
    ranks_sp_euclidean, ranks_po_euclidean = euclidean_ranks
    ranks_sp_cosine, ranks_po_cosine = cosine_ranks

    all_ranks = {
        "sp_euclidean": ranks_sp_euclidean,
        "po_euclidean": ranks_po_euclidean,
        "sp_cosine": ranks_sp_cosine,
        "po_cosine": ranks_po_cosine
    }

    results = dict()
    for subgroup, ranks in all_ranks.items():
        results[subgroup] = {
            "mr": mr_score(ranks),
            "mrr": mrr_score(ranks),
            "h@1": hits_at_n_score(ranks, n=1),
            "h@3": hits_at_n_score(ranks, n=3),
            "h@10": hits_at_n_score(ranks, n=10),
            "h@100": hits_at_n_score(ranks, n=100)
        }

    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Pipeline to control the experiments for the latent search feature.')

    parser.add_argument('--dataset', default='fb15k_237', type=str, help='Dataset to process')
    parser.add_argument('--output-folder', default="/data/alberto/ampligraph_results/latent_search/v0.2.1",
                        type=str, help='Base folder for the output of the experiment.')
    parser.add_argument('--save-model', default=None, action="store_true",
                        help="Whether to save or not the model trained.")
    parser.add_argument('--grid-search', '-grid', default=None, type=str,
                        help="Whether to carry out a grid search or not.")

    parser.add_argument('--debug', action='store_true',
                        help="Whether we are debugging or not (smaller dataset, run eagerly=True...)")

    parser.add_argument('--no-train', action="store_false", dest='train',
                        help="Whether to train a model or not.")
    parser.add_argument('--no-evaluate', action="store_false", dest='evaluate',
                        help="Whether to evaluate the model.")
    parser.add_argument('--load-model', default=None, type=str,
                        help="Path to model checkpoint.")

    parser.add_argument('--seed', default=0, type=int, help="Random seed for the experiment.")

    parser.add_argument('--batch-size', '-batch', default=1, type=int)
    parser.add_argument('--scoring-type', '-score', default="ComplEx", type=str,
                        help="KGE Scoring function to use for the experiment.")
    parser.add_argument('--optimizer', default='adam', type=str, help="Optimizer to be used through training.")
    parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float,
                        help="Learning rate of the optimizer.")
    parser.add_argument('-reg', '--regularization', default=1e-4, type=float)
    parser.add_argument('--loss-function', '-loss', dest='loss', default="cosine", type=str,
                        help="Loss function to be used during training.")
    parser.add_argument('--alpha-loss', default=1.0, type=float,
                        help="Hyperparameter to weigh contribution of the two terms in the "
                             "StructureAndLatent loss function.")
    parser.add_argument('--padding-quantile', '-quantile', default=0.9, type=float, dest='quantile',
                        help="Quantile to compute the padding shape (i.e., for what proportion"
                             "of the triples do we want to cover the entirety of their neighbourhood)")

    parser.add_argument('--latent-search', action='store_true', default=None,
                        help="Whether to perform a latent search before decoding the latent.")
    parser.add_argument('--train-embeddings', action='store_true', default=False,
                        help="Whether to use as the entity and relation embeddings some trainable "
                             "weights initialized with a random uniform distribution.")
    parser.add_argument('--llm-embeddings', default=None, type=str,
                        help="Path to the file storing the npy files with the embeddings of "
                             "entities pre-computed with an LLM. The path to the same file "
                             "for relations and to the dictionaries used to extract the labels"
                             "for entities and relations are inferred from it, so make sure to keep "
                             "everything in the same folder.")

    parser.add_argument('--neighbourhood', action='store_true', default=None,
                        help="Whether to use the neighbourhood of a triple, instead of only triples "
                             "sharing (s,p) or (p,o) in our triple.")
    parser.add_argument('--latent-aggregator', default="average", type=str,
                        help="Whether to aggregate the input triples using a simple average (`'average') "
                             "or a Feed Forward Neural Network (`'ffnn'`).")


    args = parser.parse_args()

    tf.keras.utils.set_random_seed(seed=args.seed)

    if args.grid_search and args.train:
        grid, grid_search_hyperparams = generate_grid_search_combinations(args.grid_search)
        print(f"\n Grid to be searched: {grid} \n")
        for hyperparams in tqdm(grid_search_hyperparams, desc="Grid searching...", total=len(grid_search_hyperparams)):
            main(args, hyperparams)
        print(f"Completed grid search using grid: {grid}")
    else:
        hyperparams = {
            'batch_size': args.batch_size, 'seed': args.seed,
            'epochs': 10, 'num_heads': 3,
            'encoder_hidden_dim': 400, 'encoder_latent_dim': 200,
            'decoder_hidden_dim': 1000,
            'learning_rate': args.learning_rate,
            'quantile': args.quantile,
            'latent_search': args.latent_search,
            'alpha_loss': args.alpha_loss,
            'latent_aggregator': args.latent_aggregator
        }
        main(args, hyperparams)

CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 100 --output-folder /mnt/new_disk/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /mnt/new_disk/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid


cd ~/ampligraph_remote/experiments/LatentSearch/code/
deactivate
source /home/alberto/.virtualenvs/AmpliGraph-Lab/bin/activate
dataset=yago310
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /mnt/new_disk/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/latency_experiments --freeze-relations --llm-integration /mnt/new_disk/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
dataset=fb15k-237
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /mnt/new_disk/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/latency_experiments --freeze-relations --llm-integration /mnt/new_disk/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
# CURRENTLY RUNNING

cd ~/ampligraph_remote/experiments/LatentSearch/code/
deactivate
source /home/alberto/.virtualenvs/AmpliGraph-Lab/bin/activate
dataset=wn18rr
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

dataset=fb15k-237
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=0 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid


cd ~/ampligraph_remote/experiments/LatentSearch/code/
deactivate
source /home/alberto/.virtualenvs/AmpliGraph-Lab/bin/activate
dataset=yago310
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top5 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top5_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top10 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top10_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=1 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top20 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top20_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid


cd ~/ampligraph_remote/experiments/LatentSearch/code/
deactivate
source /home/alberto/.virtualenvs/AmpliGraph-Lab/bin/activate
dataset=yago310
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=2 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid


cd ~/ampligraph_remote/experiments/LatentSearch/code/
deactivate
source /home/alberto/.virtualenvs/AmpliGraph-Lab/bin/activate
dataset=fb15k-237
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top3 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top3_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top7 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top7_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid

CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top15 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top15_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid


cd ~/ampligraph_remote/experiments/LatentSearch/code/
deactivate
source /home/alberto/.virtualenvs/AmpliGraph-Lab/bin/activate
dataset=wn18rr
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid

dataset=fb15k-237
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=4 python main_batch_latent_search.py --batch-size 30 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid


cd ~/ampligraph_remote/experiments/LatentSearch/code/
deactivate
source /home/alberto/.virtualenvs/AmpliGraph-Lab/bin/activate
dataset=yago310
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score TransE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid
CUDA_VISIBLE_DEVICES=5 python main_batch_latent_search.py --batch-size 15 -score DistMult --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.1.0_final --freeze-relations -valid



dataset=fb15k-237
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid --only-llm
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid --only-llm
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score RotatE --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid --only-llm
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-3 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid --only-llm
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-2 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid --only-llm
CUDA_VISIBLE_DEVICES=3 python main_batch_latent_search.py --batch-size 30 -score ComplEx --dataset $dataset -lr 1e-4 --epochs 30 --output-folder /data/alberto/ampligraph_results/latent_search/KGE_inference_search/v0.6.0/batched_freeze_relations_top30 --freeze-relations --llm-integration /data/alberto/latent_llm_clusters/$dataset/${dataset}_top30_closest_NovaSearch.stella_en_400M_v5.pkl --tune-llm-ents -valid --only-llm