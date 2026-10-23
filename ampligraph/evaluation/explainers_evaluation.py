# Copyright 2019-2026 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ROAR (Remove And Retrain) fidelity evaluation for KGE explainers.
# NOTE: retrain_model() uses the AmpliGraph compat layer (ampligraph.compat.*
# models) which exposes get_hyperparameter_dict() / calibrate() / predict_proba().
# It will NOT work directly with ScoringBasedEmbeddingModel instances.
# evaluate_explainer() expects the explainer object to be passed in (not the
# model) since predict_explain lives on the explainer, not the model.

from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import os
import json
import pickle as pkl
from scipy import stats
import tempfile


def get_correlation(post, pred, show=False, save=None):
    """Get Pearson coefficient to measure correlation between
       probabilities before and after retraining.

       Parameters
       ----------
       post: array-like
           Predicted probabilities after retraining.
       pred: array-like
           Original predicted probabilities.
       show: bool
           Whether to display chart.
       save: str
           Save path for correlation plot.

       Returns
       -------
       pearson_coef: float
       p_value: float
       support: int
       all_values: int
    """
    ii = np.flatnonzero(np.isfinite(post))
    pearson_coef, p_value = stats.pearsonr(pred[ii], np.array(post)[ii])
    msg = "Pearson Correlation Coefficient: {},  P-value: {} (support={}/{})"
    support = len(np.array(post)[ii])
    all_values = len(post)
    print(msg.format(pearson_coef, p_value, support, all_values))
    if save is not None or show:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 10))
        p = sns.regplot(x=post, y=pred)
        font = 18
        p.set_yticks(p.get_yticks().tolist())
        p.set_yticklabels(["{:.2}".format(t) for t in p.get_yticks()], size=font)
        p.set_xticks(p.get_xticks().tolist())
        p.set_xticklabels(["{:.2}".format(t) for t in p.get_xticks()], size=font)
        msg = "Probabilities before and after retraining\nPearson: {:.3}, p-value: {:.1}"
        plt.title(msg.format(pearson_coef, p_value), fontsize=font * 1.4)
        plt.ylabel("Original", fontsize=font)
        plt.xlabel("After retraining", fontsize=font)
        sns.despine(bottom=True, left=True)
        sns.set_style("white", {'axes.grid': False})
        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()
    return pearson_coef, p_value, support, all_values


def remove_triples(X, triples):
    """Remove a set of triples from an array.

       Parameters
       ----------
       X: np.array, shape (n, 3)
       triples: np.array, shape (m, 3)

       Returns
       -------
       updated: np.array
    """
    df = pd.DataFrame({'s': X[:, 0], 'p': X[:, 1], 'o': X[:, 2]})
    df.drop_duplicates(inplace=True)
    inds = []
    for t in triples:
        mask = (df['s'] == t[0]) & (df['p'] == t[1]) & (df['o'] == t[2])
        inds.extend(df.index[mask].tolist())
    return df.drop(inds).values


def get_calibration_set(valid, train):
    """Filter validation set to triples whose entities and predicates appear in train.

       Parameters
       ----------
       valid: np.array
       train: np.array

       Returns
       -------
       updated: np.array
    """
    nodes = set(train[:, 0]) | set(train[:, 2])
    preds = set(train[:, 1])
    trips = [t for t in valid if t[0] in nodes and t[2] in nodes and t[1] in preds]
    return np.array(trips)


def check_coverage(train, query):
    """Check whether all components of a query triple appear in the training set.

       Parameters
       ----------
       train: np.array
       query: np.array, shape (3,)

       Returns
       -------
       bool
    """
    nodes = set(train[:, 0]) | set(train[:, 2])
    preds = set(train[:, 1])
    return query[0] in nodes and query[2] in nodes and query[1] in preds


def _concat(a, b):
    """Concatenate two arrays; handles empty inputs."""
    if len(a) > 0 and len(b) > 0:
        return np.concatenate([a, b], 0)
    return a if len(a) > 0 else (b if len(b) > 0 else np.array([]))


def retrain_model(model, X, explanation, get_explanation, verbose=False, local=True):
    """Retrain a model for the ROAR experiment.

       Parameters
       ----------
       model: ampligraph.compat model instance (TransE/DistMult/ComplEx/HolE)
           Must expose get_hyperparameter_dict(), fit(), calibrate(), predict_proba().
       X: dict
           Dataset dictionary with 'train' / 'valid' / 'test' splits.
       explanation: dict
           Single explanation dictionary from predict_explain().
       get_explanation: callable
           Function that extracts the explanation triples from the dict.
       verbose: bool
       local: bool
           True  → retrain on the explanation graph only.
           False → retrain on training data minus the explanation triples.

       Returns
       -------
       re_model or None
    """
    # NOTE: uses compat-layer API (get_hyperparameter_dict, batches_count).
    parameters = model.get_hyperparameter_dict()
    parameters['verbose'] = verbose
    parameters['batches_count'] = 1

    from ampligraph.compat import TransE, DistMult, ComplEx, HolE
    _COMPAT = {'TransE': TransE, 'DistMult': DistMult, 'ComplEx': ComplEx, 'HolE': HolE}
    re_model = _COMPAT[model.name](**parameters)

    explanation_graph = get_explanation(explanation)
    if len(explanation_graph) == 0:
        return None

    if not local:
        explanation_graph = remove_triples(explanation_graph, [explanation['query_triple']])
        train = remove_triples(X['train'], explanation_graph)
        if not check_coverage(train, explanation['query_triple']):
            return None
        re_model.fit(train)
    else:
        ex = [tuple(e) for e in explanation_graph]
        if tuple(explanation['query_triple']) not in ex:
            explanation_graph = np.concatenate(
                [explanation_graph, explanation['query_triple'].reshape(-1, 3)], 0
            )
        explanation_graph = explanation_graph.reshape(-1, 3)
        re_model.fit(explanation_graph)
        calibration_set = get_calibration_set(X['valid'], explanation_graph)
        if calibration_set is None or len(calibration_set) == 0:
            calibration_set = explanation_graph
        re_model.calibrate(calibration_set, positive_base_rate=0.5)

    return re_model


def retraining_experiment(model, X, explanations, get_explanation,
                           path=tempfile.gettempdir(), local=True):
    """Run retraining for all explanations and save post-retraining probabilities.

       Parameters
       ----------
       model: compat model instance
       X: dict
       explanations: list of explanation dicts
       get_explanation: callable
       path: str
       local: bool

       Returns
       -------
       pred_post: list
    """
    proba_path = os.path.join(path, "post_probability.csv")
    pred_post = []
    if os.path.exists(proba_path):
        df = pd.read_csv(proba_path)
        pred_post = df['probability'].values.tolist()

    print("3. Start retraining experiment.")
    for i, explanation in enumerate(tqdm(explanations[len(pred_post):])):
        re_model = retrain_model(model, X, explanation, get_explanation, local=local)
        if re_model is not None:
            pred_post.append(re_model.predict_proba(np.array([explanation['query_triple']]))[0])
        else:
            pred_post.append(None)
        if i % 100 == 0:
            pd.DataFrame({'probability': pred_post}).to_csv(proba_path, index=False)

    pd.DataFrame({'probability': pred_post}).to_csv(proba_path, index=False)
    return pred_post


def calculate_differences(pred_pre, pred_post, path=tempfile.gettempdir(), plot=False, local=True):
    """Calculate probability differences before and after retraining.

       Parameters
       ----------
       pred_pre: array-like
       pred_post: list (may contain None)
       path: str
       plot: bool
       local: bool

       Returns
       -------
       diffs: list
    """
    pre_vals = np.array(pred_pre).flatten().tolist()
    if plot:
        import matplotlib.pyplot as plt
        diffs = [pre - post for pre, post in zip(pre_vals, pred_post) if post is not None]
        plt.bar(range(len(diffs)), diffs)
        label = "on explanation graph" if local else "without explanations"
        plt.title("Probability difference: original vs retrained {}".format(label))
        plt.xlabel("Instances")
        plt.ylabel("Probability Difference")
    else:
        diffs = [pre - post if post is not None else None
                 for pre, post in zip(pre_vals, pred_post)]
    pd.DataFrame({'diff': diffs}).to_csv(os.path.join(path, 'proba_differences.csv'), index=False)
    return diffs


def evaluate_explainer(model, X, path=tempfile.gettempdir(), n=-1,
                        explainer='example', params={}, name="evaluation",
                        plot=False, local=True):
    """ROAR fidelity experiment (Hooker et al. 2019) for a KGE explainer.

       Parameters
       ----------
       model: ScoringBasedEmbeddingModel or compat model instance
           Must expose predict_explain() and get_hyperparameter_dict().
       X: dict
           Dataset dictionary with 'train' / 'valid' / 'test' splits.
       path: str
           Directory where explanations and CSVs are cached.
       n: int
           Number of test triples to explain; -1 = all.
       explainer: str
           Registered explainer name: 'example' or 'random_explainer'.
       params: dict
           Extra kwargs forwarded to predict_explain().
       name: str
       plot: bool
       local: bool

       Returns
       -------
       info: dict
    """
    if explainer == 'example':
        get_explanation = lambda elem: _concat(elem['prototype'], elem['examples'])
    elif explainer == 'random_explainer':
        get_explanation = lambda elem: elem['relevant_triples']
    else:
        raise NotImplementedError("Unknown explainer: {}".format(explainer))

    explanation_path = os.path.join(path, 'explanations.pkl')
    proba_path = os.path.join(path, "pred_probability.csv")
    info_path = os.path.join(path, "explanation_info.json")

    if (os.path.exists(explanation_path)
            and os.path.exists(proba_path)
            and os.path.exists(info_path)):
        print("Loading cached files from: {}".format(path))
        with open(explanation_path, 'rb') as f:
            explanations = pkl.loads(f.read())
        pred_pre = pd.read_csv(proba_path)
        with open(info_path) as f:
            info = json.loads(f.read())
    else:
        print("Creating files in: {}".format(path))
        os.makedirs(path, exist_ok=True)

        start = time.time()
        print("1. Generating explanations.")
        explanations = model.predict_explain(X['test'][:n], explainer=explainer, params=params)
        end = time.time()

        n_actual = len(explanations)
        avg_size = np.mean([len(get_explanation(e)) for e in explanations])
        std_size = np.std([len(get_explanation(e)) for e in explanations])
        all_trips = len(X['train'])
        avg_sparse = float(np.mean([1 - len(get_explanation(e)) / all_trips for e in explanations]))
        std_sparse = float(np.std([1 - len(get_explanation(e)) / all_trips for e in explanations]))

        info = {
            'predict_explain_test_set_time [s]': end - start,
            'predict_explain_time per triple [s]': (end - start) / n_actual,
            'n_elements': n_actual,
            'average_expl_size [n triples]': avg_size,
            'std_expl_size [n triples]': std_size,
            'average_sparsity': avg_sparse,
            'std_sparsity': std_sparse,
        }
        with open(info_path, 'w') as f:
            f.write(json.dumps(info))

        pred_pre = [e['probability'][0] for e in explanations]
        with open(explanation_path, 'wb') as f:
            f.write(pkl.dumps(explanations))
        pd.DataFrame({'probability': pred_pre}).to_csv(proba_path, index=False)

    print("2. Retraining and evaluating probability drop.")
    pred_post = retraining_experiment(model, X, explanations, get_explanation, path, local=local)

    diffs = calculate_differences(pred_pre, pred_post, path=path, plot=plot, local=local)
    mean_diff = float(np.nanmean(diffs)) if not np.all(np.isnan(
        [d for d in diffs if d is not None])) else -1.0
    std_diff = float(np.nanstd(diffs)) if not np.all(np.isnan(
        [d for d in diffs if d is not None])) else -1.0

    save_path = os.path.join(path, "correlation_{}.pdf".format(name)) if plot else None
    pearson_coef, p_value, support, all_values = get_correlation(
        pred_post, np.array(pred_pre), show=False, save=save_path
    )

    info.update({
        'mean_proba_diff': mean_diff,
        'std_proba_diff': std_diff,
        'pearson': pearson_coef,
        'pvalue': p_value,
        'support': float(support),
        'all_values': float(all_values),
    })
    with open(info_path, 'w') as f:
        f.write(json.dumps(info))

    return info
