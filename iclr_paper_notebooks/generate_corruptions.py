import numpy as np
from functools import partial
import tensorflow as tf
from sklearn.utils import check_random_state
from tqdm import tqdm
from ampligraph.datasets import AmpligraphDatasetAdapter, NumpyDatasetAdapter
from ampligraph.evaluation import generate_corruptions_for_fit, to_idx, generate_corruptions_for_eval, \
    hits_at_n_score, mrr_score

from sklearn.utils.validation import column_or_1d
from sklearn.metrics.classification import *

from sklearn.preprocessing import LabelBinarizer, label_binarize


def generate_corruptions(self, X_pos, batches_count, epochs):
    try:
        tf.reset_default_graph()
        self.rnd = check_random_state(self.seed)
        tf.random.set_random_seed(self.seed)

        self._load_model_from_trained_params()

        dataset_handle = NumpyDatasetAdapter()
        dataset_handle.use_mappings(self.rel_to_idx, self.ent_to_idx)

        dataset_handle.set_data(X_pos, "pos")

        gen_fn = partial(dataset_handle.get_next_batch, batches_count=batches_count, dataset_type="pos")
        dataset = tf.data.Dataset.from_generator(gen_fn,
                                                 output_types=tf.int32,
                                                 output_shapes=(None, 3))
        dataset = dataset.repeat().prefetch(1)
        dataset_iter = tf.data.make_one_shot_iterator(dataset)

        x_pos_tf = dataset_iter.get_next()

        e_s, e_p, e_o = self._lookup_embeddings(x_pos_tf)
        scores_pos = self._fn(e_s, e_p, e_o)

        x_neg_tf = generate_corruptions_for_fit(x_pos_tf,
                                                entities_list=None,
                                                eta=1,
                                                corrupt_side='s+o',
                                                entities_size=0,
                                                rnd=self.seed)

        e_s_neg, e_p_neg, e_o_neg = self._lookup_embeddings(x_neg_tf)
        scores = self._fn(e_s_neg, e_p_neg, e_o_neg)

        epoch_iterator_with_progress = tqdm(range(1, epochs + 1), disable=(not self.verbose), unit='epoch')

        scores_list = []
        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            for _ in epoch_iterator_with_progress:
                losses = []
                for batch in range(batches_count):
                    scores_list.append(sess.run(scores))

        dataset_handle.cleanup()
        return np.concatenate(scores_list)
    
    except Exception as e:
        dataset_handle.cleanup()
        raise e
        
        
def pos_iso(cal_model, pos_scores, neg_scores, positive_base_rate):
    weigths_pos = len(neg_scores) / len(pos_scores)
    weights_neg = (1.0 - positive_base_rate) / positive_base_rate
    weights = np.concatenate((np.full(pos_scores.shape, weigths_pos),
                              np.full(neg_scores.shape, weights_neg))).astype(float)
    target =  np.concatenate((np.ones(pos_scores.shape), np.zeros(neg_scores.shape))).astype(float)
    x = np.concatenate((pos_scores, neg_scores)).astype(float)
    
    cal_model.fit(x, target, sample_weight=weights)
    return cal_model


def _check_binary_probabilistic_predictions(y_true, y_prob):
    """Check that y_true is binary and y_prob contains valid probabilities"""
    check_consistent_length(y_true, y_prob)

    labels = np.unique(y_true)

    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         "Provided labels %s." % labels)

    if y_prob.max() > 1:
        raise ValueError("y_prob contains values greater than 1.")

    if y_prob.min() < 0:
        raise ValueError("y_prob contains values less than 0.")

    return label_binarize(y_true, labels)[:, 0]

def calibration_loss(y_true, y_prob, sample_weight=None, reducer="avg",
                     bin_size_ratio=0.1, sliding_window=False, pos_label=None):
    """Compute calibration loss.
    Across all items in a set of N predictions, the calibration loss measures
    the aggregated difference between (1) the average predicted probabilities
    assigned to the positive class, and (2) the frequencies
    of the positive class in the actual outcome.
    The calibration loss is appropriate for binary and categorical outcomes
    that can be structured as true or false.
    Which label is considered to be the positive label is controlled via the
    parameter pos_label, which defaults to 1.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    sample_weight : array-like, shape (n_samples,), optional
        Sample weights.
    reducer : 'avg' | 'max'
        Aggregation method.
    bin_size_ratio : float, optional (default=0.1)
        Ratio of the size of bins over the size of input arrays.
        Each bins will contain N * bin_size_ratio elements.
        Smaller bin_size_ratio might increase the accuracy of the calibration
        loss, provided sufficient data in bins.
    sliding_window : bool, optional (default=False)
        If true, compute the loss based on overlapping bins. Each neighboring
        bins share all but 2 elements.
    pos_label : int or str, optional (default=None)
        Label of the positive class. If None, the maximum label is used as
        positive class
    Returns
    -------
    score : float
        calibration loss
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import calibration_loss
    >>> y_true = np.array([0, 0, 0, 1] + [0, 1, 1, 1])
    >>> y_pred = np.array([0.25, 0.25, 0.25, 0.25] + [0.75, 0.75, 0.75, 0.75])
    >>> calibration_loss(y_true, y_pred, bin_size_ratio=0.5)
    0.0
    >>> calibration_loss(y_true, y_pred, bin_size_ratio=0.5, \
                         reducer="max")
    0.0
    >>> y_true = np.array([0, 0, 0, 0] + [1, 1, 1, 1])
    >>> calibration_loss(y_true, y_pred, bin_size_ratio=0.5)
    0.25
    >>> calibration_loss(y_true, y_pred, bin_size_ratio=0.5, \
                         reducer="max")
    0.25
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    assert_all_finite(y_true)
    assert_all_finite(y_prob)
    check_consistent_length(y_true, y_prob, sample_weight)

    if pos_label is None:
        pos_label = y_true.max()
    y_true = np.array(y_true == pos_label, int)
    y_true = _check_binary_probabilistic_predictions(y_true, y_prob)

    loss = 0.
    count = 0.
    remapping = np.argsort(y_prob)
    y_true = y_true[remapping]
    y_prob = y_prob[remapping]
    if sample_weight is not None:
        sample_weight = sample_weight[remapping]

    if sliding_window:
        if sample_weight:
            raise ValueError("sample_weight is incompatible with "
                             "sliding_window set to True")
        bin_size = int(bin_size_ratio * y_true.shape[0])
        # compute averages over a sliding window of size bin_size
        cumsum_true = np.zeros(y_true.shape[0] + 1)
        cumsum_true[1:] = np.cumsum(y_true)
        cumsum_prob = np.zeros(y_prob.shape[0] + 1)
        cumsum_prob[1:] = np.cumsum(y_prob)
        avg_pos = ((cumsum_true[bin_size:] - cumsum_true[:-bin_size])
                   / bin_size)
        avg_pred = ((cumsum_prob[bin_size:] - cumsum_prob[:-bin_size])
                    / bin_size)
        deltas = np.abs(avg_pos - avg_pred)
        if reducer == "max":
            loss = deltas.max()
        elif reducer == "avg":
            loss = deltas.sum()
            count = deltas.shape[0]
    else:
        i_thres = np.searchsorted(y_prob,
                                  np.arange(0, 1, bin_size_ratio)).tolist()
        i_thres.append(y_true.shape[0])
        for i, i_start in enumerate(i_thres[:-1]):
            i_end = i_thres[i+1]
            if sample_weight is None:
                delta_count = float(i_end - i_start)
                avg_pred_true = y_true[i_start:i_end].sum() / delta_count
                bin_centroid = y_prob[i_start:i_end].sum() / delta_count
            else:
                delta_count = float(sample_weight[i_start:i_end].sum())
                avg_pred_true = (np.dot(y_true[i_start:i_end],
                                        sample_weight[i_start:i_end])
                                 / delta_count)
                bin_centroid = (np.dot(y_prob[i_start:i_end],
                                       sample_weight[i_start:i_end])
                                / delta_count)
            count += delta_count
            if reducer == "max":
                loss = max(loss, abs(avg_pred_true - bin_centroid))
            elif reducer == "avg":
                delta_loss = abs(avg_pred_true - bin_centroid) * delta_count
                if not np.isnan(delta_loss):
                    loss += delta_loss
            else:
                raise ValueError("reducer is neither 'avg' nor 'max'")
    if reducer == "avg":
        loss /= count
    return loss