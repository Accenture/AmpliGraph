import tensorflow as tf


def pairwise_loss(scores_pos, scores_neg, margin=1):
    """Pairwise, max-margin loss.

     Introduced in :cite:`bordes2013translating`.

    .. math::

        \mathcal{L}(\Theta) = \sum_{t^+ \in \mathcal{G}}\sum_{t^- \in \mathcal{C}}max(0, [\gamma + f_{model}(t^-;\Theta) - f_{model}(t^+;\Theta)])

    where :math:`\gamma` is the margin, :math:`\mathcal{G}` is the set of positives,
    :math:`\mathcal{C}` is the set of corruptions, :math:`f_{model}(t;\Theta)` is the model-specific scoring function.

    Parameters
    ----------
    scores_pos : tf.Tensor, shape [n, 1]
        A tensor of scores assigned to positive statements.
    scores_neg : tf.Tensor, shape [n, 1]
        A tensor of scores assigned to negative statements.
    margin : float
        The margin used by the function to distinguish positives form negatives.

    Returns
    -------
    loss : float
        The loss value that must be minimized.

    """

    margin_tf = tf.constant(margin, dtype=tf.float32, name='margin')
    loss = tf.reduce_sum(tf.maximum(margin_tf - scores_pos + scores_neg, 0))
    return loss


def negative_log_likelihood_loss(scores_pos, scores_neg, **kwargs):
    """Negative log-likelihood loss.

    As described in :cite:`trouillon2016complex`.

    .. math::

        \mathcal{L}(\Theta) = \sum_{t \in \mathcal{G} \cup \mathcal{C}}log(1 + exp(-yf_{model}(t;\Theta)))

    where :math:`y` is the label of the statement :math:` \in [-1, 1]`, :math:`\mathcal{G}` is the set of positives,
    :math:`\mathcal{C}` is the set of corruptions, :math:`f_{model}(t;\Theta)` is the model-specific scoring function.

    Parameters
    ----------
    scores_pos : tf.Tensor, shape [n, 1]
        A tensor of scores assigned to positive statements.
    scores_neg : tf.Tensor, shape [n, 1]
        A tensor of scores assigned to negative statements.
    kwargs : dict
        Not used.

    Returns
    -------
    loss : float
        The loss value that must be minimized.

    """
    scores = tf.concat([-scores_pos, scores_neg], 0)
    return tf.reduce_sum(tf.log(1 + tf.exp(scores)))
    # TODO (consider add regularization term - now handled in params constructor)
    #   + lambda_reg * (tf.norm(e_s + e_p - e_o, ord=2, axis=1) +  tf.norm(e_s + e_p - e_o, ord=2, axis=1))

def absolute_margin_loss(scores_pos, scores_neg, margin=1):
    """Absolute margin , max-margin loss.

        Introduced in :cite:`Hamaguchi2017`.

       .. math::

           \mathcal{L}(\Theta) = \sum_{t^+ \in \mathcal{G}}\sum_{t^- \in \mathcal{C}} f_{model}(t^-;\Theta) - max(0, [\gamma - f_{model}(t^+;\Theta)])

       where :math:`\gamma` is the margin, :math:`\mathcal{G}` is the set of positives,
       :math:`\mathcal{C}` is the set of corruptions, :math:`f_{model}(t;\Theta)` is the model-specific scoring function.

       Parameters
       ----------
       scores_pos : tf.Tensor, shape [n, 1]
           A tensor of scores assigned to positive statements.
       scores_neg : tf.Tensor, shape [n, 1]
           A tensor of scores assigned to negative statements.
       margin : float
           The margin used by the function to distinguish positives form negatives.

       Returns
       -------
       loss : float
           The loss value that must be minimized.

       """

    margin_tf = tf.constant(margin, dtype=tf.float32, name='margin')
    loss = tf.reduce_sum(scores_neg + tf.maximum(margin_tf - scores_pos, 0))
    return loss
