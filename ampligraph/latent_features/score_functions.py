import tensorflow as tf


def trans_e_score(e_s, e_p, e_o, norm):
    """The TransE scoring function.

        .. math::

            f_{TransE}=-||(\mathbf{e}_s + \mathbf{r}_p) - \mathbf{e}_o||_n

    Parameters
    ----------
    e_s : Tensor, shape [n]
        The embeddings of a list of subjects.
    e_p : Tensor, shape [n]
        The embeddings of a list of predicates.
    e_o : Tensor, shape [n]
        The embeddings of a list of objects.

    Returns
    -------
    score : TensorFlow operation
        The operation corresponding to the TransE scoring function.

    """

    return tf.negative(tf.norm(e_s + e_p - e_o, ord=norm, axis=1))