# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf

# Precision for floating point comparison
COMPARISION_PRECISION = 1e3

# Scoring layer registry. Every scoring function must be registered in
# this registry.
SCORING_LAYER_REGISTRY = {}


def register_layer(name, external_params=None, class_params=None):
    """Register the scoring function using this decorator.

    Parameters
    -----------
    name: str
        Name of the scoring function to be used to register the class.
    external_params: list of strings
        If there are any scoring function hyperparams, register their names.
    class_params: dict
        Parameters that may be used internally across various models.
    """
    if external_params is None:
        external_params = []
    if class_params is None:
        class_params = {}

    def insert_in_registry(class_handle):
        assert (
            name not in SCORING_LAYER_REGISTRY.keys()
        ), "Scoring Layer with name {} \
        already exists!".format(
            name
        )

        # store the class handle in the registry with name as key
        SCORING_LAYER_REGISTRY[name] = class_handle
        # create a class level variable and store the name
        class_handle.name = name

        # store other params related to the scoring function in the registry
        # this will be used later during model selection, etc
        SCORING_LAYER_REGISTRY[name].external_params = external_params
        SCORING_LAYER_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry


class AbstractScoringLayer(tf.keras.layers.Layer):
    """Abstract class for scoring layer."""

    def get_config(self):
        config = super(AbstractScoringLayer, self).get_config()
        config.update({"k": self.internal_k})
        return config

    def __init__(self, k):
        """Initializes the scoring layer.

        Parameters
        ----------
        k: int
            Embedding size.
        """
        super(AbstractScoringLayer, self).__init__()
        # store the embedding size. (concrete models may overwrite this)
        self.internal_k = k

    def call(self, triples):
        """Interface to the external world.
        Computes the scores of the triples.

        Parameters
        ----------
        triples: array-like, shape (n, 3)
            Batch of input triples.

        Returns
        -------
        scores: tf.Tensor, shape (n,1)
            Tensor of scores of inputs.
        """
        return self._compute_scores(triples)

    def _compute_scores(self, triples):
        """Abstract function to compute scores. Override this method in concrete classes.

        Parameters
        -----------
        triples: array-like, shape (n, 3)
            Batch of input triples.

        Returns
        --------
        scores: tf.Tensor, shape (n,1)
            Tensor of scores of inputs.
        """
        raise NotImplementedError("Abstract method not implemented!")

    def _get_object_corruption_scores(self, triples, ent_matrix):
        """Abstract function to compute object corruption scores.

        Evaluate the inputs against object corruptions and scores of the corruptions.

        Parameters
        -----------
        triples: array-like, shape (n, k)
            Batch of input embeddings.
        ent_matrix: array-like, shape (m, k)
            Slice of embedding matrix (corruptions).

        Returns
        --------
        scores: tf.Tensor, shape (n,1)
            Scores of object corruptions (corruptions defined by `ent_embs` matrix).
        """
        raise NotImplementedError("Abstract method not implemented!")

    def _get_subject_corruption_scores(self, triples, ent_matrix):
        """Abstract function to compute subject corruption scores.

        Evaluate the inputs against subject corruptions and scores of the corruptions.

        Parameters
        -----------
        triples: array-like, shape (n, k)
            Batch of input embeddings.
        ent_matrix: array-like, shape (m, k)
            Slice of embedding matrix (corruptions).

        Returns
        --------
        scores: tf.Tensor, shape (n,1)
            Scores of subject corruptions (corruptions defined by `ent_embs` matrix).
        """
        raise NotImplementedError("Abstract method not implemented!")

    def get_ranks(
        self,
        triples,
        ent_matrix,
        start_ent_id,
        end_ent_id,
        filters,
        mapping_dict,
        corrupt_side="s,o",
        comparison_type="worst",
    ):
        """Computes the ranks of triples against their corruptions.

        Ranks are computed by corrupting triple subject and/or object with the embeddings in ent_matrix.

        Parameters
        -----------
        triples: array-like, shape (n, k)
            Batch of input embeddings.
        ent_matrix: array-like, shape (m, k)
            Slice of embedding matrix (corruptions).
        start_ent_id: int
            Original id of the first row of embedding matrix (used during partitioned approach).
        end_ent_id: int
            Original id of the last row of embedding matrix (used during partitioned approach).
        filters: list of lists
            Size of list is either 1 or 2 depending on ``corrupt_side``.
            Size of the internal list is equal to the size of the input triples.
            Each list contains an array of filters (True Positives) related to the specified side of triples to corrupt.
        corrupt_side: str
            Which side to corrupt during evaluation.
        comparison_type: str
            Indicates how to break ties (default: `worst`, i.e., assigns the worst rank to the test triple).
            One of the three types can be passed: `"best"`, `"middle"`, `"worst"`.

        Returns
        --------
        ranks: tf.Tensor, shape (n,2)
            Ranks of triple against subject and object corruptions (corruptions defined by `ent_embs` matrix).
        """
        # compute the score of true positives
        triple_score = self._compute_scores(triples)

        # Handle the floating point comparison by multiplying by reqd precision and casting to int
        # before comparing
        triple_score = tf.cast(triple_score * COMPARISION_PRECISION, tf.int32)

        out_ranks = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        filter_index = 0
        if tf.strings.regex_full_match(corrupt_side, ".*s.*"):
            # compute the score by corrupting the subject side of triples by
            # ent_matrix
            sub_corr_score = self._get_subject_corruption_scores(
                triples, ent_matrix
            )
            # Handle the floating point comparison by multiplying by reqd precision and casting to int
            # before comparing
            sub_corr_score = tf.cast(
                sub_corr_score * COMPARISION_PRECISION, tf.int32
            )

            # if pos score: 0.5, corr_score: 0.5, 0.5, 0.3, 0.6, 0.5, 0.5
            if comparison_type == "best":
                # returns: 1 i.e. only. 1 corruption is having score greater
                # than positive (optimistic)
                sub_rank = tf.reduce_sum(
                    tf.cast(
                        tf.expand_dims(triple_score, 1) < sub_corr_score,
                        tf.int32,
                    ),
                    1,
                )
            elif comparison_type == "middle":
                # returns: 3 i.e. 1 + (4/2) i.e. only 1  corruption is having score greater than positive
                # and 4 corruptions are having same (middle rank is 4/2 = 1),
                # so 1+2=3
                sub_rank = tf.reduce_sum(
                    tf.cast(
                        tf.expand_dims(triple_score, 1) < sub_corr_score,
                        tf.int32,
                    ),
                    1,
                )
                part = tf.cast(
                    tf.expand_dims(triple_score, 1) == sub_corr_score, tf.int32
                )
                sub_rank += tf.cast(
                    tf.math.ceil(tf.reduce_sum(part, 1) / 2), tf.int32
                )
            else:
                # returns: 5 i.e. 5 corruptions are having score >= positive
                # as you can see this strategy returns the worst rank
                # (pessimistic)

                # compare True positive score against their respective
                # corruptions and get rank.
                sub_rank = tf.reduce_sum(
                    tf.cast(
                        tf.expand_dims(triple_score, 1) <= sub_corr_score,
                        tf.int32,
                    ),
                    1,
                )

            if filters.shape[0] > 0:
                # tf.print(tf.shape(triple_score)[0])
                for i in tf.range(tf.shape(triple_score)[0]):
                    # get the ids of True positives that needs to be filtered
                    filter_ids = filters[filter_index][i]

                    if mapping_dict.size() > 0:
                        filter_ids = mapping_dict.lookup(filter_ids)
                        filter_ids = tf.reshape(filter_ids, (-1,))

                        filter_ids_selector = tf.math.greater_equal(
                            filter_ids, 0
                        )
                        filter_ids = tf.boolean_mask(
                            filter_ids, filter_ids_selector, axis=0
                        )

                    # This is done for partitioning (where the full emb matrix is not used)
                    # this gets only the filter ids of the current partition
                    # being used for generating corruption
                    filter_ids_selector = tf.logical_and(
                        filter_ids >= start_ent_id, filter_ids <= end_ent_id
                    )

                    filter_ids = tf.boolean_mask(
                        filter_ids, filter_ids_selector
                    )
                    # from entity id convert to index in the current partition
                    filter_ids = filter_ids - start_ent_id

                    # get the score of the corruptions which are actually True
                    # positives
                    score_filter = tf.gather(
                        tf.squeeze(tf.gather_nd(sub_corr_score, [[i]])),
                        filter_ids,
                    )
                    # check how many of those were ranked higher than the test
                    # triple
                    num_filters_ranked_higher = tf.reduce_sum(
                        tf.cast(
                            tf.gather(triple_score, [i]) <= score_filter,
                            tf.int32,
                        )
                    )
                    # adjust the rank of the test triple accordingly
                    sub_rank = tf.tensor_scatter_nd_sub(
                        sub_rank, [[i]], [num_filters_ranked_higher]
                    )

            out_ranks = out_ranks.write(out_ranks.size(), sub_rank)

        if tf.strings.regex_full_match(corrupt_side, ".*o.*"):
            # compute the score by corrupting the object side of triples by
            # ent_matrix
            obj_corr_score = self._get_object_corruption_scores(
                triples, ent_matrix
            )

            # Handle the floating point comparison by multiplying by reqd precision and casting to int
            # before comparing
            obj_corr_score = tf.cast(
                obj_corr_score * COMPARISION_PRECISION, tf.int32
            )

            # if pos score: 0.5, corr_score: 0.5, 0.5, 0.3, 0.6, 0.5, 0.5
            if comparison_type == "best":
                # returns: 1 i.e. only. 1 corruption is having score greater
                # than positive (optimistic)
                obj_rank = tf.reduce_sum(
                    tf.cast(
                        tf.expand_dims(triple_score, 1) < obj_corr_score,
                        tf.int32,
                    ),
                    1,
                )
            elif comparison_type == "middle":
                print("middle")
                # returns: 3 i.e. 1 + (4/2) i.e. only 1  corruption is having score greater than positive
                # and 4 corruptions are having same (middle rank is 4/2 = 1),
                # so 1+2=3
                obj_rank = tf.reduce_sum(
                    tf.cast(
                        tf.expand_dims(triple_score, 1) < obj_corr_score,
                        tf.int32,
                    ),
                    1,
                )
                part = tf.cast(
                    tf.expand_dims(triple_score, 1) == obj_corr_score, tf.int32
                )
                obj_rank += tf.cast(
                    tf.math.ceil(tf.reduce_sum(part, 1) / 2), tf.int32
                )
            else:
                # returns: 5 i.e. 5 corruptions are having score >= positive
                # as you can see this strategy returns the worst rank
                # (pessimistic)

                # compare True positive score against their respective
                # corruptions and get rank.
                obj_rank = tf.reduce_sum(
                    tf.cast(
                        tf.expand_dims(triple_score, 1) <= obj_corr_score,
                        tf.int32,
                    ),
                    1,
                )

            if filters.shape[0] > 0:
                for i in tf.range(tf.shape(triple_score)[0]):
                    if corrupt_side in ["s", "o"] and filters.shape[0] == 1:
                        filter_index = 0
                    else:
                        filter_index = 1
                    # get the ids of True positives that needs to be filtered
                    filter_ids = filters[filter_index][i]

                    if mapping_dict.size() > 0:
                        filter_ids = mapping_dict.lookup(filter_ids)
                        filter_ids = tf.reshape(filter_ids, (-1,))

                        filter_ids_selector = tf.math.greater_equal(
                            filter_ids, 0
                        )
                        filter_ids = tf.boolean_mask(
                            filter_ids, filter_ids_selector, axis=0
                        )

                    # This is done for patritioning (where the full emb matrix is not used)
                    # this gets only the filter ids of the current partition
                    # being used for generating corruption
                    filter_ids_selector = tf.logical_and(
                        filter_ids >= start_ent_id, filter_ids <= end_ent_id
                    )
                    filter_ids = tf.boolean_mask(
                        filter_ids, filter_ids_selector
                    )
                    # from entity id convert to index in the current partition
                    filter_ids = filter_ids - start_ent_id

                    # get the score of the corruptions which are actually True
                    # positives
                    score_filter = tf.gather(
                        tf.squeeze(tf.gather_nd(obj_corr_score, [[i]])),
                        filter_ids,
                    )
                    # check how many of those were ranked higher than the test
                    # triple
                    num_filters_ranked_higher = tf.reduce_sum(
                        tf.cast(
                            tf.gather(triple_score, [i]) <= score_filter,
                            tf.int32,
                        )
                    )
                    # adjust the rank of the test triple accordingly
                    obj_rank = tf.tensor_scatter_nd_sub(
                        obj_rank, [[i]], [num_filters_ranked_higher]
                    )

            out_ranks = out_ranks.write(out_ranks.size(), obj_rank)

        out_ranks = out_ranks.stack()
        return out_ranks

    def compute_output_shape(self, input_shape):
        """Returns the output shape of the outputs of the call function.

        Parameters
        -----------
        input_shape: tuple
            Shape of inputs of call function.

        Returns
        --------
        output_shape: tuple
            Shape of outputs of call function.
        """
        assert isinstance(input_shape, list)
        batch_size, _ = input_shape
        return [batch_size, 1]
