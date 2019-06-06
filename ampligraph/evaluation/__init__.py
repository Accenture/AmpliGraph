# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""The module includes performance metrics for neural graph embeddings models, along with model selection routines,
 negatives generation, and an implementation of the learning-to-rank-based evaluation protocol used in literature."""

from .metrics import mrr_score, mr_score, hits_at_n_score, rank_score
from .protocol import generate_corruptions_for_fit, evaluate_performance, to_idx, \
    generate_corruptions_for_eval, create_mappings, select_best_model_ranking, train_test_split_no_unseen, \
    filter_unseen_entities

__all__ = ['mrr_score', 'hits_at_n_score', 'rank_score', 'generate_corruptions_for_fit',
           'evaluate_performance', 'to_idx', 'generate_corruptions_for_eval', 'create_mappings',
           'select_best_model_ranking', 'train_test_split_no_unseen', 'filter_unseen_entities']
