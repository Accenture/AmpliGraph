# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import ampligraph
# Benchmark datasets are under ampligraph.datasets module
from ampligraph.datasets import load_fb15k_237
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.evaluation.metrics import mrr_score, hits_at_n_score, mr_score
import numpy as np

def test_evaluate_without_filter():
    dataset = load_fb15k_237()
    
    # create the model with transe scoring function
    model = ScoringBasedEmbeddingModel(eta=5, 
                                         k=300,
                                         scoring_type='TransE')

    # compile the model with loss and optimizer
    model.compile(optimizer='adam', loss='multiclass_nll')

    # fit the model to data.
    model.fit(dataset['train'],
                 batch_size=10000,
                 epochs=10)

    print('----------EVAL WITHOUT FILTER-----------------')
    print('----------corrupted with default protocol-----------------')
    # evaluate on the test set
    ranks = model.evaluate(dataset['test'][::100], # test set
                           batch_size=100, # evaluation batch size
                           corrupt_side='s,o', # sides to corrupt for scoring and ranking
                           )

    mr_joint = mr_score(ranks)
    mrr_joint = mrr_score(ranks)
    print('MR:', mr_joint)
    print('MRR:', mrr_joint)
    print('hits@1:', hits_at_n_score(ranks, 1))
    print('hits@10:', hits_at_n_score(ranks, 10))

    print('----------Subj and obj corrupted separately-----------------')
    ranks_sep = []
    ranks1 = model.evaluate(dataset['test'][::100], # test set
                       batch_size=100, # evaluation batch size
                       corrupt_side='s', # sides to corrupt for scoring and ranking
                       )
    ranks_sep.extend(ranks1)

    ranks2 = model.evaluate(dataset['test'][::100], # test set
                       batch_size=100, # evaluation batch size
                       corrupt_side='o', # sides to corrupt for scoring and ranking
                       )
    ranks_sep.extend(ranks2)
    mr_sep = mr_score(ranks_sep)
    print('MR:', mr_sep)
    print('MRR:', mrr_score(ranks_sep))
    print('hits@1:', hits_at_n_score(ranks_sep, 1))
    print('hits@10:', hits_at_n_score(ranks_sep, 10))
    
    np.testing.assert_equal(mr_sep, mr_joint)
    assert mrr_joint is not np.Inf, 'MRR is infinity'
    
    
def test_evaluate_with_filter():
    dataset = load_fb15k_237()
    
    # create the model with transe scoring function
    model = ScoringBasedEmbeddingModel(eta=5, 
                                         k=300,
                                         scoring_type='TransE')

    # compile the model with loss and optimizer
    model.compile(optimizer='adam', loss='multiclass_nll')

    # fit the model to data.
    model.fit(dataset['train'],
                 batch_size=10000,
                 epochs=10)

    print('----------EVAL WITH FILTER-----------------')
    print('----------corrupted with default protocol-----------------')
    # evaluate on the test set
    ranks = model.evaluate(dataset['test'][::100], # test set
                           batch_size=100, # evaluation batch size
                           corrupt_side='s,o', # sides to corrupt for scoring and ranking
                           use_filter={'train':dataset['train'], # Filter to be used for evaluation
                                   'valid':dataset['valid'],
                                   'test':dataset['test']}
                           )

    mr_joint = mr_score(ranks)
    mrr_joint = mrr_score(ranks)
    print('MR:', mr_joint)
    print('MRR:', mrr_joint)
    print('hits@1:', hits_at_n_score(ranks, 1))
    print('hits@10:', hits_at_n_score(ranks, 10))

    print('----------Subj and obj corrupted separately-----------------')
    ranks_sep = []
    ranks1 = model.evaluate(dataset['test'][::100], # test set
                       batch_size=100, # evaluation batch size
                       corrupt_side='s', # sides to corrupt for scoring and ranking
                       use_filter={'train':dataset['train'], # Filter to be used for evaluation
                                   'valid':dataset['valid'],
                                   'test':dataset['test']}
                       )
    ranks_sep.extend(ranks1)

    ranks2 = model.evaluate(dataset['test'][::100], # test set
                       batch_size=100, # evaluation batch size
                       corrupt_side='o', # sides to corrupt for scoring and ranking
                       use_filter={'train':dataset['train'], # Filter to be used for evaluation
                                   'valid':dataset['valid'],
                                   'test':dataset['test']}
                       )
    ranks_sep.extend(ranks2)
    mr_sep = mr_score(ranks_sep)
    print('MR:', mr_sep)
    print('MRR:', mrr_score(ranks_sep))
    print('hits@1:', hits_at_n_score(ranks_sep, 1))
    print('hits@10:', hits_at_n_score(ranks_sep, 10))
    
    np.testing.assert_equal(mr_sep, mr_joint)
    assert mrr_joint is not np.Inf, 'MRR is infinity'