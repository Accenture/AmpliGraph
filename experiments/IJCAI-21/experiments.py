import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import pandas as pd
import numpy as np

from ampligraph.datasets import load_onet20k, load_ppi5k, load_nl27k, load_cn15k
from ampligraph.latent_features import ComplEx, TransE, DistMult, MODEL_REGISTRY
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score, mr_score
import json

def compare_vanilla_focusE_models(dataset, model_name, vanilla_params, focusE_params):
    if dataset == 'onet20k':
        X = load_onet20k()
    elif dataset == 'ppi5k':
        X = load_onet20k()
    elif dataset == 'nl27k':
        X = load_onet20k()
    elif dataset == 'cn15k':
        X = load_onet20k()
    else:
        raise ValueError('Invalid Dataset name!')
        
    X_filter = np.concatenate([X['train'], X['valid'], X['test']], 0)
    
    X_train = X['train']
    X_train_numeric_values = X['train_numeric_values']
    
    X_valid = X['valid']
    X_valid_numeric_values = X['valid_numeric_values']
    X_valid = X_valid[X_valid_numeric_values >= 0.75]
    
    X_test = X['test']
    X_test_top_k_triples = X['test_topk']
    X_topk_numeric_values = X['test_topk_numeric_values']
    
    X_test_bottom_k_triples = X['test_bottomk']
    X_bottomk_numeric_values = X['test_bottomk_numeric_values']
    
    print('\tTraining set shape:', X_train.shape)
    print('\tValid set shape:', X_valid.shape)
    print('\tTest set shape:', X_test.shape)
    
    
    early_stopping = { 'x_valid': X_valid,
                    'criteria': 'mrr', 
                    'x_filter': X_filter, 
                    'stop_interval': 8, 
                    'burn_in': 200, 
                    'check_interval': 25 }
    
    model_regular = MODEL_REGISTRY.get(model_name)(**vanilla_params)
    model_regular.fit(X_train, True, early_stopping)
    
    ranks_topk = evaluate_performance(X_test_top_k_triples, 
                                         model=model_regular, 
                                         filter_triples=X_filter,
                                         corrupt_side='s,o', 
                                         ranking_strategy='worst')
    print('\n\tTopk Regular:')
    print('\t\tMRR: ', mrr_score(ranks_topk))
    print('\t\thits@1: ', hits_at_n_score(ranks_topk, 1))
    print('\t\thits@10: ', hits_at_n_score(ranks_topk, 10))
    
    ranks_bottomk = evaluate_performance(X_test_bottom_k_triples, 
                                         model=model_regular, 
                                         filter_triples=X_filter,
                                         corrupt_side='s,o', 
                                         ranking_strategy='worst')
    
    print('\n\tBottomk Regular:')
    print('\t\tMRR: ', mrr_score(ranks_bottomk))
    print('\t\thits@1: ', hits_at_n_score(ranks_bottomk, 1))
    print('\t\thits@10: ', hits_at_n_score(ranks_bottomk, 10))
    
    delta_mrr_regular = mrr_score(ranks_topk) - mrr_score(ranks_bottomk)
    print('\n\tDelta MRR: ', delta_mrr_regular)

    early_stopping['burn_in'] = config['focusE'][dataset][model_name]['embedding_model_params']['stop_epoch'] / 2
    model_focusE = MODEL_REGISTRY.get(model_name)(**focusE_params)
    model_focusE.fit(X_train, True, early_stopping, X_train_numeric_values)
    
    ranks_topk_focusE = evaluate_performance(X_test_top_k_triples, 
                                             model=model_focusE, 
                                             filter_triples=X_filter,
                                             corrupt_side='s,o', 
                                             ranking_strategy='worst')
    
    print('\n\tTopk FocusE:')
    print('\t\tMRR: ', mrr_score(ranks_topk_focusE))
    print('\t\thits@1: ', hits_at_n_score(ranks_topk_focusE, 1))
    print('\t\thits@10: ', hits_at_n_score(ranks_topk_focusE, 10))
    
    ranks_bottomk_focusE = evaluate_performance(X_test_bottom_k_triples, 
                                                 model=model_focusE, 
                                                 filter_triples=X_filter,
                                                 corrupt_side='s,o', 
                                                 ranking_strategy='worst')
    
    print('\n\tBottomk FocusE:')
    print('\t\tMRR: ', mrr_score(ranks_bottomk_focusE))
    print('\t\thits@1: ', hits_at_n_score(ranks_bottomk_focusE, 1))
    print('\t\thits@10: ', hits_at_n_score(ranks_bottomk_focusE, 10))
    
    delta_mrr_focuse = mrr_score(ranks_topk_focusE) - mrr_score(ranks_bottomk_focusE)
    print('\n\tDelta MRR: ', delta_mrr_focuse)
    
    print('\n\tPercentage change in delta mrr: ', (delta_mrr_focuse - delta_mrr_regular) * 100 / delta_mrr_regular)
    
    
    
if __name__ == "__main__":
    json_file_path = './config.json'

    with open(json_file_path, 'r') as j:
         config = json.loads(j.read())
            
    datasets = list(config['regular'].keys())
    for dataset in datasets:
        model_names = list(config['regular'][dataset].keys())
        for model_name in model_names:
            print(dataset, '-', model_name)
            print('-----------------')
            vanilla_params = config['regular'][dataset][model_name]
            focusE_params = config['focusE'][dataset][model_name]
            compare_vanilla_focusE_models(dataset, model_name, vanilla_params, focusE_params)
