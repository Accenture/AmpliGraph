# Examples

## Train and evaluate an embedding model

```python
import numpy as np
from ampligraph.datasets import load_wn18
from ampligraph.latent_features import ComplEx
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score

def main():

    # load Wordnet18 dataset:
    X = load_wn18()

    # Initialize a ComplEx neural embedding model with pairwise loss function:
    # The model will be trained for 300 epochs.
    model = ComplEx(batches_count=10, seed=0, epochs=20, k=150, eta=10,
                    # Use adam optimizer with learning rate 1e-3
                    optimizer='adam', optimizer_params={'lr':1e-3},
                    # Use pairwise loss with margin 0.5
                    loss='pairwise', loss_params={'margin':0.5},
                    # Use L2 regularizer with regularizer weight 1e-5
                    regularizer='LP', regularizer_params={'p':2, 'lambda':1e-5}, 
                    # Enable stdout messages (set to false if you don't want to display)
                    verbose=True)

    # For evaluation, we can use a filter which would be used to filter out 
    # positives statements created by the corruption procedure.
    # Here we define the filter set by concatenating all the positives
    filter = np.concatenate((X['train'], X['valid'], X['test']))
    
    # Fit the model on training and validation set
    model.fit(X['train'], 
              early_stopping = True,
              early_stopping_params = \
                      {
                          'x_valid': X['valid'], # validation set
                          'criteria':'hits10',   # Uses hits10 criteria for early stopping
                          'burn_in': 100,        # early stopping kicks in after 100 epochs
                          'check_interval':20,   # validates every 20th epoch
                          'stop_interval':5,     # stops if 5 successive validation checks are bad.
                          'x_filter': filter     # Use filter for filtering out positives 
                      }
              )

    

    # Run the evaluation procedure on the test set (with filtering). 
    # To disable filtering: filter_triples=None
    # Usually, we corrupt subject and object sides separately and compute ranks
    ranks = evaluate_performance(X['test'], 
                                 model=model, 
                                 filter_triples=filter,
                                 corrupt_side='s', # corrupt only the subject side
                                 verbose=True)
    
    ranks_obj = evaluate_performance(X['test'], 
                                 model=model, 
                                 filter_triples=filter,
                                 corrupt_side='o', # corrupt only the object side
                                 verbose=True)
    
    # merge the ranks before computing test statistics
    ranks.extend(ranks_obj) 

    # compute and print metrics:
    mrr = mrr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    print("MRR: %f, Hits@10: %f" % (mrr, hits_10))
    # Output: MRR: 0.886406, Hits@10: 0.935000

if __name__ == "__main__":
    main()
```


## Model selection


```python
from ampligraph.datasets import load_wn18
from ampligraph.latent_features import ComplEx
from ampligraph.evaluation import select_best_model_ranking

def main():

    # load Wordnet18 dataset:
    X_dict = load_wn18()

    model_class = ComplEx

    # Use the template given below for doing grid search. 
    param_grid = {
                     "batches_count": [10],
                     "seed": 0,
                     "epochs": [4000],
                     "k": [100, 50],
                     "eta": [5,10],
                     "loss": ["pairwise", "nll", "self_adversarial"],
                     # We take care of mapping the params to corresponding classes
                     "loss_params": {
                         #margin corresponding to both pairwise and adverserial loss
                         "margin": [0.5, 20], 
                         #alpha corresponding to adverserial loss
                         "alpha": [0.5]
                     },
                     "embedding_model_params": {
 
                     },
                     "regularizer": [None, "LP"],
                     "regularizer_params": {
                         "p": [2],
                         "lambda": [1e-4, 1e-5]
                     },
                     "optimizer": ["adam"],
                     "optimizer_params":{
                         "lr": [0.01, 0.0001]
                     },
                     "verbose": True
                 }

    # Train the model on all possibile combinations of hyperparameters.
    # Models are validated on the validation set.
    # It returnes a model re-trained on training and validation sets.
    best_model, best_params, best_mrr_train, \
    ranks_test, mrr_test = select_best_model_ranking(model_class, # Class handle of the model to be used
                                                     # Dataset 
                                                     X_dict,          
                                                     # Parameter grid
                                                     param_grid,      
                                                     # Use filtered set for eval
                                                     use_filter=True, 
                                                     # corrupt subject and objects separately during eval
                                                     use_default_protocol=True, 
                                                     # Log all the model hyperparams and evaluation stats
                                                     verbose=True)
    print(type(best_model).__name__, best_params, best_mrr_train, mrr_test)

if __name__ == "__main__":
    main()
```

## Get the embeddings

```python
import numpy as np
from ampligraph.latent_features import ComplEx

model = ComplEx(batches_count=1, seed=555, epochs=20, k=10)
X = np.array([['a', 'y', 'b'],
              ['b', 'y', 'a'],
              ['a', 'y', 'c'],
              ['c', 'y', 'a'],
              ['a', 'y', 'd'],
              ['c', 'y', 'd'],
              ['b', 'y', 'c'],
              ['f', 'y', 'e']])
model.fit(X)
model.get_embeddings(['f','e'], type='entity')
```

## Save and restore a model
```python

import numpy as np

from ampligraph.latent_features import ComplEx, save_model, restore_model

model = ComplEx(batches_count=2, seed=555, epochs=20, k=10)

X = np.array([['a', 'y', 'b'],
            ['b', 'y', 'a'],
            ['a', 'y', 'c'],
            ['c', 'y', 'a'],
            ['a', 'y', 'd'],
            ['c', 'y', 'd'],
            ['b', 'y', 'c'],
            ['f', 'y', 'e']])

model.fit(X)

EXAMPLE_LOC = 'saved_models'

# Use the trained model to predict 
y_pred_before = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
print(y_pred_before)

# Save the model
save_model(model, EXAMPLE_LOC)

# Restore the model
restored_model = restore_model(EXAMPLE_LOC)

# Use the restored model to predict
y_pred_after = restored_model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
print(y_pred_after)

# Assert that the before and after values are same
assert(y_pred_before==y_pred_after)
```