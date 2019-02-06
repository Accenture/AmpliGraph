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

    # Init a ComplEx neural embedding model with pairwise loss function:
    # The model will be trained on 30 epochs.
    # Turn stdout messages off with verbose=False
    model = ComplEx(batches_count=10, seed=0, epochs=1, k=150, lr=.1, eta=10,
                    loss='pairwise', regularizer=None,
                    optimizer='adagrad', verbose=True)

    # Fit the model on training and validation set
    model.fit(np.concatenate((X['train'], X['valid'])))

    # The entire dataset will be used to filter out false positives statements
    # created by the corruption procedure:
    filter = np.concatenate((X['train'], X['valid'], X['test']))

    # Run the evaluation procedure on the test set. Will create filtered rankings.
    # To disable filtering: filter_triples=None
    ranks = evaluate_performance(X['test'], model=model, filter_triples=filter,
                                verbose=True, splits=50)

    # compute and print metrics:
    mrr = mrr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    print("MRR: %f, Hits@10: %f" % (mrr, hits_10))


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

    # Put here the desired hyperparameter values that will be evaluated in the grid search:
    param_grid = {'batches_count': [10],
                  'seed': [0],
                  'epochs': [1],
                  'k': [50, 150],
                  'pairwise_margin': [1],
                  'lr': [.1],
                  'eta': [2],
                  'loss': ['pairwise']}

    # Train the model on all possibile combinations of hyperparameters.
    # Models are validated on the validation set.
    # It returnes a model re-trained on training and validation sets.
    best_model, best_params, best_mrr_train, \
    ranks_test, mrr_test = select_best_model_ranking(model_class, X_dict,
                                                      param_grid,
                                                      filter_retrain=True,
                                                      eval_splits=100,
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

save_model(model, EXAMPLE_LOC)

restored_model = restore_model(EXAMPLE_LOC)

y_pred_after = restored_model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
```