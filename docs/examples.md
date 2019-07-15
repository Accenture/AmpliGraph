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
                          'x_valid': X['valid'],       # validation set
                          'criteria':'hits10',         # Uses hits10 criteria for early stopping
                          'burn_in': 100,              # early stopping kicks in after 100 epochs
                          'check_interval':20,         # validates every 20th epoch
                          'stop_interval':5,           # stops if 5 successive validation checks are bad.
                          'x_filter': filter,          # Use filter for filtering out positives 
                          'corruption_entities':'all', # corrupt using all entities
                          'corrupt_side':'s+o'         # corrupt subject and object (but not at once)
                      }
              )

    

    # Run the evaluation procedure on the test set (with filtering). 
    # To disable filtering: filter_triples=None
    # Usually, we corrupt subject and object sides separately and compute ranks
    ranks = evaluate_performance(X['test'], 
                                 model=model, 
                                 filter_triples=filter,
                                 use_default_protocol=True, # corrupt subj and obj separately while evaluating
                                 verbose=True)

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
                         # generate corruption using all entities during training
                         "negative_corruption_entities":"all"
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
model.get_embeddings(['f','e'], embedding_type='entity')
```

## Save and restore a model
```python
import numpy as np
from ampligraph.latent_features import ComplEx
from ampligraph.utils import save_model, restore_model

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

#  Use the trained model to predict 
y_pred_before = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
print(y_pred_before)
#[-0.29721245, 0.07865551]

# Save the model
example_name = "helloworld.pkl"
save_model(model, model_name_path = example_name)

# Restore the model
restored_model = restore_model(model_name_path = example_name)

# Use the restored model to predict
y_pred_after = restored_model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
print(y_pred_after)
# [-0.29721245, 0.07865551]
```

## Split dataset into train/test or train/valid/test 
```python
import numpy as np
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.datasets import load_from_csv

'''
Assume we have a knowledge graph stored in my_folder/my_graph.csv,
and that the content of such file is:

a,y,b
f,y,e
b,y,a
a,y,c
c,y,a
a,y,d
c,y,d
b,y,c
f,y,e
'''

# Load the graph in memory
X = load_from_csv('my_folder', 'my_graph.csv', sep=',')

# To split the graph in train and test sets:
# (In this toy example the test set will include only two triples)
X_train, X_test = train_test_split_no_unseen(X, test_size=2)

print(X_train)

'''
X_train:[['a' 'y' 'b']
         ['f' 'y' 'e']
         ['b' 'y' 'a']
         ['c' 'y' 'a']
         ['c' 'y' 'd']
         ['b' 'y' 'c']
         ['f' 'y' 'e']]
'''

print(X_test)

'''
X_test: [['a' 'y' 'c']
         ['a' 'y' 'd']]
'''


# To split the graph in train, validation, and test the method must be called twice:
X_train_valid, X_test = train_test_split_no_unseen(X, test_size=2)
X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=2)

print(X_train)
'''
X_train:  [['a' 'y' 'b']
           ['b' 'y' 'a']
           ['c' 'y' 'd']
           ['b' 'y' 'c']
           ['f' 'y' 'e']]
'''

print(X_valid)
'''
X_valid:  [['f' 'y' 'e']
           ['c' 'y' 'a']]
'''

print(X_test)
'''
X_test:  [['a' 'y' 'c']
          ['a' 'y' 'd']]
'''

```

## Predicting score for an unseen entity 

```python
# Support we need to predict score for an unseen entity z from trained embeddings of X
import numpy as np

from ampligraph.latent_features import TransE, DistMult, HolE, ComplEx

model = ComplEx(batches_count=2, seed=555, epochs=20, k=10)

X = np.array([["a", "y", "b"],
            ["b", "y", "a"],
            ["a", "y", "c"],
            ["c", "y", "a"],
            ["a", "y", "d"],
            ["c", "y", "d"],
            ["b", "y", "c"],
            ["f", "y", "e"]])

model.fit(X)

print(model.predict(np.array(["z", "y", "f"]), approximate_unseen={
        "pool": "avg", 
        "neighbour_triples": [["z", "y", "c"],["z", "y", "d"]]
}))
```