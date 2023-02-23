# Getting Started with AmpliGraph

> **_NOTE:_**  **An interactive version of this tutorial is [available on Colab](https://colab.research.google.com/drive/1rylqOnm992AdP9z1aW8metlKpPuBTRGD).**

> **Download the [Jupyter notebook](https://github.com/Accenture/AmpliGraph/blob/master/docs/tutorials/AmpliGraphBasicsTutorial.ipynb)**

In this tutorial we will demonstrate how to use the AmpliGraph library. 

Things we will cover:

1. Exploration of a graph dataset
2. Splitting graph datasets into train and test sets
3. Training and Evaluation a Model
4. Saving and restoring a model
5. Predicting New Links
6. Visualizing embeddings using Tensorboard



## Requirements

A Python environment with the AmpliGraph library installed. Please follow [the install guide](http://docs.ampligraph.org/en/latest/install.html).



Some sanity check:


```python
import numpy as np
import pandas as pd
import ampligraph

ampligraph.__version__
```




    '2.0-dev'




## 1. Dataset Exploration

First things first! Lets import the required libraries and retrieve some data.

In this tutorial we're going to use the [**Game of Thrones knowledge Graph**](https://ampligraph.s3-eu-west-1.amazonaws.com/datasets/GoT.csv). Please note: this isn't the *greatest* dataset for demonstrating the power of knowledge graph embeddings, but is small, intuitive and should be familiar to most users. 

We downloaded the [neo4j graph published here](https://github.com/neo4j-examples/game-of-thrones). Such dataset has been generated using [these APIs](https://anapioficeandfire.com/)  which expose in a machine-readable fashion the content of open free sources such as A [Wiki of Ice and Fire](http://awoiaf.westeros.org/). We discarded all properties and saved all the directed, labeled relations in a plaintext file. Each relation (i.e. a triple) is in the form: 

    <subject, predicate, object>

The schema of the graph looks like this (image from [neo4j-examples/game-of-thrones](https://github.com/neo4j-examples/game-of-thrones)):

![](img/got-graphql-schema.jpg)

Run the following cell to pull down the dataset and load it in memory with AmpliGraph [`load_from_csv()`](http://docs.ampligraph.org/en/1.0.3/generated/ampligraph.datasets.load_from_csv.html#ampligraph.datasets.load_from_csv) utility function:


```python
import requests
from ampligraph.datasets import load_from_csv

url = 'https://ampligraph.s3-eu-west-1.amazonaws.com/datasets/GoT.csv'
open('GoT.csv', 'wb').write(requests.get(url).content)
X = load_from_csv('.', 'GoT.csv', sep=',')
X[:5, ]
```




    array([['Smithyton', 'SEAT_OF', 'House Shermer of Smithyton'],
           ['House Mormont of Bear Island', 'LED_BY', 'Maege Mormont'],
           ['Margaery Tyrell', 'SPOUSE', 'Joffrey Baratheon'],
           ['Maron Nymeros Martell', 'ALLIED_WITH',
            'House Nymeros Martell of Sunspear'],
           ['House Gargalen of Salt Shore', 'IN_REGION', 'Dorne']],
          dtype=object)



Let's list the subject and object entities found in the dataset...


```python
entities = np.unique(np.concatenate([X[:, 0], X[:, 2]]))
entities
```




    array(['Abelar Hightower', 'Acorn Hall', 'Addam Frey', ..., 'the Antlers',
           'the Paps', 'unnamed tower'], dtype=object)



... and all of the relationships that link them. Remember, these relationships only link *some* of the entities.


```python
relations = np.unique(X[:, 1])
relations
```




    array(['ALLIED_WITH', 'BRANCH_OF', 'FOUNDED_BY', 'HEIR_TO', 'IN_REGION',
           'LED_BY', 'PARENT_OF', 'SEAT_OF', 'SPOUSE', 'SWORN_TO'],
          dtype=object)




## 2. Defining Train and Test Datasets

As is typical in machine learning, we need to split our dataset into training and test (and sometimes validation) datasets. 

When dealing with Knowledge Graphs, there is a major difference with the standard method of randomly sampling N points to make up our test set. Indeed, each of our data points are two entities linked by some relationship, and we need to ensure that all entities and relationships that are represented in the test set are present in the training set in at least one triple. 

To accomplish this, AmpliGraph provides the [`train_test_split_no_unseen`](https://docs.ampligraph.org/en/latest/generated/ampligraph.evaluation.train_test_split_no_unseen.html#train-test-split-no-unseen) function.  

As an example, we will create a small test size that includes only 100 triples:


```python
from ampligraph.evaluation import train_test_split_no_unseen 

X_train, X_test = train_test_split_no_unseen(X, test_size=100) 
```

Our data is now split into train/test sets. If we needed to further obtain a validation dataset, we can just repeat the same procedure on the test set (and adjusting the split percentages). 


```python
print('Train set size: ', X_train.shape)
print('Test set size: ', X_test.shape)
```

    Train set size:  (3075, 3)
    Test set size:  (100, 3)



## 3. Training a Model 

AmpliGraph 2 has a unique class for defining [several](https://docs.ampligraph.org/en/latest/ampligraph.latent_features.html#knowledge-graph-embedding-models) Knoweldge Graph Embedding models (TransE, ComplEx, DistMult, HolE), it sufficies to specify the different scoring type. Together with that, at initialization time, we also need to define some parameters:
- **`k`** : the dimensionality of the embedding space;
- **`eta`** ($\eta$) : the number of negatives (i.e., false triples) that must be generated at training runtime for each positive (i.e., true triple).

To begin with we are going to use the [ComplEx](https://docs.ampligraph.org/en/latest/generated/ampligraph.latent_features.ComplEx.html#ampligraph.latent_features.ComplEx) model.


```python
from ampligraph.latent_features import ScoringBasedEmbeddingModel
model = ScoringBasedEmbeddingModel(k=150,
                                   eta=5,
                                   scoring_type='ComplEx',
                                   seed=0)
```

Right after defining the model, it is time to compile the model, specifying:

- **`optimizer`** : we will use the Adam optimizer, with a learning rate of 1e-3, but AmpliGraph 2 supports any _tf.keras.optimizers_;
- **`loss`** : we will consider the pairwise loss, with a margin of 0.5 set via the *loss_params* kwarg. However, many other loss functions are supported, and custom losses can be defined by the user;
- **`regularizer`** : we will use the $L_p$ regularization with $p=2$, i.e. L2 regularization. The regularization parameter $\lambda$ = 1e-5 is set via the *regularizer_params* kwarg. Also in this case, _tf.keras.regularizers_ are supported.
- **`initializer`** : we will use the Glorot Uniform initialization, but the _tf.keras.initializers_ are supported.



```python
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.regularizers import get as get_regularizer
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=1e-3)
loss = get_loss('pairwise', {'margin': 0.5})
regularizer = get_regularizer('LP', {'p': 2, 'lambda': 1e-5})

model.compile(loss=loss,
              optimizer='adam',
              entity_relation_regularizer=regularizer,
              entity_relation_initializer='glorot_uniform')
```

AmpliGraph follows the _tensorflow.keras_ style APIs, allowing, after compiling the model, to perform the main operations of the model with the **`fit`**, **`predict`**, and **`evaluate`** methods. 


### Fitting the Model

Once you run the next cell the model will start training. 

On a modern laptop this should take ~3 minutes (although your mileage may vary, especially if you have changed any of the hyper-parameters above).


```python
model.fit(X_train,
          batch_size=5000,
          epochs=200,
          verbose=True)
```

    2023-02-08 23:06:27.469961: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz


    Epoch 1/200


    2023-02-08 23:06:28.230759: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.


    2/2 [==============================] - 1s 668ms/step - loss: 7687.5562
    Epoch 2/200
    2/2 [==============================] - 0s 43ms/step - loss: 7683.3623
    Epoch 3/200
    2/2 [==============================] - 0s 48ms/step - loss: 7679.1382
    Epoch 4/200
    2/2 [==============================] - 0s 40ms/step - loss: 7674.8130
    Epoch 5/200
    2/2 [==============================] - 0s 48ms/step - loss: 7670.5405
    Epoch 6/200
    2/2 [==============================] - 0s 44ms/step - loss: 7666.1504
    Epoch 7/200
    2/2 [==============================] - 0s 41ms/step - loss: 7661.6445
    Epoch 8/200
    2/2 [==============================] - 0s 41ms/step - loss: 7657.1768
    Epoch 9/200
    2/2 [==============================] - 0s 46ms/step - loss: 7652.6147
    Epoch 10/200
    2/2 [==============================] - 0s 43ms/step - loss: 7647.9399
    Epoch 11/200
    2/2 [==============================] - 0s 41ms/step - loss: 7643.0947
    Epoch 12/200
    2/2 [==============================] - 0s 41ms/step - loss: 7638.1758
    Epoch 13/200
    2/2 [==============================] - 0s 48ms/step - loss: 7633.1064
    Epoch 14/200
    2/2 [==============================] - 0s 39ms/step - loss: 7627.8647
    Epoch 15/200
    2/2 [==============================] - 0s 39ms/step - loss: 7622.4819
    Epoch 16/200
    2/2 [==============================] - 0s 42ms/step - loss: 7616.9292
    Epoch 17/200
    2/2 [==============================] - 0s 44ms/step - loss: 7611.2173
    Epoch 18/200
    2/2 [==============================] - 0s 44ms/step - loss: 7605.2847
    Epoch 19/200
    2/2 [==============================] - 0s 40ms/step - loss: 7599.1416
    Epoch 20/200
    2/2 [==============================] - 0s 40ms/step - loss: 7592.7905
    Epoch 21/200
    2/2 [==============================] - 0s 46ms/step - loss: 7586.2324
    Epoch 22/200
    2/2 [==============================] - 0s 42ms/step - loss: 7579.4341
    Epoch 23/200
    2/2 [==============================] - 0s 40ms/step - loss: 7572.4092
    Epoch 24/200
    2/2 [==============================] - 0s 41ms/step - loss: 7565.1226
    Epoch 25/200
    2/2 [==============================] - 0s 47ms/step - loss: 7557.5513
    Epoch 26/200
    2/2 [==============================] - 0s 43ms/step - loss: 7549.6802
    Epoch 27/200
    2/2 [==============================] - 0s 40ms/step - loss: 7541.4810
    Epoch 28/200
    2/2 [==============================] - 0s 41ms/step - loss: 7533.0098
    Epoch 29/200
    2/2 [==============================] - 0s 43ms/step - loss: 7524.2012
    Epoch 30/200
    2/2 [==============================] - 0s 42ms/step - loss: 7515.0493
    Epoch 31/200
    2/2 [==============================] - 0s 40ms/step - loss: 7505.5547
    Epoch 32/200
    2/2 [==============================] - 0s 41ms/step - loss: 7495.6831
    Epoch 33/200
    2/2 [==============================] - 0s 44ms/step - loss: 7485.4292
    Epoch 34/200
    2/2 [==============================] - 0s 42ms/step - loss: 7474.8052
    Epoch 35/200
    2/2 [==============================] - 0s 41ms/step - loss: 7463.7358
    Epoch 36/200
    2/2 [==============================] - 0s 40ms/step - loss: 7452.2334
    Epoch 37/200
    2/2 [==============================] - 0s 47ms/step - loss: 7440.3193
    Epoch 38/200
    2/2 [==============================] - 0s 43ms/step - loss: 7427.9673
    Epoch 39/200
    2/2 [==============================] - 0s 40ms/step - loss: 7415.1470
    Epoch 40/200
    2/2 [==============================] - 0s 40ms/step - loss: 7401.8252
    Epoch 41/200
    2/2 [==============================] - 0s 47ms/step - loss: 7388.0103
    Epoch 42/200
    2/2 [==============================] - 0s 42ms/step - loss: 7373.6465
    Epoch 43/200
    2/2 [==============================] - 0s 40ms/step - loss: 7358.7339
    Epoch 44/200
    2/2 [==============================] - 0s 39ms/step - loss: 7343.2524
    Epoch 45/200
    2/2 [==============================] - 0s 46ms/step - loss: 7327.1968
    Epoch 46/200
    2/2 [==============================] - 0s 41ms/step - loss: 7310.5498
    Epoch 47/200
    2/2 [==============================] - 0s 40ms/step - loss: 7293.2598
    Epoch 48/200
    2/2 [==============================] - 0s 40ms/step - loss: 7275.4258
    Epoch 49/200
    2/2 [==============================] - 0s 47ms/step - loss: 7256.8550
    Epoch 50/200
    2/2 [==============================] - 0s 43ms/step - loss: 7237.6069
    Epoch 51/200
    2/2 [==============================] - 0s 39ms/step - loss: 7217.6772
    Epoch 52/200
    2/2 [==============================] - 0s 39ms/step - loss: 7197.0957
    Epoch 53/200
    2/2 [==============================] - 0s 45ms/step - loss: 7175.7148
    Epoch 54/200
    2/2 [==============================] - 0s 43ms/step - loss: 7153.5884
    Epoch 55/200
    2/2 [==============================] - 0s 40ms/step - loss: 7130.6948
    Epoch 56/200
    2/2 [==============================] - 0s 39ms/step - loss: 7106.9736
    Epoch 57/200
    2/2 [==============================] - 0s 46ms/step - loss: 7082.3877
    Epoch 58/200
    2/2 [==============================] - 0s 43ms/step - loss: 7056.9937
    Epoch 59/200
    2/2 [==============================] - 0s 40ms/step - loss: 7030.7690
    Epoch 60/200
    2/2 [==============================] - 0s 39ms/step - loss: 7003.6494
    Epoch 61/200
    2/2 [==============================] - 0s 48ms/step - loss: 6975.6445
    Epoch 62/200
    2/2 [==============================] - 0s 43ms/step - loss: 6946.6948
    Epoch 63/200
    2/2 [==============================] - 0s 40ms/step - loss: 6916.8398
    Epoch 64/200
    2/2 [==============================] - 0s 40ms/step - loss: 6885.9805
    Epoch 65/200
    2/2 [==============================] - 0s 47ms/step - loss: 6854.2222
    Epoch 66/200
    2/2 [==============================] - 0s 43ms/step - loss: 6821.4287
    Epoch 67/200
    2/2 [==============================] - 0s 40ms/step - loss: 6787.4209
    Epoch 68/200
    2/2 [==============================] - 0s 41ms/step - loss: 6752.3765
    Epoch 69/200
    2/2 [==============================] - 0s 46ms/step - loss: 6716.2451
    Epoch 70/200
    2/2 [==============================] - 0s 42ms/step - loss: 6678.9810
    Epoch 71/200
    2/2 [==============================] - 0s 40ms/step - loss: 6640.5068
    Epoch 72/200
    2/2 [==============================] - 0s 40ms/step - loss: 6600.8901
    Epoch 73/200
    2/2 [==============================] - 0s 48ms/step - loss: 6560.0127
    Epoch 74/200
    2/2 [==============================] - 0s 42ms/step - loss: 6517.9282
    Epoch 75/200
    2/2 [==============================] - 0s 40ms/step - loss: 6474.5786
    Epoch 76/200
    2/2 [==============================] - 0s 40ms/step - loss: 6429.9829
    Epoch 77/200
    2/2 [==============================] - 0s 44ms/step - loss: 6384.1543
    Epoch 78/200
    2/2 [==============================] - 0s 42ms/step - loss: 6337.0464
    Epoch 79/200
    2/2 [==============================] - 0s 39ms/step - loss: 6288.6187
    Epoch 80/200
    2/2 [==============================] - 0s 39ms/step - loss: 6238.8779
    Epoch 81/200
    2/2 [==============================] - 0s 46ms/step - loss: 6187.8340
    Epoch 82/200
    2/2 [==============================] - 0s 40ms/step - loss: 6135.7783
    Epoch 83/200
    2/2 [==============================] - 0s 37ms/step - loss: 6082.6733
    Epoch 84/200
    2/2 [==============================] - 0s 36ms/step - loss: 6028.7832
    Epoch 85/200
    2/2 [==============================] - 0s 38ms/step - loss: 5974.1294
    Epoch 86/200
    2/2 [==============================] - 0s 36ms/step - loss: 5919.1519
    Epoch 87/200
    2/2 [==============================] - 0s 31ms/step - loss: 5863.7749
    Epoch 88/200
    2/2 [==============================] - 0s 31ms/step - loss: 5808.1211
    Epoch 89/200
    2/2 [==============================] - 0s 34ms/step - loss: 5752.3843
    Epoch 90/200
    2/2 [==============================] - 0s 31ms/step - loss: 5696.8433
    Epoch 91/200
    2/2 [==============================] - 0s 29ms/step - loss: 5641.3433
    Epoch 92/200
    2/2 [==============================] - 0s 27ms/step - loss: 5586.2896
    Epoch 93/200
    2/2 [==============================] - 0s 30ms/step - loss: 5531.6914
    Epoch 94/200
    2/2 [==============================] - 0s 26ms/step - loss: 5477.6558
    Epoch 95/200
    2/2 [==============================] - 0s 25ms/step - loss: 5424.2500
    Epoch 96/200
    2/2 [==============================] - 0s 24ms/step - loss: 5371.4414
    Epoch 97/200
    2/2 [==============================] - 0s 26ms/step - loss: 5319.4209
    Epoch 98/200
    2/2 [==============================] - 0s 25ms/step - loss: 5268.1343
    Epoch 99/200
    2/2 [==============================] - 0s 22ms/step - loss: 5217.6074
    Epoch 100/200
    2/2 [==============================] - 0s 22ms/step - loss: 5167.8394
    Epoch 101/200
    2/2 [==============================] - 0s 22ms/step - loss: 5118.7812
    Epoch 102/200
    2/2 [==============================] - 0s 23ms/step - loss: 5070.6216
    Epoch 103/200
    2/2 [==============================] - 0s 21ms/step - loss: 5023.2554
    Epoch 104/200
    2/2 [==============================] - 0s 22ms/step - loss: 4976.6865
    Epoch 105/200
    2/2 [==============================] - 0s 23ms/step - loss: 4930.8843
    Epoch 106/200
    2/2 [==============================] - 0s 22ms/step - loss: 4885.9780
    Epoch 107/200
    2/2 [==============================] - 0s 21ms/step - loss: 4841.7085
    Epoch 108/200
    2/2 [==============================] - 0s 20ms/step - loss: 4798.2017
    Epoch 109/200
    2/2 [==============================] - 0s 21ms/step - loss: 4755.4863
    Epoch 110/200
    2/2 [==============================] - 0s 21ms/step - loss: 4713.4092
    Epoch 111/200
    2/2 [==============================] - 0s 20ms/step - loss: 4672.1030
    Epoch 112/200
    2/2 [==============================] - 0s 20ms/step - loss: 4631.4775
    Epoch 113/200
    2/2 [==============================] - 0s 21ms/step - loss: 4591.5752
    Epoch 114/200
    2/2 [==============================] - 0s 20ms/step - loss: 4552.3101
    Epoch 115/200
    2/2 [==============================] - 0s 20ms/step - loss: 4513.6792
    Epoch 116/200
    2/2 [==============================] - 0s 20ms/step - loss: 4475.6802
    Epoch 117/200
    2/2 [==============================] - 0s 20ms/step - loss: 4438.3237
    Epoch 118/200
    2/2 [==============================] - 0s 19ms/step - loss: 4401.5659
    Epoch 119/200
    2/2 [==============================] - 0s 19ms/step - loss: 4365.4551
    Epoch 120/200
    2/2 [==============================] - 0s 20ms/step - loss: 4329.9443
    Epoch 121/200
    2/2 [==============================] - 0s 20ms/step - loss: 4295.0410
    Epoch 122/200
    2/2 [==============================] - 0s 20ms/step - loss: 4260.6831
    Epoch 123/200
    2/2 [==============================] - 0s 19ms/step - loss: 4226.8403
    Epoch 124/200
    2/2 [==============================] - 0s 19ms/step - loss: 4193.5107
    Epoch 125/200
    2/2 [==============================] - 0s 20ms/step - loss: 4160.7246
    Epoch 126/200
    2/2 [==============================] - 0s 19ms/step - loss: 4128.4478
    Epoch 127/200
    2/2 [==============================] - 0s 19ms/step - loss: 4096.6489
    Epoch 128/200
    2/2 [==============================] - 0s 19ms/step - loss: 4065.3191
    Epoch 129/200
    2/2 [==============================] - 0s 19ms/step - loss: 4034.4797
    Epoch 130/200
    2/2 [==============================] - 0s 20ms/step - loss: 4004.1243
    Epoch 131/200
    2/2 [==============================] - 0s 20ms/step - loss: 3974.1709
    Epoch 132/200
    2/2 [==============================] - 0s 19ms/step - loss: 3944.7107
    Epoch 133/200
    2/2 [==============================] - 0s 19ms/step - loss: 3915.5930
    Epoch 134/200
    2/2 [==============================] - 0s 21ms/step - loss: 3886.9021
    Epoch 135/200
    2/2 [==============================] - 0s 19ms/step - loss: 3858.7039
    Epoch 136/200
    2/2 [==============================] - 0s 19ms/step - loss: 3830.8809
    Epoch 137/200
    2/2 [==============================] - 0s 20ms/step - loss: 3803.4395
    Epoch 138/200
    2/2 [==============================] - 0s 19ms/step - loss: 3776.4390
    Epoch 139/200
    2/2 [==============================] - 0s 20ms/step - loss: 3749.7639
    Epoch 140/200
    2/2 [==============================] - 0s 19ms/step - loss: 3723.4912
    Epoch 141/200
    2/2 [==============================] - 0s 19ms/step - loss: 3697.5691
    Epoch 142/200
    2/2 [==============================] - 0s 20ms/step - loss: 3672.0378
    Epoch 143/200
    2/2 [==============================] - 0s 20ms/step - loss: 3646.8630
    Epoch 144/200
    2/2 [==============================] - 0s 19ms/step - loss: 3621.9890
    Epoch 145/200
    2/2 [==============================] - 0s 20ms/step - loss: 3597.5039
    Epoch 146/200
    2/2 [==============================] - 0s 20ms/step - loss: 3573.3049
    Epoch 147/200
    2/2 [==============================] - 0s 20ms/step - loss: 3549.4644
    Epoch 148/200
    2/2 [==============================] - 0s 20ms/step - loss: 3525.8699
    Epoch 149/200
    2/2 [==============================] - 0s 21ms/step - loss: 3502.6682
    Epoch 150/200
    2/2 [==============================] - 0s 20ms/step - loss: 3479.8208
    Epoch 151/200
    2/2 [==============================] - 0s 20ms/step - loss: 3457.2893
    Epoch 152/200
    2/2 [==============================] - 0s 21ms/step - loss: 3435.0530
    Epoch 153/200
    2/2 [==============================] - 0s 20ms/step - loss: 3413.0737
    Epoch 154/200
    2/2 [==============================] - 0s 23ms/step - loss: 3391.3215
    Epoch 155/200
    2/2 [==============================] - 0s 21ms/step - loss: 3369.8508
    Epoch 156/200
    2/2 [==============================] - 0s 20ms/step - loss: 3348.6760
    Epoch 157/200
    2/2 [==============================] - 0s 23ms/step - loss: 3327.7073
    Epoch 158/200
    2/2 [==============================] - 0s 19ms/step - loss: 3307.0623
    Epoch 159/200
    2/2 [==============================] - 0s 19ms/step - loss: 3286.6704
    Epoch 160/200
    2/2 [==============================] - 0s 21ms/step - loss: 3266.4805
    Epoch 161/200
    2/2 [==============================] - 0s 22ms/step - loss: 3246.6299
    Epoch 162/200
    2/2 [==============================] - 0s 21ms/step - loss: 3227.0281
    Epoch 163/200
    2/2 [==============================] - 0s 20ms/step - loss: 3207.6719
    Epoch 164/200
    2/2 [==============================] - 0s 19ms/step - loss: 3188.4419
    Epoch 165/200
    2/2 [==============================] - 0s 20ms/step - loss: 3169.4727
    Epoch 166/200
    2/2 [==============================] - 0s 19ms/step - loss: 3150.6953
    Epoch 167/200
    2/2 [==============================] - 0s 20ms/step - loss: 3132.2092
    Epoch 168/200
    2/2 [==============================] - 0s 20ms/step - loss: 3113.9177
    Epoch 169/200
    2/2 [==============================] - 0s 20ms/step - loss: 3095.8491
    Epoch 170/200
    2/2 [==============================] - 0s 19ms/step - loss: 3077.9492
    Epoch 171/200
    2/2 [==============================] - 0s 19ms/step - loss: 3060.3206
    Epoch 172/200
    2/2 [==============================] - 0s 19ms/step - loss: 3042.8682
    Epoch 173/200
    2/2 [==============================] - 0s 20ms/step - loss: 3025.6821
    Epoch 174/200
    2/2 [==============================] - 0s 19ms/step - loss: 3008.5740
    Epoch 175/200
    2/2 [==============================] - 0s 19ms/step - loss: 2991.7039
    Epoch 176/200
    2/2 [==============================] - 0s 19ms/step - loss: 2975.0493
    Epoch 177/200
    2/2 [==============================] - 0s 18ms/step - loss: 2958.5591
    Epoch 178/200
    2/2 [==============================] - 0s 20ms/step - loss: 2942.2769
    Epoch 179/200
    2/2 [==============================] - 0s 19ms/step - loss: 2926.1487
    Epoch 180/200
    2/2 [==============================] - 0s 19ms/step - loss: 2910.1887
    Epoch 181/200
    2/2 [==============================] - 0s 19ms/step - loss: 2894.3938
    Epoch 182/200
    2/2 [==============================] - 0s 19ms/step - loss: 2878.7822
    Epoch 183/200
    2/2 [==============================] - 0s 19ms/step - loss: 2863.3167
    Epoch 184/200
    2/2 [==============================] - 0s 18ms/step - loss: 2848.0630
    Epoch 185/200
    2/2 [==============================] - 0s 19ms/step - loss: 2832.9565
    Epoch 186/200
    2/2 [==============================] - 0s 19ms/step - loss: 2817.9868
    Epoch 187/200
    2/2 [==============================] - 0s 19ms/step - loss: 2803.1804
    Epoch 188/200
    2/2 [==============================] - 0s 19ms/step - loss: 2788.5645
    Epoch 189/200
    2/2 [==============================] - 0s 19ms/step - loss: 2774.0742
    Epoch 190/200
    2/2 [==============================] - 0s 20ms/step - loss: 2759.7258
    Epoch 191/200
    2/2 [==============================] - 0s 19ms/step - loss: 2745.5208
    Epoch 192/200
    2/2 [==============================] - 0s 20ms/step - loss: 2731.5134
    Epoch 193/200
    2/2 [==============================] - 0s 20ms/step - loss: 2717.6428
    Epoch 194/200
    2/2 [==============================] - 0s 20ms/step - loss: 2703.9246
    Epoch 195/200
    2/2 [==============================] - 0s 19ms/step - loss: 2690.2976
    Epoch 196/200
    2/2 [==============================] - 0s 19ms/step - loss: 2676.8450
    Epoch 197/200
    2/2 [==============================] - 0s 19ms/step - loss: 2663.5295
    Epoch 198/200
    2/2 [==============================] - 0s 20ms/step - loss: 2650.3706
    Epoch 199/200
    2/2 [==============================] - 0s 19ms/step - loss: 2637.3049
    Epoch 200/200
    2/2 [==============================] - 0s 20ms/step - loss: 2624.3518





    <tensorflow.python.keras.callbacks.History at 0x15fd63520>



### Predicting with the Model
After having trained the model, we can use it to predict the score for any unseen triple given that its subject, relation and object were present in the training set.


```python
model.predict(np.array([['Leyton Hightower', 'ALLIED_WITH', 'House Ryswell of the Rills']]))

```




    array([0.09504332], dtype=float32)



### Evaluating
Now it is time to evaluate our model on the test set to see how well it's performing. 

For this we are going to use the `evaluate` method, which takes as arguments:

- **`X_test`** : the data to evaluate on. We're going to use our test set to evaluate.
- **`use_filter`** : whether to filter out the false negatives generated by the corruption strategy. If a dictionary is passed, the values of it are used as elements to filter.
- **`corrupt_side`** : specifies whether to corrupt subj and obj separately or to corrupt both during evaluation.


```python
positives_filter = {'test' : np.concatenate([X_train, X_test])}
ranks = model.evaluate(X_test, 
                       use_filter=positives_filter,   # Corruption strategy filter defined above 
                       corrupt_side='s,o', # corrupt subj and obj separately while evaluating
                       verbose=True)
```

    2023-02-09 15:57:55.492610: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.


    5/5 [==============================] - 1s 297ms/step


The ***ranks*** returned by the evaluate_performance function indicate the rank at which the test set triple was found when performing link prediction using the model. 

### Metrics
Let's compute some evaluate metrics and print them out.

We're going to use the *mrr_score* (mean reciprocal rank) and *hits_at_n_score* functions. 

- ***mrr_score***:  The function computes the mean of the reciprocal of elements of a vector of rankings ranks.
- ***hits_at_n_score***: The function computes how many elements of a vector of rankings ranks make it to the top n positions.


```python
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score

mrr = mrr_score(ranks)
print("MRR: %.2f" % (mrr))

hits_10 = hits_at_n_score(ranks, n=10)
print("Hits@10: %.2f" % (hits_10))
hits_3 = hits_at_n_score(ranks, n=3)
print("Hits@3: %.2f" % (hits_3))
hits_1 = hits_at_n_score(ranks, n=1)
print("Hits@1: %.2f" % (hits_1))
```

    MRR: 0.26
    Hits@10: 0.37
    Hits@3: 0.28
    Hits@1: 0.20


Now, how do we interpret those numbers? 

[Hits@N](http://docs.ampligraph.org/en/1.0.3/generated/ampligraph.evaluation.hits_at_n_score.html#ampligraph.evaluation.hits_at_n_score) indicates how many times in average a true triple was ranked in the top-N. Therefore, on average, we guessed the correct subject or object 53% of the time when considering the top-3 better ranked triples. The choice of which N makes more sense depends on the application.

The [Mean Reciprocal Rank (MRR)](http://docs.ampligraph.org/en/latest/generated/ampligraph.evaluation.mrr_score.html) is another popular metrics to assess the predictive power of a model.


## 4.  Saving and Restoring a Model

Before we go any further, let's save the best model found so that we can restore it in future.


```python
from ampligraph.utils import save_model, restore_model
```


```python
save_model(model, './best_model.pkl')
```

    WARNING - Found untraced functions such as _get_ranks while saving (showing 1 of 1). These functions will not be directly callable after loading.


This will save the model in the ampligraph_tutorial directory as `best_model.pkl`.

We can then delete the model...


```python
del model
```

.|.. and then restore it from disk! Ta-da! 


```python
model = restore_model('./best_model.pkl')
```

    Saved model does not include a db file. Skipping.


    2023-02-09 13:15:34.968306: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
    2023-02-09 13:15:34.990256: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.


And let's just double check that the model we restored has been fit:


```python
if model.is_fitted:
    print('The model is fit!')
else:
    print('The model is not fit! Did you skip a step?')
```

    The model is fit!



## 5. Predicting New Links

Link prediction allows us to infer missing links in a graph. This has many real-world use cases, such as predicting connections between people in a social network, interactions between proteins in a biological network, and music recommendation based on prior user taste. 

In our case, we're going to see which of the following candidate statements (that we made up) are more likely to be true:



```python
X_unseen = np.array([
    ['Jorah Mormont', 'SPOUSE', 'Daenerys Targaryen'],
    ['Tyrion Lannister', 'SPOUSE', 'Missandei'],
    ["King's Landing", 'SEAT_OF', 'House Lannister of Casterly Rock'],
    ['Sansa Stark', 'SPOUSE', 'Petyr Baelish'],
    ['Daenerys Targaryen', 'SPOUSE', 'Jon Snow'],
    ['Daenerys Targaryen', 'SPOUSE', 'Craster'],
    ['House Stark of Winterfell', 'IN_REGION', 'The North'],
    ['House Stark of Winterfell', 'IN_REGION', 'Dorne'],
    ['House Tyrell of Highgarden', 'IN_REGION', 'Beyond the Wall'],
    ['Brandon Stark', 'ALLIED_WITH', 'House Stark of Winterfell'],
    ['Brandon Stark', 'ALLIED_WITH', 'House Lannister of Casterly Rock'],    
    ['Rhaegar Targaryen', 'PARENT_OF', 'Jon Snow'],
    ['House Hutcheson', 'SWORN_TO', 'House Tyrell of Highgarden'],
    ['Daenerys Targaryen', 'ALLIED_WITH', 'House Stark of Winterfell'],
    ['Daenerys Targaryen', 'ALLIED_WITH', 'House Lannister of Casterly Rock'],
    ['Jaime Lannister', 'PARENT_OF', 'Myrcella Baratheon'],
    ['Robert I Baratheon', 'PARENT_OF', 'Myrcella Baratheon'],
    ['Cersei Lannister', 'PARENT_OF', 'Myrcella Baratheon'],
    ['Cersei Lannister', 'PARENT_OF', 'Brandon Stark'],
    ["Tywin Lannister", 'PARENT_OF', 'Jaime Lannister'],
    ["Missandei", 'SPOUSE', 'Grey Worm'],
    ["Brienne of Tarth", 'SPOUSE', 'Jaime Lannister']
])
```


```python
positives_filter['test'] = np.vstack((positives_filter['test'], X_unseen))
```


```python
ranks_unseen = model.evaluate(X_unseen,
                              use_filter=positives_filter,   # Corruption strategy filter defined above 
                              corrupt_side = 's+o',
                              verbose=True)
```

    2023-02-09 13:19:35.033726: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
    2023-02-09 13:19:35.252684: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.


    2/2 [==============================] - 1s 401ms/step



```python
scores = model.predict(X_unseen)
```

    2023-02-09 13:19:40.424379: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.


We transform the scores (real numbers) into probabilities (bound between 0 and 1) using the expit transform.

Note that the probabilities are not calibrated in any sense. 

Advanced note: To calibrate the probabilities, one may use a procedure such as [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling) or [Isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression). The challenge is to define what is a true triple and what is a false one, as the calibration of the probability of a triple being true depends on the base rate of positives and negatives.


```python
from scipy.special import expit
probs = expit(scores)
```


```python
pd.DataFrame(list(zip([' '.join(x) for x in X_unseen], 
                      ranks_unseen, 
                      np.squeeze(scores),
                      np.squeeze(probs))), 
             columns=['statement', 'rank', 'score', 'prob']).sort_values("score", ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statement</th>
      <th>rank</th>
      <th>score</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>House Hutcheson SWORN_TO House Tyrell of Highg...</td>
      <td>[-1]</td>
      <td>0.830148</td>
      <td>0.696386</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Brandon Stark ALLIED_WITH House Stark of Winte...</td>
      <td>[-1]</td>
      <td>0.630217</td>
      <td>0.652539</td>
    </tr>
    <tr>
      <th>6</th>
      <td>House Stark of Winterfell IN_REGION The North</td>
      <td>[7]</td>
      <td>0.235329</td>
      <td>0.558562</td>
    </tr>
    <tr>
      <th>8</th>
      <td>House Tyrell of Highgarden IN_REGION Beyond th...</td>
      <td>[206]</td>
      <td>0.110351</td>
      <td>0.527560</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sansa Stark SPOUSE Petyr Baelish</td>
      <td>[499]</td>
      <td>0.084068</td>
      <td>0.521005</td>
    </tr>
    <tr>
      <th>7</th>
      <td>House Stark of Winterfell IN_REGION Dorne</td>
      <td>[637]</td>
      <td>0.079129</td>
      <td>0.519772</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Daenerys Targaryen ALLIED_WITH House Stark of ...</td>
      <td>[688]</td>
      <td>0.058003</td>
      <td>0.514497</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Jaime Lannister PARENT_OF Myrcella Baratheon</td>
      <td>[674]</td>
      <td>0.055809</td>
      <td>0.513949</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Missandei SPOUSE Grey Worm</td>
      <td>[650]</td>
      <td>0.046343</td>
      <td>0.511584</td>
    </tr>
    <tr>
      <th>2</th>
      <td>King's Landing SEAT_OF House Lannister of Cast...</td>
      <td>[1066]</td>
      <td>0.032168</td>
      <td>0.508041</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Jorah Mormont SPOUSE Daenerys Targaryen</td>
      <td>[1645]</td>
      <td>0.021946</td>
      <td>0.505486</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rhaegar Targaryen PARENT_OF Jon Snow</td>
      <td>[1728]</td>
      <td>0.008681</td>
      <td>0.502170</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Cersei Lannister PARENT_OF Myrcella Baratheon</td>
      <td>[1978]</td>
      <td>0.006947</td>
      <td>0.501737</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Brienne of Tarth SPOUSE Jaime Lannister</td>
      <td>[2190]</td>
      <td>-0.004299</td>
      <td>0.498925</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Daenerys Targaryen ALLIED_WITH House Lannister...</td>
      <td>[1937]</td>
      <td>-0.008216</td>
      <td>0.497946</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Cersei Lannister PARENT_OF Brandon Stark</td>
      <td>[2337]</td>
      <td>-0.009620</td>
      <td>0.497595</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Brandon Stark ALLIED_WITH House Lannister of C...</td>
      <td>[2257]</td>
      <td>-0.020130</td>
      <td>0.494968</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Tywin Lannister PARENT_OF Jaime Lannister</td>
      <td>[2748]</td>
      <td>-0.025549</td>
      <td>0.493613</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Robert I Baratheon PARENT_OF Myrcella Baratheon</td>
      <td>[3002]</td>
      <td>-0.037298</td>
      <td>0.490677</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Daenerys Targaryen SPOUSE Craster</td>
      <td>[2982]</td>
      <td>-0.042962</td>
      <td>0.489261</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Daenerys Targaryen SPOUSE Jon Snow</td>
      <td>[3311]</td>
      <td>-0.054646</td>
      <td>0.486342</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tyrion Lannister SPOUSE Missandei</td>
      <td>[3618]</td>
      <td>-0.066496</td>
      <td>0.483382</td>
    </tr>
  </tbody>
</table>
</div>



We see that the embeddings captured some truths about Westeros. For example, **House Stark is placed in the North rather than Dorne**. It also realises **Daenerys Targaryen has no relation with Craster**, **nor Tyrion with Missandei**. It captures random trivia, as **House Hutcheson is indeed in the Reach and sworn to the Tyrells**. On the other hand, some marriages that it predicts never really happened. These mistakes are understandable: those characters were indeed close and appeared together in many different circumstances. 


## 6. Visualizing Embeddings with Tensorboard projector 

The kind folks at Google have created [Tensorboard](https://www.tensorflow.org/tensorboard), which allows us to graph how our model is learning (or not :|), peer into the innards of neural networks, and [visualize high-dimensional embeddings in the browser](https://projector.tensorflow.org/).   

Lets import the [`create_tensorboard_visualization`](http://docs.ampligraph.org/en/1.0.3/generated/ampligraph.utils.create_tensorboard_visualizations.html#ampligraph.utils.create_tensorboard_visualizations) function, which simplifies the creation of the files necessary for Tensorboard to display the embeddings.


```python
from ampligraph.utils import create_tensorboard_visualizations
```

And now we'll run the function with our model, specifying the output path:


```python
create_tensorboard_visualizations(model, 'GoT_embeddings')
```

If all went well, we should now have a number of files in the `AmpliGraph/tutorials/GoT_embeddings` directory:

```
GoT_embeddings/
    ├── checkpoint
    ├── embeddings_projector.tsv
    ├── graph_embedding.ckpt.data-00000-of-00001
    ├── graph_embedding.ckpt.index
    ├── graph_embedding.ckpt.meta
    ├── metadata.tsv
    └── projector_config.pbtxt
```

To visualize the embeddings in Tensorboard, run the following from your command line inside `AmpliGraph/tutorials`:

```bash
tensorboard --logdir=./visualizations
```
    
.. and once your browser opens up you should be able to see and explore your embeddings as below (PCA-reduced, two components):

![](img/GoT_tensoboard.png)





# The End

You made it to the end! Well done!

For more information please visit the [AmpliGraph GitHub](github.com/Accenture/AmpliGraph) (and remember to star the project!), or check out the [documentation](docs.ampligraph.org). 



