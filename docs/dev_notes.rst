Data Adapters
-------------

In order to be able to support multiple data sources to be used within AmpliGraph and to provide guidelines for the
developers to develop similar APIs, in order to ingest their data directly into AmpliGraph, we have introduced an
adapter based pattern. We have currently provided two such adapters, to adapt numpy arrays and to read data directly
from a database.

Internally, AmpliGraph uses a set of methods that are defined within the base class (``AmpligraphDatasetAdapter``).
Every data adapter must be inherited from this class to be able to be used within AmpliGraph.


Training
""""""""

While fitting a model, AmpliGraph accepts either an object of ``AmpligraphDatasetAdapter`` class or a numpy array.
To support backward compatibility, we support numpy arrays as inputs. However, internally we adapt this data in our
NumpyAdapter and then the data is consumed as described below.

AmpliGraph calls ``generate_mappings()`` of the adapter object to generate the dictionary of entity/relation to
index mappings. It then calls ``map_data`` to map the data from entity to idx if not already done.
To get batches of train data, AmpliGraph uses the ``get_next_batch`` generator.
It uses the ``get_size`` method to determine the size of the dataset.


Evaluation Procedure
""""""""""""""""""""

While evaluating the performance of the models, AmpliGraph supports either an object of ``AmpligraphDatasetAdapter``
class or a numpy array as input. Just like the fit function, we first adapt the data with the ``NumpyAdapter`` before
consuming. AmpliGraph accepts numpy array as ``filter_triples`` for backward compatibility (if the test triples are also
passed as numpy arrays); if not, it expects the Adapter to know how to filter (this is indicated by passing ``True``
to ``filter_triples`` instead of a numpy array).
The evaluate_performance method then passes the handle of this ``data_adapter`` to the ``get_ranks()`` method.
The evaluation procedure is as described below.

The ``get_ranks()`` method generates ranks for all the test triples. In order to generate the test triples it uses the
``get_next_batch()`` generator of the data_adapter with appropriate dataset type and use_filters flag,
depending on whether the filters are set or not. With ``get_next_batch()`` and ``use_filters=False``, AmpliGraph expects a batch of test triples; whereas with ``get_next_batch`` method and ``use_filters=True``, it expects the test triple along with the indices of all the subject and object entities that were involved in the ?-p-o and s-p-? relations.
It uses the ``get_size`` method to determine the size of the dataset.

Once the batch of test triples are generated (along with the filter indices - for filtering mode), the test triples
and the corresponding corruptions are scored and ranked.


Dealing with Large Graphs
-------------------------

In the context of this discussion, large graph means graphs whose embeddings do not fit in the GPU memory. For example,
with Complex model (k=200) for 10 million distinct entities,
one would need 10 million * 200 * 2(for real/imaginary) * 4(float 32) bytes of GPU memory (approximately 15 GB of
GPU just for holding the embeddings). Hence on a normal GPU, this would not fit. The user would be forced to move to
GPU to do the computations which would slow down the training/evaluation.

To avoid this, and make use of GPU cores for faster computations, we have introduced a mode to deal with large graphs.
As of now, you can specify whether a graph is large or not depending on the number of distinct entities.
It's set to 500,000 but can be changed using the ``set_entity_threshold()`` method in the ``latent_features`` module.
To reset it back to the default threshold, use the ``reset_entity_threshold()`` method.

In this mode, the entity embeddings are not created on the GPU. Instead, the embeddings are created on the CPU.
While training, we load embeddings of batch_size * 2 entities. In other words, in a batch we can get only a max of
batch_size * 2 entities i.e. subject and objects of the batch. However, in general, this number is always less than
that, as some of the entities might be repeated. In such cases we randomly select other entities that are not present
in the batch to make up that value. These entity embeddings are loaded on the GPU and the corruptions for training are
generated from these entity embeddings. This way, all the gradient computations happens on the GPU for that batch. The
updated variables are stored back on the CPU. This process is repeated for every training batch. In this way, we make
maximum use of the GPU for faster computation.

However, there is a drawback to this approach. Since we are loading and unloading the entity embeddings every batch,
we cant use optimizers other than SGD. The reason for this is that optimizers like Adam, adagrad, etc maintains
internally a different learning rate per parameter; and in our case we are changing the parameters every batch. So
these optimizers cannot be used. However, we have provided various other tricks with SGD to make up for this drawback
eg: SGD with sinusoidal/fixed decay and expanding cycling time, etc.

We use a similar approach during evaluation, were we generate corruptions in batches and load the embeddings as needed. 

In the large graph mode, the training/evaluation would be slower than usual as the embeddings need to be loaded/unloaded
from GPU every batch; however, it is still much faster than doing computations on CPU (using tensorflow cpu version and
normal AmpliGraph mode).

We have tested this approach with the fb15k dataset by explicitly setting large graph mode to just 100 entities and
using a batch count of 100. With batch count of 100, the batch size is approximately 4500. In other words we would load
approximately 4500 entity embeddings in GPU memory per batch (out of a total 14951 entities). The training slows down
by a small margin (it takes 1.5 times more per epoch than the usual mode due to the loading/unloading overhead).
However the evaluation performance is worse, since for each test triple, we generate all the possible corruptions and
this is further batched (only 4500 corruptions per batch). It takes a few hours. It is, however, much faster than
using tensorflow cpu.

If the user does not want to use this mode and prefers to use the normal mode (say, to make use of other optimizers
like Adam, etc while training), they can use the CPU version of the tensorflow and run AmpliGraph as usual.
They can increase the entity threshold to a number greater than the distinct entites in their use case and
then run AmpliGraph, so as to use the normal mode (instead of large graph mode - by default set to 500,000 entities).
However, since all the computations happen on the CPU it will be much slower.


A Note on SQLite Adapter
""""""""""""""""""""""""

This adapter can use an existing DB (if it uses AmpliGraph Schema) or can create a DB and store data in the
AmpliGraph Schema. We are providing this adapter, especially for people who want to use graph which have
billions of triples.

With our adapter, users can persist the data, in parts (if required), into the database. For example, if a user
has multiple files containing the triples data, then first they can create a mapping dictionary (concept to index)
that should be used to represent the distinct entities and relations. Next they can load each file and persist the
data in sql by specifying whether to use the data as train/test/valid. This can be repeated for each file and the
data can be extended in the database.

Once the data is created this way, the user can pass the adapter handle to the fit and evaluate function.
These functions will internally use the required APIs and consume data appropriately
as specified (i.e. train/test/valid).

.. code-block:: python

    #Usage for extremely large datasets:
    from AmpliGraph.datasets import SQLiteAdapter
    adapt = SQLiteAdapter()

    #compute the mappings from the large dataset.
    #Let's assume that the mappings are already computed in rel_to_idx, ent_to_idx. 
    #Set the mappings
    adapt.use_mappings(rel_to_idx, ent_to_idx)

    #load and store parts of data in the db as train test or valid
    #if you have already mapped the entity names to index, set mapped_status = True
    adapt.set_data(load_part1, 'train', mapped_status = True)
    adapt.set_data(load_part2, 'train', mapped_status = True)
    adapt.set_data(load_part3, 'train', mapped_status = True)

    #if mapped_status = False, then the adapter will map the entities to index before persisting
    adapt.set_data(load_part1, 'test', mapped_status = False)
    adapt.set_data(load_part2, 'test', mapped_status = False)

    adapt.set_data(load_part1, 'valid', mapped_status = False)
    adapt.set_data(load_part2, 'valid', mapped_status = False)

    #create the model
    model = ComplEx(batches_count=10000, seed=0, epochs=10, k=50, eta=10)
    model.fit(adapt)









