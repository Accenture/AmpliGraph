import tensorflow as tf
import numpy as np
from ampligraph.latent_features.models import ScoringBasedEmbeddingModel
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.optimizers import get as get_optimizer
from ampligraph.latent_features.regularizers import get as get_regularizer

BACK_COMPAT_MODELS = {}


def register_compatibility(name):
    def insert_in_registry(class_handle):
        BACK_COMPAT_MODELS[name] = class_handle
        class_handle.name = name
        return class_handle
    return insert_in_registry


class ScoringModelBase: 
    def __init__(self, k=100, eta=2, epochs=100, 
                 batches_count=100, seed=0, 
                 embedding_model_params={'corrupt_sides': ['s,o'], 
                                         'negative_corruption_entities': 'all', 
                                         'norm': 1, 'normalize_ent_emb': False}, 
                  
                 optimizer='adam', optimizer_params={'lr': 0.0005}, 
                 loss='nll', loss_params={}, regularizer=None, regularizer_params={}, 
                 initializer='xavier', initializer_params={'uniform': False}, verbose=False, 
                 model=None):
        if model is not None:
            self.model_name = model.scoring_type
        else:
            self.k = k
            self.eta = eta
            self.seed = seed

            self.batches_count = batches_count

            self.epochs = epochs
            self.embedding_model_params = embedding_model_params
            self.optimizer = optimizer
            self.optimizer_params = optimizer_params
            self.loss = loss
            self.loss_params = loss_params
            self.initializer = initializer
            self.initializer_params = initializer_params
            self.regularizer = regularizer
            self.regularizer_params = regularizer_params
            self.verbose = verbose
            
        self.model = model
        self.is_backward = True
        
    def _get_optimizer(self, optimizer, optim_params):
        learning_rate = optim_params.get('lr', 0.001)
        del optim_params['lr']
        
        if optimizer == 'adam':
            optim = tf.keras.optimizers.Adam(learning_rate=learning_rate, **optim_params)
            status = True
        elif optimizer == 'adagrad':
            optim = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, **optim_params)
            status = True
        elif optimizer == 'sgd':
            optim = tf.keras.optimizers.SGD(learning_rate=learning_rate, **optim_params)
            status = True
        else:
            optim = get_optimizer(optimizer)
            status = False

        optim_params['lr'] = learning_rate
        return optim, status
        
    def is_fit(self):
        return self.model.is_fit()
        
    def _get_initializer(self, initializer, initializer_params):
        if initializer == 'xavier':
            if initializer_params['uniform']:
                return tf.keras.initializers.GlorotUniform(seed=self.seed)
            else:
                return tf.keras.initializers.GlorotNormal(seed=self.seed)
        elif initializer == 'uniform':
            return tf.keras.initializers.RandomUniform(minval=initializer_params.get('low', -0.05), 
                                                       maxval=initializer_params.get('high', 0.05), 
                                                       seed=self.seed)
        elif initializer == 'normal':
            return tf.keras.initializers.RandomNormal(mean=initializer_params.get('mean', 0.0), 
                                                      stddev=initializer_params.get('std', 0.05), 
                                                      seed=self.seed)
        elif initializer == 'constant':
            entity_init = initializer_params.get('entity', None)
            rel_init = initializer_params.get('relation', None)
            assert entity_init is not None, 'Please pass the `entity` initializer value'
            assert rel_init is not None, 'Please pass the `relation` initializer value'
            return [tf.constant_initializer(entity_init), tf.constant_initializer(rel_init)]
        else:
            return tf.keras.initializers.get(initializer)
        
    def fit(self, X, 
            early_stopping=False, 
            early_stopping_params={}, 
            tensorboard_logs_path=None, callbacks=None):
        self.model = ScoringBasedEmbeddingModel(self.eta, self.k, scoring_type=self.model_name, seed=self.seed)
        if callbacks is None:
            callbacks = []
        if tensorboard_logs_path is not None:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_path)
            callbacks.append(tensorboard_callback)
        
        regularizer = self.regularizer
        if regularizer is not None:
            regularizer = get_regularizer(regularizer, self.regularizer_params)
            
        initializer = self.initializer
        if initializer is not None:
            initializer = self._get_initializer(initializer, self.initializer_params)
            
        loss = get_loss(self.loss, self.loss_params)
        optimizer, is_back_compat_optim = self._get_optimizer(self.optimizer, self.optimizer_params)
        
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           entity_relation_initializer=initializer,
                           entity_relation_regularizer=regularizer)
        if not is_back_compat_optim:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, 
                                       self.optimizer_params.get('lr', 0.001))

        if len(early_stopping_params) != 0:
            checkpoint = tf.keras.callbacks.EarlyStopping(
                monitor='val_{}'.format(
                    early_stopping_params.get('criteria', 'mrr')), 
                min_delta=0, 
                patience=early_stopping_params.get('stop_interval', 10), 
                verbose=self.verbose,
                mode='max', 
                restore_best_weights=True)
            callbacks.append(checkpoint)

        x_filter = early_stopping_params.get('x_filter', None)

        if isinstance(x_filter, np.ndarray) or isinstance(x_filter, list):
            x_filter = {'test': x_filter}
        elif x_filter is None or not x_filter:
            x_filter = False
        else:
            raise ValueError('Incorrect type for x_filter')
            
        self.model.fit(X,
                       batch_size=np.ceil(X.shape[0] / self.batches_count),
                       epochs=self.epochs,
                       validation_freq=early_stopping_params.get('check_interval', 10),
                       validation_burn_in=early_stopping_params.get('burn_in', 25),
                       validation_batch_size=early_stopping_params.get('batch_size', 100),
                       validation_data=early_stopping_params.get('x_valid', None),
                       validation_filter=x_filter,
                       validation_entities_subset=early_stopping_params.get('corruption_entities', None),
                       callbacks=callbacks)
        
    def get_indexes(self, X, type_of='t', order="raw2ind"):
        """Converts given data to indexes or to raw data (according to order), works for 
           both triples (type_of='t'), entities (type_of='e'), and relations (type_of='r').
           
           Parameters
           ----------
           X: np.array
               data to be indexed.
           type_of: str
               one of ['e', 't', 'r']
           order: str 
               one of ['raw2ind', 'ind2raw']

           Returns
           -------
           Y: np.array
               indexed data
        """
        return self.model.get_indexes(X, type_of, order)
        
    def get_count(self, concept_type='e'):
        ''' Returns the count of entities and relations that were present during training.
        
            Parameters
            ----------
            concept_type: str
                Indicates whether it is entity 'e' or relation 'r'. Default is 'e'
                
            Returns
            -------
            count: int
                count of the entities or relations
        '''
        if embedding_type == 'entity' or embedding_type == 'e':
            return self.model.get_count('e')
        elif embedding_type == 'relation' or embedding_type == 'r':
            return self.model.get_count('r')
        else:
            raise ValueError('Invalid value for concept_type!')
        
    def get_embeddings(self, entities, embedding_type='entity'):
        """Get the embeddings of entities or relations.
        
        .. Note ::

            Use :meth:`ampligraph.utils.create_tensorboard_visualizations` to visualize the embeddings with TensorBoard.
            
        Parameters
        ----------
        entities : array-like, dtype=int, shape=[n]
            The entities (or relations) of interest. Element of the vector must be the original string literals, and
            not internal IDs.
        embedding_type : string
            If 'e' or 'entities', ``entities`` argument will be considered as a list of knowledge graph entities (i.e. nodes).
            If set to 'r' or 'relation', they will be treated as relation types instead (i.e. predicates).
        Returns
        -------
        embeddings : ndarray, shape [n, k]
            An array of k-dimensional embeddings.
        """
        if embedding_type == 'entity' or embedding_type == 'e':
            return self.model.get_embeddings(entities, 'e')
        elif embedding_type == 'relation' or embedding_type == 'r':
            return self.model.get_embeddings(entities, 'r')
        else:
            raise ValueError('Invalid value for embedding_type!')
    
    def get_hyperparameter_dict(self):
        ent_idx = np.arange(self.model.data_indexer.get_entities_count())
        rel_idx = np.arange(self.model.data_indexer.get_relations_count())
        ent_values_raw = self.model.data_indexer.get_indexes(ent_idx, 'e', 'ind2raw')
        rel_values_raw = self.model.data_indexer.get_indexes(rel_idx, 'r', 'ind2raw')
        return dict(zip(ent_values_raw, ent_idx)), dict(zip(rel_values_raw, rel_idx))
    
    def predict(self, X):
        return self.model.predict(X)
    
    def calibrate(self, X_pos, X_neg=None, positive_base_rate=None, batches_count=100, epochs=50):
        batch_size = int(np.ceil(X_pos.shape[0] / batches_count))
        return self.model.calibrate(X_pos, 
                                    X_neg, 
                                    positive_base_rate, 
                                    batch_size, 
                                    epochs)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, 
                 x=None,
                 batch_size=32,
                 verbose=True,
                 use_filter=False,
                 corrupt_side='s,o',
                 entities_subset=None,
                 callbacks=None):
        return self.model.evaluate(x, 
                                   batch_size=batch_size, 
                                   verbose=verbose, 
                                   use_filter=use_filter, 
                                   corrupt_side=corrupt_side, 
                                   entities_subset=entities_subset, 
                                   callbacks=callbacks)

    
@register_compatibility('TransE')
class TransE(ScoringModelBase):
    def __init__(self, k=100, eta=2, epochs=100, 
                 batches_count=100, seed=0, 
                 embedding_model_params={'corrupt_sides': ['s,o'], 
                                         'negative_corruption_entities': 'all', 
                                         'norm': 1, 'normalize_ent_emb': False}, 
                  
                 optimizer='adam', optimizer_params={'lr': 0.0005}, 
                 loss='nll', loss_params={}, regularizer=None, regularizer_params={}, 
                 initializer='xavier', initializer_params={'uniform': False}, verbose=False, model=None):
        super().__init__(k, eta, epochs, 
                         batches_count, seed, 
                         embedding_model_params, 
                         optimizer, optimizer_params, 
                         loss, loss_params, regularizer, regularizer_params, 
                         initializer, initializer_params, verbose, model)
        
        self.model_name = 'TransE'

        
@register_compatibility('DistMult')
class DistMult(ScoringModelBase):
    def __init__(self, k=100, eta=2, epochs=100, 
                 batches_count=100, seed=0, 
                 embedding_model_params={'corrupt_sides': ['s,o'], 
                                         'negative_corruption_entities': 'all', 
                                         'norm': 1, 'normalize_ent_emb': False}, 
                  
                 optimizer='adam', optimizer_params={'lr': 0.0005}, 
                 loss='nll', loss_params={}, regularizer=None, regularizer_params={}, 
                 initializer='xavier', initializer_params={'uniform': False}, verbose=False, model=None):
        super().__init__(k, eta, epochs, 
                         batches_count, seed, 
                         embedding_model_params, 
                         optimizer, optimizer_params, 
                         loss, loss_params, regularizer, regularizer_params, 
                         initializer, initializer_params, verbose, model)
        
        self.model_name = 'DistMult'
        
        
@register_compatibility('ComplEx')
class ComplEx(ScoringModelBase):
    def __init__(self, k=100, eta=2, epochs=100, 
                 batches_count=100, seed=0, 
                 embedding_model_params={'corrupt_sides': ['s,o'], 
                                         'negative_corruption_entities': 'all', 
                                         'norm': 1, 'normalize_ent_emb': False}, 
                  
                 optimizer='adam', optimizer_params={'lr': 0.0005}, 
                 loss='nll', loss_params={}, regularizer=None, regularizer_params={}, 
                 initializer='xavier', initializer_params={'uniform': False}, verbose=False, model=None):
        super().__init__(k, eta, epochs, 
                         batches_count, seed, 
                         embedding_model_params, 
                         optimizer, optimizer_params, 
                         loss, loss_params, regularizer, regularizer_params, 
                         initializer, initializer_params, verbose, model)
        
        self.model_name = 'ComplEx'
        
        
@register_compatibility('HolE')
class HolE(ScoringModelBase):
    def __init__(self, k=100, eta=2, epochs=100, 
                 batches_count=100, seed=0, 
                 embedding_model_params={'corrupt_sides': ['s,o'], 
                                         'negative_corruption_entities': 'all', 
                                         'norm': 1, 'normalize_ent_emb': False}, 
                  
                 optimizer='adam', optimizer_params={'lr': 0.0005}, 
                 loss='nll', loss_params={}, regularizer=None, regularizer_params={}, 
                 initializer='xavier', initializer_params={'uniform': False}, verbose=False, model=None):
        super().__init__(k, eta, epochs, 
                         batches_count, seed, 
                         embedding_model_params, 
                         optimizer, optimizer_params, 
                         loss, loss_params, regularizer, regularizer_params, 
                         initializer, initializer_params, verbose, model)
        
        self.model_name = 'HolE'        
