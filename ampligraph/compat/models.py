import tensorflow as tf
import numpy as np
from ampligraph.latent_features.models import ScoringBasedEmbeddingModel
from ampligraph.latent_features.loss_functions import get as get_loss

from ampligraph.latent_features.regularizers import get as get_regularizer
     
class ScoringModelBase:
    def __init__(self, k=100, eta=2, epochs=100, 
                  batches_count=100, seed=0, 
                  embedding_model_params={'corrupt_sides': ['s,o'], 
                                          'negative_corruption_entities': 'all', 
                                          'norm': 1, 'normalize_ent_emb': False}, 
                  
                  optimizer='adam', optimizer_params={'lr': 0.0005}, 
                  loss='nll', loss_params={}, regularizer=None, regularizer_params={}, 
                  initializer='xavier', initializer_params={'uniform': False}, verbose=False):
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
        
    def _get_optimizer(self, learning_rate):
        if self.optimizer=='adam':
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif self.optimizer=='adagrad':
            return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif self.optimizer=='sgd':
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        
    def fit(self, X, 
            early_stopping=False, 
            early_stopping_params={}, 
            tensorboard_logs_path=None):
        self.model = ScoringBasedEmbeddingModel(self.eta, self.k, scoring_type=self.model_name, seed=self.seed)
        callbacks = []
        if tensorboard_logs_path is not None:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_path)
            callbacks.append(tensorboard_callback)
            
        
        optim = self._get_optimizer(self.optimizer_params.get('lr', 0.001))
        
        regularizer = self.regularizer
        if regularizer is not None:
            regularizer = get_regularizer(regularizer, self.regularizer_params)
        self.model.compile(optimizer=optim,
                        loss=get_loss(self.loss, self.loss_params),
                        entity_relation_initializer='glorot_uniform',
                        entity_relation_regularizer=regularizer)
        
        
    
        if len(early_stopping_params) != 0:
            checkpoint = tf.keras.callbacks.EarlyStopping(
                            monitor='val_{}'.format(early_stopping_params.get('criteria', 10)), 
                            min_delta=0, 
                            patience=early_stopping_params.get('stop_interval', 10), 
                            verbose=0,
                            mode='max', 
                            restore_best_weights=True)
            callbacks.append(checkpoint)
            
        self.model.fit(X,
                 batch_size=np.ceil(X.shape[0] / self.batches_count),
                 epochs=self.epochs,
                 validation_freq=early_stopping_params.get('check_interval', 10),
                 validation_batch_size=early_stopping_params.get('batch_size', 1),
                 validation_data=early_stopping_params.get('x_valid', None),
                 validation_filter={'test': early_stopping_params.get('x_filter', None)},
                 callbacks=callbacks)
        
    def get_embeddings(self, entities, embedding_type='entity'):
        if embedding_type == 'entity':
            self.get_embeddings(entities, 'e')
        elif embedding_type == 'relation':
            self.get_embeddings(entities, 'r')
        else:
            raise ValueError('Invalid value for embedding_type!')
    
    def get_hyperparameter_dict(self):
        ent_idx = np.arange(self.model.data_indexer.get_entities_count())
        rel_idx = np.arange(self.model.data_indexer.get_relations_count())
        ent_values_raw = self.model.data_indexer.get_indexes(ent_idx,
                                                         'e', 'ind2raw')
        rel_values_raw = self.model.data_indexer.get_indexes(rel_idx,
                                                         'r', 'ind2raw')
        return dict(zip(ent_values_raw, ent_idx)), dict(zip(rel_values_raw, rel_idx))
    
    def predict(self, X):
        return self.model.predict(X)
    
    def calibrate(self, X_pos, X_neg=None, positive_base_rate=None, batches_count=100, epochs=50):
        batch_size = int(np.ceil(X_pos.shape[0] / batches_count))
        return self.model.calibrate(X_pos, 
                                    X_neg, 
                                    positive_base_rate, 
                                    batches_count, 
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
                                   batch_size, 
                                   verbose, 
                                   use_filter, 
                                   corrupt_side, 
                                   entities_subset, 
                                   callbacks)
        
class TransE(ScoringModelBase):
    def __init__(self, k=100, eta=2, epochs=100, 
                  batches_count=100, seed=0, 
                  embedding_model_params={'corrupt_sides': ['s,o'], 
                                          'negative_corruption_entities': 'all', 
                                          'norm': 1, 'normalize_ent_emb': False}, 
                  
                  optimizer='adam', optimizer_params={'lr': 0.0005}, 
                  loss='nll', loss_params={}, regularizer=None, regularizer_params={}, 
                  initializer='xavier', initializer_params={'uniform': False}, verbose=False):
        super().__init__(k, eta, epochs, 
                          batches_count, seed, 
                          embedding_model_params, 
                          optimizer, optimizer_params, 
                          loss, loss_params, regularizer, regularizer_params, 
                          initializer, initializer_params, verbose)
        
        self.model_name = 'TransE'
        self.model = None
        
class DistMult(ScoringModelBase):
    def __init__(self, k=100, eta=2, epochs=100, 
                  batches_count=100, seed=0, 
                  embedding_model_params={'corrupt_sides': ['s,o'], 
                                          'negative_corruption_entities': 'all', 
                                          'norm': 1, 'normalize_ent_emb': False}, 
                  
                  optimizer='adam', optimizer_params={'lr': 0.0005}, 
                  loss='nll', loss_params={}, regularizer=None, regularizer_params={}, 
                  initializer='xavier', initializer_params={'uniform': False}, verbose=False):
        super().__init__(k, eta, epochs, 
                          batches_count, seed, 
                          embedding_model_params, 
                          optimizer, optimizer_params, 
                          loss, loss_params, regularizer, regularizer_params, 
                          initializer, initializer_params, verbose)
        
        self.model_name = 'DistMult'
        self.model = None
        
class ComplEx(ScoringModelBase):
    def __init__(self, k=100, eta=2, epochs=100, 
                  batches_count=100, seed=0, 
                  embedding_model_params={'corrupt_sides': ['s,o'], 
                                          'negative_corruption_entities': 'all', 
                                          'norm': 1, 'normalize_ent_emb': False}, 
                  
                  optimizer='adam', optimizer_params={'lr': 0.0005}, 
                  loss='nll', loss_params={}, regularizer=None, regularizer_params={}, 
                  initializer='xavier', initializer_params={'uniform': False}, verbose=False):
        super().__init__(k, eta, epochs, 
                          batches_count, seed, 
                          embedding_model_params, 
                          optimizer, optimizer_params, 
                          loss, loss_params, regularizer, regularizer_params, 
                          initializer, initializer_params, verbose)
        
        self.model_name = 'ComplEx'
        self.model = None
        
class HolE(ScoringModelBase):
    def __init__(self, k=100, eta=2, epochs=100, 
                  batches_count=100, seed=0, 
                  embedding_model_params={'corrupt_sides': ['s,o'], 
                                          'negative_corruption_entities': 'all', 
                                          'norm': 1, 'normalize_ent_emb': False}, 
                  
                  optimizer='adam', optimizer_params={'lr': 0.0005}, 
                  loss='nll', loss_params={}, regularizer=None, regularizer_params={}, 
                  initializer='xavier', initializer_params={'uniform': False}, verbose=False):
        super().__init__(k, eta, epochs, 
                          batches_count, seed, 
                          embedding_model_params, 
                          optimizer, optimizer_params, 
                          loss, loss_params, regularizer, regularizer_params, 
                          initializer, initializer_params, verbose)
        
        self.model_name = 'HolE'
        self.model = None
    
                          
