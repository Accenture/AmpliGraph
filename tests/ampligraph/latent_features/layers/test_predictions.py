import tensorflow as tf
import numpy as np
from ampligraph.datasets import load_fb15k_237
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.latent_features import PairwiseLoss
from ampligraph.utils import save_model, restore_model


def test_reproducible_predictions():
    X = np.array([['a', 'x', 'b'],
                  ['a', 'y', 'c'],
                  ['a', 'y', 'd'],
                  ['a', 'x', 'e'],
                  ['e', 'x', 'b'],
                  ['b', 'y', 'd'],
                  ['c', 'x', 'e']])

    model = ScoringBasedEmbeddingModel(k=50,
                                       eta=1,
                                       scoring_type='TransE',
                                       seed=0)

    adam = tf.keras.optimizers.Adam(learning_rate=0.001)

    loss = PairwiseLoss(loss_params={'margin': 0.5})

    model.compile(optimizer=adam,
                  loss=loss,
                  entity_relation_initializer='glorot_uniform',
                  entity_relation_regularizer='L2')

    model.fit(X,
              batch_size=10,
              epochs=5,
              verbose=True)

    assert (model.predict(X[:2, ]) == np.array([-2.2272136, -2.429057], dtype=np.float32).all(),
            'Prediction scores have changed.')


def test_reproducible_predictions_fb15k237():
    X = load_fb15k_237()

    model = ScoringBasedEmbeddingModel(k=50, eta=1, scoring_type='TransE', seed=0)

    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = PairwiseLoss(loss_params={'margin': 0.5})

    model.compile(optimizer=adam,
                  loss=loss,
                  entity_relation_initializer='glorot_uniform',
                  entity_relation_regularizer='L2')

    model.fit(X['train'],
              batch_size=30000,
              epochs=5,
              verbose=True)

    print(model.predict(X['test'][:2, ]))

    assert (model.predict(X['test'][:2, ]) == np.array([-0.3415385, -0.4203454], dtype=np.float32).all(),
            'Prediction scores have changed.')


def test_reproducible_predictions_restored_model():

    X = np.array([['a', 'x', 'b'],
                  ['a', 'y', 'c'],
                  ['a', 'y', 'd'],
                  ['a', 'x', 'e'],
                  ['e', 'x', 'b'],
                  ['b', 'y', 'd'],
                  ['c', 'x', 'e']])

    model = ScoringBasedEmbeddingModel(k=50,
                                       eta=1,
                                       scoring_type='TransE',
                                       seed=0)
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = PairwiseLoss(loss_params={'margin': 0.5})
    model.compile(optimizer=adam,
                  loss=loss,
                  entity_relation_initializer='glorot_uniform',
                  entity_relation_regularizer='L2')
    model.fit(X,
              batch_size=10,
              epochs=5,
              verbose=True)

    save_model(model, 'test_reproducible_predictions_restored_model')

    model_restored = restore_model(model_name_path='test_reproducible_predictions_restored_model')

    assert (model_restored.predict(X[:2, ]) == np.array([-2.2272136, -2.429057], dtype=np.float32).all(),
            'Prediction scores from restored model do not match the originals')