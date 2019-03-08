import pytest

# TODO: TEC-1819: Enable coexisting eager and graph tf tests:
# see https://stackoverflow.com/questions/50143896/both-eager-and-graph-execution-in-tensorflow-tests
pytestmark = pytest.mark.skip("Skipping to avoid failures in latent feature model tests caused by tf.eager."
                               "Will use Autograph feature in KnowEvolve code in the future to get rid of eager.")

import numpy as np
from ampligraph.evaluation.metrics import mar_score
import tensorflow as tf

X_bis = np.array([[2.0, 18.0, 0.0, 0.0],
                      [153.0, 5.0, 9.0, 0.0],
                      [103.0, 0.0, 82.0, 0.0],  # seen in the train
                      [153.0, 5.0, 56.0, 0.0],
                      [5.0, 17.0, 60.0, 0.0],
                      [9.0, 6.0, 153.0, 24.0],
                      [153.0, 5.0, 3.0, 0.0],
                      [44.0, 10.0, 1.0, 0.0],  # with 0 seen
                      [3.0, 6.0, 153.0, 0.0],
                      [242.0, 0.0, 8.0, 24.0],
                      [7.0, 1.0, 170.0, 24.0],
                      [56.0, 6.0, 153.0, 0.0],
                      [211.0, 6.0, 40.0, 24.0],
                      [60.0, 17.0, 5.0, 0.0],
                      [170.0, 1.0, 7.0, 0.0],  # seen
                      [55.0, 118.0, 43.0, 0.0],
                      [36.0, 6.0, 153.0, 0.0],
                      [40.0, 5.0, 211.0, 24.0],
                      [257.0, 0.0, 66.0, 0.0],
                      [9.0, 8.0, 26.0, 0.0]  # also seen
                      ])
X_mask = np.array([-1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

def test_load_ICEWS_r():
    datas=load_ICEWS_reduced()
    assert datas['train'][1,2]==54

def test_KnowEvolve_running():
    from ampligraph.temporal import KnowEvolve
    X = np.array([
        [0, 4, 7, 5.257244305824649],
        [4, 0, 1, 5.424818694879746],
        [9, 0, 6, 5.479774175414377],
        [8, 0, 6, 5.528975403788779],
        [6, 1, 9, 5.731247947359487],
        [3, 3, 7, 5.741413035549908],
        [6, 4, 0, 5.749306503402599],
        [6, 2, 1, 5.907582200331214],
        [8, 3, 2, 6.2633789148495005],
        [7, 1, 9, 6.3488699640368225],
        [7, 1, 5, 6.3869577883721975],
        [5, 2, 2, 6.468355387856009],
        [5, 4, 7, 7.074951676048747],
        [4, 3, 3, 7.114139126265265], ])
    KnE = KnowEvolve(max_iter=2, batch_size=3, lr=0.1, nb_nodes=10, nb_rel=5, dim_embed=6, dim_rel=3)
    KnE.fit(X)
    assert KnE.is_fitted

def test_load_and_fit_ICEWS_reduced():
    from ampligraph.temporal import KnowEvolve
    datas = load_ICEWS_reduced()
    X=datas['train']
    assert not np.any(np.isnan(X))
    KnE = KnowEvolve(max_iter=1, batch_size=20, lr=0.02, nb_nodes=500, nb_rel=260, dim_embed=10, dim_rel=5)
    KnE.fit(X[:1000])
    assert KnE.is_fitted

def test_predict():
    from ampligraph.temporal import KnowEvolve
    datas = load_ICEWS_reduced()
    X = datas['train']
    KnE = KnowEvolve(max_iter=1, batch_size=20, lr=0.02, nb_nodes=500, nb_rel=260, dim_embed=10, dim_rel=5)
    KnE.fit(X[:500])
    scores = KnE.predict(X_bis, X_mask, update_embeddings=False)
    assert (tf.reduce_all(scores>0))
    scores = KnE.predict(X_bis, X_mask, update_embeddings=True)
    assert (tf.reduce_all(scores > 0))


def test_predict_time():
    from ampligraph.temporal import KnowEvolve
    datas = load_ICEWS_reduced()
    X = datas['train']
    KnE = KnowEvolve(max_iter=1, batch_size=20, lr=0.02, nb_nodes=500, nb_rel=260, dim_embed=10, dim_rel=5)
    KnE.fit(X[:500])

    expctd_time = KnE.predict_time(X_bis)
    assert (tf.reduce_all(expctd_time > 0))

def test_feed_running():
    from ampligraph.temporal import KnowEvolve
    datas = load_ICEWS_reduced()
    X = datas['train']
    KnE = KnowEvolve(max_iter=1, batch_size=20, lr=0.02, nb_nodes=500, nb_rel=260, dim_embed=10, dim_rel=5)
    KnE.fit(X[:500])
    KnE.feed(X_bis)

def test_save_and_restore():
    from ampligraph.temporal import KnowEvolve
    path = '/media/sf_Shared_with_VM/savingTest/'
    name = 'model1'
    datas = load_ICEWS_reduced()
    X = datas['train']
    KnE = KnowEvolve(max_iter=1, batch_size=20, lr=0.02, nb_nodes=500, nb_rel=260, dim_embed=10, dim_rel=5)
    KnE.fit(X[:100])
    KnE.save(path, name)
    # need to try with differents parameters, like batch_size=25
    kne_bis = KnowEvolve(max_iter=2, batch_size=20, lr=0.02, nb_nodes=500, nb_rel=260, dim_embed=10, dim_rel=5)
    kne_bis.restore(path, name)

    kne_bis_variables = [kne_bis.model.rEmb, kne_bis.model.RREmb, kne_bis.model.V, kne_bis.model.Whh, kne_bis.model.Wh,
                         kne_bis.model.Wts, kne_bis.model.Wto]
    KnE_variables = [KnE.model.rEmb, KnE.model.RREmb, KnE.model.V, KnE.model.Whh, KnE.model.Wh, KnE.model.Wts,
                     KnE.model.Wto]
    for t1, t2 in list(zip(kne_bis_variables, KnE_variables)):
        print('testing')
        assert (tf.squeeze(tf.reduce_all(tf.equal(t1, t2))))
    assert (tf.reduce_all(tf.equal(kne_bis.predict(X[100:120], X_mask), KnE.predict(X[100:120], X_mask))))

def test_later_training():
    from ampligraph.temporal import KnowEvolve
    #work without the tensorflow seed, means the optimizer is well stored
    path = '/media/sf_Shared_with_VM/savingTest/'
    name = 'model1'
    datas = load_ICEWS_reduced()
    X = datas['train']
    KnE = KnowEvolve(max_iter=1, batch_size=20, lr=0.02, nb_nodes=500, nb_rel=260, dim_embed=10, dim_rel=5)
    KnE.fit(X[:100])
    KnE.save(path, name)
    kne_bis = KnowEvolve(max_iter=2, batch_size=20, lr=0.02, nb_nodes=500, nb_rel=260, dim_embed=10, dim_rel=5)
    kne_bis.restore(path, name)

    #second training, indepedent
    KnE.fit(X[:100])
    kne_bis.fit(X[:100])

    #test
    kne_bis_variables = [kne_bis.model.rEmb, kne_bis.model.RREmb, kne_bis.model.V, kne_bis.model.Whh, kne_bis.model.Wh,
                         kne_bis.model.Wts, kne_bis.model.Wto]
    KnE_variables = [KnE.model.rEmb, KnE.model.RREmb, KnE.model.V, KnE.model.Whh, KnE.model.Wh, KnE.model.Wts,
                     KnE.model.Wto]
    for t1, t2 in list(zip(kne_bis_variables, KnE_variables)):
        print('testing')
        assert (tf.squeeze(tf.reduce_all(tf.equal(t1, t2))))
    assert (tf.reduce_all(tf.equal(kne_bis.predict(X[100:120], X_mask), KnE.predict(X[100:120], X_mask))))


def test_preprocess_predict():
    from ampligraph.temporal import KnowEvolve
    X_bis = np.array([[2.0, 18.0, 0.0, 0.0], #-1
                      [153.0, 5.0, 9.0, 0.0], #-1
                      [103.0, 0.0, 82.0, 0.0], #-1 # seen in the train
                      [153.0, 5.0, 56.0, 0.0], #-1
                      [5.0, 17.0, 60.0, 0.0],  #1
                      [9.0, 6.0, 153.0, 24.0], #-1
                      [153.0, 5.0, 3.0, 0.0], #-1
                      [44.0, 10.0, 1.0, 0.0],  #-1 # with 0 seen
                      [3.0, 6.0, 153.0, 0.0],  #1
                      [242.0, 0.0, 8.0, 24.0], #1
                      [7.0, 1.0, 170.0, 24.0], #-1
                      [56.0, 6.0, 153.0, 0.0], #1
                      [211.0, 6.0, 40.0, 24.0], #-1
                      [60.0, 17.0, 5.0, 0.0], #-1
                      [170.0, 1.0, 7.0, 0.0], #-1 # seen
                      [55.0, 118.0, 43.0, 0.0], #-1
                      [36.0, 6.0, 153.0, 0.0], #-1
                      [40.0, 5.0, 211.0, 24.0], #-1
                      [257.0, 0.0, 66.0, 0.0], #-1
                      [9.0, 8.0, 26.0, 0.0] #-1 # also seen
                      ])
    X_mask = np.array([-1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1])
    KnE = KnowEvolve(max_iter=1, batch_size=20, lr=0.02, nb_nodes=500, nb_rel=260, dim_embed=10, dim_rel=5)
    batches, cutting_in_batches, X_idx=KnE.preprocess_predict(X_bis, X_mask)
    assert(len(batches)==2)
    assert(len(batches[0])==15)
    assert(len(batches[1])==5)
    assert(cutting_in_batches[0]==12)
    assert(cutting_in_batches[1] == 4)
    assert(X_idx[19]==9)

def test_previous_table():
    from ampligraph.temporal import KnowEvolve
    X = np.array([
        [0, 4, 7, 5.],
        [4, 0, 1, 5.],
        [9, 0, 6, 5.],
        [8, 0, 13, 5.2],
        [6, 1, 9, 5.7],
        [3, 3, 7, 5.8],
        [6, 4, 0, 5.9],
        [6, 2, 1, 6.0],
        [11, 3, 2, 6.26],
        [7, 1, 9, 6.29],
        [7, 1, 5, 6.38],
        [5, 2, 2, 6.4],
        [5, 4, 7, 7.0],
        [4, 3, 3, 7.0], ])
    KnE = KnowEvolve(max_iter=2, batch_size=3, lr=0.1, nb_nodes=20, nb_rel=5, dim_embed=6, dim_rel=3)
    KnE.fit(X)
    tp=KnE.tp
    rp=KnE.rp
    assert(tp[11,:].tolist()==[-24.0, 6.26])
    assert(tp[2,:].tolist()==[6.26, 6.4])
    assert(tp[15, :].tolist()==[-24.0, -24.0])
    assert(rp[5]==4.0)
    assert(rp[15]==-1.0)

def test_mar_ranks():
    ranks=[8,9,10,3,1,6]
    assert(mar_score(ranks)==6+1/6)




