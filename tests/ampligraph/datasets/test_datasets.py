# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from ampligraph.datasets import load_wn18, load_fb15k, load_fb15k_237, load_yago3_10, load_wn18rr
import numpy as np
import pytest


def test_load_wn18():
    wn18 = load_wn18()
    assert len(wn18['train']) == 141442
    assert len(wn18['valid']) == 5000
    assert len(wn18['test']) == 5000

    ent_train = np.union1d(np.unique(wn18["train"][:, 0]), np.unique(wn18["train"][:, 2]))
    ent_valid = np.union1d(np.unique(wn18["valid"][:, 0]), np.unique(wn18["valid"][:, 2]))
    ent_test = np.union1d(np.unique(wn18["test"][:, 0]), np.unique(wn18["test"][:, 2]))
    distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    distinct_rel = np.union1d(np.union1d(np.unique(wn18["train"][:, 1]), np.unique(wn18["train"][:, 1])),
                              np.unique(wn18["train"][:, 1]))

    assert len(distinct_ent) == 40943
    assert len(distinct_rel) == 18

def test_load_fb15k():
    fb15k = load_fb15k()
    assert len(fb15k['train']) == 483142
    assert len(fb15k['valid']) == 50000
    assert len(fb15k['test']) == 59071


    # ent_train = np.union1d(np.unique(fb15k["train"][:,0]), np.unique(fb15k["train"][:,2]))
    # ent_valid = np.union1d(np.unique(fb15k["valid"][:,0]), np.unique(fb15k["valid"][:,2]))
    # ent_test = np.union1d(np.unique(fb15k["test"][:,0]), np.unique(fb15k["test"][:,2]))
    # distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    # distinct_rel = np.union1d(np.union1d(np.unique(fb15k["train"][:,1]), np.unique(fb15k["train"][:,1])), np.unique(fb15k["train"][:,1]))

    # assert len(distinct_ent) == 14951  
    # assert len(distinct_rel) == 1345  


def test_load_fb15k_237():
    fb15k_237 = load_fb15k_237()
    assert len(fb15k_237['train']) == 272115 
    
    # - 9 because 9 triples containing unseen entities are removed
    assert len(fb15k_237['valid']) == 17535 - 9

    # - 28 because 28 triples containing unseen entities are removed
    assert len(fb15k_237['test']) == 20466 - 28

   


def test_yago_3_10():
    yago_3_10 = load_yago3_10()
    assert len(yago_3_10['train']) == 1079040 
    assert len(yago_3_10['valid']) == 5000 - 22
    assert len(yago_3_10['test']) == 5000 - 18

    # ent_train = np.union1d(np.unique(yago_3_10["train"][:,0]), np.unique(yago_3_10["train"][:,2]))
    # ent_valid = np.union1d(np.unique(yago_3_10["valid"][:,0]), np.unique(yago_3_10["valid"][:,2]))
    # ent_test = np.union1d(np.unique(yago_3_10["test"][:,0]), np.unique(yago_3_10["test"][:,2]))

    # assert len(set(ent_valid) - set(ent_train)) == 22
    # assert len (set(ent_test) - ((set(ent_valid) & set(ent_train)) | set(ent_train))) == 18

    # distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    # distinct_rel = np.union1d(np.union1d(np.unique(yago_3_10["train"][:,1]), np.unique(yago_3_10["train"][:,1])), np.unique(yago_3_10["train"][:,1]))

    # assert len(distinct_ent) == 123182  
    # assert len(distinct_rel) == 37  


def test_wn18rr():
    wn18rr = load_wn18rr()

    ent_train = np.union1d(np.unique(wn18rr["train"][:, 0]), np.unique(wn18rr["train"][:, 2]))
    ent_valid = np.union1d(np.unique(wn18rr["valid"][:, 0]), np.unique(wn18rr["valid"][:, 2]))
    ent_test = np.union1d(np.unique(wn18rr["test"][:, 0]), np.unique(wn18rr["test"][:, 2]))
    distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    distinct_rel = np.union1d(np.union1d(np.unique(wn18rr["train"][:, 1]), np.unique(wn18rr["train"][:, 1])),
                              np.unique(wn18rr["train"][:, 1]))

    assert len(wn18rr['train']) == 86835

    # - 210 because 210 triples containing unseen entities are removed
    assert len(wn18rr['valid']) == 3034 - 210

    # - 210 because 210 triples containing unseen entities are removed
    assert len(wn18rr['test']) == 3134 - 210
    
