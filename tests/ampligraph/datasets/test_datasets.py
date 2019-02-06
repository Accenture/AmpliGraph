
from ampligraph.datasets import load_wn18, load_wn11, load_fb13, load_fb15k, load_fb15k_237


def test_load_wn18():

    wn18 = load_wn18()
    assert len(wn18['train']) == 141442
    assert len(wn18['valid']) == 5000
    assert len(wn18['test']) == 5000


def test_load_wn11():
    wn11 = load_wn11()
    # Hamaguchi17 reports numbers without filtering duplicates.
    # assert len(wn11['train']) == 112581
    # assert len(wn11['valid']) == 5218
    # assert len(wn11['test']) == 21088
    assert len(wn11['train']) == 110361
    assert len(wn11['valid']) == 5215
    assert len(wn11['test']) == 21035


def test_load_fb13():
    fb13 = load_fb13()
    # Hamaguchi17 reports numbers without filtering duplicates.
    # assert len(fb13['train']) == 316232
    # assert len(fb13['valid']) == 11816
    # assert len(fb13['test']) == 47466
    assert len(fb13['train']) == 316232
    assert len(fb13['valid']) == 11816
    assert len(fb13['test']) == 47464


def test_load_fb15k():
    fb15k = load_fb15k()
    assert len(fb15k['train']) == 483142
    assert len(fb15k['valid']) == 50000
    assert len(fb15k['test']) == 59071


def test_load_fb15k_237():
    fb15k_237 = load_fb15k_237()
    assert len(fb15k_237['train']) == 272115
    assert len(fb15k_237['valid']) == 17535
    assert len(fb15k_237['test']) == 20466
