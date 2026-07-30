# ImbalancE clusters

This folder holds the **LLM-embedding nearest-neighbour clusters** consumed by the
ImbalancE experiments, together with everything needed to regenerate them.

Each cluster file (`<dataset>/<dataset>_top<k>_<model>.pkl`) is a pickle of

```python
{
    "sp": {(subject_id, relation_id): [closest_entity_id, ...], ...},
    "po": {(object_id,  relation_id): [closest_entity_id, ...], ...},
}
```

where all IDs are the **raw dataset IDs** (e.g. `/m/0102t4` for FB15k-237). The
experiment code (`experiments/LatentSearch/code/main.py`, via
`ampligraph.latent_features.models.ImbalancE`) loads these through the
`--llm-integration <path>` argument.

The build pipeline is pure **PyTorch** and is deliberately isolated from AmpliGraph's
TensorFlow environment. The only bridge to AmpliGraph is `cache_datasets.py`, used once
to materialise the dataset splits.

## Layout

```
clusters/
├── main.py                 # clustering entry point
├── cache_datasets.py       # to dump dataset splits (run using the AmpliGraph env)
├── requirements.txt        # Dependencies
├── .python-version         # 3.10.8
├── fb15k-237/
│   ├── entity2wikidata.json, mapper.csv              # description inputs
│   ├── train.txt, valid.txt, test.txt                # dataset splits
│   └── *_top{k}_*_NovaSearch.stella_en_400M_v5.pkl   # clusters output
├── wn18rr/
│   ├── wordnet-mlj12-definitions.txt
│   ├── train.txt, valid.txt, test.txt
│   └── *.pkl
└── yago3-10/
    ├── train.txt, valid.txt, test.txt
    └── *.pkl
```

Supported datasets: **fb15k-237, wn18rr, yago3-10**.

## Just use the clusters

The `*.pkl` files are versioned in this repo — point the experiments at them directly,
e.g.:

```
--llm-integration experiments/LatentSearch/clusters/fb15k-237/fb15k-237_top10_closest_NovaSearch.stella_en_400M_v5.pkl
```

For each dataset there are four files: `{,validation_} × {closest, lowest_and_closest}`.
The `validation_` variants hold the neighbours for the validation pairs (used for early
stopping); the experiment loader selects the right one automatically.

## Regenerate the clusters

Everything runs from inside this folder.

**1. Materialise the dataset splits (one-time, AmpliGraph env).**
`train/valid/test.txt` are git-ignored because they are large (yago3-10/train.txt is
~49 MB) and trivially regenerable. In an environment **that has AmpliGraph installed**
(e.g. the main project environment):

```
python cache_datasets.py
```

This writes `<dataset>/{train,valid,test}.txt`. It is the only step that touches
AmpliGraph/TensorFlow.

**2. Create the isolated PyTorch environment and run.**

```
uv venv --python 3.10
uv pip sync requirements.txt
source .venv/bin/activate

python main.py --dataset fb15k-237 --k 10 --save
python main.py --dataset fb15k-237 --k 10 --save --valid
```
- `--k` to select the number of nearest neighbors to consider.
- `--valid` produces the `validation_` file (validation pairs) instead of the test file.
- Default model is `NovaSearch/stella_en_400M_v5`; the seed is fixed to 0.

**3. Files to be generated**
```bash
### FB15k-237
# ComplEx
python main.py --dataset fb15k-237 --k 5 --save
python main.py --dataset fb15k-237 --k 5 --save --valid

# RotatE
python main.py --dataset fb15k-237 --k 10 --save
python main.py --dataset fb15k-237 --k 10 --save --valid

### WN18RR
# RotatE
python main.py --dataset wn18rr --k 5 --save
python main.py --dataset wn18rr --k 5 --save --valid

# ComplEx
python main.py --dataset wn18rr --k 30 --save
python main.py --dataset wn18rr --k 30 --save --valid

### YAGO3-10
# ComplEx & RotatE
python main.py --dataset yago310 --k 50 --save
python main.py --dataset yago310 --k 50 --save --valid
```