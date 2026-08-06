# ImbalancE — Inference-Time Latent Search Against Degree Imbalance in Link Prediction

Reproduction code for the paper *"ImbalancE: Inference-Time Latent Search Against
Degree Imbalance in Link Prediction"*, accepted at **ISWC 2026**.

ImbalancE is an **inference-time** procedure: it takes a *pretrained* Knowledge Graph
Embedding (KGE) model (ComplEx, RotatE, …) and, for each test `(subject, predicate)` or
`(predicate, object)` query, runs a short latent search that nudges the embeddings toward
a set of LLM-derived candidate entities. This counteracts the degree imbalance that makes
KGE models under-rank low-degree (long-tail) entities. No retraining of the base KGE is
required.

The pipeline has three stages:

1. **Generate the LLM clusters** — for every query pair, precompute the nearest-neighbour
   candidate entities in the latent space of a sentence-embedding LLM. Runs in an isolated
   PyTorch environment under [`clusters/`](clusters).
2. **Run the ImbalancE latent search** — load the pretrained KGE, load the clusters, and
   perform the per-pair latent search on the validation set (to fit the early stopper) and
   on the test set. Runs in the AmpliGraph environment under [`code/`](code).
3. **Post-process & evaluate** — turn the raw per-pair ranks into the MRR / Hits@k metrics
   reported in the paper, broken down by entity degree. Also under [`code/`](code).

---

## Repository layout

```
ISWC2026_ImbalancE/
├── clusters/          # Stage 1 — LLM nearest-neighbour clusters (isolated PyTorch env)
│   ├── main.py            # clustering entry point
│   ├── cache_datasets.py  # dumps dataset splits (run once, in the AmpliGraph env)
│   ├── requirements.txt   # pinned PyTorch env
│   ├── README.md          # detailed, per-dataset cluster-generation guide
│   └── <dataset>/         # splits + *.pkl cluster files
├── code/              # Stages 2 & 3 — ImbalancE latent search + evaluation
│   ├── main.py                     # latent-search entry point
│   ├── process_kge_latent_ranks.py # rank post-processing / metrics
│   ├── utils_KGE_latent_search.py
│   └── experiement_coordination    # log of the full experimental campaign + best configs
└── results/           # experiment outputs (per dataset / model / hyper-params)
```

Supported datasets (naming used by the code): **`fb15k-237`, `wn18rr`, `yago310`**.
Supported base KGE models: **ComplEx** and **RotatE** (loaded via
`ampligraph.pretrained_models.load_pretrained_model`).

---

## Environments

Two separate virtual environments are used on purpose:

- **AmpliGraph env** (TensorFlow 2.15 / Keras 2) — used by everything in `code/` and by the
  one-off `clusters/cache_datasets.py`. This is the main project environment; install
  AmpliGraph-Lab from the repository root with `uv sync` (the latest pip wheel does not include ImbalancE).
- **Isolated PyTorch env** — used only by `clusters/main.py` to run the sentence-embedding
  LLM. It is deliberately kept apart from TensorFlow. Pinned in
  [`clusters/requirements.txt`](clusters/requirements.txt) (Python 3.10).

---

## Stage 1 — Generate the LLM clusters (`clusters/`)

> The `*.pkl` cluster files are already versioned in the repo, so you can **skip this stage**
> and jump straight to Stage 2. Regenerate only if you change the embedding model, `k`, or
> the datasets. See [`clusters/README.md`](clusters/README.md) for the full details.

All commands run **from inside `clusters/`**.

**1a. Materialise the dataset splits (one-time, AmpliGraph env).**
`train/valid/test.txt` are git-ignored (too large) but are trivially regenerated. In the
AmpliGraph environment:

```bash
cd clusters
python cache_datasets.py          # writes <dataset>/{train,valid,test}.txt
```

This is the only step in Stage 1 that touches AmpliGraph/TensorFlow.

**1b. Create the isolated PyTorch env and build the clusters.**

```bash
uv venv --python 3.10
uv pip sync requirements.txt
source .venv/bin/activate
```

For each `(dataset, k)` you need **two** files: the test-pair clusters and the
validation-pair clusters (`--valid`). The default embedding model is
`NovaSearch/stella_en_400M_v5` and the seed is fixed to `0`. The exact `k` per
dataset/model used in the paper:

```bash
### FB15k-237
python main.py --dataset fb15k-237 --k 5  --save            # ComplEx
python main.py --dataset fb15k-237 --k 5  --save --valid
python main.py --dataset fb15k-237 --k 10 --save            # RotatE
python main.py --dataset fb15k-237 --k 10 --save --valid

### WN18RR
python main.py --dataset wn18rr --k 30 --save               # ComplEx
python main.py --dataset wn18rr --k 30 --save --valid
python main.py --dataset wn18rr --k 5  --save               # RotatE
python main.py --dataset wn18rr --k 5  --save --valid

### YAGO3-10  (ComplEx & RotatE both use k=50)
python main.py --dataset yago3-10 --k 50 --save
python main.py --dataset yago3-10 --k 50 --save --valid
```

Each run writes
`<dataset>/{,validation_}<dataset>_top<k>_NovaSearch.stella_en_400M_v5.pkl`, a pickle of
`{"sp": {(s, p): [neighbour_ids…]}, "po": {(o, p): [neighbour_ids…]}}` keyed by raw dataset
IDs. The Stage-2 loader picks the `validation_` variant automatically when needed.

---

## Stage 2 — Run the ImbalancE latent search (`code/`)

Switch back to the **AmpliGraph environment**. All commands run **from inside `code/`**.

`code/main.py` loads the pretrained KGE, maps the dataset to integer indices, then:

- with `-valid`, first runs the latent search on the **validation** set and builds an
  `EarlyStopper` from the resulting ranks (degree-quantile thresholds are computed
  automatically);
- runs the latent search on the **test** set (unless `--no-run-on-test`), separately for
  `sp` and `po` query pairs.

Outputs are written to `--output-folder/<dataset>/<ScoringType>_lr<lr>[…]/` as
`valid_ranks_{sp,po}.pkl`, `ranks_{sp,po}.pkl`, and `config.json`.

### Best configurations from the paper

These are the exact commands (best hyper-parameters per dataset/model) taken from
[`code/experiement_coordination`](code/experiement_coordination). Each points
`--llm-integration` at the matching cluster file from Stage 1; `--tune-llm-ents` optimises
the LLM candidate entities as well, and `--freeze-relations` keeps the relation embeddings
fixed.

```bash
cd code

### FB15k-237
DATASET=fb15k-237
python main.py -score ComplEx --dataset $DATASET -lr 1e-2 --epochs 30 --batch-size 30 \
  --freeze-relations --tune-llm-ents -valid \
  --llm-integration ../clusters/$DATASET/${DATASET}_top5_closest_NovaSearch.stella_en_400M_v5.pkl
python main.py -score RotatE  --dataset $DATASET -lr 1e-3 --epochs 30 --batch-size 30 \
  --freeze-relations --tune-llm-ents -valid \
  --llm-integration ../clusters/$DATASET/${DATASET}_top10_closest_NovaSearch.stella_en_400M_v5.pkl

### WN18RR
DATASET=wn18rr
python main.py -score ComplEx --dataset $DATASET -lr 1e-3 --epochs 30 --batch-size 30 \
  --freeze-relations --tune-llm-ents -valid \
  --llm-integration ../clusters/$DATASET/${DATASET}_top30_closest_NovaSearch.stella_en_400M_v5.pkl
python main.py -score RotatE  --dataset $DATASET -lr 1e-4 --epochs 30 --batch-size 30 \
  --freeze-relations --tune-llm-ents -valid \
  --llm-integration ../clusters/$DATASET/${DATASET}_top5_closest_NovaSearch.stella_en_400M_v5.pkl

### YAGO3-10  (uses the yago3-10 cluster folder; smaller batch size)
DATASET=yago310
python main.py -score ComplEx --dataset $DATASET -lr 1e-2 --epochs 30 --batch-size 15 \
  --freeze-relations --tune-llm-ents -valid \
  --llm-integration ../clusters/yago3-10/yago3-10_top50_closest_NovaSearch.stella_en_400M_v5.pkl
python main.py -score RotatE  --dataset $DATASET -lr 1e-3 --epochs 30 --batch-size 15 \
  --freeze-relations --tune-llm-ents -valid \
  --llm-integration ../clusters/yago3-10/yago3-10_top50_closest_NovaSearch.stella_en_400M_v5.pkl
```

Summary of the best hyper-parameters:

| Dataset    | Model   | learning rate | top-k | batch size |
|------------|---------|:-------------:|:-----:|:----------:|
| FB15k-237  | ComplEx | 1e-2          | 5     | 30         |
| FB15k-237  | RotatE  | 1e-3          | 10    | 30         |
| WN18RR     | ComplEx | 1e-3          | 30    | 30         |
| WN18RR     | RotatE  | 1e-4          | 5     | 30         |
| YAGO3-10   | ComplEx | 1e-2          | 50    | 15         |
| YAGO3-10   | RotatE  | 1e-3          | 50    | 15         |

All runs use `--epochs 30`, `--freeze-relations`, `--tune-llm-ents`, and `-valid`.

### Useful flags (`code/main.py`)

| Flag | Meaning |
|------|---------|
| `-score / --scoring-type` | Pretrained KGE to load (`ComplEx`, `RotatE`, …). |
| `--dataset` | `fb15k-237`, `wn18rr`, or `yago310`. |
| `-lr / --learning-rate` | Learning rate of the latent-search optimiser. |
| `--epochs` | Number of latent-search steps. |
| `--batch-size` | Query pairs optimised together. |
| `--llm-integration <path>` | Path to the Stage-1 cluster `.pkl`. |
| `--tune-llm-ents` | Also optimise the LLM candidate entities. |
| `--freeze-relations` | Keep relation embeddings fixed. |
| `-valid / --run-on-validation` | Run on validation first to fit the early stopper. |
| `--no-run-on-test` | Skip the test-set run (validation only). |
| `--output-folder` | Base output dir (default `../results`). |
| `--seed` | Random seed (default `0`). |

---

## Stage 3 — Post-process & evaluate (`code/`)

Once a run has produced its `valid_ranks_*.pkl` / `ranks_*.pkl` files, turn them into the
degree-stratified metrics reported in the paper. Point `--output-folder` at the specific
run directory created in Stage 2 (i.e. the `…/<Scoring>_lr<lr>/` folder), and pass
`--valid` so the early-stopper epochs derived from the validation ranks are applied:

```bash
cd code
python process_kge_latent_ranks.py --valid \
  --output-folder ../results/fb15k-237/RotatE_lr0.001
```

The script recomputes the same degree-quantile thresholds used to split entities into
degree buckets and reports the post-search ranking metrics for each bucket.

---

## Notes & reproducibility

- The base KGE checkpoints are fetched by `load_pretrained_model`; ImbalancE never retrains
  them — it only performs the inference-time latent search.
- The `results/` tree and `code/experiement_coordination` record the full grid used during
  the paper's experimental campaign; the commands above are the *best* configurations.
