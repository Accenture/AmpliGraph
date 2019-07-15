# AmpliGraph

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2595043.svg)](https://doi.org/10.5281/zenodo.2595043)

[![Documentation Status](https://readthedocs.org/projects/ampligraph/badge/?version=latest)](http://ampligraph.readthedocs.io/?badge=latest)


**Open source Python library that predicts links between concepts in a knowledge graph.**

AmpliGraph is a suite of neural machine learning models for relational Learning, a branch of machine learning
that deals with supervised learning on knowledge graphs.


**Use AmpliGraph if you need to**:

* Discover new knowledge from an existing knowledge graph.
* Complete large knowledge graphs with missing statements.
* Generate stand-alone knowledge graph embeddings.
* Develop and evaluate a new relational model.


AmpliGraph's machine learning models generate **knowledge graph embeddings**, vector representations of concepts in a metric space:

![](docs/img/kg_lp_step1.png)

It then combines embeddings with model-specific scoring functions to predict unseen and novel links:

![](docs/img/kg_lp_step2.png)


## Key Features


* **Intuitive APIs**: AmpliGraph APIs are designed to reduce the code amount required to learn models that predict links in knowledge graphs.
* **GPU-Ready**: AmpliGraph is based on TensorFlow, and it is designed to run seamlessly on CPU and GPU devices - to speed-up training.
* **Extensible**: Roll your own knowledge graph embeddings model by extending AmpliGraph base estimators.


## Modules

AmpliGraph includes the following submodules:

* **KG Loaders**: Helper functions to load datasets (knowledge graphs).
* **Latent Feature Models**: knowledge graph embedding models. AmpliGraph contains: TransE, DistMult, ComplEx, HolE. (More to come!)
* **Evaluation**: Metrics and evaluation protocols to assess the predictive power of the models.



## Installation

### Prerequisites

* Linux Box
* Python ≥ 3.6

#### Provision a Virtual Environment

Create and activate a virtual environment (conda)

```
conda create --name ampligraph python=3.6
source activate ampligraph
```

#### Install TensorFlow

AmpliGraph is built on TensorFlow 1.x.
Install from pip or conda:

**CPU-only**

```
pip install tensorflow==1.13.1

or

conda install tensorflow=1.13.1
```

**GPU support**

```
pip install tensorflow-gpu==1.13.1

or

conda install tensorflow-gpu=1.13.1
```



### Install AmpliGraph


Install the latest stable release from pip:

```
pip install ampligraph
```

If instead you want the most recent development version, you can clone the repository
and install from source (your local working copy will be on the latest commit on the `develop` branch).
The code snippet below will install the library in editable mode (`-e`):

```
git clone https://github.com/Accenture/AmpliGraph.git
cd AmpliGraph
pip install -e .
```


### Sanity Check

```python
>> import ampligraph
>> ampligraph.__version__
'1.0.3'
```


## Predictive Power Evaluation (MRR Filtered)

|          |FB15k |WN18   |WN18RR |FB15K-237|YAGO3-10 |
|----------|------|-------|-------|---------|---------|
| TransE   | 0.55 | 0.50  | 0.23  | 0.31    | 0.24    |
| DistMult | 0.79 | 0.83  | 0.44  | 0.29    | 0.49    |
| ComplEx  | 0.79 | 0.94  | 0.44  | 0.30    | 0.50    |
| HolE     | 0.80 | 0.94  | 0.47  | 0.28    | 0.50    |


## Documentation

**[Documentation available here](http://docs.ampligraph.org)**

The project documentation can be built from your local working copy with:

```
cd docs
make clean autogen html
```

## How to contribute

See [guidelines](http://docs.ampligraph.org) from AmpliGraph documentation.


## How to Cite

If you like AmpliGraph and you use it in your project, why not starring the project on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/Accenture/AmpliGraph.svg?style=social&label=Star&maxAge=3600)](https://GitHub.com/Accenture/AmpliGraph/stargazers/)


If you instead use AmpliGraph in an academic publication, cite as:

```
@misc{ampligraph,
 author= {Luca Costabello and
          Sumit Pai and
          Chan Le Van and
          Rory McGrath and
          Nicholas McCarthy},
 title = {{AmpliGraph: a Library for Representation Learning on Knowledge Graphs}},
 month = mar,
 year  = 2019,
 doi   = {10.5281/zenodo.2595043},
 url   = {https://doi.org/10.5281/zenodo.2595043}
}
```

## License

AmpliGraph is licensed under the Apache 2.0 License.