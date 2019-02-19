# AmpliGraph

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


## System Architecture


AmpliGraph includes the following submodules:

* **KG Loaders**: Helper functions to load datasets (knowledge graphs).
* **Latent Feature Models**: knowledge graph embedding models. AmpliGraph contains: TransE, DistMult, ComplEx, ConvE, and RotatE.
* **Evaluation**: Metrics and evaluation protocols to assess the predictive power of the models.



# Installation

### Provision a Virtual Environment

**Installation using Anaconda is highly recommended.**

Create & activate Virtual Environment (conda)

```
conda create --name ampligraph python=3.6
source activate ampligraph
```

### Install TensorFlow

**CPU version**

```
pip install tensorflow
```

or you could install the version packaged with conda:

```
conda install tensorflow
```

**GPU version**

```
pip install tensorflow-gpu
```

or you could install the version packaged with conda:

```
conda install tensorflow-gpu
```


## Install the library


You can install the latest stable release of AmpliGraph with `pip`, using the latest wheel (0.3.0) published by Dublin Labs:
*Note this work only from within the Dublin Labs network*

```
pip install http://dubaldeweb001.techlabs.accenture.com/wheels/ampligraph/ampligraph-0.3.dev0-py3-none-any.whl
```

If instead you want the most recent development version, you can clone the repository
and install from source (this will pull the latest commit on `develop` branch).
The code snippet below will install the library in editable mode (`-e`):

```
git clone ssh://git@innersource.accenture.com/dl/ampligraph.git
cd ampligraph
pip install -e .

```

## Download the Datasets

Datasets can be downloaded from [SharePoint](https://ts.accenture.com/sites/TechLabs-Dublin/_layouts/15/guestaccess.aspx?guestaccesstoken=Uz28P2m4hWp2TEgbvFrD%2b4BiURBHVTAw0NbPBRLzWWA%3d&folderid=2_012fd581718e74e4a9305c845a1224ee1&rev=1).
Once downloaded, decompress the archives.

**You must also set the following environment variable:**

```
export AMPLIGRAPH_DATA_HOME=/YOUR/PATH/TO/datasets
```

## Sanity Check

```python
>> import ampligraph
>> ampligraph.__version__
'0.3-dev'
```

## Installing with HDT Support
[HDT](http://www.rdfhdt.org/) is a compressed type of RDF graph data. By default, the installed ampligraph library does not support loading this data type. To enable it, you must have **`gcc` with C++11 support** installed in your Linux box.

**Ubuntu**

```
sudo add-apt-repository ppa:jonathonf/gcc-7.3
sudo apt-get update
sudo apt-get install gcc-7
sudo apt-get install g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 10
sudo update-alternatives --install /usr/bin/g++-7 g++ /usr/bin/g++-7 10
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
````

**CentOS**

Below are commands we used to install gcc 7.3.1 on CentOS 7.5:

```
sudo yum install centos-release-scl
sudo yum install devtoolset-7-gcc*
scl enable devtoolset-7 bash
```

Once finished installing gcc, you can install the ampligraph library with hdt support by:

```
pip install .[hdt]
```

##  Documentation

**[Latest documentation available here](http://10.106.43.211/docs/ampligraph/dev/index.html)**


The project documentation can be built with Sphinx:

```
cd docs
make clean autogen html
```

## Tests


```
pytest -s tests
```

