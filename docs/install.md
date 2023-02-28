# Installation

## Prerequisites

* Linux, macOS, Windows
* Python ≥ 3.8

### Provision a Virtual Environment

Create and activate a virtual environment (conda)

```
conda create --name ampligraph python=3.8
source activate ampligraph
```

### Install TensorFlow 2

AmpliGraph 2.x is built on TensorFlow 2.

Install from pip or conda:

```
pip install "tensorflow>=2.9"

or 

conda install tensorflow'>=2.9'
```

### Install TensorFlow 2 for Mac OS M1 chip

```
conda install -c apple tensorflow-deps
pip install --user tensorflow-macos==2.10
pip install --user tensorflow-metal==0.6
```

## Install AmpliGraph


Install the latest stable release from pip:

```
pip install ampligraph
```


If instead you want the most recent development version, you can clone the repository
and install from source as below (also see the [How to Contribute guide](dev.md) for details):

```
git clone https://github.com/Accenture/AmpliGraph.git
cd AmpliGraph
git checkout develop
pip install -e .
```

## Sanity Check

```python
>> import ampligraph
>> ampligraph.__version__
'2.0-dev'
```


## Support for TensorFlow 1.x
For TensorFlow 1.x-compatible AmpliGraph, use [AmpliGraph 1.x](https://docs.ampligraph.org/en/1.4.0/).
