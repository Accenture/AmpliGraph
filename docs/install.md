
# Installation

## Prerequisites

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
pip install tensorflow==1.12.0

or 

conda install tensorflow=1.12.0
```

**GPU support**

```
pip install tensorflow-gpu==1.12.0

or 

conda install tensorflow-gpu=1.12.0
```


## Install AmpliGraph


Install the latest stable release from pip:

```
pip install ampligraph
```


If instead you want the most recent development version, you can clone the repository
and install from source as: See the [How to Contribute guide](dev.md) for details.

```
git clone https://github.com/Accenture/AmpliGraph.git
git checkout develop
cd AmpliGraph
pip install -e .
```

## Sanity Check

```python
>> import ampligraph
>> ampligraph.__version__
'1.0.0'
```
