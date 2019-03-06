
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


You can install the latest stable release of `ampligraph` with pip, using the latest wheel (0.3.0) published by Dublin Labs:
*Note this work only from within the Dublin Labs network*

```
pip install http://dubaldeweb001.techlabs.accenture.com/wheels/ampligraph/ampligraph-0.3.dev0-py3-none-any.whl
```

If instead you want the most recent development version, you can clone the repository
and install from source (this will pull the latest commit on `develop` branch).
The code snippet below will install the library in editable mode (`-e`):

```
git clone https://github.com/Accenture/AmpliGraph.git
cd AmpliGraph
pip install -e .
```


## Download the Datasets

Datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/16GBu89NCVyyYetry91tMntzpV_mSQ-gK?usp=sharing).

Once downloaded, decompress the archives.

**You must also set the following environment variable:**

OSX/Linux:
```
export AMPLIGRAPH_DATA_HOME=/YOUR/PATH/TO/datasets
```

Windows:
```
setx AMPLIGRAPH_DATA_HOME /YOUR/PATH/TO/datasets
```

## Sanity Check

```python
>> import ampligraph
>> ampligraph.__version__
'0.3-dev'
```
