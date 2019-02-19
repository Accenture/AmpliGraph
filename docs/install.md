
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
git clone ssh://git@innersource.accenture.com/dl/ampligraph.git 
cd ampligraph
pip install .

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
