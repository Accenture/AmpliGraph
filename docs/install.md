# Installation

## Prerequisites

* Linux, macOS, Windows
* Python ≥ 3.8

## Provision a Virtual Environment

To provision a virtual environment for installing AmpliGraph, any option can work; here we will give provide the
instruction for using `venv` and `Conda`.

### venv

The first step is to create and activate the virtual environment.

```
python3.8 -m venv PATH/TO/NEW/VIRTUAL_ENVIRONMENT
source PATH/TO/NEW/VIRTUAL_ENVIRONMENT/bin/activate
```

Once this is done, we can proceed with the installation of TensorFlow 2:

```
pip install "tensorflow==2.9.0"
```

If you are installing Tensorflow on MacOS, instead of the following please use:

```
pip install "tensorflow-macos==2.9.0"
```

**IMPORTANT**: the installation of TensorFlow can be tricky on Mac OS with the Apple silicon chip. Though `venv` can
provide a smooth experience, we invite you to refer to the [dedicated section](#install-tensorflow-2-for-mac-os-m1-chip)
down below and consider using `conda` if some issues persist in alignment with the
[Tensorflow Plugin page on Apple developer site](https://developer.apple.com/metal/tensorflow-plugin/).


### Conda

The first step is to create and activate the virtual environment.

```
conda create --name ampligraph python=3.8
source activate ampligraph
```

Once this is done, we can proceed with the installation of TensorFlow 2, which can be done through `pip` or `conda`.

```
pip install "tensorflow==2.9.0"

or 

conda install "tensorflow==2.9.0"
```

### Install TensorFlow 2 for Mac OS M1 chip

When installing TensorFlow 2 for Mac OS with Apple silicon chip we recommend to use a conda environment. 

```
conda create --name ampligraph python=3.8
source activate ampligraph
```

After having created and activated the virtual environment, run the following to install Tensorflow. 

```
conda install -c apple tensorflow-deps
pip install --user tensorflow-macos==2.9.0
pip install --user tensorflow-metal==0.6
```

In case of problems with the installation or for further details, refer to
[Tensorflow Plugin page](https://developer.apple.com/metal/tensorflow-plugin/) on the official Apple developer website.

## Install AmpliGraph

Once the installation of Tensorflow is complete, we can proceed with the installation of AmpliGraph.

To install the latest stable release from pip:

```
pip install ampligraph
```

To sanity check the installation, run the following:

```python
>>> import ampligraph
>>> ampligraph.__version__
'2.1.0'
```

If instead you want the most recent development version, you can clone the repository from
[GitHub](https://github.com/Accenture/AmpliGraph.git), install AmpliGraph from source and checkout the `develop`
branch. In this way, your local working copy will be on the latest commit on the `develop` branch.

```
git clone https://github.com/Accenture/AmpliGraph.git
cd AmpliGraph
git checkout develop
pip install -e .
```
Notice that the code snippet above installs the library in editable mode (`-e`).

To sanity check the installation run the following:

```python
>>> import ampligraph
>>> ampligraph.__version__
'2.1-dev'
```


## Support for TensorFlow 1.x
For TensorFlow 1.x-compatible AmpliGraph, use [AmpliGraph 1.x](https://docs.ampligraph.org/en/1.4.0/), whose API are
available cloning the [repository](https://github.com/Accenture/AmpliGraph.git) from GitHub and checking out the
*ampligraph1/develop* branch. However, notice that the support for this version has been discontinued.

Finally, if you want to use AmpliGraph 1.x APIs on top of Tensorflow 2, refer to the backward compatibility APIs
provided on Ampligraph [compat](https://docs.ampligraph.org/en/2.0.0/ampligraph.latent_features.html#module-ampligraph.compat)
module.