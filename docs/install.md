# Installation

## Prerequisites

* Linux, macOS, Windows
* Python 3.9 – 3.11

## Install AmpliGraph

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
uv sync 
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