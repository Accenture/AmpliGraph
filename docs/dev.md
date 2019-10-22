# How to Contribute

## Git Repo and Issue Tracking
[![](https://img.shields.io/github/stars/Accenture/AmpliGraph.svg?style=social&label=Star&maxAge=3600)](https://GitHub.com/Accenture/AmpliGraph/stargazers/)

AmpliGraph [repository is available on GitHub](https://github.com/Accenture/AmpliGraph).

A list of open issues [is available here](https://github.com/Accenture/AmpliGraph/issues).

[Join the conversation on Slack](https://join.slack.com/t/ampligraph/shared_invite/enQtNTc2NTI0MzUxMTM5LTRkODk0MjI2OWRlZjdjYmExY2Q3M2M3NGY0MGYyMmI4NWYyMWVhYTRjZDhkZjA1YTEyMzBkMGE4N2RmNTRiZDg)
![](/img/slack_logo.png)


## How to Contribute
We welcome community contributions, whether they are new models, tests, or documentation.

You can contribute to AmpliGraph in many ways:
- Raise a [bug report](https://github.com/Accenture/AmpliGraph/issues/new?assignees=&labels=&template=bug_report.md&title=)
- File a [feature request](https://github.com/Accenture/AmpliGraph/issues/new?assignees=&labels=&template=feature_request.md&title=)
- Help other users by commenting on the [issue tracking system](https://github.com/Accenture/AmpliGraph/issues)
- Add unit tests
- Improve the documentation
- Add a new graph embedding model (see below)


## Adding Your Own Model

The landscape of knowledge graph embeddings evolves rapidly.
We welcome new models as a contribution to AmpliGraph, which has been built to provide a shared codebase to guarantee a
fair evalaution and comparison acros models.

You can add your own model by raising a pull request.

To get started, [read the documentation on how current models have been implemented](ampligraph.latent_features.html#anatomy-of-a-model).


## Developer Notes

Additional documentation on data adapters, AmpliGraph support for large graphs, and further technical details 
[is available here](dev_notes.html).

## Clone and Install in editable mode

Clone the repository and checkout the `develop` branch.
Install from source with pip. use the `-e` flag to enable [editable mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs):

```
git clone https://github.com/Accenture/AmpliGraph.git
git checkout develop
cd AmpliGraph
pip install -e .
```


## Unit Tests

To run all the unit tests:

```
$ pytest tests
```

See [pytest documentation](https://docs.pytest.org/en/latest/) for additional arguments.


## Documentation

The [project documentation](https://docs.ampligraph.org) is based on Sphinx and can be built on your local working
copy as follows:

```
cd docs
make clean autogen html
```

The above generates an HTML version of the documentation under `docs/_built/html`.


## Packaging

To build an AmpliGraph custom wheel, do the following:

```
pip wheel --wheel-dir dist --no-deps .
```