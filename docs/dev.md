# How to Contribute

## Git Repo and Issue Tracking
[![GitHub stars](https://img.shields.io/github/stars/Accenture/AmpliGraph.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/Accenture/AmpliGraph/stargazers/)

AmpliGraph [repository is available on GitHub](https://innersource.accenture.com/projects/DL/repos/xai-link-prediction/).

A list of open issues [is available here](https://github.com/Accenture/AmpliGraph/issues).

The AmpliGraph [Slack channel is available here](https://join.slack.com/t/ampligraph/shared_invite/enQtNTc2NTI0MzUxMTM5LTAxM2ViYTc0ZTI2NzNhOGZiNjkzZjNkN2NkNDc3NWUyZmU2Njg0MDMxYWY5NGUwYWVmOTNkOWI5NmI0NDJjYWI).


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