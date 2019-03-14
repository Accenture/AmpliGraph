# How to Contribute

## Git Repo and Issue Tracking 

AmpliGraph [repository is available on GitHub](https://innersource.accenture.com/projects/DL/repos/xai-link-prediction/).

A list of open issues [is available here](https://github.com/Accenture/AmpliGraph/issues).


## How to Contribute
We welcome community contributions, whether they are new models, tests, or documentation.

You can contribute to AmpliGraph in many ways:
- Raise a [bug report](https://github.com/Accenture/AmpliGraph/issues/new?assignees=&labels=&template=bug_report.md&title=)
- File a [feature request](https://github.com/Accenture/AmpliGraph/issues/new?assignees=&labels=&template=feature_request.md&title=)
- Help other users by commenting on the [issue tracking system](https://github.com/Accenture/AmpliGraph/issues)
- Add unit tests
- Improve the documentation
- Add a new graph embedding model (see below)


## How to Roll Your Own Model

AmpliGraph embeddings models include the following components:

-



TODO


## Unit Tests

To run all the unit tests:

```
$ pytest tests
```

See [pytest documentation](https://docs.pytest.org/en/latest/) for additional arguments.


## Documentation

The project documentation is [hosted at this address](https://docs.ampligraph.org).
It is based on Sphinx and can be built as follows:

```
cd docs
make clean autogen html
```


## Packaging

To build an AmpliGraph custom wheel, do the following:

```
pip wheel --wheel-dir dist --no-deps .
```