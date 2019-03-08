# How to Contribute

## Git Repo and Issue Tracking 

AmpliGraph [Git repository is available on GitHub](https://innersource.accenture.com/projects/DL/repos/xai-link-prediction/).


## How to Contribute


## How to Roll Your Own Model

TODO


## Unit Tests

To run all the unit tests:

```
$ pytest tests
```

See [pytest documentation](https://docs.pytest.org/en/latest/) for additional arguments.


## Documentation

The project documentation is based on Sphinx and can be built as follows:

```
cd docs
make clean autogen html
```

## Build your Own Wheel

```
pip wheel --wheel-dir dist --no-deps .
```