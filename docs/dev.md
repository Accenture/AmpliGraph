# Development

## Issue Tracking System

[Available on JIRA at this address](https://fleet.alm.accenture.com/thedockjira/issues/?jql=project%20%3D%20TEC%20AND%20component%20%3D%20%22KG%20embeddings%22).


## Microsoft Teams Channel

A dedicated [Microsoft Teams channel is available here](https://teams.microsoft.com/l/team/19%3ad909bb60a8254765a11c18b1645c27ed%40thread.skype/conversations?groupId=8caaa2f8-27a2-4a01-9bfa-07f60f0527ba&tenantId=e0793d39-0939-496d-b129-198edd916feb)
(Join code: `vsprdag`).


## Git

[Git repository available on InnerSource](https://innersource.accenture.com/projects/DL/repos/xai-link-prediction/).


## Unit Tests

To run all the unit tests:

```
$ pytest tests
```

See [pytest documentation](https://docs.pytest.org/en/latest/) for additional arguments.


## Build Documentation

The project documentation can be built with Sphinx:

```
cd docs
make clean autogen html
```

## Distribute

```
pip wheel --wheel-dir dist --no-deps .
```