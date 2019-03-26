# Fair search DELTR for Python

[![image](https://img.shields.io/pypi/status/fairsearchdeltr.svg)](https://pypi.org/project/fairsearchdeltr/)
[![image](https://img.shields.io/pypi/v/fairsearchdeltr.svg)](https://pypi.org/project/fairsearchdeltr/)
[![image](https://img.shields.io/pypi/pyversions/fairsearchdeltr.svg)](https://pypi.org/project/fairsearchdeltr/)
[![image](https://img.shields.io/pypi/l/fairsearchdeltr.svg)](https://pypi.org/project/fairsearchdeltr/)
[![image](https://img.shields.io/pypi/implementation/fairsearchdeltr.svg)](https://pypi.org/project/fairsearchdeltr/)

This is the Python library that implements the [DELTR](https://arxiv.org/pdf/1805.08716.pdf) model for fair ranking.

## Installation
To install `fairsearchdeltr`, simply use `pip` (or `pipenv`):
```bash
pip install fairsearchdeltr
```
And, that's it!

## Using it in your code
You need to import the class from the package first: 
```python
from fairsearchdeltr import Deltr
```

### Train a model
You need to train the model before it can rank documents. The 
```python
protected_feature = 0 # column number of the protected number
gamma = 0 # value of the gamma parameter
number_of_iteraions = 1000 # number of iterations the training should run

# create the Deltr object
dtr = Deltr(index_protected_feature, gamma, number_of_iteraions)


```

  