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

### Use the model to rank 

##

The library contains sufficient code documentation for each of the functions.
 
## Development

1. Clone this repository `git clone https://github.com/fair-search/fairsearchdeltr-python`
2. Change directory to the directory where you cloned the repository `cd WHERE_ITS_DOWNLOADED/fairsearchdeltr-python`
3. Use any IDE to work with the code

## Testing

Just run:
```
python setup.py test 
```

## Credits

The DELTR algorithm is described in this paper:

* Zehlike, Meike, and Carlos Castillo. "[Reducing Disparate Exposure in Ranking:
A Learning to Rank Approach](https://doi.org/10.1145/3132847.3132938)." arXiv preprint arXiv:1805.08716 (2018).

This code was developed by [Ivan Kitanovski](http://ivankitanovski.com/) based on the paper. See the [license](https://github.com/fair-search/fairsearchcore-python/blob/master/LICENSE) file for more information.

## See also

You can also see the [DELTR plug-in for ElasticSearch](https://github.com/fair-search/fairsearchdeltr-elasticsearch-plugin)
 and [DELTR Java library](https://github.com/fair-search/fairsearchdeltr-java).



 
  