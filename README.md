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
# load some data (this is just a sample - more is better)
sample_data = """q_id,doc_id,gender,score,judgment
    1,1,1,0.962650646167003,1
    1,2,0,0.940172822166108,0.98
    1,3,0,0.925288002880488,0.96
    1,4,1,0.896143226020877,0.94
    1,5,0,0.89180775633204,0.92
    1,6,0,0.838704766545679,0.9
    """

# import other helper libraries
import pandas as pd
from io import StringIO

data = pd.read_csv(StringIO(sample_data))

# setup the DELTR object
protected_feature = 0 # column number of the protected attribute (index after query and document id)
gamma = 1 # value of the gamma parameter
number_of_iteraions = 10000 # number of iterations the training should run
standardize = True # let's apply standardization to the features

# create the Deltr object
dtr = Deltr(protected_feature, gamma, number_of_iteraions, standardize=standardize)

# train the model
dtr.train(data)
>> array([0.02527054, 0.07692437])
# the results will be approximately the same  
```

### Use the model to rank 

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



 
  