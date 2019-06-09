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

You need to train the model before it can rank documents.
```python
# import other helper libraries
import pandas as pd
from io import StringIO

# load some train data (this is just a sample - more is better)
train_data_raw = """q_id,doc_id,gender,score,judgment
    1,1,1,0.962650646167003,1
    1,2,0,0.940172822166108,0.98
    1,3,0,0.925288002880488,0.96
    1,4,1,0.896143226020877,0.94
    1,5,0,0.89180775633204,0.92
    1,6,0,0.838704766545679,0.9
    """
train_data = pd.read_csv(StringIO(train_data_raw))

# setup the DELTR object
protected_feature = 0 # column number of the protected attribute (index after query and document id)
gamma = 1 # value of the gamma parameter
number_of_iteraions = 10000 # number of iterations the training should run
standardize = True # let's apply standardization to the features

# create the Deltr object
dtr = Deltr(protected_feature, gamma, number_of_iteraions, standardize=standardize)

# train the model
dtr.train(train_data)
>> array([0.02527054, 0.07692437])
# your run should have approximately same results  
```

### Use the model to rank 

Now, you can use the obtained model to rank some data.
```python
# load some test/prediction data
prediction_data_raw = """q_id,doc_id,gender,score
    1,7,0,0.9645
    1,8,0,0.9524
    1,9,0,0.9285
    1,10,0,0.8961
    1,11,1,0.8911
    1,12,1,0.8312
    """
prediction_data = pd.read_csv(StringIO(prediction_data_raw))

# use the model to rank the data  
dtr.rank(prediction_data)
>> doc_id  gender  judgement
4      11       1   0.074849
5      12       1   0.063770
0       7       0   0.063486
1       8       0   0.061248
2       9       0   0.056828
3      10       0   0.050836
# the result will be a re-ranked dataframe
```
The library contains sufficient code documentation for each of the functions.

### Checking the model a bit deeper

You can check how the training of the model progressed using a special property called `log`.
```python
dtr.log
>> [<TrainStep [1553844278383,[0.01926469 0.00976336],[[-0.00125304 -0.0014605 ]
  [-0.00125304 -0.0014605 ]
  [-0.00125304 -0.0014605 ]
  [-0.00125304 -0.0014605 ]
  [-0.00125304 -0.0014605 ]
  [-0.00125304 -0.0014605 ]],5.999620187652397,0.0]>,
 ...]
```
The `log` returns a list of objects from the `fairsearchdeltr.models.TrainStep` class. The class is a representation of the parameters in each step of the training.
    Contains a `timestamp`, `omega`, `omega_gradient`, `loss`, `loss_standard`, `loss_exposure`.

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

* Meike Zehlike, Gina-Theresa Diehn, Carlos Castillo. "[Reducing Disparate Exposure in Ranking:
A Learning to Rank Approach](https://doi.org/10.1145/3132847.3132938)." preprint arXiv:1805.08716 (2018).

This library was developed by [Ivan Kitanovski](http://ivankitanovski.com/) based on the paper. See the [license](https://github.com/fair-search/fairsearch-deltr-python/blob/master/LICENSE) file for more information.

For any questions contact [Mieke Zehlike](https://de.linkedin.com/in/meike-zehlike-366bba131)

## See also

You can also see the [DELTR for ElasticSearch](https://github.com/fair-search/fairsearch-deltr-for-elasticsearch)
 and [DELTR Java library](https://github.com/fair-search/fairsearchdeltr-java).



 
  
