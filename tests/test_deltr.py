import pytest

import os
import pandas as pd
import numpy as np
from io import StringIO

from fairsearchdeltr.deltr import Deltr
from tests.syntethic_dataset_creator import SyntheticDatasetCreator


class DeltrMock(Deltr):
    """
    Extend the Deltr class so as to extract some debug data for validation
    """
    def __init__(self, protected_feature: int, gamma: float, number_of_iterations=3000, learning_rate=0.001,
                 lambdaa=0.001, init_var=0.01):
        super().__init__(protected_feature, gamma, number_of_iterations, learning_rate, lambdaa, init_var)

        self.losses = []

    def _train_nn(self, tr, query_ids, feature_matrix, training_scores):
        omega, loss = tr.train_nn(query_ids, feature_matrix, training_scores, True)

        self.losses = tr.losses()

        return omega, loss


def test_create_deltr():
    assert Deltr('1', 1)
    try:
        Deltr(None, 1)
        assert False
    except:
        assert True
    try:
        Deltr(None, None)
        assert False
    except:
        assert True
    try:
        Deltr('1', None)
        assert False
    except:
        assert True

@pytest.mark.parametrize("file_name, standardize",(
                        ('test_data_1.csv', False),
                        ('test_data_1.csv', True),
))
def test_train_deltr(file_name, standardize):
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'fixtures', file_name), decimal=',')

    d = Deltr('gender', 1, 10, standardize=standardize)

    data['doc_id'] = pd.Series(range(50))

    data = data[['id', 'doc_id', 'gender', 'score', 'judgment']]

    d.train(data)

    print(">>>>" + str(d._omega))

    assert d.log != []

    if len(d.log) > 1:
        current = d.log[0].loss
        print(str(d.log[0].loss) + " " + str(d.log[0].omega))
        for log in d.log[1:]:
            print(str(log.loss) + " " + str(log.omega))
            assert log.loss < current
            current = log.loss
    else:
        assert d.log[0]


@pytest.mark.parametrize("number_of_elements, number_of_features, gamma, number_of_iterations, standardize",(
                        (20, 5, 1, 100, False),
                        (50, 10, 0.8, 500, False),
                        (1000, 3, 1, 1000, False),
                        (20, 5, 1, 100, True),
                        (50, 10, 0.8, 500, True),
                        (1000, 3, 1, 1000, True),
))
def test_train_deltr_synthetic_data(number_of_elements, number_of_features, gamma, number_of_iterations, standardize):

    # create a dataset
    sdc = SyntheticDatasetCreator(20, {'protected_feature': 2}, list(range(number_of_features-1)))
    data = sdc.dataset

    # score the elements based on some predefined weights
    weights = [10*w for w in range(number_of_features)]
    data['judgement'] = data.apply(lambda row: np.dot(row, weights), axis=1)

    # add query and document ids
    data['id'] = pd.Series([1] * number_of_elements)
    data['doc_id'] = data['doc_id'] = pd.Series(range(number_of_elements))

    # arrange the field names
    data = data[['id', 'doc_id', 'protected_feature'] + list(range(number_of_features-1)) + ['judgement']]

    # sort the elements by the judgement in a descending fashion
    data = data.sort_values(['judgement'], ascending=[0])

    # train a model
    d = Deltr('protected_feature', gamma, number_of_iterations, standardize=standardize)
    d.train(data)

    assert d.log != []

    if len(d.log) > 1:
        current = d.log[0].loss
        for log in d.log[1:]:
            assert log.loss <= current
            current = log.loss
    else:
        assert d.log[0]


@pytest.mark.parametrize("number_of_elements, number_of_features, gamma, number_of_iterations",(
                        (20, 5, 1, 100),
                        (50, 10, 0.8, 500),
                        (1000, 3, 1, 1000),
))
def test_rank_deltr(number_of_elements, number_of_features, gamma, number_of_iterations):

    # create a dataset
    sdc = SyntheticDatasetCreator(20, {'protected_feature': 2}, list(range(number_of_features - 1)))
    data = sdc.dataset

    # add query and document ids
    data['id'] = pd.Series([1] * number_of_elements)
    data['doc_id'] = data['doc_id'] = pd.Series(range(number_of_elements))

    # arrange the field names
    data = data[['id', 'doc_id', 'protected_feature'] + list(range(number_of_features - 1))]

    # score the elements based on some predefined weights
    weights = [10 * w for w in range(number_of_features)]

    # manually set omega
    d = Deltr('protected_feature', 1, 1)
    d._omega = weights

    ranked_set = d.rank(data)

    # copy the data and rank it manually
    ranked_set_manually = data.drop(['id', 'doc_id'], axis=1).copy()
    ranked_set_manually['judgement'] = ranked_set_manually.apply(lambda row: np.dot(row, weights), axis=1)
    ranked_set_manually['id'] = pd.Series([1] * number_of_elements)
    ranked_set_manually['doc_id'] = ranked_set_manually['doc_id'] = pd.Series(range(number_of_elements))
    ranked_set_manually = ranked_set_manually[['id', 'doc_id', 'protected_feature'] + list(range(number_of_features - 1))
                                      + ['judgement']]
    ranked_set_manually = ranked_set_manually.sort_values(['judgement'], ascending=[0])

    # the results should be the same (or approx close)
    assert np.allclose(ranked_set_manually['judgement'], ranked_set['judgement'])

