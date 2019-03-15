import pytest

import os
from fairsearchdeltr.deltr import Deltr
import pandas as pd
from io import StringIO


class MockDeltr(Deltr):
    def __init__(self, protected_feature: int, gamma: float, number_of_iterations=3000, learning_rate=0.001,
                 lambdaa=0.001, init_var=0.01):
        super().__init__(protected_feature, gamma, number_of_iterations, learning_rate, lambdaa, init_var)

        self.losses = []

    def _train_nn(self, tr, query_ids, feature_matrix, training_scores):
        omega, loss = tr.train_nn(query_ids, feature_matrix, training_scores, True)

        self.losses = tr.losses()

        return omega, loss


def test_create_deltr():
    assert Deltr(1, 1)
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
        Deltr(1, None)
        assert False
    except:
        assert True


def test_train_deltr():
    # data = pd.read_csv(StringIO("""
    # """), decimal=',')

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'testdata.csv'), decimal=',')

    d = MockDeltr(0, 1, 100)

    data['doc_id'] = pd.Series(range(50))

    data = data[['id', 'doc_id', 'gender', 'score', 'judgment']]

    d.train(data)

    assert d.losses != []

    if len(d.losses) > 1:
        current = d.losses[0]
        for loss in d.losses[1:]:
            assert loss < current
            current = loss
    else:
        assert d.losses[0]


def test_rank_deltr():
    pass